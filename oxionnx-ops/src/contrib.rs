//! Kernels for ONNX Runtime contrib ops (`com.microsoft` domain) and the fused
//! attention op the optimizer emits.
//!
//! * [`skip_rms_norm_into`] — `SkipSimplifiedLayerNormalization`: residual add
//!   (+ optional bias) fused with RMSNorm, optionally also returning the sum.
//! * [`ms_rotary_into`] — `com.microsoft::RotaryEmbedding` on `[B, S, H*D]` or
//!   `[B, N, S, D]` input, with a position offset or per-token position ids.
//! * [`attention_into`] — scaled dot-product attention with grouped KV heads
//!   that reads Q/K/V straight from their `[B, S, heads*D]` projections and
//!   writes `[B, S, Hq*D]`, so none of the head-split transposes, the score
//!   tensor or the probability tensor are materialised as graph tensors.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use crate::gemm_packed::PackedB;

/// Raw output pointer shared by tasks that write disjoint regions.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Run `f(lo, hi)` over `0..n` in chunks of at least `min_per_task`, in parallel
/// when there is more than one chunk and more than one thread.
fn par_ranges(n: usize, min_per_task: usize, f: impl Fn(usize, usize) + Sync + Send) {
    let t = crate::par::threads();
    let per = min_per_task.max(n.div_ceil(4 * t.max(1))).max(1);
    let tasks = n.div_ceil(per);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if tasks > 1 && t > 1 {
        use rayon::prelude::*;
        (0..tasks).into_par_iter().for_each(|i| f(i * per, ((i + 1) * per).min(n)));
        return;
    }
    let _ = tasks;
    f(0, n);
}

// ── SkipSimplifiedLayerNormalization ─────────────────────────────────────────

/// `sum = x + skip (+ bias)`; `out = sum / sqrt(mean(sum^2) + eps) * gamma`.
///
/// `skip` is either the shape of `x` or a trailing slice of it (e.g. `[S, H]`
/// against `[B, S, H]`), repeated over the leading rows. `sum_out` receives
/// `sum` and `inv_std_out` one `1/rms` per row, when requested.
#[allow(clippy::too_many_arguments)]
pub fn skip_rms_norm_into(
    x: &[f32],
    skip: &[f32],
    gamma: &[f32],
    bias: Option<&[f32]>,
    hidden: usize,
    eps: f32,
    out: &mut [f32],
    sum_out: Option<&mut [f32]>,
    inv_std_out: Option<&mut [f32]>,
) -> Result<(), String> {
    let args = crate::norm_rows::NormRows { x, skip: Some(skip), bias, gamma, beta: None, hidden, eps, rms: true };
    crate::norm_rows::norm_rows_into(&args, crate::norm_rows::NormRowsOut { out, sum: sum_out, mean: None, inv_std: inv_std_out })
        .map_err(|e| format!("SkipSimplifiedLayerNormalization: {e}"))
}

// ── com.microsoft::RotaryEmbedding ───────────────────────────────────────────

/// Geometry of one `com.microsoft::RotaryEmbedding` call.
#[derive(Clone, Copy, Debug)]
pub struct RotaryDims {
    pub batch: usize,
    pub seq: usize,
    pub num_heads: usize,
    pub head_size: usize,
    /// Leading part of each head that is rotated (`<= head_size`, even).
    pub rotary_dim: usize,
    /// `true` for `[B, N, S, D]` input, `false` for `[B, S, N*D]`.
    pub bnsd: bool,
    pub interleaved: bool,
}

impl RotaryDims {
    /// Derive the geometry from the input shape, the cos cache's last dim and the
    /// op's `num_heads` / `rotary_embedding_dim` attributes (0 = absent), following
    /// ONNX Runtime's `RotaryEmbedding` contrib kernel.
    pub fn new(
        shape: &[usize],
        cos_half: usize,
        num_heads_attr: usize,
        rotary_dim_attr: usize,
        interleaved: bool,
    ) -> Result<Self, String> {
        let err = |m: &str| Err(format!("com.microsoft::RotaryEmbedding: {m} (input shape {shape:?})"));
        let (batch, seq, num_heads, head_size, bnsd) = match *shape {
            [b, s, hidden] => {
                let head_size = if num_heads_attr > 0 {
                    if hidden % num_heads_attr != 0 {
                        return err("hidden size not divisible by num_heads");
                    }
                    hidden / num_heads_attr
                } else if rotary_dim_attr > 0 {
                    return err("rotary_embedding_dim requires num_heads for 3-D input");
                } else {
                    2 * cos_half
                };
                if head_size == 0 || hidden % head_size != 0 {
                    return err("hidden size not divisible by head size");
                }
                (b, s, hidden / head_size, head_size, false)
            }
            [b, n, s, d] => (b, s, n, d, true),
            _ => return err("input must be 3-D or 4-D"),
        };
        let rotary_dim = if rotary_dim_attr > 0 { rotary_dim_attr } else { 2 * cos_half };
        if rotary_dim > head_size || rotary_dim % 2 != 0 || rotary_dim / 2 != cos_half {
            return err(&format!(
                "rotary dim {rotary_dim} does not match head size {head_size} / cos cache half-dim {cos_half}"
            ));
        }
        Ok(Self { batch, seq, num_heads, head_size, rotary_dim, bnsd, interleaved })
    }
}

/// Apply rotary position embedding.
///
/// `position_ids` holds either one offset (`pos = offset + s`) or `B*S` ids.
/// `cos`/`sin` are `[max_pos, rotary_dim/2]` caches indexed by position, or —
/// when `per_token_cache` — `[B, S, rotary_dim/2]` rows indexed by token.
pub fn ms_rotary_into(
    x: &[f32],
    position_ids: &[f32],
    cos: &[f32],
    sin: &[f32],
    per_token_cache: bool,
    dims: RotaryDims,
    out: &mut [f32],
) -> Result<(), String> {
    let RotaryDims { batch, seq, num_heads, head_size, rotary_dim, bnsd, interleaved } = dims;
    let n = batch * seq * num_heads * head_size;
    if x.len() != n || out.len() < n {
        return Err(format!("com.microsoft::RotaryEmbedding: buffer sizes do not match {dims:?}"));
    }
    let cache = RotaryCache::new(position_ids, cos, sin, per_token_cache, batch, seq, rotary_dim / 2)?;
    let total_heads = batch * seq * num_heads;
    let op = SendPtr(out.as_mut_ptr());
    let min_heads = (crate::par::MIN_ELEMS_PER_TASK / head_size.max(1)).max(1);
    par_ranges(total_heads, min_heads, |lo, hi| {
        let op = op;
        for hrow in lo..hi {
            // Head rows are laid out [B, S, N] (3-D) or [B, N, S] (4-D).
            let (b, s) = if bnsd {
                (hrow / (num_heads * seq), hrow % seq)
            } else {
                (hrow / (seq * num_heads), (hrow / num_heads) % seq)
            };
            let (c, sn) = cache.row(b, s);
            let xi = &x[hrow * head_size..(hrow + 1) * head_size];
            // SAFETY: head row `hrow` is written by this task only.
            let o = unsafe { std::slice::from_raw_parts_mut(op.0.add(hrow * head_size), head_size) };
            rotate_head(xi, c, sn, rotary_dim, interleaved, o);
        }
    });
    Ok(())
}

/// Validated cos/sin cache lookup for `com.microsoft::RotaryEmbedding`.
#[derive(Clone, Copy)]
pub struct RotaryCache<'a> {
    position_ids: &'a [f32],
    cos: &'a [f32],
    sin: &'a [f32],
    per_token: bool,
    seq: usize,
    half: usize,
}

impl<'a> RotaryCache<'a> {
    /// `cos`/`sin` are `[max_pos, half]` (or `[B, S, half]` when `per_token`);
    /// `position_ids` is one offset or `[B, S]`. Checks every position is in range.
    pub fn new(
        position_ids: &'a [f32],
        cos: &'a [f32],
        sin: &'a [f32],
        per_token: bool,
        batch: usize,
        seq: usize,
        half: usize,
    ) -> Result<Self, String> {
        let rows = if per_token { batch * seq } else { cos.len() / half.max(1) };
        if cos.len() != sin.len() || (half > 0 && cos.len() % half != 0) {
            return Err("com.microsoft::RotaryEmbedding: cos/sin caches differ in size".into());
        }
        if per_token {
            if rows != batch * seq {
                return Err(format!(
                    "com.microsoft::RotaryEmbedding: per-token cos cache has {rows} rows, expected {}",
                    batch * seq
                ));
            }
        } else if position_ids.len() == 1 {
            if seq > 0 && (position_ids[0] < 0.0 || position_ids[0] as usize + seq > rows) {
                return Err(format!(
                    "com.microsoft::RotaryEmbedding: positions {}..{} outside cos cache ({rows} rows)",
                    position_ids[0],
                    position_ids[0] as usize + seq
                ));
            }
        } else if position_ids.len() != batch * seq {
            return Err(format!(
                "com.microsoft::RotaryEmbedding: position_ids has {} elements, expected 1 or {}",
                position_ids.len(),
                batch * seq
            ));
        } else if let Some(&p) = position_ids.iter().find(|&&p| p < 0.0 || p as usize >= rows) {
            return Err(format!("com.microsoft::RotaryEmbedding: position {p} outside cos cache ({rows} rows)"));
        }
        Ok(Self { position_ids, cos, sin, per_token, seq, half })
    }

    /// The cos and sin rows for token `(b, s)`.
    #[inline]
    pub fn row(&self, b: usize, s: usize) -> (&'a [f32], &'a [f32]) {
        let r = if self.per_token {
            b * self.seq + s
        } else if self.position_ids.len() == 1 {
            self.position_ids[0] as usize + s
        } else {
            self.position_ids[b * self.seq + s] as usize
        };
        let h = self.half;
        (&self.cos[r * h..(r + 1) * h], &self.sin[r * h..(r + 1) * h])
    }
}

/// Rotate one head `xi` into `o` (the first `rotary_dim` elements; the rest are copied).
#[inline]
pub fn rotate_head(xi: &[f32], c: &[f32], sn: &[f32], rotary_dim: usize, interleaved: bool, o: &mut [f32]) {
    let half = rotary_dim / 2;
    if interleaved {
        for i in 0..half {
            let (x1, x2) = (xi[2 * i], xi[2 * i + 1]);
            o[2 * i] = x1 * c[i] - x2 * sn[i];
            o[2 * i + 1] = x2 * c[i] + x1 * sn[i];
        }
    } else {
        let (x1, x2) = (&xi[..half], &xi[half..rotary_dim]);
        let (o1, o2) = o[..rotary_dim].split_at_mut(half);
        for i in 0..half {
            o1[i] = x1[i] * c[i] - x2[i] * sn[i];
            o2[i] = x2[i] * c[i] + x1[i] * sn[i];
        }
    }
    o[rotary_dim..xi.len()].copy_from_slice(&xi[rotary_dim..]);
}

// ── Fused attention ──────────────────────────────────────────────────────────

/// Geometry of one fused attention call.
#[derive(Clone, Copy, Debug)]
pub struct AttnDims {
    pub batch: usize,
    /// Query length.
    pub seq_q: usize,
    /// Key/value length.
    pub seq_kv: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

/// Additive attention bias, broadcast over `[B, G, R, S, T]` where query head
/// `h = g * R + r` attends with KV head `g` (`R = q_heads / kv_heads`).
/// `strides` are element strides per axis, 0 on broadcast axes.
#[derive(Clone, Copy)]
pub struct AttnBias<'a> {
    pub data: &'a [f32],
    pub strides: [usize; 5],
}

/// `out[b, s, h*D..] = softmax(scale * q·kᵀ + bias) · v` per query head `h`.
///
/// `q` is `[B, S, Hq*D]`, `k`/`v` are `[B, T, Hkv*D]`, `out` is `[B, S, Hq*D]`.
pub fn attention_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: Option<AttnBias<'_>>,
    dims: AttnDims,
    out: &mut [f32],
) -> Result<(), String> {
    attention_ext_into(q, k, v, bias, AttnExt::default(), dims, out)
}

/// Softmax sink of [`AttnExt`]: an extra logit that takes part in the
/// normalisation but has no value row (`exp(x_j) / (exp(sink) + sum exp(x))`).
#[derive(Clone, Copy, Debug, Default)]
pub enum AttnSink<'a> {
    #[default]
    None,
    /// "Smooth softmax": a sink logit of 0 for every head.
    Zero,
    /// One sink logit per query head.
    PerHead(&'a [f32]),
}

/// Per-row key windows and score transforms for [`attention_ext_into`]
/// (the `com.microsoft::GroupQueryAttention` semantics).
///
/// Query row `s` of batch `b` attends to keys `lo..hi` with
/// `hi = min(causal_offset[b] + s + 1, kv_len[b])` (just `kv_len[b]` without
/// `causal_offset`; `kv_len` defaults to all `seq_kv` keys) and
/// `lo = hi - local_window` when the window is shorter than `hi`. Keys outside
/// the window get probability 0. Scores are `softcap * tanh(x / softcap)`
/// (when `softcap > 0`) of the scaled product, then the bias is added.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnExt<'a> {
    pub kv_len: Option<&'a [usize]>,
    pub causal_offset: Option<&'a [usize]>,
    pub local_window: Option<usize>,
    pub softcap: f32,
    pub sink: AttnSink<'a>,
}

impl AttnExt<'_> {
    fn is_plain(&self) -> bool {
        self.kv_len.is_none()
            && self.causal_offset.is_none()
            && self.local_window.is_none()
            && self.softcap <= 0.0
            && matches!(self.sink, AttnSink::None)
    }

    /// Key window `lo..hi` of query row `s` in batch `b`.
    #[inline]
    fn window(&self, b: usize, s: usize, seq_kv: usize) -> (usize, usize) {
        let len = self.kv_len.map_or(seq_kv, |l| l[b].min(seq_kv));
        let hi = self.causal_offset.map_or(len, |o| (o[b] + s + 1).min(len));
        let lo = match self.local_window {
            Some(w) if hi > w => hi - w,
            _ => 0,
        };
        (lo, hi)
    }
}

/// [`attention_into`] with the key windows, soft-capping and softmax sink of `ext`.
pub fn attention_ext_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: Option<AttnBias<'_>>,
    ext: AttnExt<'_>,
    dims: AttnDims,
    out: &mut [f32],
) -> Result<(), String> {
    let AttnDims { batch, seq_q, seq_kv, q_heads, kv_heads, head_dim, scale } = dims;
    if kv_heads == 0 || q_heads % kv_heads != 0 || head_dim == 0 {
        return Err(format!("attention: invalid head geometry {dims:?}"));
    }
    if ext.kv_len.is_some_and(|l| l.len() < batch)
        || ext.causal_offset.is_some_and(|o| o.len() < batch)
        || matches!(ext.sink, AttnSink::PerHead(h) if h.len() != q_heads)
    {
        return Err(format!("attention: per-batch lengths or per-head sinks do not match {dims:?}"));
    }
    let plain = ext.is_plain();
    let (qrs, krs) = (q_heads * head_dim, kv_heads * head_dim);
    if q.len() != batch * seq_q * qrs
        || k.len() != batch * seq_kv * krs
        || v.len() != k.len()
        || out.len() < q.len()
    {
        return Err(format!("attention: buffer sizes do not match {dims:?}"));
    }
    if let Some(bb) = bias {
        let max = [batch, kv_heads, q_heads / kv_heads, seq_q, seq_kv]
            .iter()
            .zip(bb.strides)
            .map(|(&d, s)| d.saturating_sub(1) * s)
            .sum::<usize>();
        if seq_q * seq_kv > 0 && max >= bb.data.len() {
            return Err("attention: bias is smaller than its broadcast strides imply".into());
        }
    }
    if batch * seq_q == 0 {
        return Ok(());
    }
    let group = q_heads / kv_heads;
    // Query rows per task: enough tasks for every thread, at most 64 rows so the
    // score block (rows x T) stays in L2.
    let n_bh = batch * q_heads;
    let t = crate::par::threads().max(1);
    let blocks = (4 * t).div_ceil(n_bh).clamp(1, seq_q.div_ceil(8).max(1));
    let rows_per = seq_q.div_ceil(blocks).clamp(1, 64);
    let blocks = seq_q.div_ceil(rows_per);
    // x86: pack Kᵀ and V once per (batch, KV head) for the packed AVX-512/AVX2
    // GEMM, reused by the R query heads of the group and all their row blocks.
    #[cfg(target_arch = "x86_64")]
    let packs: Option<Vec<(PackedB, PackedB)>> = crate::gemm_packed::available().then(|| {
        let pack = |bg: usize| {
            let (b, g) = (bg / kv_heads, bg % kv_heads);
            let base = b * seq_kv * krs + g * head_dim;
            (
                PackedB::pack_strided(&k[base..], head_dim, seq_kv, 1, krs),
                PackedB::pack_strided(&v[base..], seq_kv, head_dim, krs, 1),
            )
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if t > 1 && batch * kv_heads > 1 {
            use rayon::prelude::*;
            return (0..batch * kv_heads).into_par_iter().map(pack).collect();
        }
        (0..batch * kv_heads).map(pack).collect()
    });
    #[cfg(not(target_arch = "x86_64"))]
    let packs: Option<Vec<()>> = None;
    let op = SendPtr(out.as_mut_ptr());
    let work = |task: usize, scores: &mut Vec<f32>| {
        let op = op;
        let (bh, blk) = (task / blocks, task % blocks);
        let (b, h) = (bh / q_heads, bh % q_heads);
        let (g, r) = (h / group, h % group);
        let s0 = blk * rows_per;
        let m = rows_per.min(seq_q - s0);
        scores.resize(m * seq_kv, 0.0);
        let qp = q[(b * seq_q + s0) * qrs + h * head_dim..].as_ptr();
        let kp = k[b * seq_kv * krs + g * head_dim..].as_ptr();
        let vp = v[b * seq_kv * krs + g * head_dim..].as_ptr();
        let pk = packs.as_ref().map(|p| &p[b * kv_heads + g]);
        // SAFETY: strides stay inside q/k/v (checked sizes above); every task
        // writes rows s0..s0+m of head h only.
        let mut row_scale = scale;
        match pk {
            #[cfg(target_arch = "x86_64")]
            Some((kt, _)) => unsafe {
                // scores[m, T] = Q[m, D] · Kᵀ; `scale` is applied with the bias below.
                crate::gemm_packed::gemm_strided_serial(m, qp, qrs, kt, scores.as_mut_ptr(), seq_kv);
            },
            _ => unsafe {
                // scores[m, T] = scale * Q[m, D] · K[T, D]ᵀ
                row_scale = 1.0;
                matrixmultiply::sgemm(
                    m, head_dim, seq_kv, scale,
                    qp, qrs as isize, 1,
                    kp, 1, krs as isize,
                    0.0, scores.as_mut_ptr(), seq_kv as isize, 1,
                );
            },
        }
        for (i, row) in scores.chunks_mut(seq_kv).enumerate() {
            let (lo, hi) = if plain { (0, seq_kv) } else { ext.window(b, s0 + i, seq_kv) };
            let win = &mut row[lo..hi];
            if row_scale != 1.0 {
                win.iter_mut().for_each(|x| *x *= row_scale);
            }
            if ext.softcap > 0.0 {
                let (c, inv) = (ext.softcap, ext.softcap.recip());
                win.iter_mut().for_each(|x| *x = c * (*x * inv).tanh());
            }
            if let Some(bb) = bias {
                let [sb, sg, sr, ss, st] = bb.strides;
                let base = b * sb + g * sg + r * sr + (s0 + i) * ss + lo * st;
                if st == 1 {
                    for (x, &y) in win.iter_mut().zip(&bb.data[base..base + (hi - lo)]) {
                        *x += y;
                    }
                } else {
                    for (j, x) in win.iter_mut().enumerate() {
                        *x += bb.data[base + j * st];
                    }
                }
            }
            match ext.sink {
                AttnSink::None if !win.is_empty() => softmax_row(win),
                AttnSink::None => {}
                AttnSink::Zero => softmax_sink_row(win, 0.0),
                AttnSink::PerHead(sinks) => softmax_sink_row(win, sinks[h]),
            }
            row[..lo].fill(0.0);
            row[hi..].fill(0.0);
        }
        let dst = unsafe { op.0.add((b * seq_q + s0) * qrs + h * head_dim) };
        match pk {
            #[cfg(target_arch = "x86_64")]
            Some((_, vp)) => unsafe {
                // out[m, D] = P[m, T] · V[T, D]
                crate::gemm_packed::gemm_strided_serial(m, scores.as_ptr(), seq_kv, vp, dst, qrs);
            },
            _ => unsafe {
                matrixmultiply::sgemm(
                    m, seq_kv, head_dim, 1.0,
                    scores.as_ptr(), seq_kv as isize, 1,
                    vp, krs as isize, 1,
                    0.0, dst, qrs as isize, 1,
                );
            },
        }
    };
    let tasks = n_bh * blocks;
    let flops = 4.0 * (n_bh * seq_q * seq_kv * head_dim) as f64;
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if tasks > 1 && t > 1 && flops > 2.0 * 64.0 * 1024.0 {
        use rayon::prelude::*;
        (0..tasks).into_par_iter().for_each_init(Vec::new, |scores, task| work(task, scores));
        return Ok(());
    }
    let _ = flops;
    let mut scores = Vec::new();
    for task in 0..tasks {
        work(task, &mut scores);
    }
    Ok(())
}

/// Softmax with an extra `sink` logit in the denominator (MLAS "smooth softmax").
fn softmax_sink_row(row: &mut [f32], sink: f32) {
    let max = row.iter().copied().fold(sink, f32::max);
    let mut sum = (sink - max).exp();
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = sum.recip();
    row.iter_mut().for_each(|x| *x *= inv);
}

fn softmax_row(row: &mut [f32]) {
    #[cfg(feature = "simd")]
    {
        crate::simd_ops::simd_softmax_inplace(row);
    }
    #[cfg(not(feature = "simd"))]
    {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        let inv = sum.recip();
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "elem {i}: {x} vs {y}");
        }
    }

    #[test]
    fn skip_rms_norm_matches_reference() {
        let (rows, h) = (37, 24);
        let (x, skip, gamma, bias) = (rand(rows * h, 1), rand(h, 2), rand(h, 3), rand(h, 4));
        let skip_full: Vec<f32> = (0..rows * h).map(|i| skip[i % h]).collect();
        for with_bias in [false, true] {
            let b = with_bias.then_some(&bias[..]);
            let mut want_sum = vec![0.0; rows * h];
            let mut want = vec![0.0; rows * h];
            let mut want_inv = vec![0.0; rows];
            for r in 0..rows {
                let mut ss = 0.0;
                for j in 0..h {
                    let v = x[r * h + j] + skip_full[r * h + j] + b.map_or(0.0, |b| b[j]);
                    want_sum[r * h + j] = v;
                    ss += v * v;
                }
                want_inv[r] = 1.0 / (ss / h as f32 + 1e-6).sqrt();
                for j in 0..h {
                    want[r * h + j] = want_sum[r * h + j] * want_inv[r] * gamma[j];
                }
            }
            for skip_in in [&skip[..], &skip_full[..]] {
                let (mut o, mut s, mut inv) = (vec![0.0; rows * h], vec![0.0; rows * h], vec![0.0; rows]);
                skip_rms_norm_into(&x, skip_in, &gamma, b, h, 1e-6, &mut o, Some(&mut s), Some(&mut inv)).unwrap();
                close(&o, &want, 1e-5);
                close(&s, &want_sum, 1e-6);
                close(&inv, &want_inv, 1e-5);
                let mut o2 = vec![0.0; rows * h];
                skip_rms_norm_into(&x, skip_in, &gamma, b, h, 1e-6, &mut o2, None, None).unwrap();
                close(&o2, &want, 1e-5);
            }
        }
    }

    #[test]
    fn rotary_matches_reference() {
        let (b, s, n, d, max_pos) = (2, 5, 3, 8, 16);
        for (rot, interleaved, bnsd, offset_ids) in [
            (8, false, false, true),
            (8, true, false, false),
            (4, false, true, false),
            (8, false, true, true),
        ] {
            let half = rot / 2;
            let x = rand(b * s * n * d, 7);
            let cos = rand(max_pos * half, 8);
            let sin = rand(max_pos * half, 9);
            let pos: Vec<f32> = if offset_ids { vec![3.0] } else { (0..b * s).map(|i| ((i * 7) % max_pos) as f32).collect() };
            let shape: Vec<usize> = if bnsd { vec![b, n, s, d] } else { vec![b, s, n * d] };
            let dims = RotaryDims::new(&shape, half, if rot < d || !bnsd { n } else { 0 }, if rot < d { rot } else { 0 }, interleaved).unwrap();
            let mut got = vec![f32::NAN; x.len()];
            ms_rotary_into(&x, &pos, &cos, &sin, false, dims, &mut got).unwrap();
            let mut want = x.clone();
            for bi in 0..b {
                for si in 0..s {
                    let p = if offset_ids { 3 + si } else { pos[bi * s + si] as usize };
                    for ni in 0..n {
                        let base = if bnsd { ((bi * n + ni) * s + si) * d } else { ((bi * s + si) * n + ni) * d };
                        for i in 0..half {
                            let (c, sn) = (cos[p * half + i], sin[p * half + i]);
                            let (i1, i2) = if interleaved { (base + 2 * i, base + 2 * i + 1) } else { (base + i, base + i + half) };
                            want[i1] = x[i1] * c - x[i2] * sn;
                            want[i2] = x[i2] * c + x[i1] * sn;
                        }
                    }
                }
            }
            close(&got, &want, 1e-6);
        }
    }

    #[test]
    fn attention_matches_reference() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for (b, s, t, hq, hkv, d) in [(2, 7, 7, 4, 2, 16), (1, 70, 70, 2, 2, 8), (3, 5, 9, 6, 1, 4)] {
            let q = rand(b * s * hq * d, 11);
            let k = rand(b * t * hkv * d, 12);
            let v = rand(b * t * hkv * d, 13);
            let group = hq / hkv;
            // Bias over [B, 1, 1, S, T] (per-batch key mask + causal-ish pattern).
            let bias: Vec<f32> = (0..b * s * t)
                .map(|i| if (i % t) > (i / t) % s + 2 { -1e9 } else { 0.1 * (i % 5) as f32 })
                .collect();
            let strides = [s * t, 0, 0, t, 1];
            let dims = AttnDims { batch: b, seq_q: s, seq_kv: t, q_heads: hq, kv_heads: hkv, head_dim: d, scale: 0.3 };
            let mut got = vec![f32::NAN; q.len()];
            pool.install(|| {
                attention_into(&q, &k, &v, Some(AttnBias { data: &bias, strides }), dims, &mut got).unwrap()
            });
            let mut want = vec![0.0; q.len()];
            for bi in 0..b {
                for h in 0..hq {
                    let g = h / group;
                    for si in 0..s {
                        let qr = &q[(bi * s + si) * hq * d + h * d..][..d];
                        let mut sc: Vec<f32> = (0..t)
                            .map(|ti| {
                                let kr = &k[(bi * t + ti) * hkv * d + g * d..][..d];
                                0.3 * qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() + bias[bi * s * t + si * t + ti]
                            })
                            .collect();
                        let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let sum: f32 = sc.iter_mut().map(|x| { *x = (*x - mx).exp(); *x }).sum();
                        for di in 0..d {
                            want[(bi * s + si) * hq * d + h * d + di] = (0..t)
                                .map(|ti| sc[ti] / sum * v[(bi * t + ti) * hkv * d + g * d + di])
                                .sum();
                        }
                    }
                }
            }
            close(&got, &want, 1e-5);
        }
    }
}
