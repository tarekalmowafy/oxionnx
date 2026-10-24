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
    if hidden == 0 || x.is_empty() {
        return Ok(());
    }
    if x.len() % hidden != 0 || skip.is_empty() || skip.len() % hidden != 0 || x.len() % skip.len() != 0 {
        return Err(format!(
            "SkipSimplifiedLayerNormalization: skip ({} elems) does not tile input ({} elems, hidden {hidden})",
            skip.len(),
            x.len()
        ));
    }
    if gamma.len() != hidden || bias.is_some_and(|b| b.len() != hidden) {
        return Err(format!(
            "SkipSimplifiedLayerNormalization: gamma/bias must have {hidden} elements"
        ));
    }
    if out.len() < x.len()
        || sum_out.as_ref().is_some_and(|s| s.len() < x.len())
        || inv_std_out.as_ref().is_some_and(|s| s.len() < x.len() / hidden)
    {
        return Err("SkipSimplifiedLayerNormalization: output buffer too small".into());
    }
    let rows = x.len() / hidden;
    let skip_rows = skip.len() / hidden;
    let op = SendPtr(out.as_mut_ptr());
    let sp = sum_out.map(|s| SendPtr(s.as_mut_ptr()));
    let ip = inv_std_out.map(|s| SendPtr(s.as_mut_ptr()));
    let min_rows = (crate::par::MIN_ELEMS_PER_TASK / hidden).max(1);
    par_ranges(rows, min_rows, |lo, hi| {
        let (op, sp, ip) = (op, sp, ip);
        for r in lo..hi {
            let xi = &x[r * hidden..(r + 1) * hidden];
            let si = &skip[(r % skip_rows) * hidden..(r % skip_rows + 1) * hidden];
            // SAFETY: row `r` of each output is written by this task only.
            let o = unsafe { std::slice::from_raw_parts_mut(op.0.add(r * hidden), hidden) };
            // Stage the sum in `o`, then normalise it in place.
            let mut lanes = [0.0f32; 8];
            let mut j = 0;
            while j + 8 <= hidden {
                for l in 0..8 {
                    let v = xi[j + l] + si[j + l] + bias.map_or(0.0, |b| b[j + l]);
                    o[j + l] = v;
                    lanes[l] += v * v;
                }
                j += 8;
            }
            let mut tail = 0.0f32;
            for jj in j..hidden {
                let v = xi[jj] + si[jj] + bias.map_or(0.0, |b| b[jj]);
                o[jj] = v;
                tail += v * v;
            }
            let inv = ((lanes.iter().sum::<f32>() + tail) / hidden as f32 + eps).sqrt().recip();
            if let Some(p) = sp {
                unsafe { std::slice::from_raw_parts_mut(p.0.add(r * hidden), hidden) }.copy_from_slice(o);
            }
            if let Some(p) = ip {
                unsafe { *p.0.add(r) = inv };
            }
            for (ov, &g) in o.iter_mut().zip(gamma) {
                *ov = *ov * inv * g;
            }
        }
    });
    Ok(())
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
    let half = rotary_dim / 2;
    let rows = if per_token_cache { batch * seq } else { cos.len() / half.max(1) };
    if cos.len() != sin.len() || (half > 0 && cos.len() % half != 0) {
        return Err("com.microsoft::RotaryEmbedding: cos/sin caches differ in size".into());
    }
    let single_offset = position_ids.len() == 1;
    if per_token_cache {
        if rows != batch * seq {
            return Err(format!(
                "com.microsoft::RotaryEmbedding: per-token cos cache has {rows} rows, expected {}",
                batch * seq
            ));
        }
    } else if single_offset {
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
    // Cache row for token (b, s); in range by the checks above.
    let cache_row = |b: usize, s: usize| -> usize {
        if per_token_cache {
            b * seq + s
        } else if single_offset {
            position_ids[0] as usize + s
        } else {
            position_ids[b * seq + s] as usize
        }
    };
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
            let r = cache_row(b, s);
            let (c, sn) = (&cos[r * half..(r + 1) * half], &sin[r * half..(r + 1) * half]);
            let xi = &x[hrow * head_size..(hrow + 1) * head_size];
            // SAFETY: head row `hrow` is written by this task only.
            let o = unsafe { std::slice::from_raw_parts_mut(op.0.add(hrow * head_size), head_size) };
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
            o[rotary_dim..].copy_from_slice(&xi[rotary_dim..]);
        }
    });
    Ok(())
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
    let AttnDims { batch, seq_q, seq_kv, q_heads, kv_heads, head_dim, scale } = dims;
    if kv_heads == 0 || q_heads % kv_heads != 0 || head_dim == 0 {
        return Err(format!("attention: invalid head geometry {dims:?}"));
    }
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
            if row_scale != 1.0 {
                row.iter_mut().for_each(|x| *x *= row_scale);
            }
            if let Some(bb) = bias {
                let [sb, sg, sr, ss, st] = bb.strides;
                let base = b * sb + g * sg + r * sr + (s0 + i) * ss;
                if st == 1 {
                    for (x, &y) in row.iter_mut().zip(&bb.data[base..base + seq_kv]) {
                        *x += y;
                    }
                } else {
                    for (j, x) in row.iter_mut().enumerate() {
                        *x += bb.data[base + j * st];
                    }
                }
            }
            softmax_row(row);
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
