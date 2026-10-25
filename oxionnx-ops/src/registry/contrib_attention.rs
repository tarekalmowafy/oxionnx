//! ONNX Runtime's attention contrib ops (`com.microsoft` domain), on top of the
//! fused attention kernel in [`crate::contrib`].
//!
//! * [`MsMultiHeadAttentionOp`] — `com.microsoft::MultiHeadAttention`. Its inputs
//!   are `(query, key, value, bias, key_padding_mask, attention_bias, past_key,
//!   past_value, past_sequence_length, cache_indirection)`, unlike oxionnx's own
//!   default-domain `MultiHeadAttention` (which projects Q/K/V itself), so the
//!   loader gives it the domain-qualified name `com.microsoft.MultiHeadAttention`.
//! * [`GroupQueryAttentionOp`] — `com.microsoft::GroupQueryAttention` (exists only
//!   in that domain, so it keeps its name).
//!
//! Semantics follow ONNX Runtime's CPU kernels (`contrib_ops/cpu/bert/
//! multihead_attention.cc`, `attention_helper.h`, `group_query_attention.cc`,
//! `gqa_attention_base.h`).

use std::borrow::Cow;

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::contrib::{self, AttnBias, AttnDims, AttnExt, AttnSink, RotaryDims};

type Buf<'a> = Cow<'a, [f32]>;

/// Whether output `i` of the node is consumed (present and named).
fn wanted(ctx: &OpContext<'_>, i: usize) -> bool {
    ctx.node.outputs.get(i).is_some_and(|n| !n.is_empty())
}

/// Size `slot` to `shape`; its contents are fully overwritten by the caller.
fn size_slot(slot: &mut Tensor, shape: &[usize]) {
    let n: usize = shape.iter().product();
    if slot.data.len() != n {
        slot.data.resize(n, 0.0);
    }
    slot.shape.clear();
    slot.shape.extend_from_slice(shape);
}

/// Empty (`[0]`) slots positionally aligned with `ctx.node.outputs`, for `execute`.
fn empty_slots(ctx: &OpContext<'_>) -> Vec<Tensor> {
    ctx.node.outputs.iter().map(|_| Tensor { data: vec![], shape: vec![0] }).collect()
}

/// `[B, N, T, H]` -> `[B, T, N*H]`.
fn bnsh_to_bsnh(src: &[f32], b: usize, n: usize, t: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; b * t * n * h];
    for bb in 0..b {
        for nn in 0..n {
            for tt in 0..t {
                let s = ((bb * n + nn) * t + tt) * h;
                let d = ((bb * t + tt) * n + nn) * h;
                out[d..d + h].copy_from_slice(&src[s..s + h]);
            }
        }
    }
    out
}

/// `[B, T, N*H]` -> `[B, N, T, H]` into `dst`.
fn bsnh_to_bnsh_into(src: &[f32], b: usize, t: usize, n: usize, h: usize, dst: &mut [f32]) {
    for bb in 0..b {
        for tt in 0..t {
            for nn in 0..n {
                let s = ((bb * t + tt) * n + nn) * h;
                let d = ((bb * n + nn) * t + tt) * h;
                dst[d..d + h].copy_from_slice(&src[s..s + h]);
            }
        }
    }
}

/// Split interleaved per-token rows of `widths` columns each into separate
/// `[rows, width]` buffers (packed QKV/KV layouts).
fn split_rows(src: &[f32], rows: usize, widths: &[usize]) -> Vec<Vec<f32>> {
    let total: usize = widths.iter().sum();
    let mut outs: Vec<Vec<f32>> = widths.iter().map(|&w| Vec::with_capacity(rows * w)).collect();
    for r in 0..rows {
        let mut off = r * total;
        for (o, &w) in outs.iter_mut().zip(widths) {
            o.extend_from_slice(&src[off..off + w]);
            off += w;
        }
    }
    outs
}

/// Add a per-column bias `[w]` to every row of `x`.
fn add_row_bias(x: &mut [f32], bias: &[f32]) {
    for row in x.chunks_mut(bias.len()) {
        for (a, &c) in row.iter_mut().zip(bias) {
            *a += c;
        }
    }
}

/// `scale` attribute, 0 meaning `1/sqrt(head_size)`.
fn scale_attr(ctx: &OpContext<'_>, head_size: usize) -> f32 {
    let s = ctx.attrs().f("scale", 0.0);
    if s == 0.0 {
        1.0 / (head_size as f32).sqrt()
    } else {
        s
    }
}

/// Element strides of a `[B|1, H|1, S, T]` attention bias in the kernel's
/// `[B, G, R, S, T]` score space (`h = g * R + r`).
fn bhst_strides(shape: &[usize], group: usize, s: usize, t: usize) -> [usize; 5] {
    let sb = if shape[0] == 1 { 0 } else { shape[1] * s * t };
    let sh = if shape[1] == 1 { 0 } else { s * t };
    [sb, sh * group, sh, t, 1]
}

/// Check an attention bias is `[B|1, H|1, S, T]`.
fn check_bhst(bias: &Tensor, b: usize, heads: usize, s: usize, t: usize) -> Result<(), String> {
    let ok = bias.ndim() == 4
        && (bias.shape[0] == b || bias.shape[0] == 1)
        && (bias.shape[1] == heads || bias.shape[1] == 1)
        && bias.shape[2] == s
        && bias.shape[3] == t;
    if ok {
        Ok(())
    } else {
        Err(format!("attention_bias {:?} is not [{b}|1, {heads}|1, {s}, {t}]", bias.shape))
    }
}

// ── com.microsoft::MultiHeadAttention ────────────────────────────────────────

/// `com.microsoft::MultiHeadAttention`.
///
/// Inputs: `query`, `key`, `value`, `bias`, `key_padding_mask`, `attention_bias`,
/// `past_key`, `past_value` (`past_sequence_length` and `cache_indirection`,
/// the shared-buffer / beam-search forms, are not supported).
/// Attributes: `num_heads`, `scale`, `mask_filter_value`, `unidirectional`.
/// Outputs: `output` `[B, S, D]`, `present_key`/`present_value` `[B, N, T, H]`
/// (`qk` is not supported).
///
/// Q/K/V formats: `[B, S, D]` / `[B, L, D]` / `[B, L, D]`; `[B, S, D]` with K/V
/// `[B, N, L, H]` (cross attention; K/V bias taken as zero, as in ORT); packed
/// KV `key` `[B, L, N, 2, H]` (no `bias`); packed QKV `query` `[B, S, N, 3, H]`.
/// `key_padding_mask` is `[B]` (key lengths), `[2B]` (ends, then starts),
/// `[B, T]` or `[B, S, T]` (1 = keep), masked scores get `mask_filter_value`
/// added; `unidirectional` scores (key after `past + s`) get `f32::MIN`; then
/// `attention_bias` `[B|1, N|1, S, T]` is added.
pub struct MsMultiHeadAttentionOp;

/// Padding-mask forms of `com.microsoft::MultiHeadAttention`'s `key_padding_mask`.
enum PadMask<'a> {
    None,
    /// Valid keys are `start[b]..end[b]` (`start` = 0 without the second half).
    Ends(&'a [f32], Option<&'a [f32]>),
    /// `[B, T]`, nonzero = keep.
    Keys(&'a [f32]),
    /// `[B, S, T]`, nonzero = keep.
    Full(&'a [f32]),
}

impl PadMask<'_> {
    /// Additive mask value for query `s`, key `t` of batch `b`.
    #[inline]
    fn value(&self, b: usize, s: usize, t: usize, seq: usize, total: usize, mfv: f32) -> f32 {
        let keep = match *self {
            PadMask::None => true,
            PadMask::Keys(m) => m[b * total + t] > 0.0,
            PadMask::Full(m) => m[(b * seq + s) * total + t] > 0.0,
            PadMask::Ends(ends, starts) => {
                let clampi = |x: f32| (x as i64).clamp(0, total as i64) as usize;
                t < clampi(ends[b]) && starts.is_none_or(|st| t >= clampi(st[b]))
            }
        };
        if keep {
            0.0
        } else {
            mfv
        }
    }
}

impl MsMultiHeadAttentionOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let name = &ctx.node.name;
        let bad = |m: String| OnnxError::ShapeMismatch(format!("com.microsoft::MultiHeadAttention '{name}': {m}"));
        let unsupported = |m: &str| OnnxError::Unsupported(format!("com.microsoft::MultiHeadAttention '{name}': {m}"));
        let attrs = ctx.attrs();
        let n = attrs.i("num_heads", 0);
        if n <= 0 {
            return Err(bad("num_heads must be positive".into()));
        }
        let n = n as usize;
        let mfv = attrs.f("mask_filter_value", -10000.0);
        let causal = attrs.i("unidirectional", 0) == 1;
        if ctx.optional_input(8).is_some() || ctx.optional_input(9).is_some() {
            return Err(unsupported("past_sequence_length / cache_indirection (shared KV buffer) inputs"));
        }
        if wanted(ctx, 3) {
            return Err(unsupported("the qk output"));
        }
        let query = ctx.input(0)?;
        let (key, value, bias) = (ctx.optional_input(1), ctx.optional_input(2), ctx.optional_input(3));
        if query.ndim() < 3 {
            return Err(bad(format!("query {:?} must be 3-D or 5-D", query.shape)));
        }
        let (b, s) = (query.shape[0], query.shape[1]);
        let qdim = |d: usize| if d % n == 0 { Ok(d / n) } else { Err(bad(format!("hidden size {d} not divisible by {n} heads"))) };
        // Q/K/V as [B, S|L, N*H]; `kv_bias`: K/V take their part of `bias`.
        let (mut q, mut k, mut v, l, h, kv_bias, kv_bnsh): (Buf, Buf, Buf, usize, usize, bool, bool) =
            match (query.ndim(), key, value) {
                (3, Some(kt), Some(vt)) if kt.ndim() == 3 => {
                    let h = qdim(query.shape[2])?;
                    if kt.shape[0] != b || kt.shape[2] != query.shape[2] || vt.shape != kt.shape {
                        return Err(bad(format!(
                            "key {:?} / value {:?} do not match query {:?} (value hidden size must equal the query's)",
                            kt.shape, vt.shape, query.shape
                        )));
                    }
                    (Cow::Borrowed(&query.data[..]), Cow::Borrowed(&kt.data[..]), Cow::Borrowed(&vt.data[..]), kt.shape[1], h, true, false)
                }
                (3, Some(kt), Some(vt)) if kt.ndim() == 4 => {
                    let h = qdim(query.shape[2])?;
                    let l = kt.shape[2];
                    if kt.shape[..] != [b, n, l, h] || vt.shape != kt.shape {
                        return Err(bad(format!("key {:?} / value {:?} are not [B, N, L, H] = [{b}, {n}, L, {h}]", kt.shape, vt.shape)));
                    }
                    (
                        Cow::Borrowed(&query.data[..]),
                        Cow::Owned(bnsh_to_bsnh(&kt.data, b, n, l, h)),
                        Cow::Owned(bnsh_to_bsnh(&vt.data, b, n, l, h)),
                        l,
                        h,
                        false,
                        true,
                    )
                }
                (3, Some(kt), None) if kt.ndim() == 5 => {
                    let h = qdim(query.shape[2])?;
                    let l = kt.shape[1];
                    if kt.shape[..] != [b, l, n, 2, h] {
                        return Err(bad(format!("packed key/value {:?} is not [B, L, N, 2, H] = [{b}, L, {n}, 2, {h}]", kt.shape)));
                    }
                    if bias.is_some() {
                        return Err(bad("bias must be absent with packed key/value".into()));
                    }
                    let mut kv = split_rows(&kt.data, b * l * n, &[h, h]).into_iter();
                    let (kk, vv) = (kv.next().unwrap_or_default(), kv.next().unwrap_or_default());
                    (Cow::Borrowed(&query.data[..]), Cow::Owned(kk), Cow::Owned(vv), l, h, false, false)
                }
                (5, None, None) => {
                    if query.shape[2] != n || query.shape[3] != 3 {
                        return Err(bad(format!("packed query/key/value {:?} is not [B, S, N, 3, H] with N = {n}", query.shape)));
                    }
                    let h = query.shape[4];
                    let mut qkv = split_rows(&query.data, b * s * n, &[h, h, h]).into_iter();
                    let (qq, kk, vv) = (qkv.next().unwrap_or_default(), qkv.next().unwrap_or_default(), qkv.next().unwrap_or_default());
                    (Cow::Owned(qq), Cow::Owned(kk), Cow::Owned(vv), s, h, true, false)
                }
                _ => {
                    return Err(bad(format!(
                        "unsupported query {:?} / key {:?} / value {:?}",
                        query.shape,
                        key.map(|t| &t.shape),
                        value.map(|t| &t.shape)
                    )))
                }
            };
        let d = n * h;
        if let Some(bt) = bias {
            if bt.data.len() != 3 * d {
                return Err(bad(format!("bias has {} elements, expected {}", bt.data.len(), 3 * d)));
            }
            add_row_bias(q.to_mut(), &bt.data[..d]);
            if kv_bias {
                add_row_bias(k.to_mut(), &bt.data[d..2 * d]);
                add_row_bias(v.to_mut(), &bt.data[2 * d..]);
            }
        }

        // Past key/value [B, N, P, H] go in front of the new keys.
        let p = match (ctx.optional_input(6), ctx.optional_input(7)) {
            (None, None) => 0,
            (Some(pk), Some(pv)) => {
                if kv_bnsh {
                    return Err(bad("past_key/past_value with [B, N, L, H] key/value".into()));
                }
                let p = pk.shape.get(2).copied().unwrap_or(0);
                if pk.shape[..] != [b, n, p, h] || pv.shape != pk.shape {
                    return Err(bad(format!("past_key {:?} / past_value {:?} are not [{b}, {n}, P, {h}]", pk.shape, pv.shape)));
                }
                if p > 0 {
                    let cat = |past: &[f32], cur: &[f32]| {
                        let t = p + l;
                        let mut out = bnsh_to_bsnh(past, b, n, p, h);
                        out.resize(b * t * d, 0.0);
                        // Spread the past rows to stride T and append the new ones.
                        for bb in (0..b).rev() {
                            out.copy_within(bb * p * d..(bb + 1) * p * d, bb * t * d);
                        }
                        for bb in 0..b {
                            out[(bb * t + p) * d..(bb + 1) * t * d].copy_from_slice(&cur[bb * l * d..(bb + 1) * l * d]);
                        }
                        out
                    };
                    k = Cow::Owned(cat(&pk.data, &k));
                    v = Cow::Owned(cat(&pv.data, &v));
                }
                p
            }
            _ => return Err(bad("past_key and past_value must both be present or both absent".into())),
        };
        let t = p + l;

        let pad = match ctx.optional_input(4) {
            None => PadMask::None,
            Some(m) => match m.shape[..] {
                [x] if x == b => PadMask::Ends(&m.data, None),
                [x] if x == 2 * b => PadMask::Ends(&m.data[..b], Some(&m.data[b..])),
                [x, y] if x == b && y == t => PadMask::Keys(&m.data),
                [x, y, z] if x == b && y == s && z == t => PadMask::Full(&m.data),
                _ => return Err(bad(format!("key_padding_mask {:?} is not [B], [2B], [B, T] or [B, S, T]", m.shape))),
            },
        };
        let ab = ctx.optional_input(5);
        if let Some(a) = ab {
            check_bhst(a, b, n, s, t).map_err(bad)?;
        }

        // Additive bias over the [B, N, 1, S, T] score space.
        let combined: Option<Vec<f32>>;
        let attn_bias = match (&pad, causal, ab) {
            (PadMask::None, false, None) => None,
            (PadMask::None, false, Some(a)) => Some(AttnBias { data: &a.data, strides: bhst_strides(&a.shape, 1, s, t) }),
            (PadMask::Ends(..) | PadMask::Keys(_), false, None) => {
                let buf: Vec<f32> =
                    (0..b * t).map(|i| pad.value(i / t.max(1), 0, i % t.max(1), s, t, mfv)).collect();
                combined = Some(buf);
                Some(AttnBias { data: combined.as_deref().unwrap_or(&[]), strides: [t, 0, 0, 0, 1] })
            }
            _ => {
                let hb = ab.map_or(1, |a| a.shape[1]);
                let ab_strides = ab.map(|a| bhst_strides(&a.shape, 1, s, t));
                let mut buf = vec![0.0f32; b * hb * s * t];
                for bb in 0..b {
                    for hh in 0..hb {
                        for ss in 0..s {
                            let row = &mut buf[((bb * hb + hh) * s + ss) * t..][..t];
                            for (tt, x) in row.iter_mut().enumerate() {
                                let m = if causal && tt > p + ss { f32::MIN } else { pad.value(bb, ss, tt, s, t, mfv) };
                                let a = match (ab, ab_strides) {
                                    (Some(a), Some([sb, sh, _, sst, _])) => a.data[bb * sb + hh * sh + ss * sst + tt],
                                    _ => 0.0,
                                };
                                *x = a + m;
                            }
                        }
                    }
                }
                combined = Some(buf);
                let sh = if hb > 1 { s * t } else { 0 };
                Some(AttnBias { data: combined.as_deref().unwrap_or(&[]), strides: [hb * s * t, sh, 0, t, 1] })
            }
        };

        let scale = scale_attr(ctx, h);
        let dims = AttnDims { batch: b, seq_q: s, seq_kv: t, q_heads: n, kv_heads: n, head_dim: h, scale };
        let first = slots.first_mut().ok_or_else(|| OnnxError::Internal("MultiHeadAttention: no output slots".into()))?;
        size_slot(first, &[b, s, d]);
        contrib::attention_into(&q, &k, &v, attn_bias, dims, &mut first.data).map_err(bad)?;
        for (i, src) in [(1, &k), (2, &v)] {
            if wanted(ctx, i) {
                if let Some(slot) = slots.get_mut(i) {
                    size_slot(slot, &[b, n, t, h]);
                    bsnh_to_bnsh_into(src, b, t, n, h, &mut slot.data);
                }
            }
        }
        Ok(())
    }
}

impl Operator for MsMultiHeadAttentionOp {
    fn op_type(&self) -> &str {
        "com.microsoft.MultiHeadAttention"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let mut slots = empty_slots(ctx);
        self.run(ctx, &mut slots)?;
        Ok(slots)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        self.run(ctx, slots)
    }
}

// ── com.microsoft::GroupQueryAttention ───────────────────────────────────────

/// `com.microsoft::GroupQueryAttention`.
///
/// Inputs: `query` `[B, S, Hq*D]` (or packed `[B, S, (Hq + 2*Hkv)*D]` with
/// `key`/`value` absent), `key`/`value` `[B, S, Hkv*D]`, `past_key`/`past_value`
/// `[B, Hkv, P, D]` (optional, may be empty), `seqlens_k` `[B]` (per-row total length
/// minus one), `total_sequence_length` (scalar), `cos_cache`/`sin_cache`
/// `[max_pos, rotary_dim/2]`, `position_ids`, `attention_bias`
/// `[B|1, Hq|1, S, T]`, `head_sink` `[Hq]`.
/// Attributes: `num_heads`, `kv_num_heads`, `scale`, `softcap`, `do_rotary`,
/// `rotary_interleaved`, `local_window_size`, `smooth_softmax`, `causal`
/// (default 1), `qk_output` (only 0).
/// Outputs: `output` `[B, S, Hq*D]`, `present_key`/`present_value`
/// `[B, Hkv, max(T, P), D]`.
///
/// Row `b` sees keys `0..seqlens_k[b] + 1`; query `s` additionally only keys
/// up to `past_b + s` when causal (`past_b` = 0 on the first prompt, where
/// `S == total_sequence_length`, else `seqlens_k[b] + 1 - S`), so right-padded
/// prefill batches work: padding queries attend to all valid keys.
/// KV-cache quantization, the sliding-window cache and the q/k norm inputs are
/// not supported.
pub struct GroupQueryAttentionOp;

impl GroupQueryAttentionOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let name = &ctx.node.name;
        let bad = |m: String| OnnxError::ShapeMismatch(format!("com.microsoft::GroupQueryAttention '{name}': {m}"));
        let unsupported = |m: &str| OnnxError::Unsupported(format!("com.microsoft::GroupQueryAttention '{name}': {m}"));
        let attrs = ctx.attrs();
        let (hq, hkv) = (attrs.i("num_heads", 0), attrs.i("kv_num_heads", 0));
        if hq <= 0 || hkv <= 0 || hq % hkv != 0 {
            return Err(bad(format!("num_heads {hq} must be a positive multiple of kv_num_heads {hkv}")));
        }
        let (hq, hkv) = (hq as usize, hkv as usize);
        if !matches!(attrs.s("k_quant_type"), "" | "NONE")
            || !matches!(attrs.s("v_quant_type"), "" | "NONE")
            || attrs.i("kv_cache_bit_width", 0) != 0
        {
            return Err(unsupported("quantized KV cache"));
        }
        if attrs.i("sliding_window_cache", 0) != 0 {
            return Err(unsupported("sliding_window_cache"));
        }
        if attrs.i("qk_output", 0) != 0 || wanted(ctx, 3) {
            return Err(unsupported("qk_output"));
        }
        if (12..16).any(|i| ctx.optional_input(i).is_some()) {
            return Err(unsupported("k_scale / v_scale / q_norm / k_norm inputs"));
        }
        let causal = attrs.i("causal", 1) != 0;
        let local_window = attrs.i("local_window_size", -1);
        let softcap = attrs.f("softcap", 0.0);
        let do_rotary = attrs.i("do_rotary", 0) != 0;
        let interleaved = attrs.i("rotary_interleaved", 0) != 0;
        let smooth = attrs.i("smooth_softmax", 0) != 0;

        let query = ctx.input(0)?;
        if query.ndim() != 3 {
            return Err(bad(format!("query {:?} must be 3-D", query.shape)));
        }
        let (b, s) = (query.shape[0], query.shape[1]);
        let (key, value) = (ctx.optional_input(1), ctx.optional_input(2));
        let (q, k, v, l, d): (Buf, Buf, Buf, usize, usize) = match (key, value) {
            (None, None) => {
                let w = query.shape[2];
                if w % (hq + 2 * hkv) != 0 {
                    return Err(bad(format!("packed QKV width {w} not divisible by {} heads", hq + 2 * hkv)));
                }
                let d = w / (hq + 2 * hkv);
                let mut parts = split_rows(&query.data, b * s, &[hq * d, hkv * d, hkv * d]).into_iter();
                let (qq, kk, vv) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
                (Cow::Owned(qq), Cow::Owned(kk), Cow::Owned(vv), s, d)
            }
            (Some(kt), Some(vt)) => {
                if query.shape[2] % hq != 0 {
                    return Err(bad(format!("query hidden size {} not divisible by {hq} heads", query.shape[2])));
                }
                let d = query.shape[2] / hq;
                if kt.ndim() != 3 || kt.shape[0] != b || kt.shape[2] != hkv * d || vt.shape != kt.shape {
                    return Err(bad(format!("key {:?} / value {:?} are not [{b}, L, {}]", kt.shape, vt.shape, hkv * d)));
                }
                (Cow::Borrowed(&query.data[..]), Cow::Borrowed(&kt.data[..]), Cow::Borrowed(&vt.data[..]), kt.shape[1], d)
            }
            _ => return Err(bad("key and value must both be present or both absent".into())),
        };
        let kvw = hkv * d;

        let past = match (ctx.optional_input(3), ctx.optional_input(4)) {
            (None, None) => None,
            (Some(pk), Some(pv)) => {
                let p = pk.shape.get(2).copied().unwrap_or(0);
                if pk.shape[..] != [b, hkv, p, d] || pv.shape != pk.shape {
                    return Err(bad(format!("past_key {:?} / past_value {:?} are not [{b}, {hkv}, P, {d}]", pk.shape, pv.shape)));
                }
                Some((pk, pv, p))
            }
            _ => return Err(bad("past_key and past_value must both be present or both absent".into())),
        };
        let p = past.map_or(0, |(_, _, p)| p);
        if past.is_some() {
            if l != s && l != 0 {
                return Err(bad(format!("key length {l} must equal the query length {s} (or 0) with a past")));
            }
        } else if l != s {
            return Err(bad(format!("key length {l} must equal the query length {s}")));
        }
        let seqlens = ctx.input(5)?;
        if seqlens.data.len() != b {
            return Err(bad(format!("seqlens_k has {} elements, expected {b}", seqlens.data.len())));
        }
        let total = ctx.input(6)?.data.first().copied().unwrap_or(0.0) as i64;
        if total <= 0 || (total as usize) < l || (total as usize > p && total as usize - l != p) {
            return Err(bad(format!("total_sequence_length {total} does not match key length {l} and past length {p}")));
        }
        let total = total as usize;
        let present_len = total.max(p);
        let first_prompt = s == total;
        if !first_prompt && s != 1 && b != 1 {
            return Err(bad("a multi-token step after the first prompt needs batch size 1".into()));
        }
        let mut kv_len = Vec::with_capacity(b);
        for &x in &seqlens.data {
            let tb = x as i64 + 1;
            if tb < 1 || tb as usize > present_len {
                return Err(bad(format!("seqlens_k value {x} outside [0, {present_len})")));
            }
            let tb = tb as usize;
            if !first_prompt && tb < s {
                return Err(bad(format!("seqlens_k value {x} too small for sequence length {s}")));
            }
            if !first_prompt && l > 0 && tb - s > p {
                return Err(bad(format!("seqlens_k value {x} needs {} past rows, past has {p}", tb - s)));
            }
            kv_len.push(tb);
        }

        // Rotary embedding on the new Q and K heads.
        let (q, k) = if do_rotary {
            let (cos, sin) = match (ctx.optional_input(7), ctx.optional_input(8)) {
                (Some(c), Some(sn)) if c.ndim() == 2 && sn.shape == c.shape => (c, sn),
                _ => return Err(bad("do_rotary needs 2-D cos_cache and sin_cache of the same shape".into())),
            };
            let half = cos.shape[1];
            if half == 0 || 2 * half > d {
                return Err(bad(format!("rotary dim {} does not fit head size {d}", 2 * half)));
            }
            let pos_ids = ctx.optional_input(9);
            // ORT: one position offset on the first prompt, per-token ids afterwards.
            let positions: Vec<f32> = if first_prompt {
                let off = pos_ids.and_then(|pi| pi.data.first().copied()).unwrap_or(0.0);
                (0..b * s).map(|i| off + (i % s) as f32).collect()
            } else if let Some(pi) = pos_ids {
                if pi.data.len() < b * s {
                    return Err(bad(format!("position_ids has {} elements, expected {}", pi.data.len(), b * s)));
                }
                pi.data[..b * s].to_vec()
            } else {
                (0..b * s).map(|i| (kv_len[i / s] - s + i % s) as f32).collect()
            };
            let rotate = |x: &[f32], heads: usize, rows: usize| -> Result<Vec<f32>, OnnxError> {
                let dims = RotaryDims { batch: b, seq: rows, num_heads: heads, head_size: d, rotary_dim: 2 * half, bnsd: false, interleaved };
                let mut out = vec![0.0; x.len()];
                contrib::ms_rotary_into(x, &positions, &cos.data, &sin.data, false, dims, &mut out).map_err(bad)?;
                Ok(out)
            };
            let qr = rotate(&q, hq, s)?;
            let kr = if l > 0 { Cow::Owned(rotate(&k, hkv, l)?) } else { k };
            (Cow::Owned(qr), kr)
        } else {
            (q, k)
        };

        // Keys/values the queries attend to, as [B, present_len, Hkv*D].
        let tbuf = present_len;
        let (k_all, v_all, offsets): (Buf, Buf, Vec<usize>) = match past {
            Some((pk, pv, _)) if l == 0 => {
                // Shared KV: the past buffer already holds every key.
                let offs = kv_len.iter().map(|&tb| if first_prompt { 0 } else { tb - s }).collect();
                let mut kk = bnsh_to_bsnh(&pk.data, b, hkv, p, d);
                let mut vv = bnsh_to_bsnh(&pv.data, b, hkv, p, d);
                kk.resize(b * tbuf * kvw, 0.0);
                vv.resize(b * tbuf * kvw, 0.0);
                (Cow::Owned(kk), Cow::Owned(vv), offs)
            }
            Some((pk, pv, _)) if !first_prompt => {
                let mut kk = vec![0.0; b * tbuf * kvw];
                let mut vv = vec![0.0; b * tbuf * kvw];
                let mut offs = Vec::with_capacity(b);
                for bb in 0..b {
                    let pb = kv_len[bb] - s;
                    for (dst, past, cur) in [(&mut kk, &pk.data, &k), (&mut vv, &pv.data, &v)] {
                        for g in 0..hkv {
                            for tt in 0..pb {
                                let src = ((bb * hkv + g) * p + tt) * d;
                                let o = (bb * tbuf + tt) * kvw + g * d;
                                dst[o..o + d].copy_from_slice(&past[src..src + d]);
                            }
                        }
                        let o = (bb * tbuf + pb) * kvw;
                        dst[o..o + l * kvw].copy_from_slice(&cur[bb * l * kvw..(bb + 1) * l * kvw]);
                    }
                    offs.push(pb);
                }
                (Cow::Owned(kk), Cow::Owned(vv), offs)
            }
            _ if l == tbuf => (k, v, vec![0; b]),
            _ => {
                let pad = |x: &[f32]| {
                    let mut out = vec![0.0; b * tbuf * kvw];
                    for bb in 0..b {
                        out[bb * tbuf * kvw..][..l * kvw].copy_from_slice(&x[bb * l * kvw..(bb + 1) * l * kvw]);
                    }
                    out
                };
                (Cow::Owned(pad(&k)), Cow::Owned(pad(&v)), vec![0; b])
            }
        };

        let group = hq / hkv;
        let attn_bias = match ctx.optional_input(10) {
            None => None,
            Some(a) => {
                check_bhst(a, b, hq, s, total).map_err(bad)?;
                if total != tbuf {
                    return Err(unsupported("attention_bias with a past buffer longer than total_sequence_length"));
                }
                Some(AttnBias { data: &a.data, strides: bhst_strides(&a.shape, group, s, tbuf) })
            }
        };
        let sink = match ctx.optional_input(11) {
            Some(hs) if hs.data.len() == hq => AttnSink::PerHead(&hs.data),
            Some(hs) => return Err(bad(format!("head_sink {:?} is not [{hq}]", hs.shape))),
            None if smooth => AttnSink::Zero,
            None => AttnSink::None,
        };
        let ext = AttnExt {
            kv_len: Some(&kv_len),
            causal_offset: causal.then_some(&offsets[..]),
            local_window: (local_window >= 0).then_some(local_window as usize),
            softcap,
            sink,
        };
        let scale = scale_attr(ctx, d);
        let dims = AttnDims { batch: b, seq_q: s, seq_kv: tbuf, q_heads: hq, kv_heads: hkv, head_dim: d, scale };
        let first = slots.first_mut().ok_or_else(|| OnnxError::Internal("GroupQueryAttention: no output slots".into()))?;
        size_slot(first, &[b, s, hq * d]);
        contrib::attention_ext_into(&q, &k_all, &v_all, attn_bias, ext, dims, &mut first.data).map_err(bad)?;
        for (i, src) in [(1, &k_all), (2, &v_all)] {
            if wanted(ctx, i) {
                if let Some(slot) = slots.get_mut(i) {
                    size_slot(slot, &[b, hkv, tbuf, d]);
                    bsnh_to_bnsh_into(src, b, tbuf, hkv, d, &mut slot.data);
                }
            }
        }
        Ok(())
    }
}

impl Operator for GroupQueryAttentionOp {
    fn op_type(&self) -> &str {
        "GroupQueryAttention"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let mut slots = empty_slots(ctx);
        self.run(ctx, &mut slots)?;
        Ok(slots)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        self.run(ctx, slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::new(data, shape.to_vec())
    }

    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}: [{i}] {g} vs {w}");
        }
    }

    /// Run `op` through both `execute` and `execute_into_slots` (on 1 and 4
    /// threads) and check every path agrees; returns the outputs.
    fn run(
        op: &dyn Operator,
        op_name: &str,
        ints: &[(&str, i64)],
        floats: &[(&str, f32)],
        inputs: &[Option<&Tensor>],
        outputs: usize,
    ) -> Vec<Tensor> {
        let mut attrs = Attributes::default();
        for &(k, v) in ints {
            attrs.ints.insert(k.into(), v);
        }
        for &(k, v) in floats {
            attrs.floats.insert(k.into(), v);
        }
        let node = Node {
            op: OpKind::Unknown(op_name.into()),
            name: "attn".into(),
            inputs: inputs.iter().enumerate().map(|(i, x)| if x.is_some() { format!("in{i}") } else { String::new() }).collect(),
            outputs: (0..outputs).map(|i| format!("out{i}")).collect(),
            attrs,
        };
        let ctx = OpContext { node: &node, inputs: inputs.to_vec(), outer_scope: None, weights: None, registry: None };
        let mut results = Vec::new();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let a = pool.install(|| op.execute(&ctx)).expect("execute");
            let mut slots: Vec<Tensor> = (0..outputs).map(|_| Tensor::new(vec![], vec![0])).collect();
            pool.install(|| op.execute_into_slots(&ctx, &mut slots)).expect("execute_into_slots");
            for (x, y) in a.iter().zip(&slots) {
                assert_eq!(x.shape, y.shape);
                close(&x.data, &y.data, 1e-6, &format!("{op_name} slots vs execute ({threads} threads)"));
            }
            results.push(a);
        }
        for (x, y) in results[0].iter().zip(&results[1]) {
            close(&x.data, &y.data, 1e-5, &format!("{op_name}: 1 vs 4 threads"));
        }
        results.swap_remove(0)
    }

    /// Scalar attention: `score(b, h, s, t)` returns the additive term, or `None`
    /// for a key outside the row's window; `sink(h)` the optional sink logit.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        (b, s, t): (usize, usize, usize),
        (hq, hkv, d): (usize, usize, usize),
        scale: f32,
        softcap: f32,
        score: &dyn Fn(usize, usize, usize, usize) -> Option<f32>,
        sink: &dyn Fn(usize) -> Option<f32>,
    ) -> Vec<f32> {
        let mut out = vec![0.0; b * s * hq * d];
        for bi in 0..b {
            for h in 0..hq {
                let g = h / (hq / hkv);
                for si in 0..s {
                    let qr = &q[(bi * s + si) * hq * d + h * d..][..d];
                    let sc: Vec<Option<f32>> = (0..t)
                        .map(|ti| {
                            score(bi, h, si, ti).map(|add| {
                                let kr = &k[(bi * t + ti) * hkv * d + g * d..][..d];
                                let mut x = scale * qr.iter().zip(kr).map(|(a, c)| a * c).sum::<f32>();
                                if softcap > 0.0 {
                                    x = softcap * (x / softcap).tanh();
                                }
                                x + add
                            })
                        })
                        .collect();
                    let mut mx = sc.iter().flatten().copied().fold(f32::NEG_INFINITY, f32::max);
                    if let Some(sk) = sink(h) {
                        mx = mx.max(sk);
                    }
                    let e: Vec<f32> = sc.iter().map(|x| x.map_or(0.0, |x| (x - mx).exp())).collect();
                    let sum = e.iter().sum::<f32>() + sink(h).map_or(0.0, |sk| (sk - mx).exp());
                    for di in 0..d {
                        out[(bi * s + si) * hq * d + h * d + di] =
                            (0..t).map(|ti| e[ti] / sum * v[(bi * t + ti) * hkv * d + g * d + di]).sum();
                    }
                }
            }
        }
        out
    }

    const MHA: &str = "com.microsoft.MultiHeadAttention";
    const GQA: &str = "GroupQueryAttention";

    #[test]
    fn mha_bias_masks_attention_bias_and_causal() {
        let (b, s, n, h) = (3, 21, 4, 16);
        let d = n * h;
        let q = rand(b * s * d, 1);
        let k = rand(b * s * d, 2);
        let v = rand(b * s * d, 3);
        let bias = rand(3 * d, 4);
        let lens = [21usize, 13, 5];
        let mask2: Vec<f32> = (0..b * s).map(|i| (i % s < lens[i / s]) as u8 as f32).collect();
        let mask1: Vec<f32> = lens.iter().map(|&x| x as f32).collect();
        let mask3: Vec<f32> = (0..b * s * s).map(|i| ((i % s) <= (i / s) % s + 3) as u8 as f32).collect();
        let ab = rand(n * s * s, 5);
        let (mut qb, mut kb, mut vb) = (q.clone(), k.clone(), v.clone());
        add_row_bias(&mut qb, &bias[..d]);
        add_row_bias(&mut kb, &bias[d..2 * d]);
        add_row_bias(&mut vb, &bias[2 * d..]);
        let scale = 0.3;
        let mfv = -10000.0;
        for (mask_kind, use_ab, causal, use_bias) in [
            (0, false, false, false),
            (0, false, false, true),
            (2, false, false, true),
            (1, false, false, false),
            (3, false, false, true),
            (0, true, false, false),
            (2, true, true, true),
            (0, false, true, false),
            (1, true, true, true),
        ] {
            let tq = t(q.clone(), &[b, s, d]);
            let tk = t(k.clone(), &[b, s, d]);
            let tv = t(v.clone(), &[b, s, d]);
            let tb = t(bias.clone(), &[3 * d]);
            let tm = match mask_kind {
                1 => Some(t(mask1.clone(), &[b])),
                2 => Some(t(mask2.clone(), &[b, s])),
                3 => Some(t(mask3.clone(), &[b, s, s])),
                _ => None,
            };
            let ta = t(ab.clone(), &[1, n, s, s]);
            let out = run(
                &MsMultiHeadAttentionOp,
                MHA,
                &[("num_heads", n as i64), ("unidirectional", causal as i64)],
                &[("scale", scale)],
                &[Some(&tq), Some(&tk), Some(&tv), use_bias.then_some(&tb), tm.as_ref(), use_ab.then_some(&ta)],
                1,
            );
            let score = |bi: usize, hi: usize, si: usize, ti: usize| {
                let keep = match mask_kind {
                    1 | 2 => ti < lens[bi],
                    3 => mask3[(bi * s + si) * s + ti] > 0.0,
                    _ => true,
                };
                let m = if causal && ti > si { f32::MIN } else if keep { 0.0 } else { mfv };
                Some(if use_ab { ab[(hi * s + si) * s + ti] } else { 0.0 } + m)
            };
            let (rq, rk, rv) = if use_bias { (&qb, &kb, &vb) } else { (&q, &k, &v) };
            let want = reference(rq, rk, rv, (b, s, s), (n, n, h), scale, 0.0, &score, &|_| None);
            assert_eq!(out[0].shape, vec![b, s, d]);
            close(&out[0].data, &want, 2e-5, &format!("mask {mask_kind} ab {use_ab} causal {causal} bias {use_bias}"));
        }
    }

    #[test]
    fn mha_packed_bnsh_and_past_layouts_match_plain() {
        let (b, s, n, h) = (2, 9, 3, 8);
        let d = n * h;
        let q = rand(b * s * d, 11);
        let k = rand(b * s * d, 12);
        let v = rand(b * s * d, 13);
        let tq = t(q.clone(), &[b, s, d]);
        let attrs = [("num_heads", n as i64)];
        let plain = run(&MsMultiHeadAttentionOp, MHA, &attrs, &[], &[Some(&tq), Some(&t(k.clone(), &[b, s, d])), Some(&t(v.clone(), &[b, s, d]))], 3);
        // Present outputs are K/V in [B, N, T, H].
        let mut kt = vec![0.0; k.len()];
        bsnh_to_bnsh_into(&k, b, s, n, h, &mut kt);
        let mut vt = vec![0.0; v.len()];
        bsnh_to_bnsh_into(&v, b, s, n, h, &mut vt);
        assert_eq!(plain[1].shape, vec![b, n, s, h]);
        close(&plain[1].data, &kt, 0.0, "present_key");
        close(&plain[2].data, &vt, 0.0, "present_value");
        // K/V as [B, N, L, H].
        let bnsh = run(&MsMultiHeadAttentionOp, MHA, &attrs, &[], &[Some(&tq), Some(&t(kt.clone(), &[b, n, s, h])), Some(&t(vt.clone(), &[b, n, s, h]))], 1);
        close(&bnsh[0].data, &plain[0].data, 1e-6, "BNSH key/value");
        // Packed KV [B, L, N, 2, H] and packed QKV [B, S, N, 3, H].
        let mut kv = Vec::new();
        let mut qkv = Vec::new();
        for r in 0..b * s * n {
            kv.extend_from_slice(&k[r * h..(r + 1) * h]);
            kv.extend_from_slice(&v[r * h..(r + 1) * h]);
            qkv.extend_from_slice(&q[r * h..(r + 1) * h]);
            qkv.extend_from_slice(&k[r * h..(r + 1) * h]);
            qkv.extend_from_slice(&v[r * h..(r + 1) * h]);
        }
        let pkv = run(&MsMultiHeadAttentionOp, MHA, &attrs, &[], &[Some(&tq), Some(&t(kv, &[b, s, n, 2, h])), None], 1);
        close(&pkv[0].data, &plain[0].data, 1e-6, "packed KV");
        let pqkv = run(&MsMultiHeadAttentionOp, MHA, &attrs, &[], &[Some(&t(qkv, &[b, s, n, 3, h])), None, None], 1);
        close(&pqkv[0].data, &plain[0].data, 1e-6, "packed QKV");
        // Past: the first 5 keys as past_key/value, the rest as new keys.
        let (p, l) = (5, s - 5);
        let split = |x: &[f32]| {
            let mut past = vec![0.0; b * n * p * h];
            let mut cur = Vec::new();
            for bb in 0..b {
                for tt in 0..s {
                    let row = &x[(bb * s + tt) * d..][..d];
                    if tt < p {
                        for nn in 0..n {
                            past[((bb * n + nn) * p + tt) * h..][..h].copy_from_slice(&row[nn * h..(nn + 1) * h]);
                        }
                    } else {
                        cur.extend_from_slice(row);
                    }
                }
            }
            (t(past, &[b, n, p, h]), t(cur, &[b, l, d]))
        };
        let ((pk, ck), (pv, cv)) = (split(&k), split(&v));
        let past = run(&MsMultiHeadAttentionOp, MHA, &attrs, &[], &[Some(&tq), Some(&ck), Some(&cv), None, None, None, Some(&pk), Some(&pv)], 3);
        close(&past[0].data, &plain[0].data, 1e-6, "past key/value");
        close(&past[1].data, &kt, 0.0, "present_key with past");
        close(&past[2].data, &vt, 0.0, "present_value with past");
    }

    /// Rotate `[B, S, heads*D]` rows at positions `pos(b, s)` (scalar reference).
    fn rotate_ref(x: &[f32], (b, s, heads, d): (usize, usize, usize, usize), cos: &[f32], sin: &[f32], half: usize, interleaved: bool, pos: &dyn Fn(usize, usize) -> usize) -> Vec<f32> {
        let mut out = x.to_vec();
        for bi in 0..b {
            for si in 0..s {
                let p = pos(bi, si);
                for hh in 0..heads {
                    let base = ((bi * s + si) * heads + hh) * d;
                    for i in 0..half {
                        let (c, sn) = (cos[p * half + i], sin[p * half + i]);
                        let (i1, i2) = if interleaved { (base + 2 * i, base + 2 * i + 1) } else { (base + i, base + i + half) };
                        out[i1] = x[i1] * c - x[i2] * sn;
                        out[i2] = x[i2] * c + x[i1] * sn;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn gqa_prefill_right_padded_batches() {
        let (b, s, hq, hkv, d) = (3, 23, 4, 2, 16);
        let (qw, kw) = (hq * d, hkv * d);
        let q = rand(b * s * qw, 21);
        let k = rand(b * s * kw, 22);
        let v = rand(b * s * kw, 23);
        let lens = [23usize, 14, 3];
        let seqlens = t(lens.iter().map(|&x| (x - 1) as f32).collect(), &[b]);
        let total = t(vec![s as f32], &[1]);
        let max_pos = 40;
        let half = 4; // rotary dim 8 < head size 16
        let cos = rand(max_pos * half, 24);
        let sin = rand(max_pos * half, 25);
        let ab = rand(b * s * s, 26);
        let sinks = rand(hq, 27);
        let scale = 0.25;
        // (rotary, interleaved, bias, softcap, sink: 0 none / 1 smooth / 2 per head, local window, causal)
        for (rot, inter, use_ab, softcap, sink, window, causal) in [
            (false, false, false, 0.0, 0, -1, 1),
            (true, false, false, 0.0, 0, -1, 1),
            (true, true, true, 0.0, 0, -1, 1),
            (false, false, true, 0.7, 0, -1, 1),
            (true, false, false, 0.0, 1, -1, 1),
            (false, false, false, 0.0, 2, 5, 1),
            (false, false, true, 0.0, 0, -1, 0),
        ] {
            let tq = t(q.clone(), &[b, s, qw]);
            let tk = t(k.clone(), &[b, s, kw]);
            let tv = t(v.clone(), &[b, s, kw]);
            let tc = t(cos.clone(), &[max_pos, half]);
            let ts = t(sin.clone(), &[max_pos, half]);
            let ta = t(ab.clone(), &[b, 1, s, s]);
            let th = t(sinks.clone(), &[hq]);
            let ints = [
                ("num_heads", hq as i64),
                ("kv_num_heads", hkv as i64),
                ("do_rotary", rot as i64),
                ("rotary_interleaved", inter as i64),
                ("smooth_softmax", (sink == 1) as i64),
                ("local_window_size", window),
                ("causal", causal),
            ];
            let floats = [("scale", scale), ("softcap", softcap)];
            let inputs = [
                Some(&tq),
                Some(&tk),
                Some(&tv),
                None,
                None,
                Some(&seqlens),
                Some(&total),
                rot.then_some(&tc),
                rot.then_some(&ts),
                None,
                use_ab.then_some(&ta),
                (sink == 2).then_some(&th),
            ];
            let what = format!("rot {rot} inter {inter} ab {use_ab} softcap {softcap} sink {sink} window {window} causal {causal}");
            let out = run(&GroupQueryAttentionOp, GQA, &ints, &floats, &inputs, 3);
            let (rq, rk) = if rot {
                let pos = |_: usize, si: usize| si;
                (rotate_ref(&q, (b, s, hq, d), &cos, &sin, half, inter, &pos), rotate_ref(&k, (b, s, hkv, d), &cos, &sin, half, inter, &pos))
            } else {
                (q.clone(), k.clone())
            };
            let score = |bi: usize, _h: usize, si: usize, ti: usize| {
                let hi = if causal == 1 { (si + 1).min(lens[bi]) } else { lens[bi] };
                let lo = if window >= 0 && hi > window as usize { hi - window as usize } else { 0 };
                (lo..hi).contains(&ti).then(|| if use_ab { ab[(bi * s + si) * s + ti] } else { 0.0 })
            };
            let sink_fn = |hh: usize| match sink {
                1 => Some(0.0),
                2 => Some(sinks[hh]),
                _ => None,
            };
            let want = reference(&rq, &rk, &v, (b, s, s), (hq, hkv, d), scale, softcap, &score, &sink_fn);
            assert_eq!(out[0].shape, vec![b, s, qw]);
            close(&out[0].data, &want, 2e-5, &what);
            let mut kt = vec![0.0; rk.len()];
            bsnh_to_bnsh_into(&rk, b, s, hkv, d, &mut kt);
            assert_eq!(out[1].shape, vec![b, hkv, s, d]);
            close(&out[1].data, &kt, 1e-6, &format!("{what}: present_key"));

            // Packed QKV gives the same result.
            let mut packed = Vec::new();
            for r in 0..b * s {
                packed.extend_from_slice(&q[r * qw..(r + 1) * qw]);
                packed.extend_from_slice(&k[r * kw..(r + 1) * kw]);
                packed.extend_from_slice(&v[r * kw..(r + 1) * kw]);
            }
            let tp = t(packed, &[b, s, qw + 2 * kw]);
            let mut pin = inputs;
            pin[0] = Some(&tp);
            pin[1] = None;
            pin[2] = None;
            let pout = run(&GroupQueryAttentionOp, GQA, &ints, &floats, &pin, 1);
            close(&pout[0].data, &out[0].data, 1e-6, &format!("{what}: packed QKV"));

            // An empty past changes nothing.
            let empty = t(vec![], &[b, hkv, 0, d]);
            let mut ein = inputs;
            ein[3] = Some(&empty);
            ein[4] = Some(&empty);
            let eout = run(&GroupQueryAttentionOp, GQA, &ints, &floats, &ein, 1);
            close(&eout[0].data, &out[0].data, 0.0, &format!("{what}: empty past"));
        }
    }

    #[test]
    fn gqa_decode_step_with_past() {
        let (b, hq, hkv, d, p) = (2, 6, 2, 16, 10);
        let lens = [7usize, 11]; // total keys per row, including the new token
        let (qw, kw) = (hq * d, hkv * d);
        let q = rand(b * qw, 31);
        let k = rand(b * kw, 32);
        let v = rand(b * kw, 33);
        let pk = rand(b * hkv * p * d, 34);
        let pv = rand(b * hkv * p * d, 35);
        let (max_pos, half) = (16, 8);
        let cos = rand(max_pos * half, 36);
        let sin = rand(max_pos * half, 37);
        let ints = [("num_heads", hq as i64), ("kv_num_heads", hkv as i64), ("do_rotary", 1)];
        let (tq, tk, tv) = (t(q.clone(), &[b, 1, qw]), t(k.clone(), &[b, 1, kw]), t(v.clone(), &[b, 1, kw]));
        let (tpk, tpv) = (t(pk.clone(), &[b, hkv, p, d]), t(pv.clone(), &[b, hkv, p, d]));
        let seqlens = t(lens.iter().map(|&x| (x - 1) as f32).collect(), &[b]);
        let total = t(vec![(p + 1) as f32], &[1]);
        let (tc, ts) = (t(cos.clone(), &[max_pos, half]), t(sin.clone(), &[max_pos, half]));
        let out = run(
            &GroupQueryAttentionOp,
            GQA,
            &ints,
            &[],
            &[Some(&tq), Some(&tk), Some(&tv), Some(&tpk), Some(&tpv), Some(&seqlens), Some(&total), Some(&tc), Some(&ts)],
            3,
        );
        let pos = |bi: usize, _: usize| lens[bi] - 1;
        let rq = rotate_ref(&q, (b, 1, hq, d), &cos, &sin, half, false, &pos);
        let rk = rotate_ref(&k, (b, 1, hkv, d), &cos, &sin, half, false, &pos);
        // Keys: past rows 0..len-1, then the new token, in [B, T, Hkv*D] with T = P + 1.
        let tt = p + 1;
        let mut kall = vec![0.0; b * tt * kw];
        let mut vall = vec![0.0; b * tt * kw];
        for bi in 0..b {
            for (dst, past, cur) in [(&mut kall, &pk, &rk), (&mut vall, &pv, &v)] {
                for g in 0..hkv {
                    for ti in 0..lens[bi] - 1 {
                        dst[(bi * tt + ti) * kw + g * d..][..d].copy_from_slice(&past[((bi * hkv + g) * p + ti) * d..][..d]);
                    }
                }
                dst[(bi * tt + lens[bi] - 1) * kw..][..kw].copy_from_slice(&cur[bi * kw..(bi + 1) * kw]);
            }
        }
        let score = |bi: usize, _: usize, _: usize, ti: usize| (ti < lens[bi]).then_some(0.0);
        let want = reference(&rq, &kall, &vall, (b, 1, tt), (hq, hkv, d), 1.0 / (d as f32).sqrt(), 0.0, &score, &|_| None);
        close(&out[0].data, &want, 2e-5, "decode step");
        assert_eq!(out[1].shape, vec![b, hkv, tt, d]);
        let mut kt = vec![0.0; kall.len()];
        bsnh_to_bnsh_into(&kall, b, tt, hkv, d, &mut kt);
        close(&out[1].data, &kt, 0.0, "decode present_key");
    }

    #[test]
    fn registry_has_both_multi_head_attention_flavours() {
        let reg = crate::registry::default_registry();
        assert!(reg.get(MHA).is_some() && reg.get(GQA).is_some());
        assert_eq!(reg.get("MultiHeadAttention").map(|o| o.op_type().to_string()), Some("MultiHeadAttention".into()));
    }
}
