//! Operators for ONNX Runtime contrib ops (`com.microsoft` domain) and the
//! optimizer-only fused attention op. Kernels live in [`crate::contrib`].
//!
//! The model loader keeps op types whose `com.microsoft` semantics differ from a
//! same-named ONNX op apart by prefixing the domain (`com.microsoft.RotaryEmbedding`);
//! `SkipSimplifiedLayerNormalization` exists only in `com.microsoft` and keeps its name.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::contrib::{self, AttnBias, AttnDims, RotaryDims};

fn err(e: String) -> OnnxError {
    OnnxError::ShapeMismatch(e)
}

/// Size `slot` to `shape`; its contents are fully overwritten by the caller.
fn shape_slot(slot: &mut Tensor, shape: &[usize]) {
    let n: usize = shape.iter().product();
    if slot.data.len() != n {
        slot.data.resize(n, 0.0);
    }
    slot.shape.clear();
    slot.shape.extend_from_slice(shape);
}

/// Empty slots positionally aligned with `ctx.node.outputs`, for `execute`.
fn empty_slots(ctx: &OpContext<'_>) -> Vec<Tensor> {
    ctx.node.outputs.iter().map(|_| Tensor { data: vec![], shape: vec![] }).collect()
}

// ── SkipSimplifiedLayerNormalization ─────────────────────────────────────────

/// `com.microsoft::SkipSimplifiedLayerNormalization`.
///
/// Inputs: `input`, `skip`, `gamma`, optional `bias`. Outputs: `output`,
/// `mean` (not produced by the simplified form; must be elided),
/// `inv_std_var` (`[.., 1]`), `input_skip_bias_sum`.
pub struct SkipSimplifiedLayerNormOp;

impl SkipSimplifiedLayerNormOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let x = ctx.input(0)?;
        let skip = ctx.input(1)?;
        let gamma = ctx.input(2)?;
        let bias = ctx.optional_input(3);
        let eps = ctx.attrs().f("epsilon", 1e-5);
        let outs = &ctx.node.outputs;
        if outs.get(1).is_some_and(|n| !n.is_empty()) {
            return Err(OnnxError::Unsupported(
                "SkipSimplifiedLayerNormalization: the `mean` output is not defined for RMS norm".into(),
            ));
        }
        let hidden = *x.shape.last().ok_or_else(|| err("SkipSimplifiedLayerNormalization: scalar input".into()))?;
        let wants = |i: usize| outs.get(i).is_some_and(|n| !n.is_empty());
        let (want_inv, want_sum) = (wants(2), wants(3));
        let mut inv_shape = x.shape.clone();
        if let Some(l) = inv_shape.last_mut() {
            *l = 1;
        }
        let (first, rest) = slots
            .split_first_mut()
            .ok_or_else(|| OnnxError::Internal("SkipSimplifiedLayerNormalization: no output slots".into()))?;
        shape_slot(first, &x.shape);
        let (mut inv, mut sum) = (None, None);
        for (i, slot) in rest.iter_mut().enumerate() {
            match i + 1 {
                2 if want_inv => {
                    shape_slot(slot, &inv_shape);
                    inv = Some(&mut slot.data[..]);
                }
                3 if want_sum => {
                    shape_slot(slot, &x.shape);
                    sum = Some(&mut slot.data[..]);
                }
                _ => {}
            }
        }
        contrib::skip_rms_norm_into(
            &x.data,
            &skip.data,
            &gamma.data,
            bias.map(|b| &b.data[..]),
            hidden,
            eps,
            &mut first.data,
            sum,
            inv,
        )
        .map_err(err)
    }
}

impl Operator for SkipSimplifiedLayerNormOp {
    fn op_type(&self) -> &str {
        "SkipSimplifiedLayerNormalization"
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

// ── com.microsoft::RotaryEmbedding ───────────────────────────────────────────

/// `com.microsoft::RotaryEmbedding`: inputs `input` (`[B, S, N*D]` or
/// `[B, N, S, D]`), `position_ids` (one offset or `[B, S]`), `cos_cache`,
/// `sin_cache` (`[max_pos, rotary_dim/2]`, or `[B, S, rotary_dim/2]`).
/// Attributes `interleaved`, `num_heads`, `rotary_embedding_dim`.
pub struct MsRotaryEmbeddingOp;

impl MsRotaryEmbeddingOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let x = ctx.input(0)?;
        let pos = ctx.input(1)?;
        let cos = ctx.input(2)?;
        let sin = ctx.input(3)?;
        let attrs = ctx.attrs();
        if attrs.i("is_packed_batching", 0) != 0 {
            return Err(OnnxError::Unsupported("com.microsoft::RotaryEmbedding: is_packed_batching".into()));
        }
        let half = *cos.shape.last().ok_or_else(|| err("com.microsoft::RotaryEmbedding: scalar cos cache".into()))?;
        let dims = RotaryDims::new(
            &x.shape,
            half,
            attrs.i("num_heads", 0).max(0) as usize,
            attrs.i("rotary_embedding_dim", 0).max(0) as usize,
            attrs.i("interleaved", 0) != 0,
        )
        .map_err(err)?;
        shape_slot(&mut slots[0], &x.shape);
        contrib::ms_rotary_into(&x.data, &pos.data, &cos.data, &sin.data, cos.ndim() == 3, dims, &mut slots[0].data)
            .map_err(err)
    }
}

impl Operator for MsRotaryEmbeddingOp {
    fn op_type(&self) -> &str {
        "com.microsoft.RotaryEmbedding"
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

// ── OxiAttention (optimizer-only) ────────────────────────────────────────────

/// Bias broadcast over the 5-D score space `[B, G, R, S, T]`.
pub const BIAS_LAYOUT_BGRST: i64 = 0;
/// Bias broadcast over the 4-D score space `[B, H, S, T]` (`H = G * R`).
pub const BIAS_LAYOUT_BHST: i64 = 1;

/// Fused grouped-query attention, emitted by the optimizer's attention fusion.
///
/// Inputs: `q` `[B, S, Hq*D]`, `k`/`v` `[B, T, Hkv*D]`, optional additive
/// `bias`, optional `scores_shape` — the shape the unfused graph reshaped the
/// `[B, G, R*S, T]` scores to before adding the bias, re-checked here at run
/// time to equal `[B, G, R, S, T]`.
/// Attributes: `q_num_heads`, `kv_num_heads`, `scale`, `bias_layout`.
/// Output: `[B, S, Hq*D]`.
pub struct OxiAttentionOp;

/// Resolve Reshape semantics (`0` copies the input dim, one `-1` is inferred).
fn resolve_reshape(target: &[f32], input: &[usize]) -> Option<Vec<usize>> {
    let numel: usize = input.iter().product();
    let mut out = Vec::with_capacity(target.len());
    let mut infer = None;
    for (i, &v) in target.iter().enumerate() {
        match v as i64 {
            0 => out.push(*input.get(i)?),
            -1 if infer.is_none() => {
                infer = Some(i);
                out.push(1);
            }
            d if d > 0 => out.push(d as usize),
            _ => return None,
        }
    }
    if let Some(i) = infer {
        let known: usize = out.iter().product();
        if known == 0 || numel % known != 0 {
            return None;
        }
        out[i] = numel / known;
    }
    Some(out)
}

/// Element strides of `shape` broadcast onto `target` (left-padded), 0 on
/// broadcast axes; `None` when not broadcastable.
fn broadcast_strides(shape: &[usize], target: &[usize]) -> Option<Vec<usize>> {
    if shape.len() > target.len() {
        return None;
    }
    let pad = target.len() - shape.len();
    let mut strides = vec![0; target.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        let (d, t) = (shape[i], target[pad + i]);
        if d != 1 && d != t {
            return None;
        }
        strides[pad + i] = if d == 1 { 0 } else { acc };
        acc *= d;
    }
    Some(strides)
}

impl OxiAttentionOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let (q, k, v) = (ctx.input(0)?, ctx.input(1)?, ctx.input(2)?);
        let bias = ctx.optional_input(3);
        let scores_shape = ctx.optional_input(4);
        let attrs = ctx.attrs();
        let hq = attrs.i("q_num_heads", 0).max(0) as usize;
        let hkv = attrs.i("kv_num_heads", 0).max(0) as usize;
        let bad = |m: String| err(format!("OxiAttention '{}': {m}", ctx.node.name));
        if q.ndim() != 3 || k.ndim() != 3 || v.shape != k.shape || hq == 0 || hkv == 0 || hq % hkv != 0 {
            return Err(bad(format!("unsupported shapes q {:?} k {:?} v {:?}, heads {hq}/{hkv}", q.shape, k.shape, v.shape)));
        }
        let (b, s, t) = (q.shape[0], q.shape[1], k.shape[1]);
        let d = q.shape[2] / hq;
        if d * hq != q.shape[2] || k.shape[0] != b || k.shape[2] != hkv * d {
            return Err(bad(format!("q {:?} / k {:?} do not split into {hq}/{hkv} heads", q.shape, k.shape)));
        }
        let (g, r) = (hkv, hq / hkv);
        if let Some(ss) = scores_shape {
            let resolved = resolve_reshape(&ss.data, &[b, g, r * s, t]);
            if resolved.as_deref() != Some(&[b, g, r, s, t][..]) {
                return Err(bad(format!("scores reshape {:?} is not [B, G, R, S, T]", ss.data)));
            }
        }
        let bias = match bias {
            None => None,
            Some(bt) => {
                let strides = match attrs.i("bias_layout", BIAS_LAYOUT_BGRST) {
                    BIAS_LAYOUT_BGRST => broadcast_strides(&bt.shape, &[b, g, r, s, t]),
                    BIAS_LAYOUT_BHST => broadcast_strides(&bt.shape, &[b, hq, s, t])
                        .map(|st| vec![st[0], st[1] * r, st[1], st[2], st[3]]),
                    other => return Err(bad(format!("unknown bias_layout {other}"))),
                }
                .ok_or_else(|| bad(format!("bias {:?} does not broadcast to the scores", bt.shape)))?;
                Some(AttnBias { data: &bt.data, strides: [strides[0], strides[1], strides[2], strides[3], strides[4]] })
            }
        };
        let scale = attrs.f("scale", 1.0 / (d as f32).sqrt());
        shape_slot(&mut slots[0], &q.shape);
        let dims = AttnDims { batch: b, seq_q: s, seq_kv: t, q_heads: hq, kv_heads: hkv, head_dim: d, scale };
        contrib::attention_into(&q.data, &k.data, &v.data, bias, dims, &mut slots[0].data).map_err(bad)
    }
}

impl Operator for OxiAttentionOp {
    fn op_type(&self) -> &str {
        "OxiAttention"
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

    #[test]
    fn reshape_and_broadcast_helpers() {
        assert_eq!(resolve_reshape(&[0.0, 0.0, 2.0, -1.0, 4.0], &[3, 2, 10, 4]), Some(vec![3, 2, 2, 5, 4]));
        assert_eq!(resolve_reshape(&[0.0, 0.0, 3.0, -1.0, 4.0], &[3, 2, 10, 4]), None);
        assert_eq!(resolve_reshape(&[0.0, 0.0, 2.0, 5.0, -1.0], &[3, 2, 10, 4]), Some(vec![3, 2, 2, 5, 4]));
        assert_eq!(broadcast_strides(&[3, 1, 1, 5, 4], &[3, 2, 2, 5, 4]), Some(vec![20, 0, 0, 4, 1]));
        assert_eq!(broadcast_strides(&[5, 4], &[3, 2, 2, 5, 4]), Some(vec![0, 0, 0, 4, 1]));
        assert_eq!(broadcast_strides(&[2, 4], &[3, 2, 2, 5, 4]), None);
    }
}
