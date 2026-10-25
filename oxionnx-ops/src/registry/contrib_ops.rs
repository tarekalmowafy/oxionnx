//! Operators for ONNX Runtime contrib ops (`com.microsoft` domain) and the
//! optimizer-only fused ops (attention, MatMul + activation). Kernels live in
//! [`crate::contrib`] and [`crate::fused_matmul`].
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

// ── OxiMatMulAct (optimizer-only) ────────────────────────────────────────────

/// `MatMul` by a constant 2-D weight with a fused activation epilogue, emitted
/// by the optimizer's MatMul-activation fusion (see [`crate::fused_matmul`]).
///
/// Inputs: `x` `[.., K]`, `w` `[K, N]`, optional `bias` `[N]`, optional `mul`.
/// Attributes: `activation` (`"gelu"`), `approximate` (`"tanh"` or `"none"`),
/// `gate` (`-1`: `act(x@w + bias) * mul`, output `[.., N]`; `0`/`1`: gated, the
/// activation applies to that half of the `N` columns and multiplies the other
/// half, output `[.., N/2]`, no `mul`).
#[derive(Default)]
pub struct OxiMatMulActOp {
    cache: WeightCache,
}

/// Constant weights packed once for [`crate::gemm_packed`], keyed by (data
/// pointer, layout tag), as in `MatMulOp`. Per session, like the registry.
#[derive(Default)]
struct WeightCache {
    #[cfg(target_arch = "x86_64")]
    packed: std::sync::RwLock<std::collections::HashMap<(usize, i64), std::sync::Arc<crate::gemm_packed::PackedB>>>,
}

#[cfg(target_arch = "x86_64")]
impl WeightCache {
    /// The packed form of input 1 when it is a 2-D session weight and the packed
    /// GEMM is available; `pack(isa)` builds it on first use (BF16 when the
    /// session enables BF16 weights).
    fn get(
        &self,
        ctx: &OpContext<'_>,
        w: &Tensor,
        tag: i64,
        pack: impl FnOnce(crate::gemm_packed::Isa) -> crate::gemm_packed::PackedB,
    ) -> Option<std::sync::Arc<crate::gemm_packed::PackedB>> {
        let is_weight = ctx.weights.and_then(|ws| ws.get(ctx.node.inputs.get(1)?)).is_some_and(|t| std::ptr::eq(t, w));
        if !crate::gemm_packed::available() || !is_weight || w.ndim() != 2 {
            return None;
        }
        let key = (w.data.as_ptr() as usize, tag);
        if let Some(p) = self.packed.read().ok().and_then(|c| c.get(&key).cloned()) {
            if (p.k, p.n) == (w.shape[0], w.shape[1]) {
                return Some(p);
            }
        }
        let p = std::sync::Arc::new(pack(crate::gemm_packed::weight_isa(crate::gemm_bf16::weights_enabled(ctx.registry))?));
        if let Ok(mut c) = self.packed.write() {
            c.insert(key, p.clone());
        }
        Some(p)
    }
}

impl OxiMatMulActOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        use crate::fused_matmul::{self as fm, Act, Spec};
        let (x, w) = (ctx.input(0)?, ctx.input(1)?);
        let (bias, mul) = (ctx.optional_input(2), ctx.optional_input(3));
        let attrs = ctx.attrs();
        let bad = |m: String| err(format!("OxiMatMulAct '{}': {m}", ctx.node.name));
        if attrs.s("activation") != "gelu" {
            return Err(bad(format!("unsupported activation {:?}", attrs.s("activation"))));
        }
        let act = if attrs.s("approximate") == "tanh" { Act::GeluTanh } else { Act::GeluErf };
        let gate = match attrs.i("gate", -1) {
            -1 => None,
            h @ (0 | 1) => Some(h as usize),
            h => return Err(bad(format!("gate must be -1, 0 or 1, got {h}"))),
        };
        if w.ndim() != 2 || x.ndim() == 0 || x.shape[x.ndim() - 1] != w.shape[0] {
            return Err(bad(format!("x {:?} @ w {:?}", x.shape, w.shape)));
        }
        let (k, n) = (w.shape[0], w.shape[1]);
        if gate.is_some() && (n % 2 != 0 || mul.is_some()) {
            return Err(bad(format!("gated form needs an even N (got {n}) and no mul")));
        }
        if let Some(b) = bias {
            if b.data.len() != n || b.shape.last() != Some(&n) {
                return Err(bad(format!("bias {:?} is not [{n}]", b.shape)));
            }
        }
        let n_out = if gate.is_some() { n / 2 } else { n };
        let m = x.data.len() / k.max(1);
        let mut out_shape = x.shape.clone();
        *out_shape.last_mut().unwrap() = n_out;
        // A multiplier that does not match the output elementwise is applied
        // afterwards with ONNX broadcasting.
        let mul_fused = mul.filter(|t| t.shape == out_shape);
        let spec = Spec { act, gate, bias: bias.map(|b| &b.data[..]), mul: mul_fused.map(|t| &t.data[..]) };
        let mut res = Tensor { data: std::mem::take(&mut slots[0].data), shape: vec![] };
        shape_slot(&mut res, &out_shape);
        if !self.packed(ctx, m, x, w, &spec, &mut res.data) {
            let mut product = crate::math::matmul(x, w).map_err(|e| bad(e.to_string()))?;
            fm::matmul_act_unfused(m, n, &mut product.data, &spec, &mut res.data);
        }
        match mul.filter(|_| mul_fused.is_none()) {
            Some(mt) => {
                let mut out = Tensor { data: vec![], shape: vec![] };
                crate::math::binary_into(&res, mt, &mut out, |a, b| a * b).map_err(bad)?;
                slots[0] = out;
            }
            None => slots[0] = res,
        }
        Ok(())
    }

    /// Packed-GEMM epilogue path; `false` when it does not apply.
    #[cfg(target_arch = "x86_64")]
    fn packed(&self, ctx: &OpContext<'_>, m: usize, x: &Tensor, w: &Tensor, spec: &crate::fused_matmul::Spec<'_>, out: &mut [f32]) -> bool {
        use crate::fused_matmul as fm;
        use crate::gemm_packed::PackedB;
        let (k, n) = (w.shape[0], w.shape[1]);
        let Some(cols) = fm::block_cols(n, spec.gate.is_some()) else { return false };
        let packed = self.cache.get(ctx, w, spec.gate.map_or(-1, |h| h as i64), |isa| match spec.gate {
            Some(h) => PackedB::pack_cols_for(isa, &w.data, k, n, &fm::gate_columns(n, h, cols)),
            None => PackedB::pack_for(isa, &w.data, k, n),
        });
        let Some(packed) = packed else { return false };
        let width = if spec.gate.is_some() { 2 * cols } else { cols };
        fm::matmul_act_packed(m, &x.data, &packed, width, spec, out, crate::par::threads());
        true
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn packed(&self, _: &OpContext<'_>, _: usize, _: &Tensor, _: &Tensor, _: &crate::fused_matmul::Spec<'_>, _: &mut [f32]) -> bool {
        false
    }
}

impl Operator for OxiMatMulActOp {
    fn op_type(&self) -> &str {
        "OxiMatMulAct"
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

// ── OxiMatMulSplit (optimizer-only) ──────────────────────────────────────────

/// Projection by a constant 2-D weight, split along its columns into several
/// outputs, emitted by the optimizer's MatMul-split fusion (see
/// [`crate::fused_matmul::SplitSpec`]).
///
/// Inputs: `x` `[.., K]`, `w` `[K, N]`, then optional `bias` `[N]`,
/// `head_gamma` `[D]` (per-head RMSNorm over `D`-wide heads, as
/// `Reshape [0,-1,D] -> SimplifiedLayerNormalization -> Reshape [0,-1,N]`),
/// `col_scale` `[N]`, and the `com.microsoft::RotaryEmbedding` inputs
/// `position_ids`, `cos_cache`, `sin_cache`.
/// Attributes: `split` (output widths), `epsilon`, `rotate` (per output, 1 =
/// rotary), `interleaved`, `num_heads`, `rotary_embedding_dim` (as for the
/// rotary op, applied to each rotated output).
/// Outputs: `[.., split[o]]` each (`[d0, M/d0, split[o]]` with the head norm,
/// as the norm's reshape pair produces).
#[derive(Default)]
pub struct OxiMatMulSplitOp {
    cache: WeightCache,
}

impl OxiMatMulSplitOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        use crate::fused_matmul::{self as fm, HeadNorm, SplitRotary, SplitSpec};
        let (x, w) = (ctx.input(0)?, ctx.input(1)?);
        let opt = |i| ctx.optional_input(i);
        let (bias, gamma, col_scale) = (opt(2), opt(3), opt(4));
        let attrs = ctx.attrs();
        let bad = |m: String| err(format!("OxiMatMulSplit '{}': {m}", ctx.node.name));
        let splits: Vec<usize> = attrs.ints("split").iter().map(|&v| v.max(0) as usize).collect();
        if w.ndim() != 2 || x.ndim() == 0 || x.shape[x.ndim() - 1] != w.shape[0] {
            return Err(bad(format!("x {:?} @ w {:?}", x.shape, w.shape)));
        }
        let (k, n) = (w.shape[0], w.shape[1]);
        if splits.iter().sum::<usize>() != n || splits.len() != slots.len() {
            return Err(bad(format!("split {splits:?} does not cover N={n} with {} outputs", slots.len())));
        }
        for (name, t) in [("bias", bias), ("col_scale", col_scale)] {
            if t.is_some_and(|t| t.data.len() != n) {
                return Err(bad(format!("{name} has {} elements, expected {n}", t.map_or(0, |t| t.data.len()))));
            }
        }
        let m = x.data.len() / k.max(1);
        // Output row geometry: the head norm's reshape pair yields [d0, M/d0, .].
        let lead: Vec<usize> = if gamma.is_some() {
            if x.ndim() < 2 {
                return Err(bad("head norm needs x of rank >= 2".into()));
            }
            vec![x.shape[0], m / x.shape[0].max(1)]
        } else {
            x.shape[..x.ndim() - 1].to_vec()
        };
        let mut rotate: Vec<bool> = attrs.ints("rotate").iter().map(|&v| v != 0).collect();
        rotate.resize(splits.len(), false);
        let (mut rotary, mut head) = (None, None);
        if rotate.iter().any(|&r| r) {
            let (pos, cos, sin) = (ctx.input(5)?, ctx.input(6)?, ctx.input(7)?);
            let [b, s] = lead[..] else { return Err(bad(format!("rotary needs 3-D outputs, got {lead:?}"))) };
            let half = *cos.shape.last().ok_or_else(|| bad("scalar cos cache".into()))?;
            let mut dims = None;
            for (o, _) in rotate.iter().enumerate().filter(|(_, &r)| r) {
                let d = RotaryDims::new(
                    &[b, s, splits[o]],
                    half,
                    attrs.i("num_heads", 0).max(0) as usize,
                    attrs.i("rotary_embedding_dim", 0).max(0) as usize,
                    attrs.i("interleaved", 0) != 0,
                )
                .map_err(bad)?;
                if dims.is_some_and(|p: RotaryDims| (p.head_size, p.rotary_dim) != (d.head_size, d.rotary_dim)) {
                    return Err(bad("rotated outputs differ in head size".into()));
                }
                dims = Some(d);
            }
            let d = dims.expect("some output is rotated");
            let cache = crate::contrib::RotaryCache::new(&pos.data, &cos.data, &sin.data, cos.ndim() == 3, b, s, d.rotary_dim / 2)
                .map_err(bad)?;
            rotary = Some(SplitRotary { cache, seq: s, rotary_dim: d.rotary_dim, interleaved: d.interleaved });
            head = Some(d.head_size);
        }
        let spec = SplitSpec {
            splits: splits.clone(),
            bias: bias.map(|t| &t.data[..]),
            norm: gamma.map(|g| HeadNorm { gamma: &g.data, eps: attrs.f("epsilon", 1e-5) }),
            col_scale: col_scale.map(|t| &t.data[..]),
            rotary,
            rotate,
        };
        let block = spec.block(head).ok_or_else(|| bad(format!("no column block fits split {splits:?} / head {head:?}")))?;
        for (slot, &w_o) in slots.iter_mut().zip(&splits) {
            let shape: Vec<usize> = lead.iter().copied().chain([w_o]).collect();
            shape_slot(slot, &shape);
        }
        let mut outs: Vec<&mut [f32]> = slots.iter_mut().map(|t| &mut t.data[..]).collect();
        #[cfg(target_arch = "x86_64")]
        if block % 32 == 0 {
            if let Some(p) = self.cache.get(ctx, w, -1, |isa| crate::gemm_packed::PackedB::pack_for(isa, &w.data, k, n)) {
                fm::matmul_split_packed(m, &x.data, &p, &spec, block, &mut outs, crate::par::threads());
                return Ok(());
            }
        }
        let mut product = crate::math::matmul(x, w).map_err(|e| bad(e.to_string()))?;
        fm::matmul_split_unfused(m, &mut product.data, &spec, block, &mut outs);
        Ok(())
    }
}

impl Operator for OxiMatMulSplitOp {
    fn op_type(&self) -> &str {
        "OxiMatMulSplit"
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
