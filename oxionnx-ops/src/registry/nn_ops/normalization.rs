//! Normalization operator implementations: LayerNorm, GroupNorm, BatchNorm,
//! RmsNorm, InstanceNorm, LpNorm, MeanVarianceNormalization.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::nn;

// ── LayerNorm ───────────────────────────────────────────────────────────────

pub struct LayerNormOp;

impl LayerNormOp {
    /// Writes `Y` and, when the node declares them, `Mean` and `InvStdDev`
    /// (shape of `X` with the normalised dims set to 1).
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let eps = ctx.attrs().f("epsilon", 1e-5);
        let axis = ctx.attrs().i("axis", -1);
        let ndim = x.ndim() as i64;
        let ax = if axis < 0 { axis + ndim } else { axis };
        if ax < 0 || ax > ndim {
            return Err(OnnxError::ShapeMismatch(format!("LayerNormalization: axis {axis} out of range for rank {ndim}")));
        }
        let mut stat_shape = x.shape.clone();
        for d in &mut stat_shape[ax as usize..] {
            *d = 1;
        }
        let wants = |i: usize| ctx.node.outputs.get(i).is_some_and(|n| !n.is_empty());
        let (first, rest) = slots
            .split_first_mut()
            .ok_or_else(|| OnnxError::Internal("LayerNormOp: no output slots".into()))?;
        size_slot(first, &x.shape);
        let (mut mean, mut inv) = (None, None);
        for (i, slot) in rest.iter_mut().enumerate() {
            if wants(i + 1) {
                size_slot(slot, &stat_shape);
                if i == 0 {
                    mean = Some(&mut slot.data[..]);
                } else if i == 1 {
                    inv = Some(&mut slot.data[..]);
                }
            }
        }
        nn::normalization::layer_norm_stats_into(x, scale, bias, eps, axis, &mut first.data, mean, inv)
            .map_err(OnnxError::ShapeMismatch)
    }
}

/// Size `slot` to `shape`; its contents are fully overwritten by the caller.
fn size_slot(slot: &mut Tensor, shape: &[usize]) {
    let n: usize = shape.iter().product();
    if slot.data.len() != n {
        slot.data.resize(n, 0.0_f32);
    }
    slot.shape.clear();
    slot.shape.extend_from_slice(shape);
}

impl Operator for LayerNormOp {
    fn op_type(&self) -> &str {
        "LayerNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let n_out = ctx.node.outputs.len().clamp(1, 3);
        let mut slots: Vec<Tensor> = (0..n_out).map(|_| Tensor { data: vec![], shape: vec![] }).collect();
        self.run(ctx, &mut slots)?;
        Ok(slots)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        self.run(ctx, slots)
    }
}

// ── GroupNorm ───────────────────────────────────────────────────────────────

pub struct GroupNormOp;
impl Operator for GroupNormOp {
    fn op_type(&self) -> &str {
        "GroupNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let num_groups = attrs.i("num_groups", 1) as usize;
        let eps = attrs.f("epsilon", 1e-5);
        Ok(vec![nn::group_norm(x, scale, bias, num_groups, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal("GroupNormOp: no output slots".into()));
        }
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.optional_input(2);
        let num_groups = ctx.attrs().i("num_groups", 1) as usize;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        let n = x.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&x.shape);
        nn::normalization::group_norm_into(x, scale, bias, num_groups, eps, &mut slots[0].data)?;
        Ok(())
    }
}

// ── BatchNorm ───────────────────────────────────────────────────────────────

pub struct BatchNormOp;
impl Operator for BatchNormOp {
    fn op_type(&self) -> &str {
        "BatchNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let mean = ctx.input(3)?;
        let var = ctx.input(4)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        Ok(vec![nn::batch_norm(x, scale, bias, mean, var, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal("BatchNormOp: no output slots".into()));
        }
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let mean = ctx.input(3)?;
        let var = ctx.input(4)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        let n = x.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&x.shape);
        nn::normalization::batch_norm_into(x, scale, bias, mean, var, eps, &mut slots[0].data)?;
        Ok(())
    }
}

// ── RMSNorm ─────────────────────────────────────────────────────────────────

pub struct RmsNormOp;
impl Operator for RmsNormOp {
    fn op_type(&self) -> &str {
        "SimplifiedLayerNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let eps = ctx.attrs().f("epsilon", 1e-6);
        Ok(vec![nn::rms_norm(x, scale, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal("RmsNormOp: no output slots".into()));
        }
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let eps = ctx.attrs().f("epsilon", 1e-6);
        let n = x.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&x.shape);
        nn::normalization::rms_norm_into(x, scale, eps, &mut slots[0].data)?;
        Ok(())
    }
}

// ── InstanceNorm ────────────────────────────────────────────────────────────

pub struct InstanceNormOp;
impl Operator for InstanceNormOp {
    fn op_type(&self) -> &str {
        "InstanceNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        Ok(vec![nn::instance_norm(x, scale, bias, eps)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        if slots.is_empty() {
            return Err(OnnxError::Internal(
                "InstanceNormOp: no output slots".into(),
            ));
        }
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let bias = ctx.input(2)?;
        let eps = ctx.attrs().f("epsilon", 1e-5);
        let n = x.numel();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&x.shape);
        nn::normalization::instance_norm_into(x, scale, bias, eps, &mut slots[0].data)?;
        Ok(())
    }
}

// ── LpNorm ──────────────────────────────────────────────────────────────────

pub struct LpNormOp;
impl Operator for LpNormOp {
    fn op_type(&self) -> &str {
        "LpNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let axis = attrs.i("axis", -1);
        let p = attrs.i("p", 2);
        Ok(vec![nn::lp_norm(ctx.input(0)?, axis, p)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let axis = ctx.attrs().i("axis", -1);
        let p = ctx.attrs().i("p", 2);
        let ndim = input.ndim();
        let ax = if axis < 0 {
            (axis + ndim as i64) as usize
        } else {
            axis as usize
        };
        if ax >= ndim {
            return Err(OnnxError::from(format!(
                "lp_norm: axis {ax} out of range for ndim {ndim}"
            )));
        }
        let n = input.data.len();
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        slots[0].data.copy_from_slice(&input.data);

        let axis_size = input.shape[ax];
        let outer: usize = input.shape[..ax].iter().product();
        let inner: usize = input.shape[ax + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                let norm: f32 = if p == 1 {
                    (0..axis_size)
                        .map(|a| input.data[(o * axis_size + a) * inner + i].abs())
                        .sum()
                } else {
                    (0..axis_size)
                        .map(|a| {
                            let v = input.data[(o * axis_size + a) * inner + i];
                            v * v
                        })
                        .sum::<f32>()
                        .sqrt()
                };
                let norm = if norm == 0.0 { 1.0 } else { norm };
                for a in 0..axis_size {
                    slots[0].data[(o * axis_size + a) * inner + i] /= norm;
                }
            }
        }
        Ok(())
    }
}

// ── MeanVarianceNormalization ───────────────────────────────────────────────

pub struct MeanVarianceNormalizationOp;
impl Operator for MeanVarianceNormalizationOp {
    fn op_type(&self) -> &str {
        "MeanVarianceNormalization"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axes_list = ctx.attrs().ints("axes");
        let axes = if axes_list.is_empty() {
            vec![0, 2, 3]
        } else {
            axes_list.to_vec()
        };
        Ok(vec![nn::mean_variance_normalization(ctx.input(0)?, &axes)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let input = ctx.input(0)?;
        let axes_list = ctx.attrs().ints("axes");
        let axes: Vec<i64> = if axes_list.is_empty() {
            vec![0, 2, 3]
        } else {
            axes_list.to_vec()
        };

        let ndim = input.ndim();
        if ndim == 0 {
            return Err(OnnxError::from(
                "mean_variance_normalization: input must have at least 1 dimension".to_string(),
            ));
        }

        let norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + ndim as i64) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        for &ax in &norm_axes {
            if ax >= ndim {
                return Err(OnnxError::from(format!(
                    "mean_variance_normalization: axis {ax} out of range for {ndim}D tensor"
                )));
            }
        }

        let total = input.numel();
        let n = total;
        if slots[0].data.len() != n {
            slots[0].data.resize(n, 0.0_f32);
        }
        slots[0].shape.clone_from(&input.shape);
        slots[0].data.copy_from_slice(&input.data);

        let mut strides = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * input.shape[d + 1];
        }

        let mut is_reduced = vec![false; ndim];
        for &ax in &norm_axes {
            is_reduced[ax] = true;
        }

        let reduced_size: usize = (0..ndim)
            .filter(|&d| is_reduced[d])
            .map(|d| input.shape[d])
            .product::<usize>()
            .max(1);
        let non_reduced_size = total / reduced_size;

        let non_reduced_dims: Vec<usize> = (0..ndim).filter(|&d| !is_reduced[d]).collect();
        let reduced_dims: Vec<usize> = (0..ndim).filter(|&d| is_reduced[d]).collect();

        for nr_idx in 0..non_reduced_size {
            let mut nr_coords = vec![0usize; non_reduced_dims.len()];
            let mut rem = nr_idx;
            for j in (0..non_reduced_dims.len()).rev() {
                let dim = non_reduced_dims[j];
                nr_coords[j] = rem % input.shape[dim];
                rem /= input.shape[dim];
            }

            let mut flat_indices = Vec::with_capacity(reduced_size);
            nn::collect_reduced_indices(
                input,
                &strides,
                &reduced_dims,
                &non_reduced_dims,
                &nr_coords,
                0,
                0,
                &mut flat_indices,
            );

            let mean: f32 =
                flat_indices.iter().map(|&fi| input.data[fi]).sum::<f32>() / reduced_size as f32;
            let var: f32 = flat_indices
                .iter()
                .map(|&fi| (input.data[fi] - mean) * (input.data[fi] - mean))
                .sum::<f32>()
                / reduced_size as f32;
            let inv_std = 1.0 / (var + 1e-9_f32).sqrt();

            for &fi in &flat_indices {
                slots[0].data[fi] = (input.data[fi] - mean) * inv_std;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod layer_norm_op_tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};

    #[test]
    fn layer_norm_mean_and_inv_std_outputs() {
        // [2, 3, 4] normalised over the last two axes (axis = 1).
        let x: Vec<f32> = (0..24).map(|i| ((i * 7) % 11) as f32 * 0.3 - 1.0).collect();
        let x = Tensor::new(x, vec![2, 3, 4]);
        let scale = Tensor::new((0..12).map(|i| 0.5 + i as f32 * 0.1).collect(), vec![3, 4]);
        let bias = Tensor::new((0..12).map(|i| i as f32 * 0.01).collect(), vec![3, 4]);
        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".into(), 1);
        attrs.floats.insert("epsilon".into(), 1e-5);
        let node = Node {
            op: OpKind::LayerNorm,
            name: "ln".into(),
            inputs: vec!["x".into(), "s".into(), "b".into()],
            outputs: vec!["y".into(), "mean".into(), "inv".into()],
            attrs,
        };
        let ctx = OpContext {
            node: &node,
            inputs: vec![Some(&x), Some(&scale), Some(&bias)],
            outer_scope: None,
            weights: None,
            registry: None,
        };
        let out = LayerNormOp.execute(&ctx).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].shape, vec![2, 1, 1]);
        assert_eq!(out[2].shape, vec![2, 1, 1]);
        for r in 0..2 {
            let row = &x.data[r * 12..(r + 1) * 12];
            let m = row.iter().map(|&v| v as f64).sum::<f64>() / 12.0;
            let var = row.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / 12.0;
            let inv = 1.0 / (var + 1e-5).sqrt();
            assert!((out[1].data[r] as f64 - m).abs() < 1e-5);
            assert!((out[2].data[r] as f64 - inv).abs() < 1e-4 * inv);
            for j in 0..12 {
                let want = (row[j] as f64 - m) * inv * scale.data[j] as f64 + bias.data[j] as f64;
                assert!((out[0].data[r * 12 + j] as f64 - want).abs() < 1e-5, "y[{r},{j}]");
            }
        }
        // Slot path matches.
        let mut slots: Vec<Tensor> = (0..3).map(|_| Tensor::new(vec![], vec![0])).collect();
        LayerNormOp.execute_into_slots(&ctx, &mut slots).unwrap();
        for (a, b) in slots.iter().zip(&out) {
            assert_eq!(a.data, b.data);
            assert_eq!(a.shape, b.shape);
        }
    }
}
