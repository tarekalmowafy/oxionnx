//! Operator trait implementations for indexing and scatter/gather operations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::indexing;

// ── Gather ──────────────────────────────────────────────────────────────────

pub struct GatherOp;

impl GatherOp {
    /// Gather along axis 0 of a table the session left in the mapped model
    /// file (its tensor is released, see [`oxionnx_core::mapped`]): only the
    /// requested rows are read. Writes `out` and returns the output shape.
    fn gather_mapped(ctx: &OpContext<'_>, data: &Tensor, indices: &Tensor, out: &mut Vec<f32>) -> Result<Vec<usize>, OnnxError> {
        let bad = |m: String| OnnxError::Internal(format!("Gather '{}': {m}", ctx.node.name));
        let name = ctx.node.inputs.first().map(String::as_str).unwrap_or_default();
        let table = ctx
            .registry
            .and_then(|r| r.mapped_weight(name))
            .ok_or_else(|| bad(format!("table '{name}' was released and has no mapped copy")))?;
        let axis = indexing::normalize_axis(ctx.attrs().i("axis", 0), data.shape.len(), "Gather")?;
        if axis != 0 {
            return Err(bad(format!("mapped table '{name}' gathered along axis {axis}")));
        }
        let rows = data.shape[0];
        let inner: usize = data.shape[1..].iter().product();
        if table.data_type() == oxionnx_core::mapped::data_type::UINT8 || table.len() != rows * inner {
            return Err(bad(format!("mapped table '{name}' does not match shape {:?}", data.shape)));
        }
        let shape: Vec<usize> = indices.shape.iter().chain(&data.shape[1..]).copied().collect();
        out.resize(indices.data.len() * inner, 0.0);
        let direct = table.as_f32();
        for (&i, dst) in indices.data.iter().zip(out.chunks_exact_mut(inner.max(1))) {
            let r = indexing::normalize_index(i, rows, "gather")?;
            match direct {
                Some(src) => dst.copy_from_slice(&src[r * inner..(r + 1) * inner]),
                None => table.decode_f32(r * inner, dst),
            }
        }
        Ok(shape)
    }
}

/// A table the session released and serves from the mapped model file.
fn released(t: &Tensor) -> bool {
    t.data.len() != t.shape.iter().product::<usize>()
}

impl Operator for GatherOp {
    fn op_type(&self) -> &str {
        "Gather"
    }
    fn reads_mapped(&self, ctx: &OpContext<'_>, input: usize) -> bool {
        let rank = ctx.inputs.first().copied().flatten().map_or(0, |t| t.shape.len());
        input == 0 && indexing::normalize_axis(ctx.attrs().i("axis", 0), rank, "Gather").is_ok_and(|a| a == 0)
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let idx = ctx.input(1)?;
        if released(x) {
            let mut out = Vec::new();
            let shape = Self::gather_mapped(ctx, x, idx, &mut out)?;
            return Ok(vec![Tensor::new(out, shape)]);
        }
        let axis = ctx.attrs().i("axis", 0);
        Ok(vec![indexing::gather(x, idx, axis)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let data = ctx.input(0)?;
        let indices = ctx.input(1)?;
        if released(data) {
            slots[0].shape = Self::gather_mapped(ctx, data, indices, &mut slots[0].data)?;
            return Ok(());
        }
        let axis_raw = ctx.attrs().i("axis", 0);
        let axis = indexing::normalize_axis(axis_raw, data.shape.len(), "Gather")?;
        // Output shape: data.shape[..axis] ++ indices.shape ++ data.shape[axis+1..]
        let out_shape: Vec<usize> = data.shape[..axis]
            .iter()
            .chain(indices.shape.iter())
            .chain(data.shape[axis + 1..].iter())
            .copied()
            .collect();
        let out_len: usize = out_shape.iter().product();
        if slots[0].data.len() != out_len {
            slots[0].data.resize(out_len, 0.0_f32);
        }
        slots[0].shape.clone_from(&out_shape);
        indexing::gather_into(data, &indices.data, axis, &mut slots[0].data)?;
        Ok(())
    }
}

// ── GatherElements ──────────────────────────────────────────────────────────

pub struct GatherElementsOp;
impl Operator for GatherElementsOp {
    fn op_type(&self) -> &str {
        "GatherElements"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let idx = ctx.input(1)?;
        let axis = ctx.attrs().i("axis", 0);
        Ok(vec![indexing::gather_elements(x, idx, axis)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── GatherND ────────────────────────────────────────────────────────────────

pub struct GatherNDOp;
impl Operator for GatherNDOp {
    fn op_type(&self) -> &str {
        "GatherND"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let batch_dims = ctx.attrs().i("batch_dims", 0);
        Ok(vec![indexing::gather_nd(
            ctx.input(0)?,
            ctx.input(1)?,
            batch_dims,
        )?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── Where ───────────────────────────────────────────────────────────────────

pub struct WhereOp;
impl Operator for WhereOp {
    fn op_type(&self) -> &str {
        "Where"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let cond = ctx.input(0)?;
        let x = ctx.input(1)?;
        let y = ctx.input(2)?;
        Ok(vec![indexing::where_op(cond, x, y)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── ScatterElements ─────────────────────────────────────────────────────────

pub struct ScatterElementsOp;
impl Operator for ScatterElementsOp {
    fn op_type(&self) -> &str {
        "ScatterElements"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let data = ctx.input(0)?;
        let indices = ctx.input(1)?;
        let updates = ctx.input(2)?;
        let axis = ctx.attrs().i("axis", 0);
        let reduction = indexing::ScatterReduction::parse(ctx.attrs().s("reduction"))?;
        Ok(vec![indexing::scatter_elements_reduce(
            data, indices, updates, axis, reduction,
        )?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let data = ctx.input(0)?;
        let indices = ctx.input(1)?;
        let updates = ctx.input(2)?;
        let axis_raw = ctx.attrs().i("axis", 0);
        let axis = indexing::normalize_axis(axis_raw, data.shape.len(), "ScatterElements")?;
        let reduction = indexing::ScatterReduction::parse(ctx.attrs().s("reduction"))?;
        if slots[0].data.len() != data.data.len() {
            slots[0].data.resize(data.data.len(), 0.0_f32);
        }
        slots[0].shape.clone_from(&data.shape);
        slots[0].data.copy_from_slice(&data.data);
        indexing::scatter_elements_into(
            &data.shape,
            &indices.shape,
            &indices.data,
            &updates.data,
            axis,
            reduction,
            &mut slots[0].data,
        )?;
        Ok(())
    }
}

// ── ScatterND ───────────────────────────────────────────────────────────────

pub struct ScatterNDOp;
impl Operator for ScatterNDOp {
    fn op_type(&self) -> &str {
        "ScatterND"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let data = ctx.input(0)?;
        let indices = ctx.input(1)?;
        let updates = ctx.input(2)?;
        let reduction = indexing::ScatterReduction::parse(ctx.attrs().s("reduction"))?;
        Ok(vec![indexing::scatter_nd_reduce(
            data, indices, updates, reduction,
        )?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let data = ctx.input(0)?;
        let indices = ctx.input(1)?;
        let updates = ctx.input(2)?;
        let k = *indices.shape.last().ok_or_else(|| {
            OnnxError::ShapeMismatch("ScatterND: indices must be at least 1D".to_string())
        })?;
        let reduction = indexing::ScatterReduction::parse(ctx.attrs().s("reduction"))?;
        if slots[0].data.len() != data.data.len() {
            slots[0].data.resize(data.data.len(), 0.0_f32);
        }
        slots[0].shape.clone_from(&data.shape);
        slots[0].data.copy_from_slice(&data.data);
        indexing::scatter_nd_into(
            &data.shape,
            k,
            &indices.data,
            &updates.data,
            reduction,
            &mut slots[0].data,
        )?;
        Ok(())
    }
}

// ── QuantizeLinear ──────────────────────────────────────────────────────────

pub struct QuantizeLinearOp;
impl Operator for QuantizeLinearOp {
    fn op_type(&self) -> &str {
        "QuantizeLinear"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let zp = ctx.optional_input(2);
        let axis = ctx.attrs().i("axis", 1);
        Ok(vec![indexing::quantize_linear_axis(x, scale, zp, axis)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── DequantizeLinear ────────────────────────────────────────────────────────

pub struct DequantizeLinearOp;
impl Operator for DequantizeLinearOp {
    fn op_type(&self) -> &str {
        "DequantizeLinear"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let x = ctx.input(0)?;
        let scale = ctx.input(1)?;
        let zp = ctx.optional_input(2);
        let axis = ctx.attrs().i("axis", 1);
        Ok(vec![indexing::dequantize_linear_axis(x, scale, zp, axis)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── OneHot ──────────────────────────────────────────────────────────────────

pub struct OneHotOp;
impl Operator for OneHotOp {
    fn op_type(&self) -> &str {
        "OneHot"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let indices = ctx.input(0)?;
        let depth = ctx.input(1)?.data[0] as usize;
        let values_t = ctx.input(2)?;
        let off_val = values_t.data[0];
        let on_val = if values_t.data.len() > 1 {
            values_t.data[1]
        } else {
            1.0
        };
        let axis = ctx.attrs().i("axis", -1);
        Ok(vec![indexing::one_hot(
            indices,
            depth,
            (off_val, on_val),
            axis,
        )?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── Compress ────────────────────────────────────────────────────────────────

pub struct CompressOp;
impl Operator for CompressOp {
    fn op_type(&self) -> &str {
        "Compress"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let axis = {
            let v = ctx.attrs().i("axis", i64::MIN);
            if v == i64::MIN {
                None
            } else {
                Some(v)
            }
        };
        Ok(vec![indexing::compress(
            ctx.input(0)?,
            ctx.input(1)?,
            axis,
        )?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

// ── Unique ──────────────────────────────────────────────────────────────────

pub struct UniqueOp;
impl Operator for UniqueOp {
    fn op_type(&self) -> &str {
        "Unique"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let attrs = ctx.attrs();
        let axis = {
            let v = attrs.i("axis", i64::MIN);
            if v == i64::MIN {
                None
            } else {
                Some(v)
            }
        };
        let sorted = attrs.i("sorted", 1) != 0;
        let (y, idx, inv, counts) = indexing::unique(ctx.input(0)?, axis, sorted)?;
        Ok(vec![y, idx, inv, counts])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod mapped_tests {
    use super::*;
    use oxionnx_core::graph::{Attributes, Node, OpKind};
    use oxionnx_core::mapped::{data_type, SharedBytes};
    use oxionnx_core::{MappedWeight, OperatorRegistry};
    use std::sync::Arc;

    fn gather_node(axis: Option<i64>) -> Node {
        let mut attrs = Attributes::default();
        if let Some(a) = axis {
            attrs.ints.insert("axis".into(), a);
        }
        Node { op: OpKind::Unknown("Gather".into()), name: "g".into(), inputs: vec!["emb".into(), "ids".into()], outputs: vec!["y".into()], attrs }
    }

    /// Rows of a released table served from mapped bytes equal the decoded
    /// table's, through both execute paths, for FP32 (aligned and not) and
    /// FP16/BF16 storage.
    #[test]
    fn gather_reads_mapped_rows() {
        let (rows, inner) = (7usize, 5usize);
        let vals: Vec<f32> = (0..rows * inner).map(|i| i as f32 * 0.37 - 3.0).collect();
        let ids = Tensor::new(vec![6.0, 0.0, -1.0, 3.0, 3.0, 2.0], vec![3, 2]);
        let dense = Tensor::new(vals.clone(), vec![rows, inner]);
        let nd = gather_node(None);
        {
            let ctx = OpContext { node: &nd, inputs: vec![Some(&dense), Some(&ids)], outer_scope: None, weights: None, registry: None };
            let y = GatherOp.execute(&ctx).unwrap().remove(0);
            assert_eq!(y.shape, vec![3, 2, inner]);
            assert_eq!(&y.data[..inner], &vals[6 * inner..]);
        }
        let f16: Vec<u8> = vals.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
        let bf16: Vec<u8> = vals.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
        let f32b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let cases: Vec<(Vec<u8>, i32, Vec<f32>)> = vec![
            (f32b, data_type::FLOAT, vals.clone()),
            (f16, data_type::FLOAT16, vals.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()),
            (bf16, data_type::BFLOAT16, vals.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect()),
        ];
        for (bytes, dt, decoded) in cases {
            let want = {
                let d = Tensor::new(decoded, vec![rows, inner]);
                let ctx = OpContext { node: &nd, inputs: vec![Some(&d), Some(&ids)], outer_scope: None, weights: None, registry: None };
                GatherOp.execute(&ctx).unwrap().remove(0)
            };
            for pad in [0usize, 1, 2, 3] {
                let mut file = vec![0x5Au8; pad];
                file.extend(&bytes);
                let owner: SharedBytes = Arc::new(file);
                let reg = OperatorRegistry::new();
                reg.set_mapped_weight("emb".into(), MappedWeight::new(owner, pad..pad + bytes.len(), dt).unwrap());
                let released = Tensor { data: vec![], shape: vec![rows, inner] };
                for axis in [None, Some(0), Some(-2)] {
                    let nd = gather_node(axis);
                    let ctx = OpContext { node: &nd, inputs: vec![Some(&released), Some(&ids)], outer_scope: None, weights: None, registry: Some(&reg) };
                    assert!(GatherOp.reads_mapped(&ctx, 0));
                    assert!(!GatherOp.reads_mapped(&ctx, 1));
                    let y = GatherOp.execute(&ctx).unwrap().remove(0);
                    assert_eq!((&y.shape, &y.data), (&want.shape, &want.data), "dt {dt} pad {pad}");
                    let mut slots = vec![Tensor { data: vec![9.0; 3], shape: vec![3] }];
                    GatherOp.execute_into_slots(&ctx, &mut slots).unwrap();
                    assert_eq!((&slots[0].shape, &slots[0].data), (&want.shape, &want.data));
                    let bad = Tensor::new(vec![7.0], vec![1]);
                    let ctx = OpContext { inputs: vec![Some(&released), Some(&bad)], ..ctx };
                    assert!(GatherOp.execute(&ctx).is_err());
                }
                let nd = gather_node(Some(1));
                let ctx = OpContext { node: &nd, inputs: vec![Some(&released), Some(&ids)], outer_scope: None, weights: None, registry: Some(&reg) };
                assert!(!GatherOp.reads_mapped(&ctx, 0), "only axis 0 reads rows");
                assert!(GatherOp.execute(&ctx).is_err());
                let nd = gather_node(None);
                let ctx = OpContext { node: &nd, inputs: vec![Some(&released), Some(&ids)], outer_scope: None, weights: None, registry: None };
                assert!(GatherOp.execute(&ctx).is_err(), "no mapped copy");
            }
        }
    }
}
