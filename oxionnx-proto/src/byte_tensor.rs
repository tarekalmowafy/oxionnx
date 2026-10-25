//! Raw-byte storage for the uint8 initializers of block-quantized contrib ops.
//!
//! The runtime `Tensor` is f32-only, so a uint8 initializer normally becomes
//! one f32 per byte, which is 4x its file size. The packed weights of
//! `com.microsoft::MatMulNBits` (`B`, `zero_points`) and
//! `com.microsoft::GatherBlockQuantized` (`data`, `zero_points`) are only ever
//! read as bytes by their kernels, so the loader keeps them as **byte
//! tensors**: each innermost row of `last` bytes is zero-padded to a multiple
//! of 4 and packed little-endian into f32 words, giving shape
//! `[d0, .., d_{r-2}, ceil(last / 4)]`.
//!
//! A consumer learns which of its inputs are byte tensors from node attributes
//! the loader adds:
//! * [`ATTR_INPUTS`]: input indices held as byte tensors;
//! * [`ATTR_LAST`]: their original innermost dimension, in bytes, parallel to
//!   [`ATTR_INPUTS`];
//! * [`ATTR_DTYPES`]: the ONNX `data_type` of every input that is an
//!   initializer (`0` for the others), since the f32 runtime tensors no longer
//!   carry it.
//!
//! An initializer is stored this way only when every reference to it is such
//! a slot, so no other operator ever sees the packed form.

use std::collections::{HashMap, HashSet};

use crate::reader::{dtype_code, ReaderError};
use crate::types::{AttributeProto, AttributeValue, GraphProto, TensorProto};
use oxionnx_core::Tensor;

/// Int-list attribute: indices of the inputs that are byte tensors.
pub const ATTR_INPUTS: &str = "__oxi_u8_inputs";
/// Int-list attribute: original innermost dimension (bytes) of each byte tensor.
pub const ATTR_LAST: &str = "__oxi_u8_last";
/// Int-list attribute: ONNX `data_type` of each initializer input (0 = not one).
pub const ATTR_DTYPES: &str = "__oxi_init_dtypes";

/// Input slots of `(domain, op_type)` that are read as raw bytes.
fn byte_slots(domain: &str, op_type: &str) -> &'static [usize] {
    match (domain, op_type) {
        ("com.microsoft", "MatMulNBits") => &[1, 3],
        ("com.microsoft", "GatherBlockQuantized") => &[0, 3],
        _ => &[],
    }
}

/// Every value name a subgraph (or its nested subgraphs) reads.
fn subgraph_refs(g: &GraphProto, out: &mut HashSet<String>) {
    for n in &g.nodes {
        out.extend(n.inputs.iter().cloned());
        for a in &n.attributes {
            if let Some(sg) = &a.value.g {
                subgraph_refs(sg, out);
            }
            for sg in &a.value.graphs {
                subgraph_refs(sg, out);
            }
        }
    }
}

fn int_list(name: &str, ints: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        value: AttributeValue { ints, attr_type: crate::reader::attr_type::INTS, ..Default::default() },
    }
}

/// Pick the uint8 initializers of `graph` to hold as byte tensors and annotate
/// their consumers (see the module docs). Returns the chosen names.
pub fn plan(graph: &mut GraphProto) -> HashSet<String> {
    let dtypes: HashMap<&str, i32> = graph.initializers.iter().map(|t| (t.name.as_str(), t.data_type)).collect();
    // Names referenced anywhere other than a byte slot are left alone.
    let mut other: HashSet<String> = graph.outputs.iter().cloned().collect();
    let mut eligible: HashSet<String> = HashSet::new();
    for n in &graph.nodes {
        let slots = byte_slots(&n.domain, &n.op_type);
        for (i, name) in n.inputs.iter().enumerate() {
            if slots.contains(&i) && dtypes.get(name.as_str()) == Some(&dtype_code::UINT8) {
                eligible.insert(name.clone());
            } else {
                other.insert(name.clone());
            }
        }
        for a in &n.attributes {
            if let Some(sg) = &a.value.g {
                subgraph_refs(sg, &mut other);
            }
            for sg in &a.value.graphs {
                subgraph_refs(sg, &mut other);
            }
        }
    }
    eligible.retain(|name| !other.contains(name));
    let dims: HashMap<String, i64> = graph
        .initializers
        .iter()
        .filter(|t| eligible.contains(&t.name))
        .map(|t| (t.name.clone(), t.dims.last().copied().unwrap_or(1)))
        .collect();
    let dtypes: HashMap<String, i32> = dtypes.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    for n in &mut graph.nodes {
        if byte_slots(&n.domain, &n.op_type).is_empty() {
            continue;
        }
        let (mut idx, mut last) = (vec![], vec![]);
        for (i, name) in n.inputs.iter().enumerate() {
            if let Some(&d) = dims.get(name) {
                idx.push(i as i64);
                last.push(d);
            }
        }
        let types = n.inputs.iter().map(|name| i64::from(*dtypes.get(name).unwrap_or(&0))).collect();
        n.attributes.push(int_list(ATTR_DTYPES, types));
        if !idx.is_empty() {
            n.attributes.push(int_list(ATTR_INPUTS, idx));
            n.attributes.push(int_list(ATTR_LAST, last));
        }
    }
    eligible
}

/// Pack `bytes` (a uint8 tensor of shape `dims`) into a byte tensor.
pub fn pack(name: &str, dims: &[i64], bytes: &[u8]) -> Result<Tensor, ReaderError> {
    let mut shape = Vec::with_capacity(dims.len().max(1));
    for &d in dims {
        shape.push(usize::try_from(d).map_err(|_| ReaderError::InvalidDim { tensor: name.to_string(), dim: d })?);
    }
    let count = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    if count != Some(bytes.len()) {
        return Err(ReaderError::ElementCountMismatch {
            tensor: name.to_string(),
            dims: dims.to_vec(),
            expected: count.unwrap_or(usize::MAX),
            found: bytes.len(),
        });
    }
    if shape.is_empty() {
        shape.push(1);
    }
    let last = *shape.last().expect("non-empty shape");
    let words = last.div_ceil(4);
    let rows = bytes.len().checked_div(last).unwrap_or(0);
    let mut data = Vec::with_capacity(rows * words);
    if last > 0 {
        for row in bytes.chunks(last) {
            for w in row.chunks(4) {
                let mut b = [0u8; 4];
                b[..w.len()].copy_from_slice(w);
                data.push(f32::from_bits(u32::from_le_bytes(b)));
            }
        }
    }
    *shape.last_mut().expect("non-empty shape") = words;
    Ok(Tensor { data, shape })
}

/// Raw uint8 payload of an inline initializer (`raw_data`, or `int32_data`).
pub fn inline_bytes(tp: &TensorProto) -> Vec<u8> {
    if !tp.raw_data.is_empty() {
        tp.raw_data.clone()
    } else {
        tp.int32_data.iter().map(|&v| v as u8).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeProto;

    #[test]
    fn pack_pads_rows_to_words() {
        let t = pack("t", &[2, 5], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        let words: Vec<u32> = t.data.iter().map(|f| f.to_bits()).collect();
        assert_eq!(words, vec![0x0403_0201, 0x05, 0x0908_0706, 0x0a]);
        assert!(pack("t", &[3], &[1, 2]).is_err());
    }

    #[test]
    fn plan_only_takes_exclusive_byte_slots() {
        let init = |name: &str, dt| TensorProto { name: name.into(), data_type: dt, dims: vec![4, 1, 16], ..Default::default() };
        let node = |op: &str, inputs: &[&str]| NodeProto {
            op_type: op.into(),
            domain: if op == "Add" { String::new() } else { "com.microsoft".into() },
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let mut g = GraphProto {
            initializers: vec![init("b", 2), init("s", 1), init("z", 2), init("shared", 2)],
            nodes: vec![
                node("MatMulNBits", &["a", "b", "s", "z"]),
                node("MatMulNBits", &["a", "shared", "s"]),
                node("Add", &["shared", "a"]),
            ],
            ..Default::default()
        };
        let chosen = plan(&mut g);
        assert_eq!(chosen, ["b", "z"].iter().map(|s| s.to_string()).collect());
        let attr = |n: &NodeProto, k: &str| n.attributes.iter().find(|a| a.name == k).map(|a| a.value.ints.clone());
        assert_eq!(attr(&g.nodes[0], ATTR_INPUTS), Some(vec![1, 3]));
        assert_eq!(attr(&g.nodes[0], ATTR_LAST), Some(vec![16, 16]));
        assert_eq!(attr(&g.nodes[0], ATTR_DTYPES), Some(vec![0, 2, 1, 2]));
        assert_eq!(attr(&g.nodes[1], ATTR_INPUTS), None);
        assert_eq!(attr(&g.nodes[2], ATTR_DTYPES), None);
    }
}
