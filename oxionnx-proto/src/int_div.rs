//! Mark `Div` nodes that divide integer tensors.
//!
//! The runtime `Tensor` is f32-only, so `Div` on int64 values (e.g. the
//! `floor_divide` / `//` that PyTorch exporters emit for index arithmetic)
//! would otherwise produce fractions where ONNX truncates. The loader works
//! out each value's element type from graph inputs, initializers,
//! `value_info` and a forward pass over dtype-preserving ops, and adds
//! [`ATTR_INT_DIV`] `= 1` to every `Div` whose operands are integers. The
//! `Div` kernel truncates its result toward zero when it sees the attribute.

use std::collections::HashMap;

use crate::reader::{attr_type, dtype_code};
use crate::types::{AttributeProto, AttributeValue, GraphProto};

/// Int attribute set to 1 on `Div` nodes whose inputs are integer tensors.
pub const ATTR_INT_DIV: &str = "__oxi_int_div";

fn is_int(dt: i32) -> bool {
    matches!(
        dt,
        dtype_code::UINT8
            | dtype_code::INT8
            | dtype_code::UINT16
            | dtype_code::INT16
            | dtype_code::INT32
            | dtype_code::INT64
            | dtype_code::UINT32
            | dtype_code::UINT64
    )
}

/// Element type of `op`'s first output when it can be derived from its inputs.
fn output_dtype(op: &str, domain: &str, input_dt: &[i32], attrs: &[AttributeProto]) -> Option<i32> {
    if !domain.is_empty() && domain != "ai.onnx" {
        return None;
    }
    let attr_i = |n: &str| attrs.iter().find(|a| a.name == n).map(|a| a.value.i);
    let first = || input_dt.first().copied().filter(|&d| d != dtype_code::UNDEFINED);
    match op {
        "Cast" => attr_i("to").map(|t| t as i32),
        "Shape" | "Size" | "NonZero" | "ArgMax" | "ArgMin" => Some(dtype_code::INT64),
        "Constant" => attrs.iter().find(|a| a.name == "value").and_then(|a| a.value.t.as_ref()).map(|t| t.data_type),
        "Where" => input_dt.get(1).copied().filter(|&d| d != dtype_code::UNDEFINED),
        "Add" | "Sub" | "Mul" | "Div" | "Mod" | "Neg" | "Abs" | "Min" | "Max" | "Clip" | "Identity"
        | "Reshape" | "Squeeze" | "Unsqueeze" | "Flatten" | "Transpose" | "Expand" | "Concat" | "Slice"
        | "Gather" | "GatherElements" | "GatherND" | "Tile" | "ReduceSum" | "ReduceMax" | "ReduceMin"
        | "ReduceProd" | "CumSum" | "Range" | "Pad" | "ScatterElements" | "ScatterND" => first(),
        _ => None,
    }
}

/// Annotate integer `Div` nodes of `graph` (top level only) with [`ATTR_INT_DIV`].
pub fn mark(graph: &mut GraphProto) {
    let mut dt: HashMap<String, i32> = HashMap::new();
    for vi in graph.input_value_infos.iter().chain(&graph.value_infos).chain(&graph.output_value_infos) {
        if vi.elem_type != dtype_code::UNDEFINED {
            dt.insert(vi.name.clone(), vi.elem_type);
        }
    }
    for t in &graph.initializers {
        dt.insert(t.name.clone(), t.data_type);
    }
    // ONNX nodes are topologically sorted, so one forward pass fills the gaps.
    for n in &mut graph.nodes {
        let ins: Vec<i32> = n.inputs.iter().map(|i| dt.get(i).copied().unwrap_or(dtype_code::UNDEFINED)).collect();
        if n.op_type == "Div" && n.domain.is_empty() {
            let out = n.outputs.first().and_then(|o| dt.get(o)).copied();
            if out.map_or(ins.iter().any(|&d| is_int(d)), is_int) {
                n.attributes.push(AttributeProto {
                    name: ATTR_INT_DIV.to_string(),
                    value: AttributeValue { i: 1, attr_type: attr_type::INT, ..Default::default() },
                });
            }
        }
        if let Some(out) = n.outputs.first() {
            if !dt.contains_key(out) {
                if let Some(d) = output_dtype(&n.op_type, &n.domain, &ins, &n.attributes) {
                    dt.insert(out.clone(), d);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeProto, TensorProto, ValueInfoProto};

    #[test]
    fn marks_only_integer_divs() {
        let node = |op: &str, inputs: &[&str], out: &str| NodeProto {
            op_type: op.into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec![out.into()],
            ..Default::default()
        };
        let vi = |name: &str, elem_type| ValueInfoProto { name: name.into(), elem_type, ..Default::default() };
        let mut g = GraphProto {
            input_value_infos: vec![vi("ids", dtype_code::INT64), vi("x", dtype_code::FLOAT32)],
            initializers: vec![TensorProto { name: "three".into(), data_type: dtype_code::INT64, ..Default::default() }],
            nodes: vec![
                node("Div", &["ids", "three"], "q"),     // int64 graph input
                node("Shape", &["x"], "s"),              // int64 by rule
                node("Div", &["s", "three"], "q2"),      // int64 via propagation
                node("Div", &["x", "x"], "f"),           // float
                node("Cast", &["q"], "qf"),              // no `to` attr: unknown
                node("Div", &["qf", "qf"], "unknown"),
            ],
            ..Default::default()
        };
        mark(&mut g);
        let marked: Vec<bool> = g
            .nodes
            .iter()
            .filter(|n| n.op_type == "Div")
            .map(|n| n.attributes.iter().any(|a| a.name == ATTR_INT_DIV && a.value.i == 1))
            .collect();
        assert_eq!(marked, vec![true, true, false, false]);
    }
}
