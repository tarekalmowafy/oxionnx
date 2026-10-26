//! Which initializers the loader may leave in the model file ("lazy"), to be
//! read on demand instead of being decoded at load time.
//!
//! These are large tables read only by row lookups, i.e. embedding tables:
//! every reference to the initializer is the `data` input of a `Gather` along
//! axis 0, or one of the table inputs of `com.microsoft::GatherBlockQuantized`.
//! A table that is also read any other way (a tied embedding reused as an
//! output projection, a graph output, a subgraph capture) is decoded as usual.
//! The session can still decide to decode a lazy table later, e.g. when the
//! optimizer gave it a consumer that cannot read it from the file.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::PathBuf;

use crate::reader::dtype_code;
use crate::types::{GraphProto, NodeProto};

/// Tables smaller than this are always decoded.
pub const DEFAULT_MIN_LAZY_BYTES: usize = 4 << 20;

/// Where a lazy initializer's raw (little-endian) bytes are stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LazySource {
    /// `bytes[range]` of the buffer the model was parsed from.
    Inline(Range<usize>),
    /// `length` bytes at `offset` of an external-data file.
    External { path: PathBuf, offset: u64, length: usize },
}

/// An initializer the loader left in the model file. Its entry in the weight
/// map is *released*: the shape it would have had, with empty data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LazyInit {
    pub name: String,
    /// ONNX `data_type` of the stored elements.
    pub data_type: i32,
    pub source: LazySource,
}

fn attr_i(n: &NodeProto, name: &str, default: i64) -> i64 {
    n.attributes.iter().find(|a| a.name == name).map_or(default, |a| a.value.i)
}

fn is_default_domain(d: &str) -> bool {
    d.is_empty() || d == "ai.onnx"
}

/// Input slots of `n` that read a table by rows, with the element types each
/// accepts. `rank` gives an initializer's rank (for negative axes).
fn table_slots(n: &NodeProto, rank: impl Fn(&str) -> Option<usize>) -> &'static [(usize, &'static [i32])] {
    const FLOATS: &[i32] = &[dtype_code::FLOAT32, dtype_code::FLOAT16, dtype_code::BFLOAT16];
    const U8: &[i32] = &[dtype_code::UINT8];
    const F32: &[i32] = &[dtype_code::FLOAT32];
    const U8_OR_F32: &[i32] = &[dtype_code::UINT8, dtype_code::FLOAT32];
    let first_axis = |attr: &str, default: i64| {
        let a = attr_i(n, attr, default);
        a == 0 || n.inputs.first().and_then(|x| rank(x)).is_some_and(|r| a == -(r as i64))
    };
    match (n.domain.as_str(), n.op_type.as_str()) {
        (d, "Gather") if is_default_domain(d) && first_axis("axis", 0) => &[(0, FLOATS)],
        ("com.microsoft", "GatherBlockQuantized") if first_axis("gather_axis", 0) => {
            &[(0, U8), (2, F32), (3, U8_OR_F32)]
        }
        _ => &[],
    }
}

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

/// Names of the initializers of `graph` (at least `min_bytes` large) that only
/// table lookups read. `byte_names` are the uint8 initializers the loader
/// holds as byte tensors; one is eligible only when its innermost dimension is
/// a multiple of 4, so that its stored bytes already are the byte-tensor
/// layout.
pub fn plan(graph: &GraphProto, byte_names: &HashSet<String>, min_bytes: usize) -> HashSet<String> {
    let inits: HashMap<&str, &crate::types::TensorProto> =
        graph.initializers.iter().map(|t| (t.name.as_str(), t)).collect();
    let rank = |name: &str| inits.get(name).map(|t| t.dims.len());
    let mut other: HashSet<String> = graph.outputs.iter().cloned().collect();
    let mut eligible: HashSet<String> = HashSet::new();
    for n in &graph.nodes {
        let slots = table_slots(n, rank);
        for (i, name) in n.inputs.iter().enumerate() {
            let Some(t) = inits.get(name.as_str()) else { continue };
            match slots.iter().find(|(s, _)| *s == i) {
                Some((_, types)) if types.contains(&t.data_type) => {
                    eligible.insert(name.clone());
                }
                _ => {
                    other.insert(name.clone());
                }
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
    eligible.retain(|name| {
        let t = inits[name.as_str()];
        let elem = match t.data_type {
            dtype_code::FLOAT32 => 4,
            dtype_code::UINT8 => 1,
            _ => 2,
        };
        let numel = t.dims.iter().try_fold(1usize, |a, &d| a.checked_mul(usize::try_from(d).ok()?));
        let big = numel.and_then(|c| c.checked_mul(elem)).is_some_and(|b| b >= min_bytes.max(1));
        let bytes_ok = match t.data_type {
            dtype_code::UINT8 => byte_names.contains(name) && t.dims.last().is_some_and(|&d| d > 0 && d % 4 == 0),
            _ => true,
        };
        !other.contains(name) && big && bytes_ok
    });
    eligible
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AttributeProto, AttributeValue, TensorProto};

    fn init(name: &str, dt: i32, dims: &[i64]) -> TensorProto {
        TensorProto { name: name.into(), data_type: dt, dims: dims.to_vec(), ..Default::default() }
    }

    fn node(op: &str, inputs: &[&str], axis: Option<i64>) -> NodeProto {
        NodeProto {
            op_type: op.into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            attributes: axis
                .map(|a| AttributeProto { name: "axis".into(), value: AttributeValue { i: a, ..Default::default() } })
                .into_iter()
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn only_tables_read_by_row_lookups() {
        let g = GraphProto {
            initializers: vec![
                init("emb", 1, &[1000, 64]),
                init("tied", 1, &[1000, 64]),
                init("axis1", 1, &[1000, 64]),
                init("neg_axis", 1, &[1000, 64]),
                init("small", 1, &[10, 64]),
                init("ints", 7, &[1000, 64]),
                init("out", 1, &[1000, 64]),
            ],
            nodes: vec![
                node("Gather", &["emb", "ids"], None),
                node("Gather", &["emb", "ids2"], Some(0)),
                node("Gather", &["tied", "ids"], None),
                node("MatMul", &["x", "tied"], None),
                node("Gather", &["axis1", "ids"], Some(1)),
                node("Gather", &["neg_axis", "ids"], Some(-2)),
                node("Gather", &["small", "ids"], None),
                node("Gather", &["ints", "ids"], None),
                node("Gather", &["out", "ids"], None),
                node("Gather", &["ids_table", "emb"], None),
            ],
            outputs: vec!["out".into()],
            ..Default::default()
        };
        // `emb` is also read as Gather's *indices* by the last node: not a table read.
        assert_eq!(plan(&g, &HashSet::new(), 4096), ["neg_axis".to_string()].into_iter().collect());
        let g2 = GraphProto { nodes: g.nodes[..2].to_vec(), ..g.clone() };
        assert!(plan(&g2, &HashSet::new(), 4096).contains("emb"));
    }

    #[test]
    fn block_quantized_tables_need_word_rows() {
        let gbq = |inputs: &[&str]| NodeProto {
            op_type: "GatherBlockQuantized".into(),
            domain: "com.microsoft".into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let g = GraphProto {
            initializers: vec![
                init("q", 2, &[1000, 64]),
                init("s", 1, &[1000, 4]),
                init("z", 2, &[1000, 2]),
                init("q_odd", 2, &[1000, 66]),
            ],
            nodes: vec![gbq(&["q", "ids", "s", "z"]), gbq(&["q_odd", "ids", "s"])],
            ..Default::default()
        };
        let bytes: HashSet<String> = ["q", "z", "q_odd"].iter().map(|s| s.to_string()).collect();
        let got = plan(&g, &bytes, 1);
        // `z` rows are 2 bytes: not the word layout.
        assert_eq!(got, ["q", "s"].iter().map(|s| s.to_string()).collect());
        assert!(plan(&g, &HashSet::new(), 1).contains("s"), "uint8 tables must be byte tensors");
        assert!(!plan(&g, &HashSet::new(), 1).contains("q"));
    }
}
