//! `MatMulNBits` weight prepacking → `OxiMatMulNBits`.
//!
//! Repacks each `com.microsoft::MatMulNBits` weight (`B`, `scales`,
//! `zero_points`) into the int8 kernels' layout once, at session build, and
//! stores it under `B`'s name, dropping the consumed `scales` / `zero_points`
//! weights. The session then holds one kernel-ready copy of every quantized
//! weight instead of the ORT layout plus a lazily built repack. The node
//! becomes `OxiMatMulNBits(A, packed, bias)`.
//!
//! A node is left alone (and runs the unprepacked path) when any of its weight
//! inputs is not an initializer, is shared with another consumer or a graph
//! output, or fails to pack (the op then reports the error at run time).

use crate::graph::{Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Names read by `graph`'s nodes (recursively through nested subgraphs).
fn subgraph_refs(graph: &Graph, out: &mut HashSet<String>) {
    for n in &graph.nodes {
        out.extend(n.inputs.iter().cloned());
        for g in n.attrs.graphs.values() {
            subgraph_refs(g, out);
        }
    }
}

/// Output width of a projection the MatMul fusions (`OxiMatMulAct`,
/// `OxiMatMulSplit`) may absorb, and whether it is quantized: a `MatMul` by a
/// 2-D weight, or a bias-free `OxiMatMulNBits` (its fused node then carries
/// `nbits = 1` and the packed weight).
pub(crate) fn projection_cols(node: &Node, w: &Tensor) -> Option<(usize, bool)> {
    match node.op.as_str() {
        "MatMul" if w.shape.len() == 2 => Some((w.shape[1], false)),
        "OxiMatMulNBits" if node.inputs.len() == 2 => {
            oxionnx_ops::matmul_nbits::Packed::from_words(&w.data).ok().map(|p| (p.s.n, true))
        }
        _ => None,
    }
}

/// Whether `node` is a projection the MatMul fusions consider.
pub(crate) fn is_projection(node: &Node) -> bool {
    node.op == OpKind::MatMul || node.op.as_str() == "OxiMatMulNBits"
}

/// See the module docs.
pub fn prepack_matmul_nbits(nodes: Vec<Node>, weights: &mut HashMap<String, Tensor>, output_names: &[String]) -> Vec<Node> {
    if !nodes.iter().any(|n| n.op.as_str() == "MatMulNBits") {
        return nodes;
    }
    let mut uses: HashMap<&str, usize> = HashMap::new();
    let mut pinned: HashSet<String> = output_names.iter().cloned().collect();
    for n in &nodes {
        for i in n.inputs.iter().filter(|i| !i.is_empty()) {
            *uses.entry(i.as_str()).or_default() += 1;
        }
        for g in n.attrs.graphs.values() {
            subgraph_refs(g, &mut pinned);
        }
    }
    let exclusive = |name: &str| uses.get(name) == Some(&1) && !pinned.contains(name);
    let mut plan = Vec::new();
    for (idx, n) in nodes.iter().enumerate() {
        if n.op.as_str() != "MatMulNBits" || n.inputs.get(4).is_some_and(|g| !g.is_empty()) {
            continue;
        }
        let input = |i: usize| n.inputs.get(i).filter(|s| !s.is_empty());
        let (Some(b), Some(scales)) = (input(1), input(2)) else { continue };
        let zp = input(3);
        let consts = [Some(b), Some(scales), zp].into_iter().flatten();
        if !consts.clone().all(|name| weights.contains_key(name) && exclusive(name)) {
            continue;
        }
        let packed = oxionnx_ops::registry::nbits_ops::pack_matmul_nbits(
            &n.attrs,
            &weights[b],
            &weights[scales],
            zp.map(|z| &weights[z]),
        );
        match packed {
            Ok(words) => plan.push((idx, words)),
            Err(e) => tracing::warn!(node = %n.name, error = %e, "MatMulNBits: weight not prepacked"),
        }
    }
    if plan.is_empty() {
        return nodes;
    }
    let count = plan.len();
    let mut nodes = nodes;
    for (idx, words) in plan {
        let n = &mut nodes[idx];
        let (b, scales, zp) = (n.inputs[1].clone(), n.inputs[2].clone(), n.inputs.get(3).cloned());
        let len = words.len();
        weights.insert(b.clone(), Tensor { data: words, shape: vec![len] });
        weights.remove(&scales);
        if let Some(z) = zp.filter(|z| !z.is_empty()) {
            weights.remove(&z);
        }
        let mut inputs = vec![n.inputs[0].clone(), b];
        if let Some(bias) = n.inputs.get(5).filter(|s| !s.is_empty()) {
            inputs.push(bias.clone());
        }
        n.op = OpKind::Unknown("OxiMatMulNBits".into());
        n.inputs = inputs;
    }
    tracing::debug!(count, "prepacked MatMulNBits weights into OxiMatMulNBits");
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Attributes;

    fn graph() -> (Vec<Node>, HashMap<String, Tensor>, Vec<f32>) {
        // K = 48 (two blocks of 32, the second partial), N = 20, 8 bits.
        let (k, n, kb) = (48usize, 20usize, 2usize);
        let b: Vec<f32> = (0..n * kb * 32).map(|i| ((i * 73 + 11) % 256) as f32).collect();
        let scales: Vec<f32> = (0..n * kb).map(|i| 0.01 + (i % 7) as f32 * 0.003).collect();
        let zp: Vec<f32> = (0..n * kb).map(|i| ((i * 29) % 256) as f32).collect();
        let mut w = vec![0f32; k * n];
        for j in 0..n {
            for p in 0..k {
                let bl = p / 32;
                w[p * n + j] = (b[(j * kb + bl) * 32 + p % 32] - zp[j * kb + bl]) * scales[j * kb + bl];
            }
        }
        let weights = [
            ("B", Tensor::new(b, vec![n, kb, 32])),
            ("S", Tensor::new(scales, vec![n, kb])),
            ("Z", Tensor::new(zp, vec![n, kb])),
            ("bias", Tensor::new((0..n).map(|i| i as f32 * 0.1).collect(), vec![n])),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let mut attrs = Attributes::default();
        for (key, v) in [("K", k as i64), ("N", n as i64), ("bits", 8), ("block_size", 32)] {
            attrs.ints.insert(key.into(), v);
        }
        let node = Node {
            op: OpKind::Unknown("MatMulNBits".into()),
            name: "mm".into(),
            inputs: ["A", "B", "S", "Z", "", "bias"].iter().map(|s| s.to_string()).collect(),
            outputs: vec!["Y".into()],
            attrs,
        };
        (vec![node], weights, w)
    }

    #[test]
    fn prepacks_exclusive_weights() {
        let (nodes, mut weights, _) = graph();
        let out = prepack_matmul_nbits(nodes, &mut weights, &["Y".to_string()]);
        assert_eq!(out[0].op.as_str(), "OxiMatMulNBits");
        assert_eq!(out[0].inputs, vec!["A", "B", "bias"]);
        assert!(!weights.contains_key("S") && !weights.contains_key("Z"));
        assert!(oxionnx_ops::matmul_nbits::Packed::is_packed(&weights["B"].data));

        // A scale shared with another consumer keeps the node as it is.
        let (mut nodes, mut weights, _) = graph();
        let mut other = nodes[0].clone();
        other.op = OpKind::Add;
        other.inputs = vec!["S".into(), "S".into()];
        other.outputs = vec!["T".into()];
        nodes.push(other);
        let out = prepack_matmul_nbits(nodes, &mut weights, &["Y".to_string(), "T".to_string()]);
        assert_eq!(out[0].op.as_str(), "MatMulNBits");
        assert!(weights.contains_key("S"));
    }

    #[test]
    fn session_runs_matmul_nbits() {
        let (nodes, weights, w) = graph();
        let graph = Graph {
            nodes,
            input_names: vec!["A".into()],
            output_names: vec!["Y".into()],
            ..Default::default()
        };
        let (m, k, n) = (3usize, 48usize, 20usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 37) % 19) as f32 / 19.0 - 0.5).collect();
        let session = crate::Session::from_graph(graph, weights).unwrap();
        let inputs: HashMap<&str, Tensor> = [("A", Tensor::new(a.clone(), vec![m, k]))].into();
        let y = &session.run(&inputs).unwrap()["Y"];
        assert_eq!(y.shape, vec![m, n]);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| a[i * k + p] * w[p * n + j]).sum::<f32>() + j as f32 * 0.1;
                let got = y.data[i * n + j];
                assert!((got - want).abs() < 0.02 * (1.0 + want.abs()), "[{i},{j}] {got} vs {want}");
            }
        }
    }

    #[test]
    fn gated_mlp_on_quantized_weights_fuses() {
        // MatMulNBits -> Split -> Gelu(first half) * second half.
        let (mut nodes, mut weights, _) = graph();
        nodes[0].inputs.truncate(4);
        let mk = |op: OpKind, inputs: &[&str], outputs: &[&str]| Node {
            op,
            name: outputs[0].to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: Attributes::default(),
        };
        let mut split = mk(OpKind::Split, &["Y"], &["g", "u"]);
        split.attrs.ints.insert("axis".into(), -1);
        let mut gelu = mk(OpKind::Gelu, &["g"], &["ga"]);
        gelu.attrs.strings.insert("approximate".into(), "tanh".into());
        nodes.extend([split, gelu, mk(OpKind::Mul, &["ga", "u"], &["out"])]);
        let registry = oxionnx_ops::default_registry();
        let out = crate::optimizer::optimize(nodes, &mut weights, &["out".to_string()], &registry);
        let ops: Vec<&str> = out.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, vec!["OxiMatMulAct"]);
        assert_eq!(out[0].attrs.i("nbits", 0), 1);
        assert!(oxionnx_ops::matmul_nbits::Packed::is_packed(&weights[&out[0].inputs[1]].data));
    }
}
