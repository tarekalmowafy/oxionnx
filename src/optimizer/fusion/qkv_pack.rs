//! Pack projections that share an input into one GEMM.
//!
//! HF exports (Qwen3, Gemma 3 / harrier, ModernBERT ONNX-community exports)
//! often project the same hidden state with separate Q, K and V `MatMul`s:
//!
//! ```text
//! q = MatMul(x, Wq) [+ bq];  k = MatMul(x, Wk) [+ bk];  v = MatMul(x, Wv) [+ bv]
//! ```
//!
//! This pass rewrites every such group into
//!
//! ```text
//! q, k, v = Split(MatMul(x, [Wq | Wk | Wv]) [+ [bq | bk | bv]], axis=-1)
//! ```
//!
//! keeping the output names, so [`super::fuse_matmul_split`] then turns it into
//! one `OxiMatMulSplit` (with each output's head norm and rotary). One GEMM
//! reads `x` once and runs one parallel region instead of three.
//!
//! A member's weight must be a 2-D constant used by that `MatMul` only (the
//! originals are dropped, so memory does not grow), all weights must have the
//! same `K`, and every width must be a multiple of 32 (the packed epilogue's
//! block). A missing bias in a group that has some is filled with zeros
//! (`y + 0 = y`).

use crate::graph::{Node, OpKind};
use crate::optimizer::graph_utils::{NameAllocator, TensorUsage};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

struct Member {
    mm: usize,
    w: String,
    n: usize,
    /// `(Add node, bias weight)`.
    bias: Option<(usize, String)>,
    /// The tensor the group's Split produces for this member.
    out: String,
}

/// See the module docs.
pub fn pack_shared_projections(nodes: Vec<Node>, weights: &mut HashMap<String, Tensor>, output_names: &[String]) -> Vec<Node> {
    let usage = TensorUsage::new(&nodes, output_names);
    let mut consumer: HashMap<&str, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for inp in n.inputs.iter().filter(|s| !s.is_empty()) {
            consumer.entry(inp.as_str()).or_insert(i);
        }
    }
    // Candidate members grouped by (input, K), in node order.
    let mut groups: Vec<(String, usize, Vec<Member>)> = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        if n.op != OpKind::MatMul || n.inputs.len() != 2 || n.outputs.len() != 1 {
            continue;
        }
        let (x, w) = (&n.inputs[0], &n.inputs[1]);
        let Some(t) = weights.get(w) else { continue };
        if x.is_empty() || weights.contains_key(x) || t.shape.len() != 2 || t.data.len() != t.shape[0] * t.shape[1] || t.data.is_empty() {
            continue;
        }
        if usage.consumers(w) != 1 || usage.is_graph_output(w) {
            continue;
        }
        let (k, nn) = (t.shape[0], t.shape[1]);
        if nn % 32 != 0 {
            continue;
        }
        let y = &n.outputs[0];
        let mut out = y.clone();
        let mut bias = None;
        if usage.is_fusable_intermediate(y) {
            if let Some(&a) = consumer.get(y.as_str()) {
                let an = &nodes[a];
                if an.op == OpKind::Add && an.inputs.len() == 2 && an.outputs.len() == 1 {
                    let b = if an.inputs[0] == *y { &an.inputs[1] } else { &an.inputs[0] };
                    if weights.get(b).is_some_and(|t| t.data.len() == nn && t.shape.last() == Some(&nn)) && b != y {
                        bias = Some((a, b.clone()));
                        out = an.outputs[0].clone();
                    }
                }
            }
        }
        let m = Member { mm: i, w: w.clone(), n: nn, bias, out };
        match groups.iter_mut().find(|(gx, gk, _)| gx == x && *gk == k) {
            Some(g) => g.2.push(m),
            None => groups.push((x.clone(), k, vec![m])),
        }
    }
    groups.retain(|g| g.2.len() >= 2);
    if groups.is_empty() {
        return nodes;
    }
    let mut names = NameAllocator::new(&nodes, weights);
    let mut drop: HashSet<usize> = HashSet::new();
    let mut member_biases: Vec<String> = Vec::new();
    let mut insert: HashMap<usize, Vec<Node>> = HashMap::new();
    for (x, k, members) in groups {
        let ntot: usize = members.iter().map(|m| m.n).sum();
        let first = &members[0];
        // Packed weight, row by row; the originals go.
        let mut data = Vec::with_capacity(k * ntot);
        let ws: Vec<Tensor> = members.iter().map(|m| weights.remove(&m.w).expect("member weight")).collect();
        for r in 0..k {
            for (m, t) in members.iter().zip(&ws) {
                data.extend_from_slice(&t.data[r * m.n..(r + 1) * m.n]);
            }
        }
        std::mem::drop(ws);
        let wname = names.allocate(&first.w, "_packed");
        weights.insert(wname.clone(), Tensor::new(data, vec![k, ntot]));
        let mm_out = names.allocate(&first.out, "_packed");
        let base = nodes[first.mm].name.clone();
        let mut new_nodes = vec![Node {
            op: OpKind::MatMul,
            name: format!("{base}/packed"),
            inputs: vec![x, wname],
            outputs: vec![mm_out.clone()],
            attrs: Default::default(),
        }];
        let mut split_in = mm_out;
        if members.iter().any(|m| m.bias.is_some()) {
            let mut b = Vec::with_capacity(ntot);
            for m in &members {
                match &m.bias {
                    Some((_, bn)) => b.extend_from_slice(&weights[bn].data),
                    None => b.extend(std::iter::repeat(0.0).take(m.n)),
                }
            }
            let bname = names.allocate(&first.w, "_packed_bias");
            weights.insert(bname.clone(), Tensor::new(b, vec![ntot]));
            let add_out = names.allocate(&first.out, "_packed_biased");
            new_nodes.push(Node {
                op: OpKind::Add,
                name: format!("{base}/packed_bias"),
                inputs: vec![split_in, bname],
                outputs: vec![add_out.clone()],
                attrs: Default::default(),
            });
            split_in = add_out;
        }
        let sname = names.allocate(&first.w, "_packed_split");
        weights.insert(sname.clone(), Tensor::new(members.iter().map(|m| m.n as f32).collect(), vec![members.len()]));
        let mut split = Node {
            op: OpKind::Split,
            name: format!("{base}/packed_split"),
            inputs: vec![split_in, sname],
            outputs: members.iter().map(|m| m.out.clone()).collect(),
            attrs: Default::default(),
        };
        split.attrs.ints.insert("axis".into(), -1);
        new_nodes.push(split);
        for m in &members {
            drop.insert(m.mm);
            if let Some((a, b)) = &m.bias {
                drop.insert(*a);
                member_biases.push(b.clone());
            }
        }
        insert.insert(first.mm, new_nodes);
        tracing::debug!(members = members.len(), "packed projections sharing an input into one MatMul");
    }
    let mut out = Vec::with_capacity(nodes.len());
    for (i, n) in nodes.into_iter().enumerate() {
        if let Some(v) = insert.remove(&i) {
            out.extend(v);
        }
        if !drop.contains(&i) {
            out.push(n);
        }
    }
    // Member biases nothing else reads.
    let used: HashSet<&str> = out.iter().flat_map(|n| n.inputs.iter().map(String::as_str)).collect();
    for b in member_biases {
        if !used.contains(b.as_str()) && !output_names.contains(&b) {
            weights.remove(&b);
        }
    }
    out
}

#[cfg(test)]
#[path = "qkv_pack_tests.rs"]
mod tests;
