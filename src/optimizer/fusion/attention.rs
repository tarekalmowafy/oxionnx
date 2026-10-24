//! Scaled dot-product attention fusion → `OxiAttention`.
//!
//! Exporters that do not use a fused attention op spell attention out as
//! head-split Reshape/Transpose chains around two batched MatMuls and a
//! Softmax. Run as written, that materialises Q/K/V in head-major layout, the
//! `[B, H, S, T]` scores and the probabilities as full graph tensors. This pass
//! replaces the whole chain with one `OxiAttention` node that reads Q/K/V from
//! their `[B, S, heads*D]` projections and writes `[B, S, Hq*D]` (see
//! [`oxionnx_ops::registry::contrib_ops::OxiAttentionOp`]).
//!
//! # Patterns
//!
//! **Grouped-query, 5-D** (onnx-community Gemma 3 exports; `G` KV heads, `R`
//! query heads per KV head, `h = g*R + r`):
//!
//! ```text
//! q5 = Reshape(q, [0,0,G,R,D]);  qt = Transpose(q5, [0,2,3,1,4]);  qr = Reshape(qt, [0,0,-1,D])
//! k4 = Reshape(k, [0,0,G,D]);    kt = Transpose(k4, [0,2,3,1])
//! v4 = Reshape(v, [0,0,G,D]);    vt = Transpose(v4, [0,2,1,3])
//! sc = Reshape([Mul|Div](MatMul(qr, kt), c)?, scores_shape)      // -> [B,G,R,S,T]
//! p  = Reshape(Softmax(Add(sc, bias)?, axis=-1), [0,0,-1,0])
//! o  = Reshape(Transpose(Reshape(MatMul(p, vt), [0,0,R,-1,D]), [0,3,1,2,4]), [0,0,G*R*D])
//! ```
//!
//! **Multi-head, 4-D** (the common BERT-style form; `Hq == Hkv == H`):
//!
//! ```text
//! qt = Transpose(Reshape(q, [0,0,H,D]), [0,2,1,3])
//! kt = Transpose(Reshape(k, [0,0,H,D]), [0,2,3,1])
//! vt = Transpose(Reshape(v, [0,0,H,D]), [0,2,1,3])
//! p  = Softmax(Add([Mul|Div](MatMul(qt, kt), c)?, bias)?, axis=-1)
//! o  = Reshape(Transpose(MatMul(p, vt), [0,2,1,3]), [0,0,H*D])
//! ```
//!
//! Every removed intermediate must have exactly one consumer and must not be a
//! graph output; reshape targets and permutations must be constants with the
//! exact values above (with `0` = "copy dim", so the match is independent of
//! batch and sequence length). The scores reshape target of the 5-D form is
//! usually computed at run time; it becomes an input of the fused node, which
//! re-checks it against `[B, G, R, S, T]` on every run.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::{const_scalar, TensorUsage};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Bias layouts understood by `OxiAttention` (mirrors `oxionnx_ops::registry::contrib_ops`).
const BIAS_LAYOUT_BGRST: i64 = 0;
const BIAS_LAYOUT_BHST: i64 = 1;

struct Matcher<'a> {
    nodes: &'a [Node],
    weights: &'a HashMap<String, Tensor>,
    usage: TensorUsage,
    producer: HashMap<&'a str, usize>,
    consumer: HashMap<&'a str, usize>,
}

/// A successful match: the nodes to delete and the fused node replacing the last one.
struct Fused {
    remove: Vec<usize>,
    at: usize,
    node: Node,
}

impl<'a> Matcher<'a> {
    fn new(nodes: &'a [Node], weights: &'a HashMap<String, Tensor>, output_names: &[String]) -> Self {
        let mut producer = HashMap::new();
        let mut consumer = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for o in &n.outputs {
                if !o.is_empty() {
                    producer.insert(o.as_str(), i);
                }
            }
            for inp in &n.inputs {
                if !inp.is_empty() {
                    consumer.entry(inp.as_str()).or_insert(i);
                }
            }
        }
        Self { nodes, weights, usage: TensorUsage::new(nodes, output_names), producer, consumer }
    }

    /// Producer of `tensor` if it is an `op` node whose output can be removed.
    fn prod(&self, tensor: &str, op: OpKind) -> Option<usize> {
        if !self.usage.is_fusable_intermediate(tensor) {
            return None;
        }
        let i = *self.producer.get(tensor)?;
        (self.nodes[i].op == op && self.nodes[i].outputs.len() == 1).then_some(i)
    }

    /// The single consumer of `tensor`, if it is an `op` node and `tensor` can be removed.
    fn cons(&self, tensor: &str, op: OpKind) -> Option<usize> {
        if !self.usage.is_fusable_intermediate(tensor) {
            return None;
        }
        let i = *self.consumer.get(tensor)?;
        (self.nodes[i].op == op).then_some(i)
    }

    fn const_ints(&self, name: &str) -> Option<Vec<i64>> {
        let t = self.weights.get(name)?;
        Some(t.data.iter().map(|&v| v as i64).collect())
    }

    /// `Reshape(x, target)` with a constant `target` and `allowzero = 0`.
    fn reshape_target(&self, i: usize) -> Option<Vec<i64>> {
        let n = &self.nodes[i];
        if n.attrs.i("allowzero", 0) != 0 {
            return None;
        }
        self.const_ints(n.inputs.get(1)?)
    }

    fn perm_is(&self, i: usize, perm: &[i64]) -> bool {
        self.nodes[i].attrs.ints("perm") == perm
    }

    fn input(&self, i: usize, k: usize) -> Option<&'a str> {
        self.nodes[i].inputs.get(k).map(String::as_str).filter(|s| !s.is_empty())
    }

    fn out(&self, i: usize) -> &'a str {
        &self.nodes[i].outputs[0]
    }

    /// `Transpose(Reshape(src, target), perm)` producing `tensor`: returns
    /// `(src, reshape target, [reshape, transpose])`.
    fn head_split(&self, tensor: &str, perm: &[i64]) -> Option<(&'a str, Vec<i64>, [usize; 2])> {
        let tr = self.prod(tensor, OpKind::Transpose)?;
        if !self.perm_is(tr, perm) {
            return None;
        }
        let rs = self.prod(self.input(tr, 0)?, OpKind::Reshape)?;
        Some((self.input(rs, 0)?, self.reshape_target(rs)?, [rs, tr]))
    }

    /// Optional scalar `Mul`/`Div` by a constant on top of the QK MatMul.
    /// Returns `(qk matmul, scale factor, nodes)`.
    fn scaled_qk(&self, tensor: &str) -> Option<(usize, f32, Vec<usize>)> {
        if let Some(mm) = self.prod(tensor, OpKind::MatMul) {
            return Some((mm, 1.0, vec![mm]));
        }
        for (op, inv) in [(OpKind::Mul, false), (OpKind::Div, true)] {
            let Some(sc) = self.prod(tensor, op) else { continue };
            let (a, b) = (self.input(sc, 0)?, self.input(sc, 1)?);
            let candidates: &[(&str, &str)] = if inv { &[(a, b)] } else { &[(a, b), (b, a)] };
            for &(x, c) in candidates {
                let (Some(mm), Some(v)) = (self.prod(x, OpKind::MatMul), const_scalar(self.weights, c)) else {
                    continue;
                };
                if inv && v == 0.0 {
                    return None;
                }
                return Some((mm, if inv { 1.0 / v } else { v }, vec![mm, sc]));
            }
        }
        None
    }

    /// Softmax input: optional `Add(scores, bias)`; returns `(scores, bias, add node)`.
    fn masked(&self, tensor: &'a str, scores_op: OpKind) -> Option<(&'a str, Option<&'a str>, Option<usize>)> {
        if let Some(add) = self.prod(tensor, OpKind::Add) {
            let (a, b) = (self.input(add, 0)?, self.input(add, 1)?);
            for (s, bias) in [(a, b), (b, a)] {
                if self.prod(s, scores_op.clone()).is_some() || self.prod(s, OpKind::Mul).is_some() || self.prod(s, OpKind::Div).is_some() {
                    return Some((s, Some(bias), Some(add)));
                }
            }
            return None;
        }
        Some((tensor, None, None))
    }

    fn try_fuse(&self, sm: usize) -> Option<Fused> {
        let ax = self.nodes[sm].attrs.ints.get("axis").copied();
        let sm_in = self.input(sm, 0)?;
        let sm_out = self.out(sm);
        self.try_5d(sm, sm_in, sm_out, ax).or_else(|| self.try_4d(sm, sm_in, sm_out, ax))
    }

    fn try_5d(&self, sm: usize, sm_in: &'a str, sm_out: &'a str, ax: Option<i64>) -> Option<Fused> {
        if !matches!(ax, Some(-1) | Some(4)) {
            return None;
        }
        let (scores5, bias, add) = self.masked(sm_in, OpKind::Reshape)?;
        let sc_rs = self.prod(scores5, OpKind::Reshape)?;
        if self.nodes[sc_rs].attrs.i("allowzero", 0) != 0 {
            return None;
        }
        let scores_shape = self.input(sc_rs, 1)?;
        let (qk, scale, qk_nodes) = self.scaled_qk(self.input(sc_rs, 0)?)?;
        // Q: Reshape [0,0,G,R,D] -> Transpose [0,2,3,1,4] -> Reshape [0,0,-1,D]
        let qr = self.prod(self.input(qk, 0)?, OpKind::Reshape)?;
        let qr_t = self.reshape_target(qr)?;
        let (q_src, q5, q_nodes) = self.head_split(self.input(qr, 0)?, &[0, 2, 3, 1, 4])?;
        let [0, 0, g, r, d] = q5[..] else { return None };
        if g <= 0 || r <= 0 || d <= 0 || qr_t != [0, 0, -1, d] {
            return None;
        }
        // K: Reshape [0,0,G,D] -> Transpose [0,2,3,1]
        let (k_src, k4, k_nodes) = self.head_split(self.input(qk, 1)?, &[0, 2, 3, 1])?;
        if k4 != [0, 0, g, d] {
            return None;
        }
        // P: Reshape [0,0,-1,0] -> MatMul(P, V)
        let p_rs = self.cons(sm_out, OpKind::Reshape)?;
        if self.reshape_target(p_rs)? != [0, 0, -1, 0] || self.input(p_rs, 0)? != sm_out {
            return None;
        }
        let pv = self.cons(self.out(p_rs), OpKind::MatMul)?;
        if self.input(pv, 0)? != self.out(p_rs) {
            return None;
        }
        let (v_src, v4, v_nodes) = self.head_split(self.input(pv, 1)?, &[0, 2, 1, 3])?;
        if v4 != [0, 0, g, d] {
            return None;
        }
        // Out: [B,G,R*S,D] -> Reshape [0,0,R,-1,D] -> Transpose [0,3,1,2,4] -> Reshape [0,0,G*R*D]
        let o5 = self.cons(self.out(pv), OpKind::Reshape)?;
        if self.reshape_target(o5)? != [0, 0, r, -1, d] {
            return None;
        }
        let ot = self.cons(self.out(o5), OpKind::Transpose)?;
        if !self.perm_is(ot, &[0, 3, 1, 2, 4]) {
            return None;
        }
        let o = self.consumer.get(self.out(ot)).copied()?;
        if !self.usage.is_fusable_intermediate(self.out(ot)) || self.nodes[o].op != OpKind::Reshape {
            return None;
        }
        if self.reshape_target(o)? != [0, 0, g * r * d] {
            return None;
        }
        let mut remove = vec![sc_rs, qr, sm, p_rs, pv, o5, ot];
        remove.extend(qk_nodes.iter().chain(&q_nodes).chain(&k_nodes).chain(&v_nodes).chain(add.iter()));
        Some(self.build(remove, o, [q_src, k_src, v_src], bias, Some(scores_shape), g * r, g, scale, BIAS_LAYOUT_BGRST))
    }

    fn try_4d(&self, sm: usize, sm_in: &'a str, sm_out: &'a str, ax: Option<i64>) -> Option<Fused> {
        if !matches!(ax, Some(-1) | Some(3)) {
            return None;
        }
        let (scores, bias, add) = self.masked(sm_in, OpKind::MatMul)?;
        let (qk, scale, qk_nodes) = self.scaled_qk(scores)?;
        let (q_src, q4, q_nodes) = self.head_split(self.input(qk, 0)?, &[0, 2, 1, 3])?;
        let [0, 0, h, d] = q4[..] else { return None };
        if h <= 0 || d <= 0 {
            return None;
        }
        let (k_src, k4, k_nodes) = self.head_split(self.input(qk, 1)?, &[0, 2, 3, 1])?;
        let pv = self.cons(sm_out, OpKind::MatMul)?;
        if self.input(pv, 0)? != sm_out {
            return None;
        }
        let (v_src, v4, v_nodes) = self.head_split(self.input(pv, 1)?, &[0, 2, 1, 3])?;
        if k4 != [0, 0, h, d] || v4 != k4 {
            return None;
        }
        let ot = self.cons(self.out(pv), OpKind::Transpose)?;
        if !self.perm_is(ot, &[0, 2, 1, 3]) {
            return None;
        }
        let o = self.consumer.get(self.out(ot)).copied()?;
        if !self.usage.is_fusable_intermediate(self.out(ot)) || self.nodes[o].op != OpKind::Reshape {
            return None;
        }
        if self.reshape_target(o)? != [0, 0, h * d] {
            return None;
        }
        let mut remove = vec![sm, pv, ot];
        remove.extend(qk_nodes.iter().chain(&q_nodes).chain(&k_nodes).chain(&v_nodes).chain(add.iter()));
        Some(self.build(remove, o, [q_src, k_src, v_src], bias, None, h, h, scale, BIAS_LAYOUT_BHST))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        remove: Vec<usize>,
        at: usize,
        qkv: [&str; 3],
        bias: Option<&str>,
        scores_shape: Option<&str>,
        q_heads: i64,
        kv_heads: i64,
        scale: f32,
        bias_layout: i64,
    ) -> Fused {
        let mut inputs: Vec<String> = qkv.iter().map(|s| s.to_string()).collect();
        inputs.push(bias.unwrap_or("").to_string());
        inputs.push(scores_shape.unwrap_or("").to_string());
        while inputs.last().is_some_and(String::is_empty) {
            inputs.pop();
        }
        let mut attrs = Attributes::default();
        attrs.ints.insert("q_num_heads".into(), q_heads);
        attrs.ints.insert("kv_num_heads".into(), kv_heads);
        attrs.ints.insert("bias_layout".into(), bias_layout);
        attrs.floats.insert("scale".into(), scale);
        let final_node = &self.nodes[at];
        Fused {
            remove,
            at,
            node: Node {
                op: OpKind::Unknown("OxiAttention".into()),
                name: format!("{}/OxiAttention", final_node.name),
                inputs,
                outputs: final_node.outputs.clone(),
                attrs,
            },
        }
    }
}

/// Fuse attention chains into `OxiAttention`; see the module docs.
pub fn fuse_attention(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    output_names: &[String],
) -> Vec<Node> {
    let fused: Vec<Fused> = {
        let m = Matcher::new(&nodes, weights, output_names);
        let mut claimed = HashSet::new();
        let mut out = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            if n.op != OpKind::Softmax {
                continue;
            }
            if let Some(f) = m.try_fuse(i) {
                if f.remove.iter().chain([&f.at]).all(|j| !claimed.contains(j)) {
                    claimed.extend(f.remove.iter().copied());
                    claimed.insert(f.at);
                    out.push(f);
                }
            }
        }
        out
    };
    if fused.is_empty() {
        return nodes;
    }
    tracing::debug!(count = fused.len(), "fused attention chains into OxiAttention");
    let mut drop = HashSet::new();
    let mut replace = HashMap::new();
    for f in fused {
        drop.extend(f.remove);
        replace.insert(f.at, f.node);
    }
    nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| {
            if let Some(f) = replace.remove(&i) {
                Some(f)
            } else if drop.contains(&i) {
                None
            } else {
                Some(n)
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
