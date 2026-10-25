//! MatMul + Split projection fusion → `OxiMatMulSplit`.
//!
//! Rewrites a projection by a constant weight whose result is split along its
//! columns — the fused QKV projection of attention blocks — into one node whose
//! kernel finishes each head block of the product while it is in cache and
//! writes it straight into its output (see `oxionnx_ops::fused_matmul::SplitSpec`).
//!
//! # Pattern
//!
//! ```text
//! y = MatMul(x, W) [+ b]                                    // W: constant [K, N]; b: constant [N]
//! [y = Reshape(SimplifiedLayerNormalization(Reshape(y, [0,-1,D]), gamma), [0,-1,N])]   // per-head RMSNorm
//! [y = Mul(y, c)]                                           // c: constant [N]
//! y_0, .., y_k = Split(y, axis=-1)
//! [y_o = com.microsoft.RotaryEmbedding(y_o, position_ids, cos, sin)]   // any of the outputs
//! ```
//!
//! The rotary nodes must share their position/cos/sin inputs and attributes.
//! Every removed intermediate must have exactly one consumer and must not be a
//! graph output.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

const ROTARY: &str = "com.microsoft.RotaryEmbedding";

struct Matcher<'a> {
    nodes: &'a [Node],
    weights: &'a HashMap<String, Tensor>,
    usage: TensorUsage,
    consumer: HashMap<&'a str, usize>,
}

struct Fused {
    remove: Vec<usize>,
    at: usize,
    node: Node,
}

impl<'a> Matcher<'a> {
    fn new(nodes: &'a [Node], weights: &'a HashMap<String, Tensor>, output_names: &[String]) -> Self {
        let mut consumer = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for inp in &n.inputs {
                if !inp.is_empty() {
                    consumer.entry(inp.as_str()).or_insert(i);
                }
            }
        }
        Self { nodes, weights, usage: TensorUsage::new(nodes, output_names), consumer }
    }

    /// The single consumer of `tensor`, if its op type is `op` and `tensor` can be removed.
    fn cons(&self, tensor: &str, op: &str) -> Option<usize> {
        if !self.usage.is_fusable_intermediate(tensor) {
            return None;
        }
        let i = *self.consumer.get(tensor)?;
        (self.nodes[i].op.as_str() == op).then_some(i)
    }

    fn out(&self, i: usize) -> &'a str {
        &self.nodes[i].outputs[0]
    }

    fn other_input(&self, i: usize, tensor: &str) -> Option<&'a str> {
        let ins = &self.nodes[i].inputs;
        match (ins.len(), ins.first().map(|s| s == tensor), ins.get(1).map(|s| s == tensor)) {
            (2, Some(true), Some(false)) => Some(&ins[1]),
            (2, Some(false), Some(true)) => Some(&ins[0]),
            _ => None,
        }
    }

    /// A constant with `n` elements laid out along the last axis (`[n]`, `[1, 1, n]`, ...).
    fn row_const(&self, name: &str, n: usize) -> bool {
        self.weights.get(name).is_some_and(|t| t.data.len() == n && t.shape.last() == Some(&n))
    }

    fn reshape_to(&self, i: usize, target: &[i64]) -> bool {
        let node = &self.nodes[i];
        node.attrs.i("allowzero", 0) == 0
            && node.inputs.get(1).and_then(|t| self.weights.get(t)).is_some_and(|t| {
                t.data.len() == target.len() && t.data.iter().zip(target).all(|(&v, &e)| v as i64 == e)
            })
    }

    /// Optional `Reshape -> SimplifiedLayerNormalization -> Reshape` over `D`-wide heads.
    /// Returns `(gamma, epsilon, nodes, output)`.
    fn head_norm(&self, y: &'a str, n: usize) -> Option<(&'a str, f32, [usize; 3], &'a str)> {
        let r1 = self.cons(y, "Reshape")?;
        let ln = self.cons(self.out(r1), OpKind::RMSNorm.as_str())?;
        let node = &self.nodes[ln];
        let gamma = node.inputs.get(1)?.as_str();
        let d = self.weights.get(gamma)?.data.len();
        let extra_outputs_unused = node.outputs.iter().skip(1).all(|o| o.is_empty() || self.usage.consumers(o) == 0);
        if d == 0
            || n % d != 0
            || !self.row_const(gamma, d)
            || !self.reshape_to(r1, &[0, -1, d as i64])
            || node.inputs.get(2).is_some_and(|b| !b.is_empty())
            || node.attrs.i("axis", -1) != -1
            || !extra_outputs_unused
        {
            return None;
        }
        let r2 = self.cons(self.out(ln), "Reshape")?;
        if !self.reshape_to(r2, &[0, -1, n as i64]) {
            return None;
        }
        Some((gamma, node.attrs.f("epsilon", 1e-5), [r1, ln, r2], self.out(r2)))
    }

    fn split_sizes(&self, i: usize, n: usize) -> Option<Vec<i64>> {
        let s = &self.nodes[i];
        if s.attrs.i("axis", 0) != -1 || s.outputs.iter().any(String::is_empty) {
            return None;
        }
        let k = s.outputs.len() as i64;
        let sizes: Vec<i64> = match s.inputs.get(1).filter(|t| !t.is_empty()) {
            Some(t) => self.weights.get(t)?.data.iter().map(|&v| v as i64).collect(),
            None if !s.attrs.ints("split").is_empty() => s.attrs.ints("split").to_vec(),
            None if n as i64 % k == 0 => vec![n as i64 / k; k as usize],
            None => return None,
        };
        (sizes.len() == s.outputs.len() && sizes.iter().all(|&v| v > 0) && sizes.iter().sum::<i64>() == n as i64)
            .then_some(sizes)
    }

    fn try_fuse(&self, mm: usize) -> Option<Fused> {
        let node = &self.nodes[mm];
        let (x, w) = (node.inputs.first()?.as_str(), node.inputs.get(1)?.as_str());
        let n = *self.weights.get(w).filter(|t| t.shape.len() == 2)?.shape.last()?;
        if node.outputs.len() != 1 || x.is_empty() {
            return None;
        }
        let mut remove = vec![mm];
        let mut y = self.out(mm);
        let mut inputs: Vec<String> = vec![x.into(), w.into(), String::new(), String::new(), String::new()];
        let mut attrs = Attributes::default();
        if let Some(add) = self.cons(y, "Add") {
            let b = self.other_input(add, y)?;
            if !self.row_const(b, n) {
                return None;
            }
            inputs[2] = b.into();
            remove.push(add);
            y = self.out(add);
        }
        if let Some((gamma, eps, nodes, out)) = self.head_norm(y, n) {
            inputs[3] = gamma.into();
            attrs.floats.insert("epsilon".into(), eps);
            remove.extend(nodes);
            y = out;
        }
        if let Some(mul) = self.cons(y, "Mul") {
            let c = self.other_input(mul, y)?;
            if !self.row_const(c, n) {
                return None;
            }
            inputs[4] = c.into();
            remove.push(mul);
            y = self.out(mul);
        }
        let split = self.cons(y, "Split")?;
        let sizes = self.split_sizes(split, n)?;
        remove.push(split);
        // Rotary on any of the split outputs, all with the same cache inputs and attributes.
        let mut outputs = Vec::new();
        let mut rotate = Vec::new();
        let mut rotary: Option<&Node> = None;
        for o in &self.nodes[split].outputs {
            match self.cons(o, ROTARY).filter(|&r| self.nodes[r].inputs.first() == Some(o)) {
                Some(r) => {
                    let rn = &self.nodes[r];
                    let same = |a: &Node| {
                        a.inputs[1..] == rn.inputs[1..]
                            && ["interleaved", "num_heads", "rotary_embedding_dim"]
                                .iter()
                                .all(|k| a.attrs.i(k, 0) == rn.attrs.i(k, 0))
                    };
                    if rn.inputs.len() != 4
                        || rn.outputs.len() != 1
                        || rn.attrs.i("is_packed_batching", 0) != 0
                        || rotary.is_some_and(|p| !same(p))
                    {
                        return None;
                    }
                    rotary = Some(rn);
                    remove.push(r);
                    outputs.push(rn.outputs[0].clone());
                    rotate.push(1);
                }
                None => {
                    outputs.push(o.clone());
                    rotate.push(0);
                }
            }
        }
        if let Some(rn) = rotary {
            inputs.extend(rn.inputs[1..].iter().cloned());
            for k in ["interleaved", "num_heads", "rotary_embedding_dim"] {
                attrs.ints.insert(k.into(), rn.attrs.i(k, 0));
            }
        }
        while inputs.last().is_some_and(String::is_empty) {
            inputs.pop();
        }
        attrs.int_lists.insert("split".into(), sizes);
        attrs.int_lists.insert("rotate".into(), rotate);
        let at = *remove.iter().max()?;
        Some(Fused {
            remove: remove.into_iter().filter(|&i| i != at).collect(),
            at,
            node: Node {
                op: OpKind::Unknown("OxiMatMulSplit".into()),
                name: format!("{}/OxiMatMulSplit", self.nodes[split].name),
                inputs,
                outputs,
                attrs,
            },
        })
    }
}

/// Fuse projection + split chains into `OxiMatMulSplit` nodes (see the module docs).
pub fn fuse_matmul_split(nodes: Vec<Node>, weights: &HashMap<String, Tensor>, output_names: &[String]) -> Vec<Node> {
    let fused: Vec<Fused> = {
        let m = Matcher::new(&nodes, weights, output_names);
        let mut claimed = HashSet::new();
        let mut out = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            if n.op != OpKind::MatMul {
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
    tracing::debug!(count = fused.len(), "fused projection + split chains into OxiMatMulSplit");
    let mut drop = HashSet::new();
    let mut replace = HashMap::new();
    for f in fused {
        drop.extend(f.remove);
        replace.insert(f.at, f.node);
    }
    nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| if let Some(f) = replace.remove(&i) { Some(f) } else { (!drop.contains(&i)).then_some(n) })
        .collect()
}

#[cfg(test)]
#[path = "matmul_split_tests.rs"]
mod tests;
