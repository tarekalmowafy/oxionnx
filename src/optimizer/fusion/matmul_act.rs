//! MatMul + activation fusion → `OxiMatMulAct`.
//!
//! Rewrites a MatMul by a constant 2-D weight followed by a GELU into one node
//! whose kernel applies the activation to each block of the product while it
//! is in cache (see `oxionnx_ops::fused_matmul`), so the product is never
//! written out in full.
//!
//! # Patterns
//!
//! ```text
//! y = MatMul(x, W) [+ b]                       // W: constant [K, N]; b: constant [N]
//!
//! gated (GeGLU):  y0, y1 = Split(y, axis=-1, [N/2, N/2]);  out = Mul(Gelu(y_h), y_(1-h))
//! plain:          out = Gelu(y)  or  Mul(Gelu(y), z)
//! ```
//!
//! `Add` and `Mul` may take their operands in either order. Every removed
//! intermediate must have exactly one consumer (the gated `Split` outputs: one
//! each) and must not be a graph output.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

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

    /// The single consumer of `tensor`, if it is an `op` node and `tensor` can be removed.
    fn cons(&self, tensor: &str, op: OpKind) -> Option<usize> {
        if !self.usage.is_fusable_intermediate(tensor) {
            return None;
        }
        let i = *self.consumer.get(tensor)?;
        (self.nodes[i].op == op).then_some(i)
    }

    fn out(&self, i: usize) -> &'a str {
        &self.nodes[i].outputs[0]
    }

    /// The input of binary node `i` that is not `tensor`.
    fn other_input(&self, i: usize, tensor: &str) -> Option<&'a str> {
        let ins = &self.nodes[i].inputs;
        if ins.len() != 2 {
            return None;
        }
        match (ins[0] == tensor, ins[1] == tensor) {
            (true, false) => Some(&ins[1]),
            (false, true) => Some(&ins[0]),
            _ => None,
        }
    }

    /// `Gelu` node `i`'s `approximate` attribute, if it is one we fuse.
    fn gelu_approx(&self, i: usize) -> Option<&'static str> {
        match self.nodes[i].attrs.s("approximate") {
            "tanh" => Some("tanh"),
            "" | "none" => Some("none"),
            _ => None,
        }
    }

    fn try_fuse(&self, mm: usize) -> Option<Fused> {
        let node = &self.nodes[mm];
        let (x, w) = (node.inputs.first()?.as_str(), node.inputs.get(1)?.as_str());
        let wt = self.weights.get(w)?;
        if wt.shape.len() != 2 || node.outputs.len() != 1 || x.is_empty() {
            return None;
        }
        let n = wt.shape[1];
        let mut remove = vec![mm];
        let mut y = self.out(mm);
        // Optional bias.
        let mut bias = None;
        if let Some(add) = self.cons(y, OpKind::Add) {
            let b = self.other_input(add, y)?;
            let bt = self.weights.get(b)?;
            if bt.data.len() != n || bt.shape.last() != Some(&n) || bt.shape.iter().rev().skip(1).any(|&d| d != 1) {
                return None;
            }
            bias = Some(b);
            remove.push(add);
            y = self.out(add);
        }
        if let Some(gelu) = self.cons(y, OpKind::Gelu) {
            // Plain: Gelu(y), optionally times another tensor.
            let approx = self.gelu_approx(gelu)?;
            remove.push(gelu);
            let g = self.out(gelu);
            let mul = self.cons(g, OpKind::Mul).and_then(|m| Some((m, self.other_input(m, g)?)));
            let (at, z) = match mul {
                Some((m, z)) => (m, Some(z)),
                None => (gelu, None),
            };
            return Some(self.build(remove, at, [x, w, bias.unwrap_or(""), z.unwrap_or("")], approx, -1));
        }
        // Gated: Split into two equal halves along the last axis.
        let split = self.cons(y, OpKind::Split)?;
        if !self.splits_in_halves(split, n) {
            return None;
        }
        let halves = [self.nodes[split].outputs[0].as_str(), self.nodes[split].outputs[1].as_str()];
        let (h, gelu) = (0..2).find_map(|h| Some((h, self.cons(halves[h], OpKind::Gelu)?)))?;
        let approx = self.gelu_approx(gelu)?;
        let g = self.out(gelu);
        let mul = self.cons(g, OpKind::Mul)?;
        if self.other_input(mul, g)? != halves[1 - h] || self.cons(halves[1 - h], OpKind::Mul) != Some(mul) {
            return None;
        }
        remove.extend([split, gelu]);
        Some(self.build(remove, mul, [x, w, bias.unwrap_or(""), ""], approx, h as i64))
    }

    /// `Split` node `i` cuts its input's last axis (of width `n`) into two equal halves.
    fn splits_in_halves(&self, i: usize, n: usize) -> bool {
        let s = &self.nodes[i];
        if s.outputs.len() != 2 || s.outputs.iter().any(String::is_empty) || s.attrs.i("axis", 0) != -1 || n % 2 != 0 {
            return false;
        }
        let sizes: Option<Vec<i64>> = match s.inputs.get(1).filter(|t| !t.is_empty()) {
            Some(t) => self.weights.get(t).map(|w| w.data.iter().map(|&v| v as i64).collect()),
            None if !s.attrs.ints("split").is_empty() => Some(s.attrs.ints("split").to_vec()),
            None => Some(vec![n as i64 / 2; 2]),
        };
        sizes.is_some_and(|v| v == [n as i64 / 2; 2])
    }

    fn build(&self, remove: Vec<usize>, at: usize, inputs: [&str; 4], approx: &str, gate: i64) -> Fused {
        let mut inputs: Vec<String> = inputs.iter().map(|s| s.to_string()).collect();
        while inputs.last().is_some_and(String::is_empty) {
            inputs.pop();
        }
        let mut attrs = Attributes::default();
        attrs.strings.insert("activation".into(), "gelu".into());
        attrs.strings.insert("approximate".into(), approx.into());
        attrs.ints.insert("gate".into(), gate);
        let last = &self.nodes[at];
        let remove = remove.into_iter().filter(|&i| i != at).collect();
        Fused {
            remove,
            at,
            node: Node {
                op: OpKind::Unknown("OxiMatMulAct".into()),
                name: format!("{}/OxiMatMulAct", last.name),
                inputs,
                outputs: last.outputs.clone(),
                attrs,
            },
        }
    }
}

/// Fuse MatMul + GELU chains into `OxiMatMulAct` nodes (see the module docs).
pub fn fuse_matmul_act(nodes: Vec<Node>, weights: &HashMap<String, Tensor>, output_names: &[String]) -> Vec<Node> {
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
    tracing::debug!(count = fused.len(), "fused MatMul + activation chains into OxiMatMulAct");
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
#[path = "matmul_act_tests.rs"]
mod tests;
