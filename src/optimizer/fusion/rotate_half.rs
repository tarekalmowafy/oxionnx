//! Written-out rotary embedding fusion → `OxiRotateHalf`.
//!
//! HF exports that do not use `com.microsoft::RotaryEmbedding` spell
//! `apply_rotary_pos_emb` out op by op:
//!
//! ```text
//! x_lo = Slice(x, 0, D/2, axis=-1);  x_hi = Slice(x, D/2, INT_MAX, axis=-1)   // or Split(x, 2, axis=-1)
//! out  = Add(Mul(x, cos), Mul(Concat(Neg(x_hi), x_lo, axis=-1), sin))        // operands in any order
//! ```
//!
//! Seven nodes and five full-size intermediates become one `OxiRotateHalf(x,
//! cos, sin)` that reads `x` once (see `oxionnx_ops::contrib::rotate_half_into`).
//! `cos`/`sin` may be any tensors; the kernel broadcasts them to `x` at run
//! time, so this works in either head layout (arctic rotates `[B, S, H, D]`,
//! the ModernBERT exports `[B, H, S, D]`).
//!
//! The slice/split axis must be provably the last one (`-1`, or the last index
//! of a rank known from [`crate::optimizer::shape_values`]). Every removed
//! intermediate must have exactly one consumer and must not be a graph output.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::optimizer::shape_values::ShapeFacts;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

struct Matcher<'a> {
    nodes: &'a [Node],
    weights: &'a HashMap<String, Tensor>,
    facts: &'a ShapeFacts,
    usage: TensorUsage,
    producer: HashMap<&'a str, usize>,
}

/// The two halves of `x` and how to describe them to the kernel.
struct Halves<'a> {
    x: &'a str,
    nodes: Vec<usize>,
    attrs: Attributes,
}

impl<'a> Matcher<'a> {
    fn prod(&self, t: &str, op: OpKind) -> Option<usize> {
        if !self.usage.is_fusable_intermediate(t) {
            return None;
        }
        let i = *self.producer.get(t)?;
        (self.nodes[i].op == op).then_some(i)
    }

    fn input(&self, i: usize, k: usize) -> Option<&'a str> {
        self.nodes[i].inputs.get(k).map(String::as_str).filter(|s| !s.is_empty())
    }

    fn ints(&self, name: &str) -> Option<Vec<i64>> {
        Some(self.weights.get(name)?.data.iter().map(|&v| v as i64).collect())
    }

    /// `axis` names the last axis of `x`.
    fn last_axis(&self, x: &str, axis: i64) -> bool {
        axis == -1 || self.facts.dims.get(x).is_some_and(|d| axis >= 0 && axis as usize + 1 == d.len())
    }

    /// `Slice(x, [s], [e], [last axis], [1]?)`: returns `(x, [s, e])`.
    fn slice(&self, t: &str) -> Option<(&'a str, [i64; 2], usize)> {
        let i = self.prod(t, OpKind::Slice)?;
        let x = self.input(i, 0)?;
        let one = |k: usize| self.input(i, k).and_then(|n| self.ints(n)).filter(|v| v.len() == 1).map(|v| v[0]);
        let (s, e, axis) = (one(1)?, one(2)?, one(3)?);
        if self.input(i, 4).is_some_and(|_| one(4) != Some(1)) || !self.last_axis(x, axis) {
            return None;
        }
        Some((x, [s, e], i))
    }

    /// `rot = Concat(Neg(x_hi), x_lo)` along the last axis.
    fn halves(&self, rot: &str) -> Option<Halves<'a>> {
        let cat = self.prod(rot, OpKind::Concat)?;
        let node = &self.nodes[cat];
        if node.inputs.len() != 2 {
            return None;
        }
        let neg = self.prod(&node.inputs[0], OpKind::Neg)?;
        let (hi, lo) = (self.input(neg, 0)?, self.input(cat, 1)?);
        let axis = node.attrs.i("axis", 0);
        let mut attrs = Attributes::default();
        if let (Some((x, hs, hn)), Some((x2, ls, ln))) = (self.slice(hi), self.slice(lo)) {
            if x != x2 || !self.last_axis(x, axis) {
                return None;
            }
            attrs.int_lists.insert("lo".into(), ls.to_vec());
            attrs.int_lists.insert("hi".into(), hs.to_vec());
            return Some(Halves { x, nodes: vec![cat, neg, hn, ln], attrs });
        }
        // Split(x, 2, axis = -1): hi = output 1, lo = output 0.
        let sp = *self.producer.get(hi)?;
        let s = &self.nodes[sp];
        if s.op != OpKind::Split
            || s.outputs.len() != 2
            || s.outputs[0] != lo
            || s.outputs[1] != hi
            || !self.usage.is_fusable_intermediate(lo)
            || !self.usage.is_fusable_intermediate(hi)
        {
            return None;
        }
        let x = self.input(sp, 0)?;
        if !self.last_axis(x, s.attrs.i("axis", 0)) || !self.last_axis(x, axis) {
            return None;
        }
        let sizes = match self.input(sp, 1) {
            Some(t) => Some(self.ints(t)?),
            None if !s.attrs.ints("split").is_empty() => Some(s.attrs.ints("split").to_vec()),
            None => None,
        };
        match sizes {
            Some(v) if v.len() == 2 => {
                attrs.int_lists.insert("split".into(), v);
            }
            Some(_) => return None,
            None => {
                attrs.ints.insert("halves".into(), 1);
            }
        }
        Some(Halves { x, nodes: vec![cat, neg, sp], attrs })
    }

    /// `Mul(a, b)` with one operand `x`: returns the other.
    fn other_of_mul(&self, m: usize, x: &str) -> Option<&'a str> {
        let (a, b) = (self.input(m, 0)?, self.input(m, 1)?);
        match (a == x, b == x) {
            (true, false) => Some(b),
            (false, true) => Some(a),
            _ => None,
        }
    }

    fn try_fuse(&self, add: usize) -> Option<(Vec<usize>, Node)> {
        let node = &self.nodes[add];
        if node.inputs.len() != 2 || node.outputs.len() != 1 {
            return None;
        }
        for (p, q) in [(0, 1), (1, 0)] {
            let (Some(m1), Some(m2)) = (self.prod(&node.inputs[p], OpKind::Mul), self.prod(&node.inputs[q], OpKind::Mul)) else {
                continue;
            };
            for (r, s) in [(0, 1), (1, 0)] {
                let (Some(rot), Some(sin)) = (self.input(m2, r), self.input(m2, s)) else { continue };
                let Some(h) = self.halves(rot) else { continue };
                let Some(cos) = self.other_of_mul(m1, h.x) else { continue };
                let mut remove = h.nodes;
                remove.extend([m1, m2]);
                let fused = Node {
                    op: OpKind::Unknown("OxiRotateHalf".into()),
                    name: format!("{}/OxiRotateHalf", node.name),
                    inputs: vec![h.x.to_string(), cos.to_string(), sin.to_string()],
                    outputs: node.outputs.clone(),
                    attrs: h.attrs,
                };
                return Some((remove, fused));
            }
        }
        None
    }
}

/// Fuse written-out rotary chains into `OxiRotateHalf` nodes (see the module
/// docs). `facts` supplies ranks for non-negative axes.
pub fn fuse_rotate_half(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    facts: &ShapeFacts,
    output_names: &[String],
) -> Vec<Node> {
    let mut drop = HashSet::new();
    let mut replace = HashMap::new();
    {
        let mut producer = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for o in n.outputs.iter().filter(|o| !o.is_empty()) {
                producer.insert(o.as_str(), i);
            }
        }
        let m = Matcher { nodes: &nodes, weights, facts, usage: TensorUsage::new(&nodes, output_names), producer };
        for (i, n) in nodes.iter().enumerate() {
            if n.op != OpKind::Add {
                continue;
            }
            if let Some((remove, fused)) = m.try_fuse(i) {
                if remove.iter().all(|j| !drop.contains(j) && !replace.contains_key(j)) {
                    drop.extend(remove);
                    replace.insert(i, fused);
                }
            }
        }
    }
    if replace.is_empty() {
        return nodes;
    }
    tracing::debug!(count = replace.len(), "fused written-out rotary chains into OxiRotateHalf");
    nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| replace.remove(&i).or_else(|| (!drop.contains(&i)).then_some(n)))
        .collect()
}

#[cfg(test)]
#[path = "rotate_half_tests.rs"]
mod tests;
