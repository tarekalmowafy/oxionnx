//! Written-out exact GELU → `Gelu` (`approximate = "none"`).
//!
//! PyTorch exports `nn.GELU()` / `F.gelu` (and HF's `GELUActivation`) below
//! opset 20 as
//!
//! ```text
//! d = Div(x, sqrt 2)            // or Mul(x, 1/sqrt 2)
//! a = Add(Erf(d), 1)
//! out = Mul(Mul(x, a), 0.5)     // (A) GTE, ModernBERT-style MLPs
//!     | Mul(Mul(x, 0.5), a)     // (B) BERT's `x * 0.5 * (1 + erf(..))`
//!     | Mul(x, Mul(a, 0.5))     // (C)
//! ```
//!
//! `Add` and `Mul` may take their operands in either order; the constants are
//! scalar initializers or `Constant` nodes (which stay: CSE shares them). Every removed intermediate must have exactly one
//! consumer and must not be a graph output (`x` itself is kept). The `Gelu`
//! node can then fuse into a preceding MatMul (`OxiMatMulAct`), or run as one
//! vectorised kernel instead of five elementwise passes.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// The values of constant tensor `t`: an initializer, or the output of
/// `Constant` node `producer`.
pub(super) fn const_data(t: &str, weights: &HashMap<String, Tensor>, producer: Option<&Node>) -> Option<Vec<f32>> {
    if let Some(w) = weights.get(t) {
        return Some(w.data.clone());
    }
    let p = producer.filter(|p| p.op == OpKind::Constant && p.outputs.first().map(String::as_str) == Some(t))?;
    let a = &p.attrs;
    if let Some(v) = a.tensors.get("value") {
        return (v.data.len() == v.shape.iter().product::<usize>()).then(|| v.data.clone());
    }
    a.floats.get("value_float").copied().or_else(|| a.ints.get("value_int").map(|&v| v as f32)).map(|v| vec![v])
}

/// Drop `Constant` nodes nothing reads any more (after a fusion removed their
/// consumers; CSE may have shared them, so they are not removed with it).
pub(super) fn drop_unread_constants(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
    // Subgraphs may read outer-scope names without listing them as inputs.
    if nodes.iter().any(|n| !n.attrs.graphs.is_empty()) {
        return nodes;
    }
    let read: HashSet<&str> = nodes.iter().flat_map(|n| n.inputs.iter().map(String::as_str)).chain(output_names.iter().map(String::as_str)).collect();
    let keep: Vec<bool> = nodes.iter().map(|n| n.op != OpKind::Constant || n.outputs.iter().any(|o| read.contains(o.as_str()))).collect();
    nodes.into_iter().zip(keep).filter_map(|(n, k)| k.then_some(n)).collect()
}

struct Matcher<'a> {
    nodes: &'a [Node],
    weights: &'a HashMap<String, Tensor>,
    usage: TensorUsage,
    producer: HashMap<&'a str, usize>,
    consumer: HashMap<&'a str, usize>,
}

impl<'a> Matcher<'a> {
    /// `t` is a scalar constant within float rounding of `v`.
    fn is_const(&self, t: &str, v: f32) -> bool {
        let p = self.producer.get(t).map(|&i| &self.nodes[i]);
        const_data(t, self.weights, p).is_some_and(|d| d.len() == 1 && (d[0] - v).abs() <= 1e-5 * v.abs())
    }

    /// The single consumer of `t`, if it is an `op` node and `t` can be removed.
    fn cons(&self, t: &str, op: OpKind) -> Option<usize> {
        if !self.usage.is_fusable_intermediate(t) {
            return None;
        }
        let i = *self.consumer.get(t)?;
        (self.nodes[i].op == op && self.nodes[i].inputs.len() == 2).then_some(i)
    }

    /// The input of binary node `i` that is not `t`.
    fn other(&self, i: usize, t: &str) -> Option<&'a str> {
        let ins = &self.nodes[i].inputs;
        match (ins[0] == t, ins[1] == t) {
            (true, false) => Some(&ins[1]),
            (false, true) => Some(&ins[0]),
            _ => None,
        }
    }

    /// Binary node `i` is `t * c` (either order) with `c` the scalar `v`.
    fn mul_by(&self, i: usize, t: &str, v: f32) -> bool {
        self.nodes[i].op == OpKind::Mul && self.other(i, t).is_some_and(|o| self.is_const(o, v))
    }

    fn out(&self, i: usize) -> &'a str {
        &self.nodes[i].outputs[0]
    }

    /// `(x, nodes to remove, the final node)` for the chain through Erf node `e`.
    fn try_match(&self, e: usize) -> Option<(&'a str, Vec<usize>, usize)> {
        let d = *self.producer.get(self.nodes[e].inputs.first()?.as_str())?;
        let dn = &self.nodes[d];
        if dn.inputs.len() != 2 || !self.usage.is_fusable_intermediate(self.out(d)) {
            return None;
        }
        let x: &str = match dn.op {
            OpKind::Div if self.is_const(&dn.inputs[1], std::f32::consts::SQRT_2) => &dn.inputs[0],
            OpKind::Mul => {
                let c = [0, 1].into_iter().find(|&j| self.is_const(&dn.inputs[j], std::f32::consts::FRAC_1_SQRT_2))?;
                &dn.inputs[1 - c]
            }
            _ => return None,
        };
        if x.is_empty() {
            return None;
        }
        let a = self.cons(self.out(e), OpKind::Add)?;
        if !self.is_const(self.other(a, self.out(e))?, 1.0) {
            return None;
        }
        let m1 = self.cons(self.out(a), OpKind::Mul)?;
        let o1 = self.other(m1, self.out(a))?;
        if o1 == x {
            // (A) Mul(Mul(x, a), 0.5)
            let m2 = self.cons(self.out(m1), OpKind::Mul)?;
            return self.mul_by(m2, self.out(m1), 0.5).then(|| (x, vec![d, e, a, m1, m2], m2));
        }
        if self.is_const(o1, 0.5) {
            // (C) Mul(x, Mul(a, 0.5))
            let m2 = self.cons(self.out(m1), OpKind::Mul)?;
            return (self.other(m2, self.out(m1))? == x).then(|| (x, vec![d, e, a, m1, m2], m2));
        }
        // (B) Mul(Mul(x, 0.5), a)
        let h = *self.producer.get(o1)?;
        (self.usage.is_fusable_intermediate(o1) && self.nodes[h].inputs.len() == 2 && self.mul_by(h, x, 0.5))
            .then(|| (x, vec![d, e, a, h, m1], m1))
    }
}

/// Rewrite written-out exact GELU chains into `Gelu` nodes (see the module docs).
pub fn fuse_gelu_erf(nodes: Vec<Node>, weights: &HashMap<String, Tensor>, output_names: &[String]) -> Vec<Node> {
    let mut replace: HashMap<usize, Node> = HashMap::new();
    let mut drop: HashSet<usize> = HashSet::new();
    {
        let mut producer = HashMap::new();
        let mut consumer = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for o in n.outputs.iter().filter(|o| !o.is_empty()) {
                producer.insert(o.as_str(), i);
            }
            for inp in n.inputs.iter().filter(|t| !t.is_empty()) {
                consumer.entry(inp.as_str()).or_insert(i);
            }
        }
        let m = Matcher { nodes: &nodes, weights, usage: TensorUsage::new(&nodes, output_names), producer, consumer };
        for (e, n) in nodes.iter().enumerate() {
            if n.op != OpKind::Erf || n.outputs.len() != 1 {
                continue;
            }
            let Some((x, remove, last)) = m.try_match(e) else { continue };
            if remove.iter().any(|i| drop.contains(i) || replace.contains_key(i)) {
                continue;
            }
            let mut attrs = Attributes::default();
            attrs.strings.insert("approximate".into(), "none".into());
            replace.insert(
                last,
                Node {
                    op: OpKind::Gelu,
                    name: format!("{}/Gelu", nodes[last].name),
                    inputs: vec![x.to_string()],
                    outputs: nodes[last].outputs.clone(),
                    attrs,
                },
            );
            drop.extend(remove.into_iter().filter(|&i| i != last));
        }
    }
    if replace.is_empty() {
        return nodes;
    }
    tracing::debug!(count = replace.len(), "fused written-out exact GELU chains into Gelu");
    let nodes = nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| match replace.remove(&i) {
            Some(g) => Some(g),
            None => (!drop.contains(&i)).then_some(n),
        })
        .collect();
    drop_unread_constants(nodes, output_names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: OpKind, inputs: &[&str], output: &str) -> Node {
        Node {
            op,
            name: output.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec![output.to_string()],
            attrs: Attributes::default(),
        }
    }

    fn consts() -> HashMap<String, Tensor> {
        [("sqrt2", std::f32::consts::SQRT_2), ("isqrt2", std::f32::consts::FRAC_1_SQRT_2), ("one", 1.0), ("half", 0.5), ("two", 2.0)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), Tensor::new(vec![v], vec![])))
            .collect()
    }

    /// `x -> Relu -> h`, then the GELU chain on `h` in `form` ending in `y`, then `Relu(y)`.
    fn graph(form: char, div: bool, swap: bool) -> Vec<Node> {
        let sw = |a: &'static str, b: &'static str| if swap { [b, a] } else { [a, b] };
        let mut v = vec![node(OpKind::Relu, &["x"], "h")];
        v.push(if div { node(OpKind::Div, &["h", "sqrt2"], "d") } else { node(OpKind::Mul, &sw("h", "isqrt2"), "d") });
        v.push(node(OpKind::Erf, &["d"], "e"));
        v.push(node(OpKind::Add, &sw("e", "one"), "a"));
        match form {
            'A' => {
                v.push(node(OpKind::Mul, &sw("h", "a"), "m"));
                v.push(node(OpKind::Mul, &sw("m", "half"), "y"));
            }
            'B' => {
                v.push(node(OpKind::Mul, &sw("h", "half"), "m"));
                v.push(node(OpKind::Mul, &sw("m", "a"), "y"));
            }
            _ => {
                v.push(node(OpKind::Mul, &sw("a", "half"), "m"));
                v.push(node(OpKind::Mul, &sw("h", "m"), "y"));
            }
        }
        v.push(node(OpKind::Relu, &["y"], "z"));
        v
    }

    #[test]
    fn fuses_every_form_and_operand_order() {
        for form in ['A', 'B', 'C'] {
            for div in [true, false] {
                for swap in [false, true] {
                    let out = fuse_gelu_erf(graph(form, div, swap), &consts(), &["z".to_string()]);
                    let ops: Vec<&str> = out.iter().map(|n| n.op.as_str()).collect();
                    assert_eq!(ops, ["Relu", "Gelu", "Relu"], "form {form} div={div} swap={swap}");
                    assert_eq!(out[1].inputs, ["h"]);
                    assert_eq!(out[1].outputs, ["y"]);
                    assert_eq!(out[1].attrs.s("approximate"), "none");
                }
            }
        }
    }

    /// The constants as (shared) `Constant` nodes, as CSE leaves them.
    #[test]
    fn fuses_with_constant_nodes() {
        let mut g = graph('A', true, false);
        let mut cs = Vec::new();
        for (name, v) in [("sqrt2", std::f32::consts::SQRT_2), ("one", 1.0), ("half", 0.5)] {
            let mut c = node(OpKind::Constant, &[], name);
            if name == "one" {
                c.attrs.floats.insert("value_float".into(), v);
            } else {
                c.attrs.tensors.insert("value".into(), Tensor::new(vec![v], vec![]));
            }
            cs.push(c);
        }
        cs.append(&mut g);
        let out = fuse_gelu_erf(cs, &HashMap::new(), &["z".to_string()]);
        let ops: Vec<&str> = out.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, ["Relu", "Gelu", "Relu"]);
    }

    #[test]
    fn keeps_chains_that_differ() {
        let outputs = ["z".to_string()];
        // Wrong constant.
        let mut g = graph('A', true, false);
        g[1].inputs[1] = "two".into();
        assert_eq!(fuse_gelu_erf(g, &consts(), &outputs).len(), 7);
        // The final multiply is by another tensor.
        let mut g = graph('A', true, false);
        g[5].inputs[1] = "x".into();
        assert_eq!(fuse_gelu_erf(g, &consts(), &outputs).len(), 7);
        // An intermediate with a second consumer.
        let mut g = graph('B', true, false);
        g.push(node(OpKind::Relu, &["e"], "extra"));
        assert_eq!(fuse_gelu_erf(g, &consts(), &outputs).len(), 8);
        // An intermediate that is a graph output.
        let g = graph('C', false, true);
        assert_eq!(fuse_gelu_erf(g, &consts(), &["z".to_string(), "a".to_string()]).len(), 7);
        // Erf of a tensor other than the multiplied one.
        let mut g = graph('A', true, false);
        g[4].inputs[0] = "x".into();
        assert_eq!(fuse_gelu_erf(g, &consts(), &outputs).len(), 7);
    }
}
