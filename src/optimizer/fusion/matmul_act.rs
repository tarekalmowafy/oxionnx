//! MatMul + activation fusion → `OxiMatMulAct`.
//!
//! Rewrites a MatMul by a constant 2-D weight followed by a GELU or SiLU into
//! one node whose kernel applies the activation to each block of the product
//! while it is in cache (see `oxionnx_ops::fused_matmul`), so the product is
//! never written out in full.
//!
//! # Patterns
//!
//! ```text
//! y = MatMul(x, W) [+ b]                       // W: constant [K, N]; b: constant [N]
//! act = Gelu | SiLU                            // SiLU: from `Sigmoid + Mul`, see fuse_mul_sigmoid_to_silu
//!
//! gated (GeGLU / SwiGLU, one weight):
//!     y0, y1 = Split(y, axis=-1, [N/2, N/2])   // or two Slices of the halves, see below
//!     out = Mul(act(y_h), y_(1-h))
//! gated (two weights, the usual SwiGLU / GeGLU MLP):
//!     out = Mul(act(MatMul(x, Wg)), MatMul(x, Wu))   // Wg, Wu: constant FP32 [K, H]
//!     → one gated node on the concatenated weight [Wg | Wu]
//! plain:  out = act(y)  or  Mul(act(y), z)
//! ```
//!
//! The Slice form is `Slice(y, 0, H, axes=[-1])` and `Slice(y, H, >= N,
//! axes=[-1])`; the bounds may be computed from `Shape(y)` (e.g. `(N + 1) / 2`
//! as granite / ModernBERT-style exports write them), which is evaluated with
//! the known `N`, and that bound subgraph is removed with the slices.
//!
//! `Add` and `Mul` may take their operands in either order. Every removed
//! intermediate must have exactly one consumer (the gated `Split` outputs: one
//! each) and must not be a graph output. The two-weight form replaces both
//! weights by their concatenation (each original is dropped when nothing else
//! reads it), so the session holds the same amount of weight data.

use crate::graph::{Attributes, Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

struct Matcher<'a> {
    nodes: &'a [Node],
    weights: &'a HashMap<String, Tensor>,
    usage: TensorUsage,
    consumer: HashMap<&'a str, usize>,
    consumers: HashMap<&'a str, Vec<usize>>,
    producer: HashMap<&'a str, usize>,
    output_names: HashSet<&'a str>,
}

/// The two-weight gated form: `name` is to hold `[gate | up]` along the columns.
struct Concat {
    name: String,
    gate: String,
    up: String,
}

struct Fused {
    remove: Vec<usize>,
    at: usize,
    node: Node,
    concat: Option<Concat>,
}

/// A value in a Slice-bound subgraph: integers, or `Shape(y)` itself.
#[derive(Clone)]
enum Val {
    Ints(Vec<i64>),
    ShapeOfY,
}

impl<'a> Matcher<'a> {
    fn new(nodes: &'a [Node], weights: &'a HashMap<String, Tensor>, output_names: &'a [String]) -> Self {
        let mut consumer = HashMap::new();
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut producer = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for inp in &n.inputs {
                if !inp.is_empty() {
                    consumer.entry(inp.as_str()).or_insert(i);
                    let c = consumers.entry(inp.as_str()).or_default();
                    if c.last() != Some(&i) {
                        c.push(i);
                    }
                }
            }
            for o in n.outputs.iter().filter(|o| !o.is_empty()) {
                producer.insert(o.as_str(), i);
            }
        }
        Self {
            nodes,
            weights,
            usage: TensorUsage::new(nodes, output_names),
            consumer,
            consumers,
            producer,
            output_names: output_names.iter().map(String::as_str).collect(),
        }
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

    /// The activation consuming `t` (its only consumer): `(node, activation, approximate)`.
    fn act(&self, t: &str) -> Option<(usize, &'static str, &'static str)> {
        if let Some(i) = self.cons(t, OpKind::Gelu) {
            let approx = match self.nodes[i].attrs.s("approximate") {
                "tanh" => "tanh",
                "" | "none" => "none",
                _ => return None,
            };
            return Some((i, "gelu", approx));
        }
        self.cons(t, OpKind::SiLU).filter(|&i| self.nodes[i].inputs.len() == 1).map(|i| (i, "silu", ""))
    }

    fn try_fuse(&self, mm: usize) -> Option<Fused> {
        let node = &self.nodes[mm];
        let (x, w) = (node.inputs.first()?.as_str(), node.inputs.get(1)?.as_str());
        let wt = self.weights.get(w)?;
        let (n, nbits) = crate::optimizer::nbits::projection_cols(node, wt)?;
        if node.outputs.len() != 1 || x.is_empty() {
            return None;
        }
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
        if let Some((a, activation, approx)) = self.act(y) {
            // Plain: act(y), optionally times another tensor.
            remove.push(a);
            let g = self.out(a);
            let mul = self.cons(g, OpKind::Mul).and_then(|m| Some((m, self.other_input(m, g)?)));
            let Some((m, z)) = mul else {
                return Some(self.build(remove, a, [x, w, bias.unwrap_or(""), ""], activation, approx, -1, nbits, None));
            };
            // Two weights: z = MatMul(x, Wu) → one gated GEMM on [W | Wu].
            if let Some((up, wu)) = self.up_projection(z, x, wt).filter(|_| bias.is_none() && !nbits) {
                remove.push(up);
                let concat = Concat { name: format!("{w}+{wu}/gated"), gate: w.to_string(), up: wu.to_string() };
                let inputs = [x, concat.name.as_str(), "", ""];
                let mut f = self.build(remove, m, inputs, activation, approx, 0, false, None);
                f.concat = Some(concat);
                return Some(f);
            }
            return Some(self.build(remove, m, [x, w, bias.unwrap_or(""), z], activation, approx, -1, nbits, None));
        }
        // Gated: Split (or two Slices) into two equal halves along the last axis.
        let (halves, cut) = self.halves(y, n)?;
        let (h, (a, activation, approx)) = (0..2).find_map(|h| Some((h, self.act(halves[h])?)))?;
        let g = self.out(a);
        let mul = self.cons(g, OpKind::Mul)?;
        if self.other_input(mul, g)? != halves[1 - h] || self.cons(halves[1 - h], OpKind::Mul) != Some(mul) {
            return None;
        }
        remove.extend(cut);
        remove.push(a);
        Some(self.build(remove, mul, [x, w, bias.unwrap_or(""), ""], activation, approx, h as i64, nbits, None))
    }

    /// `z` is `MatMul(x, Wu)` by a constant FP32 weight shaped like `w`, read
    /// only by the gating multiply: `(node, Wu)`.
    fn up_projection(&self, z: &str, x: &str, w: &Tensor) -> Option<(usize, &'a str)> {
        let up = *self.producer.get(z)?;
        let n = &self.nodes[up];
        if n.op != OpKind::MatMul || n.inputs.len() != 2 || n.inputs[0] != x || n.outputs.len() != 1 || !self.usage.is_fusable_intermediate(z) {
            return None;
        }
        let wu = self.weights.get(n.inputs[1].as_str())?;
        (wu.shape == w.shape && w.shape.len() == 2 && wu.data.len() == wu.shape.iter().product::<usize>() && w.data.len() == wu.data.len())
            .then_some((up, n.inputs[1].as_str()))
    }

    /// The two halves of `y`'s last axis (width `n`), cut by one `Split` or by
    /// two `Slice`s: `(half tensors, nodes to remove with them)`.
    fn halves(&self, y: &str, n: usize) -> Option<([&'a str; 2], Vec<usize>)> {
        if let Some(split) = self.cons(y, OpKind::Split) {
            if !self.splits_in_halves(split, n) {
                return None;
            }
            let o = &self.nodes[split].outputs;
            return Some(([o[0].as_str(), o[1].as_str()], vec![split]));
        }
        self.slice_halves(y, n)
    }

    /// `y` is read by two `Slice`s cutting its last axis into halves, and
    /// otherwise only by `Shape` nodes feeding their bounds (see the module docs).
    fn slice_halves(&self, y: &str, n: usize) -> Option<([&'a str; 2], Vec<usize>)> {
        if n % 2 != 0 || self.output_names.contains(y) {
            return None;
        }
        let h = (n / 2) as i64;
        let mut halves: [Option<&'a str>; 2] = [None, None];
        let mut slices = Vec::new();
        let mut bound_nodes = Vec::new();
        for &c in self.consumers.get(y)? {
            let cn = &self.nodes[c];
            if cn.op == OpKind::Shape && cn.inputs.len() == 1 && cn.attrs.ints.is_empty() {
                continue;
            }
            if cn.op != OpKind::Slice || cn.inputs.first().map(String::as_str) != Some(y) || cn.outputs.len() != 1 {
                return None;
            }
            let mut dep = Vec::new();
            let mut ev = |i: usize| -> Option<Option<Vec<i64>>> {
                match cn.inputs.get(i).filter(|t| !t.is_empty()) {
                    None => Some(None),
                    Some(t) => match self.eval(t, y, n, &mut dep, 0)? {
                        Val::Ints(v) => Some(Some(v)),
                        Val::ShapeOfY => None,
                    },
                }
            };
            let (starts, ends, axes, steps) = (ev(1)??, ev(2)??, ev(3)?, ev(4)?);
            if axes.as_deref() != Some(&[-1][..]) || steps.as_ref().is_some_and(|s| s[..] != [1]) || starts.len() != 1 || ends.len() != 1 {
                return None;
            }
            let half = match (starts[0], ends[0]) {
                (0, e) if e == h => 0,
                (s, e) if s == h && e >= n as i64 => 1,
                _ => return None,
            };
            if halves[half].replace(cn.outputs[0].as_str()).is_some() {
                return None;
            }
            slices.push(c);
            bound_nodes.extend(dep);
        }
        let halves = [halves[0]?, halves[1]?];
        // The bound nodes (those derived from `Shape(y)`) go too: each may feed
        // only other bound nodes or the slices.
        bound_nodes.sort_unstable();
        bound_nodes.dedup();
        let shapes: Vec<usize> = self.consumers[y].iter().copied().filter(|c| !slices.contains(c)).collect();
        if !shapes.iter().all(|s| bound_nodes.contains(s)) {
            return None;
        }
        let gone: HashSet<usize> = bound_nodes.iter().chain(&slices).copied().collect();
        for &b in &bound_nodes {
            for o in &self.nodes[b].outputs {
                if self.output_names.contains(o.as_str()) || self.consumers.get(o.as_str()).is_some_and(|cs| cs.iter().any(|c| !gone.contains(c))) {
                    return None;
                }
            }
        }
        slices.extend(bound_nodes);
        Some((halves, slices))
    }

    /// Integer value of `t` in a Slice-bound subgraph, with `Shape(y)`'s last
    /// entry known to be `n`. Nodes computed from `Shape(y)` are added to `dep`.
    fn eval(&self, t: &str, y: &str, n: usize, dep: &mut Vec<usize>, depth: usize) -> Option<Val> {
        let p = self.producer.get(t).copied();
        // Initializers and (shared) `Constant` nodes, which are left in place.
        if let Some(d) = super::gelu_erf::const_data(t, self.weights, p.map(|p| &self.nodes[p])) {
            return d.iter().map(|&v| (v.fract() == 0.0 && v.abs() < 9.0e15).then_some(v as i64)).collect::<Option<Vec<i64>>>().map(Val::Ints);
        }
        if depth > 16 {
            return None;
        }
        let p = p?;
        let pn = &self.nodes[p];
        if pn.outputs.len() != 1 {
            return None;
        }
        let mut arg = |i: usize| -> Option<Val> { self.eval(pn.inputs.get(i).filter(|s| !s.is_empty())?, y, n, dep, depth + 1) };
        let ints = |v: Val| match v {
            Val::Ints(v) => Some(v),
            Val::ShapeOfY => None,
        };
        let v = match pn.op {
            OpKind::Shape if pn.inputs.first().map(String::as_str) == Some(y) && pn.attrs.ints.is_empty() => Val::ShapeOfY,
            OpKind::Gather => {
                let (data, idx) = (arg(0)?, ints(arg(1)?)?);
                match data {
                    // Only the last axis of `Shape(y)` is known.
                    Val::ShapeOfY if pn.attrs.i("axis", 0) == 0 && idx.iter().all(|&i| i == -1) => Val::Ints(vec![n as i64; idx.len()]),
                    Val::Ints(d) if pn.attrs.i("axis", 0) == 0 => {
                        let len = d.len() as i64;
                        Val::Ints(idx.iter().map(|&i| d.get((if i < 0 { i + len } else { i }) as usize).copied()).collect::<Option<_>>()?)
                    }
                    _ => return None,
                }
            }
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
                let (a, b) = (ints(arg(0)?)?, ints(arg(1)?)?);
                let len = a.len().max(b.len());
                if !(a.len() == len || a.len() == 1) || !(b.len() == len || b.len() == 1) {
                    return None;
                }
                let f = |l: i64, r: i64| match pn.op {
                    OpKind::Add => l.checked_add(r),
                    OpKind::Sub => l.checked_sub(r),
                    OpKind::Mul => l.checked_mul(r),
                    _ => l.checked_div(r),
                };
                Val::Ints((0..len).map(|i| f(a[i.min(a.len() - 1)], b[i.min(b.len() - 1)])).collect::<Option<_>>()?)
            }
            OpKind::Cast | OpKind::Identity | OpKind::Unsqueeze | OpKind::Squeeze | OpKind::Reshape => {
                Val::Ints(ints(arg(0)?)?)
            }
            OpKind::Concat => {
                let mut all = Vec::new();
                for i in 0..pn.inputs.len() {
                    all.extend(ints(arg(i)?)?);
                }
                Val::Ints(all)
            }
            _ => return None,
        };
        // Every node reached here sits on a path from `Shape(y)` unless all of
        // its inputs were constants, which only happens for unfolded nodes.
        dep.push(p);
        Some(v)
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

    #[allow(clippy::too_many_arguments)]
    fn build(&self, remove: Vec<usize>, at: usize, inputs: [&str; 4], activation: &str, approx: &str, gate: i64, nbits: bool, concat: Option<Concat>) -> Fused {
        let mut inputs: Vec<String> = inputs.iter().map(|s| s.to_string()).collect();
        while inputs.last().is_some_and(String::is_empty) {
            inputs.pop();
        }
        let mut attrs = Attributes::default();
        attrs.strings.insert("activation".into(), activation.into());
        if !approx.is_empty() {
            attrs.strings.insert("approximate".into(), approx.into());
        }
        attrs.ints.insert("gate".into(), gate);
        if nbits {
            attrs.ints.insert("nbits".into(), 1);
        }
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
            concat,
        }
    }
}

/// Fuse MatMul + GELU / SiLU chains into `OxiMatMulAct` nodes (see the
/// module docs). The two-weight gated form adds the concatenated weight to
/// `weights` and removes the originals nothing else reads.
pub fn fuse_matmul_act(nodes: Vec<Node>, weights: &mut HashMap<String, Tensor>, output_names: &[String]) -> Vec<Node> {
    let fused: Vec<Fused> = {
        let m = Matcher::new(&nodes, weights, output_names);
        let mut claimed = HashSet::new();
        let mut out = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            if !crate::optimizer::nbits::is_projection(n) {
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
    let mut concats = Vec::new();
    for f in fused {
        drop.extend(f.remove);
        replace.insert(f.at, f.node);
        concats.extend(f.concat);
    }
    let nodes: Vec<Node> = nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| if let Some(f) = replace.remove(&i) { Some(f) } else { (!drop.contains(&i)).then_some(n) })
        .collect();
    let nodes = super::gelu_erf::drop_unread_constants(nodes, output_names);
    if !concats.is_empty() {
        let mut read: HashSet<&str> = nodes.iter().flat_map(|n| n.inputs.iter().map(String::as_str)).collect();
        read.extend(output_names.iter().map(String::as_str));
        for c in concats {
            let (g, u) = (&weights[&c.gate], &weights[&c.up]);
            let (k, h) = (g.shape[0], g.shape[1]);
            let mut data = Vec::with_capacity(2 * k * h);
            for (gr, ur) in g.data.chunks_exact(h.max(1)).zip(u.data.chunks_exact(h.max(1))) {
                data.extend_from_slice(gr);
                data.extend_from_slice(ur);
            }
            weights.insert(c.name, Tensor::new(data, vec![k, 2 * h]));
            for name in [c.gate, c.up] {
                if !read.contains(name.as_str()) {
                    weights.remove(&name);
                }
            }
        }
    }
    nodes
}

#[cfg(test)]
#[path = "matmul_act_tests.rs"]
mod tests;
