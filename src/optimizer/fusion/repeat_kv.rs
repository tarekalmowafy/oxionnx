//! `repeat_kv` and attention-bias broadcasts feeding `com.microsoft.MultiHeadAttention`
//! (PERF_PLAN P1.2).
//!
//! Grouped-query decoders exported with a plain `MultiHeadAttention` (Qwen3,
//! voyage-4-nano, pplx-embed onnx-community exports) copy every K/V head to its
//! query-head group first:
//!
//! ```text
//! x  = [B, Hkv, T, H]            (usually Transpose(Reshape(k, [0, 0, Hkv, -1]), [0, 2, 1, 3]))
//! e  = Expand(Unsqueeze(x, [2]), [B, Hkv, G, T, H])
//! kr = Reshape(Transpose(Reshape(e, [B, Hkv*G, T, H]), [0, 2, 1, 3]), [0, 0, Hq*H])
//! MultiHeadAttention(q, kr, vr, ...)
//! ```
//!
//! The pass feeds the attention the un-repeated K/V instead, with the
//! `kv_num_heads = Hq / G` attribute the kernel shares with GroupQueryAttention:
//! the 3-D `k` when `x` is that Reshape/Transpose of it, else `x` itself as
//! `[B, Hkv, T, H]` (which the kernel reads in place). The repeat factor `G` must be a constant entry of the
//! Expand shape (also when that shape is built at run time by
//! `Concat(.., G, ..)`), and the reshape that merges `Hkv` and `G` must be the
//! exporter's `Concat(b, Mul(hkv, G), t, h)` of the same dimensions. The
//! kernel re-checks the K/V head count and sizes on every run, so anything the
//! Expand broadcast besides the group axis turns into an error, not a silent
//! change. The copies are left to dead-node elimination.
//!
//! The same pass drops a broadcast in front of `attention_bias`: an
//! `Expand(m, ..)`, or a `Tile(m, [1, num_heads, 1, 1])`, of the additive mask
//! is replaced by `m`, which the kernel broadcasts itself (`[B|1, N|1, S|1, T|1]`).

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

const MHA: &str = "com.microsoft.MultiHeadAttention";

struct Graph<'a> {
    nodes: &'a [Node],
    weights: &'a HashMap<String, Tensor>,
    producer: HashMap<&'a str, usize>,
}

impl<'a> Graph<'a> {
    fn new(nodes: &'a [Node], weights: &'a HashMap<String, Tensor>) -> Self {
        let mut producer = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for o in &n.outputs {
                if !o.is_empty() {
                    producer.insert(o.as_str(), i);
                }
            }
        }
        Self { nodes, weights, producer }
    }

    /// The node producing `tensor` as its first output, if it is an `op`.
    fn prod(&self, tensor: &str, op: OpKind) -> Option<&'a Node> {
        let n = &self.nodes[*self.producer.get(tensor)?];
        (n.op == op && n.outputs.first().is_some_and(|o| o == tensor)).then_some(n)
    }

    fn input(n: &'a Node, k: usize) -> Option<&'a str> {
        n.inputs.get(k).map(String::as_str).filter(|s| !s.is_empty())
    }

    /// Integer values of an initializer or of a `Constant` node's output (the
    /// onnx-community exports keep their shape constants as `Constant` nodes).
    fn const_ints(&self, name: &str) -> Option<Vec<i64>> {
        let ints = |t: &Tensor| t.data.iter().map(|&v| v as i64).collect();
        if let Some(t) = self.weights.get(name) {
            return Some(ints(t));
        }
        let n = self.prod(name, OpKind::Constant)?;
        if let Some(t) = n.attrs.tensors.get("value") {
            Some(ints(t))
        } else if let Some(&v) = n.attrs.ints.get("value_int") {
            Some(vec![v])
        } else {
            n.attrs.int_lists.get("value_ints").cloned()
        }
    }

    /// `Reshape(x, target)` producing `tensor` (with `allowzero = 0`): `(x, target name)`.
    fn reshape(&self, tensor: &str) -> Option<(&'a str, &'a str)> {
        let n = self.prod(tensor, OpKind::Reshape)?;
        if n.attrs.i("allowzero", 0) != 0 {
            return None;
        }
        Some((Self::input(n, 0)?, Self::input(n, 1)?))
    }

    /// `Transpose(x, [0, 2, 1, 3])` producing `tensor`: `x`.
    fn swap12(&self, tensor: &str) -> Option<&'a str> {
        let n = self.prod(tensor, OpKind::Transpose)?;
        (n.attrs.ints("perm") == [0, 2, 1, 3]).then(|| Self::input(n, 0)).flatten()
    }

    /// `Unsqueeze(x, [2])` producing `tensor`: `x`.
    fn unsqueeze2(&self, tensor: &str) -> Option<&'a str> {
        let n = self.prod(tensor, OpKind::Unsqueeze)?;
        let axes = match Self::input(n, 1) {
            Some(a) => self.const_ints(a)?,
            None => n.attrs.ints("axes").to_vec(),
        };
        (axes == [2]).then(|| Self::input(n, 0)).flatten()
    }

    /// The 5 entries of an Expand shape: either a constant, or (looking through
    /// the exporter's `Where(Equal(s, -1), 1, s)` and `Reshape(s, [-1])`) a
    /// `Concat` of 5 one-element tensors. Each entry is `Ok(constant)` or
    /// `Err(tensor name)`.
    fn shape_entries(&self, mut tensor: &'a str, len: usize) -> Option<Vec<Result<i64, &'a str>>> {
        loop {
            if let Some(v) = self.const_ints(tensor) {
                return (v.len() == len).then(|| v.into_iter().map(Ok).collect());
            }
            if let Some(w) = self.prod(tensor, OpKind::Where) {
                tensor = Self::input(w, 2)?;
            } else if let Some((x, target)) = self.reshape(tensor) {
                if self.const_ints(target)? != [-1] {
                    return None;
                }
                tensor = x;
            } else {
                let c = self.prod(tensor, OpKind::Concat)?;
                if c.inputs.len() != len || c.inputs.iter().any(|i| i.is_empty()) {
                    return None;
                }
                return c
                    .inputs
                    .iter()
                    .map(|i| match self.const_ints(i) {
                        Some(v) if v.len() == 1 => Some(Ok(v[0])),
                        Some(_) => None,
                        None => Some(Err(i.as_str())),
                    })
                    .collect();
            }
        }
    }

    /// `true` when `tensor` is `Mul(a, G)` / `Mul(G, a)` with constant `G`.
    fn is_mul_by(&self, tensor: &str, a: &str, g: i64) -> bool {
        let Some(m) = self.prod(tensor, OpKind::Mul) else { return false };
        let (Some(x), Some(y)) = (Self::input(m, 0), Self::input(m, 1)) else { return false };
        let is_g = |n: &str| self.const_ints(n).is_some_and(|v| v == [g]);
        (x == a && is_g(y)) || (y == a && is_g(x))
    }

    /// Match the repeat_kv chain ending at MHA input `kr`: `(x, G)` with
    /// `x = [B, Hkv, T, H]` the un-repeated tensor.
    fn repeat_kv(&self, kr: &str) -> Option<(&'a str, i64)> {
        let (t2, r4_target) = self.reshape(kr)?;
        match self.const_ints(r4_target)?[..] {
            [0, 0, d] if d > 0 || d == -1 => {}
            _ => return None,
        }
        let r3 = self.swap12(t2)?;
        let (e, r3_target) = self.reshape(r3)?;
        let ex = self.prod(e, OpKind::Expand)?;
        let x = self.unsqueeze2(Self::input(ex, 0)?)?;
        let s5 = self.shape_entries(Self::input(ex, 1)?, 5)?;
        let Ok(g) = s5[2] else { return None };
        if g < 1 {
            return None;
        }
        // The reshape must merge exactly the Hkv and G axes of the Expand result.
        let s4 = self.shape_entries(r3_target, 4)?;
        let merges = match (&s5[1], &s4[1]) {
            (Err(hkv), Err(prod)) => self.is_mul_by(prod, hkv, g),
            (Ok(hkv), Ok(prod)) => *prod == hkv * g,
            _ => false,
        };
        let same = |a: &Result<i64, &str>, b: &Result<i64, &str>| match (a, b) {
            (Ok(x), Ok(y)) => x == y && *x > 0,
            (Err(x), Err(y)) => x == y,
            _ => false,
        };
        if !merges || !same(&s5[0], &s4[0]) || !same(&s5[3], &s4[2]) || !same(&s5[4], &s4[3]) {
            return None;
        }
        Some((x, g))
    }

    /// `x = Transpose(Reshape(k, [0, 0, Hkv, -1 | H]), [0, 2, 1, 3])`: `(k, Hkv)`.
    fn split_heads(&self, x: &str) -> Option<(&'a str, i64)> {
        let (k, target) = self.reshape(self.swap12(x)?)?;
        match self.const_ints(target)?[..] {
            [0, 0, hkv, h] if hkv > 0 && (h > 0 || h == -1) => Some((k, hkv)),
            _ => None,
        }
    }

    /// The un-broadcast source of an attention bias, if `bias` is `Expand(m, ..)`
    /// or `Tile(m, [1, num_heads, 1, 1])`.
    fn bias_source(&self, bias: &str, num_heads: i64) -> Option<&'a str> {
        if let Some(e) = self.prod(bias, OpKind::Expand) {
            return Self::input(e, 0);
        }
        let t = self.prod(bias, OpKind::Tile)?;
        // A head-axis repeat by `num_heads` only yields the [.., num_heads, ..] the
        // kernel accepts from a head axis of 1, i.e. a broadcast.
        (self.const_ints(Self::input(t, 1)?)? == [1, num_heads, 1, 1]).then(|| Self::input(t, 0)).flatten()
    }
}

/// Rewire `com.microsoft.MultiHeadAttention` nodes past `repeat_kv` copies and
/// attention-bias broadcasts (see the module docs); dead copies are removed.
pub fn fuse_repeat_kv(nodes: Vec<Node>, weights: &HashMap<String, Tensor>, output_names: &[String]) -> Vec<Node> {
    // (node, [(input slot, new input)], kv_num_heads)
    type Edit = (usize, Vec<(usize, String)>, Option<i64>);
    let mut edits: Vec<Edit> = Vec::new();
    {
        let g = Graph::new(&nodes, weights);
        for (i, n) in nodes.iter().enumerate() {
            if n.op != OpKind::Unknown(MHA.into()) {
                continue;
            }
            let heads = n.attrs.i("num_heads", 0);
            if heads <= 0 {
                continue;
            }
            let mut rewire = Vec::new();
            let mut kv_heads = None;
            // K/V: no bias, past or present (their head counts would change), plain
            // 3-D query/key/value inputs.
            let free = |k: usize| n.inputs.get(k).map_or(true, |s| s.is_empty());
            let unused_out = |k: usize| n.outputs.get(k).map_or(true, |s| s.is_empty());
            if n.attrs.i("kv_num_heads", 0) == 0
                && [3, 6, 7, 8, 9].into_iter().all(free)
                && unused_out(1)
                && unused_out(2)
                && unused_out(3)
            {
                if let (Some(k), Some(v)) = (Graph::input(n, 1), Graph::input(n, 2)) {
                    if let (Some((xk, gk)), Some((xv, gv))) = (g.repeat_kv(k), g.repeat_kv(v)) {
                        if gk == gv && heads % gk == 0 && gk > 1 {
                            let nkv = heads / gk;
                            // Prefer the 3-D projections, which leaves the head
                            // transposes dead; else the head-major `x` (e.g. a KV-cache
                            // Concat that is a graph output anyway), read in place.
                            match (g.split_heads(xk), g.split_heads(xv)) {
                                (Some((k3, hk)), Some((v3, hv))) if hk == nkv && hv == nkv => {
                                    rewire.extend([(1, k3.to_string()), (2, v3.to_string())]);
                                }
                                _ => rewire.extend([(1, xk.to_string()), (2, xv.to_string())]),
                            }
                            kv_heads = Some(nkv);
                        }
                    }
                }
            }
            if let Some(src) = Graph::input(n, 5).and_then(|b| g.bias_source(b, heads)) {
                rewire.push((5, src.to_string()));
            }
            if !rewire.is_empty() {
                edits.push((i, rewire, kv_heads));
            }
        }
    }
    if edits.is_empty() {
        return nodes;
    }
    let mut nodes = nodes;
    for (i, rewire, kv_heads) in edits {
        let n = &mut nodes[i];
        for (k, name) in rewire {
            n.inputs[k] = name;
        }
        if let Some(h) = kv_heads {
            n.attrs.ints.insert("kv_num_heads".into(), h);
        }
    }
    remove_dead(nodes, output_names)
}

/// Drop nodes none of whose outputs is consumed or a graph output, to a fixed
/// point. Unlike `dead_code::dead_node_elimination` this does not assume the
/// node list is topologically sorted (it is not yet, at this stage, for some
/// exports).
fn remove_dead(mut nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
    let outputs: HashSet<&str> = output_names.iter().map(String::as_str).collect();
    loop {
        let used: HashSet<String> = nodes.iter().flat_map(|n| n.inputs.iter()).filter(|s| !s.is_empty()).cloned().collect();
        let before = nodes.len();
        nodes.retain(|n| n.outputs.is_empty() || n.outputs.iter().any(|o| !o.is_empty() && (used.contains(o) || outputs.contains(o.as_str()))));
        if nodes.len() == before {
            return nodes;
        }
    }
}

#[cfg(test)]
#[path = "repeat_kv_tests.rs"]
mod tests;
