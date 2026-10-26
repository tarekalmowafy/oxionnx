//! Partially known shapes and shape values, and folding of the shape
//! arithmetic they resolve.
//!
//! [`super::shape_inference`] only records shapes whose every dimension is
//! known, so with a symbolic batch or sequence axis it learns nothing about
//! most activations. Exports compute a lot from such shapes anyway — head
//! sizes (`Shape(q)[3] / 2` as rotary slice bounds, `1 / sqrt(Shape(q)[-1])`
//! as the attention scale), reshape targets (`Concat(batch, -1, 3, H, D)`) —
//! and those small computations block the fusions, because a matcher can only
//! check constants.
//!
//! [`analyse`] propagates, per tensor, the rank and whichever dimensions are
//! static ([`Dims`]), and for small integer tensors that hold shapes, the
//! elements that are known ([`ShapeFacts::values`]). [`fold_shape_values`]
//! then turns every `Shape`-derived tensor whose value is fully known into a
//! weight, and lets [`super::constant_fold`] evaluate what follows.

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use oxionnx_core::OperatorRegistry;
use std::collections::{HashMap, HashSet};

/// Rank and per-axis size of a tensor; `None` for an axis whose size is not
/// static.
pub type Dims = Vec<Option<usize>>;

/// Elements of a shape-like (small, integer) tensor; `None` where unknown.
pub type Values = Vec<Option<i64>>;

/// Shape-like tensors are tracked up to this many elements.
const MAX_VALUES: usize = 64;

/// What [`analyse`] learned: per tensor, its [`Dims`], and for shape-like
/// tensors their [`Values`].
#[derive(Default)]
pub struct ShapeFacts {
    pub dims: HashMap<String, Dims>,
    pub values: HashMap<String, Values>,
}

impl ShapeFacts {
    pub fn rank(&self, name: &str) -> Option<usize> {
        self.dims.get(name).map(Vec::len)
    }

    /// The value of `name` if every element is known.
    pub fn known_values(&self, name: &str) -> Option<Vec<i64>> {
        self.values.get(name)?.iter().copied().collect()
    }
}

fn broadcast(a: &Dims, b: &Dims) -> Dims {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = if i + a.len() >= n { a[i + a.len() - n] } else { Some(1) };
            let y = if i + b.len() >= n { b[i + b.len() - n] } else { Some(1) };
            match (x, y) {
                (Some(1), y) => y,
                (x, Some(1)) => x,
                (Some(x), Some(y)) if x == y => Some(x),
                // A valid graph broadcasts the unknown side to the known
                // non-1 size.
                (Some(x), None) | (None, Some(x)) => Some(x),
                _ => None,
            }
        })
        .collect()
}

fn norm_axis(axis: i64, rank: usize) -> Option<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    (0..rank as i64).contains(&a).then_some(a as usize)
}

/// Elementwise ops whose output has the shape of input 0.
fn same_shape(op: &OpKind) -> bool {
    use OpKind::*;
    matches!(
        op,
        Identity | Cast | Neg | Sqrt | Reciprocal | Relu | Sigmoid | Tanh | Erf | Gelu | SiLU | Exp | Log | Abs | Cos
            | Sin | Softmax | LayerNorm | RMSNorm | Not | Floor | Ceil | Round | Sign | Dropout | Clip | LeakyRelu
    )
}

/// Binary/ternary broadcasting ops.
fn broadcasting(op: &OpKind) -> bool {
    use OpKind::*;
    matches!(op, Add | Sub | Mul | Div | Pow | Equal | Less | Greater | LessOrEqual | GreaterOrEqual | And | Or | Where | VariadicMax | VariadicMin)
}

struct Analysis<'a> {
    weights: &'a HashMap<String, Tensor>,
    facts: ShapeFacts,
}

impl Analysis<'_> {
    fn dims(&self, name: &str) -> Option<&Dims> {
        self.facts.dims.get(name)
    }

    fn vals(&self, name: &str) -> Option<Values> {
        if let Some(v) = self.facts.values.get(name) {
            return Some(v.clone());
        }
        let t = self.weights.get(name)?;
        (t.data.len() <= MAX_VALUES).then(|| t.data.iter().map(|&v| (v.fract() == 0.0).then_some(v as i64)).collect())
    }

    fn ints(&self, name: &str) -> Option<Vec<i64>> {
        self.vals(name)?.into_iter().collect()
    }

    fn set_dims(&mut self, name: &str, d: Dims) {
        if !name.is_empty() {
            self.facts.dims.insert(name.to_string(), d);
        }
    }

    fn set_vals(&mut self, name: &str, v: Values) {
        if !name.is_empty() && v.len() <= MAX_VALUES {
            self.facts.values.insert(name.to_string(), v);
        }
    }

    fn node(&mut self, n: &Node) {
        let input = |k: usize| n.inputs.get(k).map(String::as_str).filter(|s| !s.is_empty());
        let out = n.outputs.first().map(String::as_str).unwrap_or("");
        let d0 = input(0).and_then(|x| self.dims(x).cloned());
        match &n.op {
            OpKind::Shape => {
                // Always 1-D, even when the input's rank is unknown.
                let Some(d) = d0 else {
                    self.set_dims(out, vec![None]);
                    return;
                };
                let (s, e) = oxionnx_ops::registry::misc_ops::shape_slice_bounds(d.len(), &n.attrs);
                let v: Values = d[s..e].iter().map(|x| x.map(|x| x as i64)).collect();
                self.set_dims(out, vec![Some(v.len())]);
                self.set_vals(out, v);
            }
            op if same_shape(op) => {
                if let Some(d) = d0 {
                    self.set_dims(out, d);
                }
                // Values pass through a Cast to another numeric type, and Identity.
                let numeric = matches!(op, OpKind::Identity)
                    || (matches!(op, OpKind::Cast) && matches!(n.attrs.i("to", 0), 1 | 6 | 7 | 11));
                if numeric {
                    if let Some(v) = input(0).and_then(|x| self.facts.values.get(x).cloned()) {
                        self.set_vals(out, v);
                    }
                }
            }
            op if broadcasting(op) => {
                let ds: Option<Vec<Dims>> =
                    n.inputs.iter().filter(|s| !s.is_empty()).map(|s| self.dims(s).cloned()).collect();
                if let Some(ds) = ds {
                    if let Some(first) = ds.first() {
                        let d = ds[1..].iter().fold(first.clone(), |acc, x| broadcast(&acc, x));
                        self.set_dims(out, d);
                    }
                }
            }
            OpKind::MatMul => {
                let (Some(a), Some(b)) = (d0, input(1).and_then(|x| self.dims(x).cloned())) else { return };
                if a.len() < 2 || b.len() < 2 {
                    return;
                }
                let mut d = broadcast(&a[..a.len() - 2].to_vec(), &b[..b.len() - 2].to_vec());
                d.push(a[a.len() - 2]);
                d.push(b[b.len() - 1]);
                self.set_dims(out, d);
            }
            OpKind::Reshape => {
                let Some(target) = input(1).and_then(|t| self.vals(t)) else { return };
                let mut d: Dims = target
                    .iter()
                    .enumerate()
                    .map(|(i, v)| match v {
                        Some(0) if n.attrs.i("allowzero", 0) == 0 => d0.as_ref().and_then(|x| x.get(i).copied().flatten()),
                        Some(v) if *v > 0 => Some(*v as usize),
                        _ => None,
                    })
                    .collect();
                // One unknown axis can be inferred from a fully known input.
                let numel = d0.as_ref().and_then(|x| x.iter().copied().product::<Option<usize>>());
                if let (Some(numel), [i]) = (numel, &d.iter().enumerate().filter(|(_, x)| x.is_none()).map(|(i, _)| i).collect::<Vec<_>>()[..]) {
                    let known: usize = d.iter().flatten().product();
                    if known > 0 && numel % known == 0 {
                        d[*i] = Some(numel / known);
                    }
                }
                self.set_dims(out, d);
                if let Some(v) = input(0).and_then(|x| self.facts.values.get(x).cloned()) {
                    self.set_vals(out, v);
                }
            }
            OpKind::Transpose => {
                let Some(d) = d0 else { return };
                let perm = n.attrs.ints("perm");
                let perm: Vec<i64> = if perm.is_empty() { (0..d.len() as i64).rev().collect() } else { perm.to_vec() };
                if perm.len() != d.len() {
                    return;
                }
                let p: Option<Dims> = perm.iter().map(|&p| norm_axis(p, d.len()).map(|p| d[p])).collect();
                if let Some(p) = p {
                    self.set_dims(out, p);
                }
            }
            OpKind::Split => {
                let Some(d) = d0 else { return };
                let Some(axis) = norm_axis(n.attrs.i("axis", 0), d.len()) else { return };
                let k = n.outputs.len();
                let sizes: Option<Vec<i64>> = match input(1) {
                    Some(t) => self.ints(t),
                    None if !n.attrs.ints("split").is_empty() => Some(n.attrs.ints("split").to_vec()),
                    None => d[axis].filter(|&x| k > 0 && x % k == 0).map(|x| vec![(x / k) as i64; k]),
                };
                for (i, o) in n.outputs.iter().enumerate() {
                    let mut od = d.clone();
                    od[axis] = sizes.as_ref().and_then(|s| s.get(i)).map(|&s| s.max(0) as usize);
                    self.set_dims(o, od);
                }
            }
            OpKind::Squeeze => {
                let (Some(d), Some(axes)) = (d0, input(1).and_then(|t| self.ints(t)).or_else(|| {
                    let a = n.attrs.ints("axes");
                    (!a.is_empty()).then(|| a.to_vec())
                })) else {
                    return;
                };
                let axes: Option<HashSet<usize>> = axes.iter().map(|&a| norm_axis(a, d.len())).collect();
                let Some(axes) = axes else { return };
                self.set_dims(out, d.iter().enumerate().filter(|(i, _)| !axes.contains(i)).map(|(_, x)| *x).collect());
                if let Some(v) = input(0).and_then(|x| self.facts.values.get(x).cloned()) {
                    self.set_vals(out, v);
                }
            }
            OpKind::Unsqueeze => {
                let axes = input(1).and_then(|t| self.ints(t)).or_else(|| {
                    let a = n.attrs.ints("axes");
                    (!a.is_empty()).then(|| a.to_vec())
                });
                let (Some(d), Some(axes)) = (d0, axes) else { return };
                let rank = d.len() + axes.len();
                let axes: Option<HashSet<usize>> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
                let Some(axes) = axes else { return };
                let mut it = d.iter();
                let od: Dims = (0..rank).map(|i| if axes.contains(&i) { Some(1) } else { it.next().copied().flatten() }).collect();
                self.set_dims(out, od);
                if let Some(v) = input(0).and_then(|x| self.facts.values.get(x).cloned()) {
                    self.set_vals(out, v);
                }
            }
            OpKind::Concat => {
                let ds: Option<Vec<Dims>> = n.inputs.iter().map(|s| self.dims(s).cloned()).collect();
                let Some(ds) = ds else { return };
                let Some(first) = ds.first() else { return };
                let Some(axis) = norm_axis(n.attrs.i("axis", 0), first.len()) else { return };
                if ds.iter().any(|x| x.len() != first.len()) {
                    return;
                }
                let mut d = first.clone();
                d[axis] = ds.iter().map(|x| x[axis]).sum();
                for i in (0..d.len()).filter(|&i| i != axis) {
                    d[i] = ds.iter().find_map(|x| x[i]);
                }
                // 1-D shape vectors: concatenate the element values.
                if first.len() == 1 {
                    let mut v = Values::new();
                    for (s, x) in n.inputs.iter().zip(&ds) {
                        match (self.vals(s), x[0]) {
                            (Some(vs), _) => v.extend(vs),
                            (None, Some(len)) => v.extend(std::iter::repeat(None).take(len)),
                            (None, None) => {
                                v.clear();
                                break;
                            }
                        }
                    }
                    if d[0] == Some(v.len()) && !v.is_empty() {
                        self.set_vals(out, v);
                    }
                }
                self.set_dims(out, d);
            }
            OpKind::Gather => {
                let (Some(data), Some(idx)) = (d0, input(1).and_then(|x| self.dims(x).cloned())) else { return };
                let Some(axis) = norm_axis(n.attrs.i("axis", 0), data.len()) else { return };
                let d: Dims = data[..axis].iter().chain(&idx).chain(&data[axis + 1..]).copied().collect();
                self.set_dims(out, d);
                // Elements of a 1-D shape vector at constant indices.
                if data.len() == 1 && idx.len() <= 1 {
                    if let (Some(v), Some(ix)) = (input(0).and_then(|x| self.vals(x)), input(1).and_then(|t| self.ints(t))) {
                        let picked: Option<Values> = ix
                            .iter()
                            .map(|&i| {
                                let i = if i < 0 { i + v.len() as i64 } else { i };
                                v.get(usize::try_from(i).ok()?).copied()
                            })
                            .collect();
                        if let Some(p) = picked {
                            self.set_vals(out, p);
                        }
                    }
                }
            }
            OpKind::Slice => {
                let Some(d) = d0 else { return };
                let (starts, ends) = (input(1).and_then(|t| self.ints(t)), input(2).and_then(|t| self.ints(t)));
                let axes = match input(3) {
                    Some(t) => self.ints(t),
                    None => starts.as_ref().map(|s| (0..s.len() as i64).collect()),
                };
                let steps = match input(4) {
                    Some(t) => self.ints(t),
                    None => starts.as_ref().map(|s| vec![1; s.len()]),
                };
                let mut od = d.clone();
                let mut ok = true;
                match (&starts, &ends, &axes, &steps) {
                    (Some(st), Some(en), Some(ax), Some(sp)) if st.len() == en.len() && ax.len() == st.len() && sp.len() == st.len() => {
                        for i in 0..st.len() {
                            let Some(a) = norm_axis(ax[i], d.len()) else {
                                ok = false;
                                break;
                            };
                            od[a] = match (d[a], sp[i]) {
                                (Some(dim), 1) => {
                                    let clamp = |v: i64| (if v < 0 { v + dim as i64 } else { v }).clamp(0, dim as i64);
                                    Some((clamp(en[i]) - clamp(st[i])).max(0) as usize)
                                }
                                _ => None,
                            };
                        }
                    }
                    _ => {
                        // Unknown bounds: the sliced axes are unknown, but the rank holds.
                        od = d.iter().map(|_| None).collect();
                        for (i, x) in od.iter_mut().enumerate() {
                            if axes.as_ref().is_some_and(|ax| ax.iter().all(|&a| norm_axis(a, d.len()) != Some(i))) {
                                *x = d[i];
                            }
                        }
                    }
                }
                if !ok {
                    return;
                }
                // Slices of a 1-D shape vector.
                if d.len() == 1 && ok {
                    if let (Some(v), Some(st), Some(en), Some(sp)) = (input(0).and_then(|x| self.vals(x)), &starts, &ends, &steps) {
                        if st.len() == 1 && sp[0] == 1 {
                            let len = v.len() as i64;
                            let clamp = |x: i64| (if x < 0 { x + len } else { x }).clamp(0, len) as usize;
                            let (s, e) = (clamp(st[0]), clamp(en[0]));
                            self.set_vals(out, if s < e { v[s..e].to_vec() } else { vec![] });
                        }
                    }
                }
                self.set_dims(out, od);
            }
            OpKind::Expand => {
                let (Some(d), Some(shape)) = (d0, input(1).and_then(|t| self.vals(t))) else { return };
                let target: Dims = shape.iter().map(|v| v.filter(|&v| v >= 0).map(|v| v as usize)).collect();
                self.set_dims(out, broadcast(&d, &target));
            }
            OpKind::ConstantOfShape => {
                if let Some(v) = input(0).and_then(|t| self.vals(t)) {
                    self.set_dims(out, v.iter().map(|x| x.filter(|&x| x >= 0).map(|x| x as usize)).collect());
                }
            }
            OpKind::Unknown(name) if name == "SkipLayerNormalization" || name == "SkipSimplifiedLayerNormalization" => {
                if let Some(d) = d0 {
                    self.set_dims(out, d);
                }
            }
            OpKind::Unknown(name) if name == "OxiRotateHalf" => {
                if let Some(d) = d0 {
                    self.set_dims(out, d);
                }
            }
            _ => {}
        }
    }
}

/// Propagate partially known shapes (and shape values) through `nodes`, which
/// must be topologically sorted. `input_shapes` seeds fully static graph inputs.
pub fn analyse(nodes: &[Node], weights: &HashMap<String, Tensor>, input_shapes: &HashMap<String, Vec<usize>>) -> ShapeFacts {
    let mut a = Analysis { weights, facts: ShapeFacts::default() };
    for (name, t) in weights {
        a.facts.dims.insert(name.clone(), t.shape.iter().map(|&d| Some(d)).collect());
    }
    for (name, s) in input_shapes {
        a.facts.dims.insert(name.clone(), s.iter().map(|&d| Some(d)).collect());
    }
    for n in nodes {
        a.node(n);
    }
    a.facts
}

/// Ops whose fully known output value is folded into a weight.
fn foldable(op: &OpKind) -> bool {
    use OpKind::*;
    matches!(op, Shape | Gather | Slice | Concat | Unsqueeze | Squeeze | Cast | Reshape | Identity)
}

/// Fold every shape-like tensor whose value [`analyse`] resolves completely
/// into a weight, then constant-fold what depends on it. Repeats while that
/// resolves more. `nodes` must be topologically sorted.
pub fn fold_shape_values(
    mut nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    registry: &OperatorRegistry,
    output_names: &[String],
    input_shapes: &HashMap<String, Vec<usize>>,
) -> Vec<Node> {
    let outputs: HashSet<&str> = output_names.iter().map(String::as_str).collect();
    for _ in 0..4 {
        let facts = analyse(&nodes, weights, input_shapes);
        let before = nodes.len();
        nodes.retain(|n| {
            if !foldable(&n.op) || n.outputs.len() != 1 {
                return true;
            }
            let o = &n.outputs[0];
            if o.is_empty() || outputs.contains(o.as_str()) || weights.contains_key(o) {
                return true;
            }
            let (Some(v), Some(d)) = (facts.known_values(o), facts.dims.get(o)) else { return true };
            let Some(shape) = d.iter().copied().collect::<Option<Vec<usize>>>() else { return true };
            if shape.iter().product::<usize>() != v.len() {
                return true;
            }
            weights.insert(o.clone(), Tensor::new(v.iter().map(|&x| x as f32).collect(), shape));
            false
        });
        if nodes.len() == before {
            break;
        }
        nodes = super::constant_fold::constant_fold(nodes, weights, registry, output_names);
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::test_utils::make_node;

    fn w(v: &[f32], shape: &[usize]) -> Tensor {
        Tensor::new(v.to_vec(), shape.to_vec())
    }

    /// `Reshape(y, Concat(Shape(y)[0], -1, 3, 4, 8))` -> Transpose -> `Shape[3] / 2`
    /// folds to 4 although batch and sequence are unknown.
    #[test]
    fn head_size_from_partial_shapes_folds() {
        let mut gather_b = make_node(OpKind::Gather, "g0", vec!["ys", "i0"], vec!["b"]);
        gather_b.attrs.ints.insert("axis".into(), 0);
        let mut concat = make_node(OpKind::Concat, "cat", vec!["b1", "m1", "c3", "c4", "c8"], vec!["target"]);
        concat.attrs.ints.insert("axis".into(), 0);
        let mut tr = make_node(OpKind::Transpose, "tr", vec!["y5"], vec!["yt"]);
        tr.attrs.int_lists.insert("perm".into(), vec![0, 3, 2, 1, 4]);
        let mut gather_d = make_node(OpKind::Gather, "g1", vec!["yts", "i4"], vec!["d"]);
        gather_d.attrs.ints.insert("axis".into(), 0);
        let mut div = make_node(OpKind::Div, "div", vec!["d", "two"], vec!["half"]);
        div.attrs.ints.insert("__oxi_int_div".into(), 1);
        let nodes = vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "wq"], vec!["y"]),
            make_node(OpKind::Shape, "s0", vec!["y"], vec!["ys"]),
            gather_b,
            make_node(OpKind::Unsqueeze, "u", vec!["b", "ax0"], vec!["b1"]),
            concat,
            make_node(OpKind::Reshape, "rs", vec!["y", "target"], vec!["y5"]),
            tr,
            make_node(OpKind::Shape, "s1", vec!["yt"], vec!["yts"]),
            gather_d,
            div,
            make_node(OpKind::Relu, "use", vec!["yt"], vec!["out"]),
        ];
        let mut weights: HashMap<String, Tensor> = [
            ("wq", w(&vec![0.0; 16 * 96], &[16, 96])),
            ("i0", w(&[0.0], &[])),
            ("i4", w(&[4.0], &[])),
            ("ax0", w(&[0.0], &[1])),
            ("m1", w(&[-1.0], &[1])),
            ("c3", w(&[3.0], &[1])),
            ("c4", w(&[4.0], &[1])),
            ("c8", w(&[8.0], &[1])),
            ("two", w(&[2.0], &[])),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let registry = oxionnx_ops::default_registry();
        let outputs = vec!["out".to_string(), "half".to_string()];
        let facts = analyse(&nodes, &weights, &HashMap::new());
        assert_eq!(facts.dims["y5"], vec![None, None, Some(3), Some(4), Some(8)]);
        assert_eq!(facts.dims["yt"], vec![None, Some(4), Some(3), None, Some(8)]);
        assert_eq!(facts.values["target"], vec![None, Some(-1), Some(3), Some(4), Some(8)]);
        // `half` is a graph output, so its Div stays; `d` becomes a weight.
        let out = fold_shape_values(nodes, &mut weights, &registry, &outputs, &HashMap::new());
        assert_eq!(weights["d"].data, [8.0]);
        // The Shape node is dead once its only consumer is folded.
        let out = crate::optimizer::dead_code::dead_node_elimination(out, &outputs);
        assert!(out.iter().all(|n| n.name != "g1" && n.name != "s1"), "{:?}", out.iter().map(|n| &n.name).collect::<Vec<_>>());
    }
}
