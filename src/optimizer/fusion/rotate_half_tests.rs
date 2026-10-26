use super::*;
use crate::optimizer::test_utils::make_node;
use oxionnx_core::OpContext;

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// Execute `nodes` in order with the default registry; returns every tensor.
fn run(nodes: &[Node], weights: &HashMap<String, Tensor>, feeds: &[(&str, Tensor)]) -> HashMap<String, Tensor> {
    let registry = oxionnx_ops::default_registry();
    let mut env: HashMap<String, Tensor> = feeds.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    for node in nodes {
        let op = registry.get(node.op.as_str()).unwrap_or_else(|| panic!("no op {}", node.op.as_str()));
        let inputs: Vec<Option<&Tensor>> =
            node.inputs.iter().map(|n| if n.is_empty() { None } else { weights.get(n).or_else(|| env.get(n)) }).collect();
        let ctx = OpContext { node, inputs, outer_scope: None, weights: Some(weights), registry: Some(&registry) };
        let results = op.execute(&ctx).unwrap_or_else(|e| panic!("{}: {e}", node.name));
        for (name, t) in node.outputs.iter().zip(results) {
            if !name.is_empty() {
                env.insert(name.clone(), t);
            }
        }
    }
    env
}

fn node(op: OpKind, name: &str, inputs: &[&str], out: &[&str]) -> Node {
    make_node(op, name, inputs.to_vec(), out.to_vec())
}

fn c(v: &[f32]) -> Tensor {
    Tensor::new(v.to_vec(), vec![v.len()])
}

/// The rotate-half chain on `x` with the halves taken by `Slice` (`split = false`)
/// or `Split`; `swap` reverses the operand order of every commutative op.
fn chain(split: bool, swap: bool, axis: i64, half: i64) -> (Vec<Node>, HashMap<String, Tensor>) {
    let mut w = HashMap::new();
    let mut nodes = Vec::new();
    if split {
        let mut s = node(OpKind::Split, "split", &["x"], &["x_lo", "x_hi"]);
        s.attrs.ints.insert("axis".into(), axis);
        s.attrs.ints.insert("num_outputs".into(), 2);
        nodes.push(s);
    } else {
        w.insert("zero".into(), c(&[0.0]));
        w.insert("half".into(), c(&[half as f32]));
        w.insert("max".into(), c(&[9.223372e18]));
        w.insert("axis".into(), c(&[axis as f32]));
        w.insert("one".into(), c(&[1.0]));
        nodes.push(node(OpKind::Slice, "lo", &["x", "zero", "half", "axis", "one"], &["x_lo"]));
        nodes.push(node(OpKind::Slice, "hi", &["x", "half", "max", "axis"], &["x_hi"]));
    }
    nodes.push(node(OpKind::Neg, "neg", &["x_hi"], &["n_hi"]));
    let mut cat = node(OpKind::Concat, "cat", &["n_hi", "x_lo"], &["rot"]);
    cat.attrs.ints.insert("axis".into(), -1);
    nodes.push(cat);
    let (m1, m2, add): (&[&str], &[&str], &[&str]) =
        if swap { (&["cos", "x"], &["sin", "rot"], &["t2", "t1"]) } else { (&["x", "cos"], &["rot", "sin"], &["t1", "t2"]) };
    nodes.push(node(OpKind::Mul, "m1", m1, &["t1"]));
    nodes.push(node(OpKind::Mul, "m2", m2, &["t2"]));
    nodes.push(node(OpKind::Add, "add", add, &["out"]));
    nodes.push(node(OpKind::Relu, "after", &["out"], &["y"]));
    (nodes, w)
}

fn count(nodes: &[Node], op: &str) -> usize {
    nodes.iter().filter(|n| n.op.as_str() == op).count()
}

/// Fuse, expect `fused` OxiRotateHalf nodes, and compare `y` bit for bit.
fn check(nodes: Vec<Node>, w: &HashMap<String, Tensor>, facts: &ShapeFacts, feeds: &[(&str, Tensor)], fused: usize) {
    let outs = vec!["y".to_string()];
    let got = fuse_rotate_half(nodes.clone(), w, facts, &outs);
    assert_eq!(count(&got, "OxiRotateHalf"), fused, "{got:#?}");
    if fused > 0 {
        assert_eq!(got.len(), 2, "{got:#?}");
    }
    let want = &run(&nodes, w, feeds)["y"];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for threads in [1, 4] {
        let r = if threads == 1 { run(&got, w, feeds) } else { pool.install(|| run(&got, w, feeds)) };
        assert_eq!(r["y"].shape, want.shape);
        assert!(r["y"].data.iter().zip(&want.data).all(|(a, b)| a.to_bits() == b.to_bits()), "threads={threads}");
    }
}

#[test]
fn fuses_both_layouts_and_both_half_forms() {
    let no_facts = ShapeFacts::default();
    // (x shape, cos shape): [B, S, H, D] rotated with [B, S, 1, D] (arctic) and
    // [B, H, S, D] with [1, 1, S, D] (ModernBERT); the last is large enough to
    // run the parallel path.
    for (xs, cs) in [(vec![2, 5, 3, 8], vec![2, 5, 1, 8]), (vec![2, 3, 5, 8], vec![1, 1, 5, 8]), (vec![2, 8, 64, 64], vec![2, 1, 64, 64])] {
        let n: usize = xs.iter().product();
        let m: usize = cs.iter().product();
        let feeds = [
            ("x", Tensor::new(rand(n, 1), xs.clone())),
            ("cos", Tensor::new(rand(m, 2), cs.clone())),
            ("sin", Tensor::new(rand(m, 3), cs.clone())),
        ];
        let d = *xs.last().unwrap() as i64;
        for split in [false, true] {
            for swap in [false, true] {
                let (nodes, w) = chain(split, swap, -1, d / 2);
                check(nodes, &w, &no_facts, &feeds, 1);
            }
        }
    }
}

#[test]
fn positive_axis_needs_a_known_rank() {
    let feeds = [
        ("x", Tensor::new(rand(2 * 3 * 4 * 8, 1), vec![2, 3, 4, 8])),
        ("cos", Tensor::new(rand(4 * 8, 2), vec![4, 8])),
        ("sin", Tensor::new(rand(4 * 8, 3), vec![4, 8])),
    ];
    let (nodes, w) = chain(false, false, 3, 4);
    check(nodes.clone(), &w, &ShapeFacts::default(), &feeds, 0);
    let mut facts = ShapeFacts::default();
    facts.dims.insert("x".into(), vec![None, Some(3), None, Some(8)]);
    check(nodes.clone(), &w, &facts, &feeds, 1);
    // Axis 2 is not the last one: not a rotary.
    let (nodes, w) = chain(false, false, 2, 2);
    assert_eq!(count(&fuse_rotate_half(nodes, &w, &facts, &["y".to_string()]), "OxiRotateHalf"), 0);
}

#[test]
fn shared_intermediates_and_mismatched_sources_are_left_alone() {
    let feeds = [
        ("x", Tensor::new(rand(2 * 4 * 8, 1), vec![2, 4, 8])),
        ("z", Tensor::new(rand(2 * 4 * 8, 4), vec![2, 4, 8])),
        ("cos", Tensor::new(rand(4 * 8, 2), vec![4, 8])),
        ("sin", Tensor::new(rand(4 * 8, 3), vec![4, 8])),
    ];
    let facts = ShapeFacts::default();
    // `rot` also feeds another node.
    let (mut nodes, w) = chain(false, false, -1, 4);
    nodes.push(node(OpKind::Relu, "extra", &["rot"], &["e"]));
    let mut with_extra = nodes.clone();
    with_extra.push(node(OpKind::Add, "join", &["y", "e"], &["y2"]));
    let got = fuse_rotate_half(with_extra, &w, &facts, &["y2".to_string()]);
    assert_eq!(count(&got, "OxiRotateHalf"), 0);
    // The cos term multiplies another tensor than the one rotated.
    let (mut nodes, w) = chain(false, false, -1, 4);
    nodes.iter_mut().find(|n| n.name == "m1").unwrap().inputs[0] = "z".into();
    check(nodes, &w, &facts, &feeds, 0);
}
