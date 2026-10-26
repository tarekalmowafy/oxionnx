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
        let inputs: Vec<Option<&Tensor>> = node
            .inputs
            .iter()
            .map(|n| if n.is_empty() { None } else { weights.get(n).or_else(|| env.get(n)) })
            .collect();
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

fn node(op: OpKind, name: &str, inputs: &[&str], out: &str) -> Node {
    make_node(op, name, inputs.to_vec(), vec![out])
}

fn norm(rms: bool, input: &str, beta: bool, out: &str, eps: f32) -> Node {
    let op = if rms { OpKind::RMSNorm } else { OpKind::LayerNorm };
    let ins: Vec<&str> = if beta { vec![input, "gamma", "beta"] } else { vec![input, "gamma"] };
    let mut n = node(op, "norm", &ins, out);
    n.attrs.floats.insert("epsilon".into(), eps);
    n.attrs.ints.insert("axis".into(), -1);
    n
}

fn weights(h: usize) -> HashMap<String, Tensor> {
    let mut w = HashMap::new();
    w.insert("gamma".into(), Tensor::new(rand(h, 1).iter().map(|v| v + 1.0).collect(), vec![h]));
    w.insert("beta".into(), Tensor::new(rand(h, 2), vec![h]));
    w.insert("bias".into(), Tensor::new(rand(h, 3), vec![h]));
    w
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape);
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn count(nodes: &[Node], op: &str) -> usize {
    nodes.iter().filter(|n| n.op.as_str() == op).count()
}

/// Fuse, check the op counts, and compare `outs` between the two graphs.
fn check(
    nodes: Vec<Node>,
    w: &HashMap<String, Tensor>,
    feeds: &[(&str, Tensor)],
    graph_outputs: &[&str],
    fused_op: Option<&str>,
    removed: usize,
) {
    let outs: Vec<String> = graph_outputs.iter().map(|s| s.to_string()).collect();
    let fused = fuse_skip_norm(nodes.clone(), w, &outs, true, true);
    match fused_op {
        Some(op) => {
            assert_eq!(count(&fused, op), 1, "{fused:#?}");
            assert_eq!(fused.len(), nodes.len() - removed, "{fused:#?}");
        }
        None => assert_eq!(fused.len(), nodes.len(), "should not fuse: {fused:#?}"),
    }
    let (want, got) = (run(&nodes, w, feeds), run(&fused, w, feeds));
    for o in graph_outputs {
        let d = max_diff(&want[*o], &got[*o]);
        assert!(d < 1e-5, "{o}: max diff {d}");
    }
}

fn feeds(shape_a: &[usize], shape_b: &[usize]) -> Vec<(&'static str, Tensor)> {
    let na: usize = shape_a.iter().product();
    let nb: usize = shape_b.iter().product();
    vec![("a", Tensor::new(rand(na, 10), shape_a.to_vec())), ("b", Tensor::new(rand(nb, 11), shape_b.to_vec()))]
}

#[test]
fn layer_norm_with_sum_reused() {
    // Pre-norm (ModernBERT): `s` feeds the norm and the next residual Add.
    let h = 40;
    let w = weights(h);
    for beta in [false, true] {
        let nodes = vec![
            node(OpKind::Add, "add", &["a", "b"], "s"),
            norm(false, "s", beta, "y", 1e-5),
            node(OpKind::Mul, "mm", &["y", "y"], "z"),
            node(OpKind::Add, "next", &["s", "z"], "r"),
        ];
        check(nodes, &w, &feeds(&[2, 3, h], &[2, 3, h]), &["r", "y"], Some("SkipLayerNormalization"), 1);
    }
}

#[test]
fn layer_norm_sum_only_for_norm() {
    // Post-norm (BERT): no sum output, and a constant-bias Add is folded in.
    let h = 33;
    let w = weights(h);
    let nodes = vec![
        node(OpKind::Add, "bias_add", &["bias", "b"], "bb"),
        node(OpKind::Add, "add", &["a", "bb"], "s"),
        norm(false, "s", true, "y", 1e-12),
    ];
    let fused = fuse_skip_norm(nodes.clone(), &w, &["y".into()], true, true);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].inputs, ["a", "b", "gamma", "beta", "bias"]);
    assert_eq!(fused[0].outputs, ["y"]);
    check(nodes, &w, &feeds(&[3, 5, h], &[3, 5, h]), &["y"], Some("SkipLayerNormalization"), 2);
}

#[test]
fn rms_norm_gemma_style() {
    let h = 64;
    let w = weights(h);
    let nodes = vec![
        node(OpKind::Add, "add", &["a", "b"], "s"),
        norm(true, "s", false, "y", 1e-6),
        node(OpKind::Add, "next", &["s", "y"], "r"),
    ];
    check(nodes.clone(), &w, &feeds(&[1, 7, h], &[1, 7, h]), &["r"], Some("SkipSimplifiedLayerNormalization"), 1);
    // With a folded bias.
    let nodes = vec![
        node(OpKind::Add, "bias_add", &["b", "bias"], "bb"),
        node(OpKind::Add, "add", &["bb", "a"], "s"),
        norm(true, "s", false, "y", 1e-6),
        node(OpKind::Add, "next", &["s", "y"], "r"),
    ];
    check(nodes, &w, &feeds(&[2, 7, h], &[2, 7, h]), &["r", "y"], Some("SkipSimplifiedLayerNormalization"), 2);
}

#[test]
fn broadcast_operands() {
    let h = 24;
    let w = weights(h);
    // Smaller operand first, trailing-rows broadcast.
    let nodes = vec![node(OpKind::Add, "add", &["b", "a"], "s"), norm(false, "s", true, "y", 1e-5)];
    check(nodes, &w, &feeds(&[2, 5, h], &[5, h]), &["y"], Some("SkipLayerNormalization"), 1);
    // Leading-1 operand.
    let nodes = vec![node(OpKind::Add, "add", &["a", "b"], "s"), norm(true, "s", false, "y", 1e-6)];
    check(nodes, &w, &feeds(&[3, 4, h], &[1, 4, h]), &["y"], Some("SkipSimplifiedLayerNormalization"), 1);
    // Broadcast over a middle axis: materialised sum, still exact.
    let nodes = vec![
        node(OpKind::Add, "add", &["a", "b"], "s"),
        norm(false, "s", true, "y", 1e-5),
        node(OpKind::Mul, "m", &["s", "y"], "r"),
    ];
    check(nodes, &w, &feeds(&[3, 4, h], &[3, 1, h]), &["r"], Some("SkipLayerNormalization"), 1);
}

#[test]
fn sum_as_graph_output() {
    let h = 16;
    let w = weights(h);
    let nodes = vec![node(OpKind::Add, "add", &["a", "b"], "s"), norm(false, "s", false, "y", 1e-5)];
    let fused = fuse_skip_norm(nodes.clone(), &w, &["s".into(), "y".into()], true, true);
    assert_eq!(fused[0].outputs, ["y", "", "", "s"]);
    check(nodes, &w, &feeds(&[4, h], &[4, h]), &["s", "y"], Some("SkipLayerNormalization"), 1);
}

#[test]
fn not_fused() {
    let h = 16;
    let w = weights(h);
    let f = feeds(&[4, h], &[4, h]);
    // Not the last axis.
    let mut n = norm(false, "s", false, "y", 1e-5);
    n.attrs.ints.insert("axis".into(), 0);
    check(vec![node(OpKind::Add, "add", &["a", "b"], "s"), n], &w, &f, &["y"], None, 0);
    // Constant Add operand (that is a bias, not a residual).
    check(
        vec![node(OpKind::Add, "add", &["a", "bias"], "s"), norm(false, "s", false, "y", 1e-5)],
        &w,
        &f,
        &["y"],
        None,
        0,
    );
    // Not an Add.
    check(
        vec![node(OpKind::Mul, "mul", &["a", "b"], "s"), norm(true, "s", false, "y", 1e-6)],
        &w,
        &f,
        &["y"],
        None,
        0,
    );
    // Norm disabled by the caller.
    let nodes = vec![node(OpKind::Add, "add", &["a", "b"], "s"), norm(true, "s", false, "y", 1e-6)];
    assert_eq!(fuse_skip_norm(nodes.clone(), &w, &["y".into()], true, false).len(), 2);
}

#[test]
fn computed_beta_placed_at_norm() {
    let h = 20;
    let mut w = weights(h);
    let beta = w.remove("beta").unwrap();
    let mut c = node(OpKind::Constant, "c", &[], "beta");
    c.attrs.tensors.insert("value".into(), beta);
    let f = feeds(&[2, 3, h], &[2, 3, h]);
    // Sum reused after the norm: fused at the norm's position.
    let nodes = vec![
        node(OpKind::Add, "add", &["a", "b"], "s"),
        c.clone(),
        norm(false, "s", true, "y", 1e-5),
        node(OpKind::Add, "next", &["s", "y"], "r"),
    ];
    let fused = fuse_skip_norm(nodes.clone(), &w, &["r".into()], true, true);
    assert_eq!(fused.iter().map(|n| n.op.as_str()).collect::<Vec<_>>(), ["Constant", "SkipLayerNormalization", "Add"]);
    check(nodes, &w, &f, &["r"], Some("SkipLayerNormalization"), 1);
    // Sum used before the norm: left alone.
    let nodes = vec![
        node(OpKind::Add, "add", &["a", "b"], "s"),
        node(OpKind::Mul, "early", &["s", "s"], "e"),
        c,
        norm(false, "s", true, "y", 1e-5),
        node(OpKind::Add, "next", &["e", "y"], "r"),
    ];
    check(nodes, &w, &f, &["r"], None, 0);
}
