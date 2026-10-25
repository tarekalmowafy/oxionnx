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

/// Execute `nodes` in order with the default registry. Weights are passed by
/// reference (as a session does), so the fused op can take its packed path.
fn run(nodes: &[Node], weights: &HashMap<String, Tensor>, feeds: &[(&str, Tensor)], out: &str) -> Tensor {
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
            env.insert(name.clone(), t);
        }
    }
    env.remove(out).unwrap()
}

fn node(op: OpKind, name: &str, inputs: &[&str], out: &str) -> Node {
    make_node(op, name, inputs.to_vec(), vec![out])
}

fn gelu(input: &str, out: &str, approximate: Option<&str>) -> Node {
    let mut n = node(OpKind::Gelu, "gelu", &[input], out);
    if let Some(a) = approximate {
        n.attrs.strings.insert("approximate".into(), a.into());
    }
    n
}

fn split_halves(input: &str, outs: [&str; 2], sizes: Option<&str>) -> Node {
    let inputs: Vec<&str> = std::iter::once(input).chain(sizes).collect();
    let mut n = make_node(OpKind::Split, "split", inputs, outs.to_vec());
    n.attrs.ints.insert("axis".into(), -1);
    n
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape);
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// x [2, 5, K] @ W [K, N] (+ b), as a weights map and feeds.
fn base(k: usize, n: usize) -> (HashMap<String, Tensor>, Vec<(&'static str, Tensor)>) {
    let mut w = HashMap::new();
    w.insert("W".to_string(), Tensor::new(rand(k * n, 1), vec![k, n]));
    w.insert("b".to_string(), Tensor::new(rand(n, 2), vec![n]));
    w.insert("halves".to_string(), Tensor::new(vec![(n / 2) as f32; 2], vec![2]));
    (w, vec![("x", Tensor::new(rand(2 * 5 * k, 3), vec![2, 5, k]))])
}

fn check(nodes: Vec<Node>, w: &HashMap<String, Tensor>, feeds: &[(&str, Tensor)], gate: i64) {
    let fused = fuse_matmul_act(nodes.clone(), w, &["out".to_string()]);
    assert_eq!(fused.len(), 1, "{:?}", fused.iter().map(|n| n.op.as_str()).collect::<Vec<_>>());
    assert_eq!(fused[0].op.as_str(), "OxiMatMulAct");
    assert_eq!(fused[0].attrs.i("gate", 9), gate);
    let want = run(&nodes, w, feeds, "out");
    let got = run(&fused, w, feeds, "out");
    assert!(max_diff(&got, &want) < 1e-5, "diff {}", max_diff(&got, &want));
}

#[test]
fn fuses_gated_gelu_in_both_orders_with_and_without_bias() {
    // N/2 = 64 takes the packed epilogue; N/2 = 20 has no block width and falls back.
    for n in [128, 40] {
        let (w, feeds) = base(24, n);
        for h in [0, 1] {
            for bias in [false, true] {
                let mut nodes = vec![node(OpKind::MatMul, "mm", &["x", "W"], "y")];
                let y = if bias {
                    nodes.push(node(OpKind::Add, "add", &["b", "y"], "yb"));
                    "yb"
                } else {
                    "y"
                };
                nodes.push(split_halves(y, ["y0", "y1"], (n == 40).then_some("halves")));
                let (act, other) = if h == 0 { ("y0", "y1") } else { ("y1", "y0") };
                nodes.push(gelu(act, "g", Some("tanh")));
                nodes.push(node(OpKind::Mul, "mul", &[other, "g"], "out"));
                check(nodes, &w, &feeds, h);
            }
        }
    }
}

#[test]
fn fuses_plain_gelu_with_optional_multiplier() {
    let (mut w, mut feeds) = base(24, 64);
    feeds.push(("z", Tensor::new(rand(2 * 5 * 64, 4), vec![2, 5, 64])));
    w.insert("zb".to_string(), Tensor::new(rand(64, 5), vec![64]));
    for approx in [Some("tanh"), None] {
        // Plain GELU.
        check(vec![node(OpKind::MatMul, "mm", &["x", "W"], "y"), gelu("y", "out", approx)], &w, &feeds, -1);
        // Times a same-shaped tensor (fused into the epilogue) or a broadcast one (applied after).
        for z in ["z", "zb"] {
            let nodes = vec![
                node(OpKind::MatMul, "mm", &["x", "W"], "y"),
                node(OpKind::Add, "add", &["y", "b"], "yb"),
                gelu("yb", "g", approx),
                node(OpKind::Mul, "mul", &["g", z], "out"),
            ];
            check(nodes, &w, &feeds, -1);
        }
    }
}

#[test]
fn declines_unfusable_chains() {
    let (w, _) = base(8, 32);
    let gated = || {
        vec![
            node(OpKind::MatMul, "mm", &["x", "W"], "y"),
            split_halves("y", ["y0", "y1"], None),
            gelu("y0", "g", Some("tanh")),
            node(OpKind::Mul, "mul", &["g", "y1"], "out"),
        ]
    };
    // The activation is a graph output.
    let kept = fuse_matmul_act(gated(), &w, &["out".to_string(), "g".to_string()]);
    assert_eq!(kept.len(), 4);
    // A second consumer of one half.
    let mut extra = gated();
    extra.push(node(OpKind::Identity, "peek", &["y1"], "y1_copy"));
    let kept = fuse_matmul_act(extra.clone(), &w, &["out".to_string(), "y1_copy".to_string()]);
    assert_eq!(kept.len(), extra.len());
    // Unequal split.
    let mut w2 = w.clone();
    w2.insert("sizes".to_string(), Tensor::new(vec![8.0, 24.0], vec![2]));
    let mut uneven = gated();
    uneven[1] = split_halves("y", ["y0", "y1"], Some("sizes"));
    assert_eq!(fuse_matmul_act(uneven, &w2, &["out".to_string()]).len(), 4);
    // The weight is not a constant.
    let mut dynamic = gated();
    dynamic[0] = node(OpKind::MatMul, "mm", &["x", "x2"], "y");
    assert_eq!(fuse_matmul_act(dynamic, &w, &["out".to_string()]).len(), 4);
}
