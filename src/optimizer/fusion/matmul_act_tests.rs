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

/// Fuse `nodes` (output `out`) into one `OxiMatMulAct` with `gate` and compare
/// it with the unfused graph; returns the fused graph's weights.
fn check(nodes: Vec<Node>, w: &HashMap<String, Tensor>, feeds: &[(&str, Tensor)], gate: i64) -> HashMap<String, Tensor> {
    let mut fw = w.clone();
    let fused = fuse_matmul_act(nodes.clone(), &mut fw, &["out".to_string()]);
    assert_eq!(fused.len(), 1, "{:?}", fused.iter().map(|n| n.op.as_str()).collect::<Vec<_>>());
    assert_eq!(fused[0].op.as_str(), "OxiMatMulAct");
    assert_eq!(fused[0].attrs.i("gate", 9), gate);
    let want = run(&nodes, w, feeds, "out");
    let got = run(&fused, &fw, feeds, "out");
    assert!(max_diff(&got, &want) < 1e-5, "diff {}", max_diff(&got, &want));
    fw
}

/// `fuse_matmul_act` on a copy of `w`.
fn fuse(nodes: Vec<Node>, w: &HashMap<String, Tensor>, outputs: &[String]) -> Vec<Node> {
    fuse_matmul_act(nodes, &mut w.clone(), outputs)
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
    let kept = fuse(gated(), &w, &["out".to_string(), "g".to_string()]);
    assert_eq!(kept.len(), 4);
    // A second consumer of one half.
    let mut extra = gated();
    extra.push(node(OpKind::Identity, "peek", &["y1"], "y1_copy"));
    let kept = fuse(extra.clone(), &w, &["out".to_string(), "y1_copy".to_string()]);
    assert_eq!(kept.len(), extra.len());
    // Unequal split.
    let mut w2 = w.clone();
    w2.insert("sizes".to_string(), Tensor::new(vec![8.0, 24.0], vec![2]));
    let mut uneven = gated();
    uneven[1] = split_halves("y", ["y0", "y1"], Some("sizes"));
    assert_eq!(fuse(uneven, &w2, &["out".to_string()]).len(), 4);
    // The weight is not a constant.
    let mut dynamic = gated();
    dynamic[0] = node(OpKind::MatMul, "mm", &["x", "x2"], "y");
    assert_eq!(fuse(dynamic, &w, &["out".to_string()]).len(), 4);
}

fn silu(input: &str, out: &str) -> Node {
    node(OpKind::SiLU, "silu", &[input], out)
}

fn int_const(w: &mut HashMap<String, Tensor>, name: &str, v: &[i64]) {
    w.insert(name.to_string(), Tensor::new(v.iter().map(|&x| x as f32).collect(), vec![v.len()]));
}

#[test]
fn fuses_silu_plain_split_and_two_weight_forms() {
    for n in [128, 40] {
        let (mut w, feeds) = base(24, n);
        w.insert("Wu".to_string(), Tensor::new(rand(24 * n, 7), vec![24, n]));
        // Plain, with and without a bias.
        check(vec![node(OpKind::MatMul, "mm", &["x", "W"], "y"), silu("y", "out")], &w, &feeds, -1);
        check(
            vec![node(OpKind::MatMul, "mm", &["x", "W"], "y"), node(OpKind::Add, "add", &["y", "b"], "yb"), silu("yb", "out")],
            &w,
            &feeds,
            -1,
        );
        // One weight, Split into halves.
        for h in [0, 1] {
            let (act, other) = if h == 0 { ("y0", "y1") } else { ("y1", "y0") };
            let nodes = vec![
                node(OpKind::MatMul, "mm", &["x", "W"], "y"),
                split_halves("y", ["y0", "y1"], (n == 40).then_some("halves")),
                silu(act, "g"),
                node(OpKind::Mul, "mul", &["g", other], "out"),
            ];
            check(nodes, &w, &feeds, h);
        }
        // Two weights (SwiGLU), multiply operands in both orders, the up
        // projection before or after the gate.
        for (swap, up_first) in [(false, false), (true, false), (false, true), (true, true)] {
            let up = node(OpKind::MatMul, "up", &["x", "Wu"], "u");
            let mut nodes = vec![node(OpKind::MatMul, "gate", &["x", "W"], "y"), silu("y", "g")];
            nodes.insert(if up_first { 0 } else { 2 }, up);
            nodes.push(node(OpKind::Mul, "mul", if swap { &["u", "g"] } else { &["g", "u"] }, "out"));
            let fw = check(nodes, &w, &feeds, 0);
            assert!(!fw.contains_key("W") && !fw.contains_key("Wu"), "originals dropped");
            assert_eq!(fw["W+Wu/gated"].shape, vec![24, 2 * n]);
        }
        // Two weights where the up weight is also read elsewhere: kept.
        let nodes = vec![
            node(OpKind::MatMul, "gate", &["x", "W"], "y"),
            silu("y", "g"),
            node(OpKind::MatMul, "up", &["x", "Wu"], "u"),
            node(OpKind::Mul, "mul", &["g", "u"], "out"),
            node(OpKind::MatMul, "other", &["x", "Wu"], "out2"),
        ];
        let mut fw = w.clone();
        let fused = fuse_matmul_act(nodes, &mut fw, &["out".to_string(), "out2".to_string()]);
        assert_eq!(fused.len(), 2);
        assert!(fw.contains_key("Wu") && !fw.contains_key("W"));
    }
}

#[test]
fn two_weight_form_needs_the_same_input_and_shape() {
    let (mut w, mut feeds) = base(24, 64);
    w.insert("Wu".to_string(), Tensor::new(rand(24 * 32, 7), vec![24, 32]));
    feeds.push(("x2", Tensor::new(rand(2 * 5 * 24, 9), vec![2, 5, 24])));
    let graph = |up_in: &'static str, wu: &'static str| {
        vec![
            node(OpKind::MatMul, "gate", &["x", "W"], "y"),
            silu("y", "g"),
            node(OpKind::MatMul, "up", &[up_in, wu], "u"),
            node(OpKind::Mul, "mul", &["g", "u"], "out"),
        ]
    };
    // A different input or weight width: the plain form with the multiplier.
    for (up_in, wu) in [("x2", "W"), ("x", "Wu")] {
        let nodes = graph(up_in, wu);
        let mut fw = w.clone();
        let fused = fuse_matmul_act(nodes.clone(), &mut fw, &["out".to_string()]);
        let ops: Vec<&str> = fused.iter().map(|n| n.op.as_str()).collect();
        if wu == "Wu" {
            // Up is [.., 32], gate [.., 64]: broadcasting fails either way; just check no gated rewrite.
            assert!(fused.iter().all(|n| n.attrs.i("gate", -1) == -1), "{ops:?}");
            continue;
        }
        assert_eq!(ops, ["MatMul", "OxiMatMulAct"]);
        assert_eq!(fused[1].attrs.i("gate", 9), -1);
        let want = run(&nodes, &w, &feeds, "out");
        let got = run(&fused, &fw, &feeds, "out");
        assert!(max_diff(&got, &want) < 1e-5);
    }
}

/// `y -> Shape -> Gather(-1) -> Add 1 -> Div 2` as the half width, then
/// `Slice(y, 0, h)` and `Slice(y, h, 2h)` along axis -1 (granite's export).
fn slice_halves_graph(act_half: usize, gelu_act: bool) -> (Vec<Node>, Vec<(&'static str, &'static str)>) {
    let mut nodes = vec![
        node(OpKind::MatMul, "mm", &["x", "W"], "y"),
        node(OpKind::Shape, "shape", &["y"], "s"),
        node(OpKind::Gather, "gather", &["s", "minus1"], "last"),
        node(OpKind::Add, "add1", &["last", "one"], "lp1"),
        node(OpKind::Div, "div2", &["lp1", "two"], "half"),
        node(OpKind::Mul, "end0", &["half", "one"], "e0"),
        node(OpKind::Mul, "end1", &["half", "two"], "e1"),
        node(OpKind::Slice, "slice0", &["y", "zero", "e0", "minus1"], "y0"),
        node(OpKind::Slice, "slice1", &["y", "e0", "e1", "minus1"], "y1"),
    ];
    let (a, o) = if act_half == 0 { ("y0", "y1") } else { ("y1", "y0") };
    nodes.push(if gelu_act { gelu(a, "g", None) } else { silu(a, "g") });
    nodes.push(node(OpKind::Mul, "mul", &["g", o], "out"));
    (nodes, vec![])
}

/// [`slice_halves_graph`] with its integer constants as `Constant` nodes.
fn with_constant_nodes(mut nodes: Vec<Node>) -> Vec<Node> {
    let mut cs = Vec::new();
    for (name, v) in [("minus1", -1.0f32), ("one", 1.0), ("two", 2.0), ("zero", 0.0)] {
        let mut c = make_node(OpKind::Constant, name, vec![], vec![name]);
        c.attrs.tensors.insert("value".into(), Tensor::new(vec![v], vec![1]));
        cs.push(c);
    }
    cs.append(&mut nodes);
    cs
}

fn slice_consts(w: &mut HashMap<String, Tensor>) {
    int_const(w, "minus1", &[-1]);
    int_const(w, "one", &[1]);
    int_const(w, "two", &[2]);
    int_const(w, "zero", &[0]);
}

#[test]
fn fuses_gated_slice_halves_with_shape_computed_bounds() {
    for n in [128, 40] {
        let (mut w, feeds) = base(24, n);
        slice_consts(&mut w);
        for h in [0, 1] {
            for gelu_act in [false, true] {
                let (nodes, _) = slice_halves_graph(h, gelu_act);
                check(nodes, &w, &feeds, h as i64);
                // The same with `Constant` nodes, which stay (shared after CSE).
                let (bare, _) = base(24, n);
                let nodes = with_constant_nodes(slice_halves_graph(h, gelu_act).0);
                let fused = fuse(nodes.clone(), &bare, &["out".to_string()]);
                let ops: Vec<&str> = fused.iter().map(|n| n.op.as_str()).collect();
                assert_eq!(ops, ["OxiMatMulAct"]);
                let want = run(&nodes, &bare, &feeds, "out");
                let got = run(&fused, &bare, &feeds, "out");
                assert!(max_diff(&got, &want) < 1e-5);
            }
        }
    }
}

#[test]
fn declines_slices_that_are_not_halves() {
    let (mut w, _) = base(24, 64);
    slice_consts(&mut w);
    int_const(&mut w, "three", &[3]);
    let outputs = ["out".to_string()];
    // Bounds that cut at a third.
    let (mut nodes, _) = slice_halves_graph(0, false);
    nodes[4].inputs[1] = "three".into();
    assert_eq!(fuse(nodes.clone(), &w, &outputs).len(), nodes.len());
    // A bound tensor read by something else.
    let (mut nodes, _) = slice_halves_graph(0, false);
    nodes.push(node(OpKind::Identity, "peek", &["half"], "peek"));
    let outs = ["out".to_string(), "peek".to_string()];
    assert_eq!(fuse(nodes.clone(), &w, &outs).len(), nodes.len());
    // A slice along another axis.
    let (mut nodes, _) = slice_halves_graph(0, false);
    nodes[7].inputs[3] = "zero".into();
    assert_eq!(fuse(nodes.clone(), &w, &outputs).len(), nodes.len());
    // Gather of an axis other than the last of Shape(y).
    let (mut nodes, _) = slice_halves_graph(1, false);
    nodes[2].inputs[1] = "zero".into();
    assert_eq!(fuse(nodes.clone(), &w, &outputs).len(), nodes.len());
}

/// The optimizer pipeline turns `Sigmoid + Mul` and written-out erf GELU into
/// fused MatMul activations.
#[test]
fn pipeline_fuses_written_out_activations() {
    let registry = oxionnx_ops::default_registry();
    let (mut w, feeds) = base(24, 64);
    w.insert("Wu".to_string(), Tensor::new(rand(24 * 64, 7), vec![24, 64]));
    w.insert("sqrt2".to_string(), Tensor::new(vec![std::f32::consts::SQRT_2], vec![]));
    w.insert("onef".to_string(), Tensor::new(vec![1.0], vec![]));
    w.insert("halff".to_string(), Tensor::new(vec![0.5], vec![]));
    let swiglu = vec![
        node(OpKind::MatMul, "gate", &["x", "W"], "y"),
        node(OpKind::Sigmoid, "sig", &["y"], "sg"),
        node(OpKind::Mul, "silu", &["y", "sg"], "g"),
        node(OpKind::MatMul, "up", &["x", "Wu"], "u"),
        node(OpKind::Mul, "mul", &["g", "u"], "out"),
    ];
    let geglu_erf = vec![
        node(OpKind::MatMul, "mm", &["x", "W"], "y"),
        split_halves("y", ["y0", "y1"], None),
        node(OpKind::Div, "div", &["y1", "sqrt2"], "d"),
        node(OpKind::Erf, "erf", &["d"], "e"),
        node(OpKind::Add, "add", &["e", "onef"], "a"),
        node(OpKind::Mul, "m1", &["y1", "a"], "m"),
        node(OpKind::Mul, "m2", &["m", "halff"], "g"),
        node(OpKind::Mul, "mul", &["g", "y0"], "out"),
    ];
    for (nodes, act, gate) in [(swiglu, "silu", 0), (geglu_erf, "gelu", 1)] {
        let mut fw = w.clone();
        let opt = crate::optimizer::optimize(nodes.clone(), &mut fw, &["out".to_string()], &registry);
        let ops: Vec<&str> = opt.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, ["OxiMatMulAct"], "{act}");
        assert_eq!(opt[0].attrs.s("activation"), act);
        assert_eq!(opt[0].attrs.i("gate", 9), gate);
        let want = run(&nodes, &w, &feeds, "out");
        let got = run(&opt, &fw, &feeds, "out");
        assert!(max_diff(&got, &want) < 1e-5, "{act}: diff {}", max_diff(&got, &want));
    }
}
