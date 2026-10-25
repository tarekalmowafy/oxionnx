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

/// Execute `nodes` in order with the default registry (weights by reference, as
/// in a session) and return the named outputs.
fn run(nodes: &[Node], weights: &HashMap<String, Tensor>, feeds: &[(&str, Tensor)], outs: &[&str]) -> Vec<Tensor> {
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
    outs.iter().map(|o| env.remove(*o).unwrap()).collect()
}

fn node(op: OpKind, name: &str, inputs: &[&str], out: &str) -> Node {
    make_node(op, name, inputs.to_vec(), vec![out])
}

fn ints(w: &mut HashMap<String, Tensor>, name: &str, v: &[i64]) -> String {
    w.insert(name.into(), Tensor::new(v.iter().map(|&x| x as f32).collect(), vec![v.len()]));
    name.into()
}

fn rotary(input: &str, out: &str, interleaved: i64) -> Node {
    let mut n = node(OpKind::Unknown(ROTARY.into()), &format!("rot_{out}"), &[input, "pos", "cos", "sin"], out);
    n.attrs.ints.insert("interleaved".into(), interleaved);
    n
}

/// Gemma 3 QKV: x [2, 5, K] @ W [K, (Hq + 2) * D], per-head RMSNorm, column
/// scale, split into q/k/v, rotary on q and k.
fn qkv_graph(d: usize, hq: usize, interleaved: i64) -> (Vec<Node>, HashMap<String, Tensor>, Vec<(&'static str, Tensor)>) {
    let (k, n) = (24, (hq + 2) * d);
    let mut w = HashMap::new();
    w.insert("W".to_string(), Tensor::new(rand(k * n, 1), vec![k, n]));
    w.insert("gamma".to_string(), Tensor::new(rand(d, 2), vec![d]));
    w.insert("cs".to_string(), Tensor::new(rand(n, 3), vec![n]));
    w.insert("pos".to_string(), Tensor::new(vec![0.0], vec![1]));
    let heads = ints(&mut w, "c_heads", &[0, -1, d as i64]);
    let flat = ints(&mut w, "c_flat", &[0, -1, n as i64]);
    let sizes = ints(&mut w, "c_sizes", &[(hq * d) as i64, d as i64, d as i64]);
    let mut ln = node(OpKind::RMSNorm, "ln", &["r1", "gamma"], "ln");
    ln.attrs.floats.insert("epsilon".into(), 1e-6);
    ln.attrs.ints.insert("axis".into(), -1);
    let mut split = make_node(OpKind::Split, "split", vec!["m", &sizes], vec!["q", "k", "v"]);
    split.attrs.ints.insert("axis".into(), -1);
    let nodes = vec![
        node(OpKind::MatMul, "mm", &["x", "W"], "y"),
        node(OpKind::Reshape, "r1", &["y", &heads], "r1"),
        ln,
        node(OpKind::Reshape, "r2", &["ln", &flat], "r2"),
        node(OpKind::Mul, "mul", &["cs", "r2"], "m"),
        split,
        rotary("q", "qr", interleaved),
        rotary("k", "kr", interleaved),
    ];
    let feeds = vec![
        ("x", Tensor::new(rand(2 * 5 * k, 4), vec![2, 5, k])),
        ("cos", Tensor::new(rand(8 * d / 2, 5), vec![8, d / 2])),
        ("sin", Tensor::new(rand(8 * d / 2, 6), vec![8, d / 2])),
    ];
    (nodes, w, feeds)
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape);
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn fuses_gemma_qkv_projection_and_matches_unfused() {
    // D = 32 and 64 take the packed epilogue; D = 48 is not a multiple of 32 and falls back.
    for (d, hq, il) in [(32, 2, 0), (64, 4, 0), (32, 1, 1), (48, 2, 0)] {
        let (nodes, w, feeds) = qkv_graph(d, hq, il);
        let outs = ["qr", "kr", "v"];
        let names: Vec<String> = outs.iter().map(|s| s.to_string()).collect();
        let fused = fuse_matmul_split(nodes.clone(), &w, &names);
        assert_eq!(fused.len(), 1, "{:?}", fused.iter().map(|n| n.op.as_str()).collect::<Vec<_>>());
        assert_eq!(fused[0].op.as_str(), "OxiMatMulSplit");
        assert_eq!(fused[0].outputs, names);
        let want = run(&nodes, &w, &feeds, &outs);
        let got = run(&fused, &w, &feeds, &outs);
        for (o, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(max_diff(g, e) < 1e-5, "d={d} output {o}: diff {}", max_diff(g, e));
        }
    }
}

#[test]
fn fuses_plain_biased_split() {
    let (k, n) = (16, 96);
    let mut w = HashMap::new();
    w.insert("W".to_string(), Tensor::new(rand(k * n, 1), vec![k, n]));
    w.insert("b".to_string(), Tensor::new(rand(n, 2), vec![n]));
    let mut split = make_node(OpKind::Split, "split", vec!["yb"], vec!["q", "k", "v"]);
    split.attrs.ints.insert("axis".into(), -1);
    split.attrs.ints.insert("num_outputs".into(), 3);
    let nodes = vec![node(OpKind::MatMul, "mm", &["x", "W"], "y"), node(OpKind::Add, "add", &["y", "b"], "yb"), split];
    let feeds = vec![("x", Tensor::new(rand(3 * 4 * k, 3), vec![3, 4, k]))];
    let outs = ["q", "k", "v"];
    let names: Vec<String> = outs.iter().map(|s| s.to_string()).collect();
    let fused = fuse_matmul_split(nodes.clone(), &w, &names);
    assert_eq!(fused.len(), 1);
    let (want, got) = (run(&nodes, &w, &feeds, &outs), run(&fused, &w, &feeds, &outs));
    for (g, e) in got.iter().zip(&want) {
        assert!(max_diff(g, e) < 1e-5);
    }
}

#[test]
fn declines_shared_intermediates_and_mismatched_rotary() {
    let (nodes, w, _) = qkv_graph(32, 2, 0);
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    // The normalised heads are a graph output.
    assert_eq!(fuse_matmul_split(nodes.clone(), &w, &names(&["qr", "kr", "v", "ln"])).len(), nodes.len());
    // A second consumer of the scaled product.
    let mut extra = nodes.clone();
    extra.push(node(OpKind::Identity, "peek", &["m"], "m_copy"));
    assert_eq!(fuse_matmul_split(extra.clone(), &w, &names(&["qr", "kr", "v", "m_copy"])).len(), extra.len());
    // The two rotary nodes use different caches.
    let mut mixed = nodes.clone();
    mixed[7].inputs[2] = "cos2".into();
    assert_eq!(fuse_matmul_split(mixed.clone(), &w, &names(&["qr", "kr", "v"])).len(), mixed.len());
}
