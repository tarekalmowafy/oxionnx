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

/// Execute `nodes` in order with the default registry.
fn run(nodes: &[Node], weights: &HashMap<String, Tensor>, feeds: &[(&str, Tensor)], out: &str) -> Tensor {
    let registry = oxionnx_ops::default_registry();
    let mut env: HashMap<String, Tensor> = weights.clone();
    for (k, v) in feeds {
        env.insert(k.to_string(), v.clone());
    }
    for node in nodes {
        let op = registry.get(node.op.as_str()).unwrap_or_else(|| panic!("no op {}", node.op.as_str()));
        let inputs: Vec<Option<&Tensor>> =
            node.inputs.iter().map(|n| if n.is_empty() { None } else { env.get(n) }).collect();
        let ctx = OpContext { node, inputs, outer_scope: None, weights: Some(weights), registry: Some(&registry) };
        let results = op.execute(&ctx).unwrap_or_else(|e| panic!("{}: {e}", node.name));
        for (name, t) in node.outputs.iter().zip(results) {
            env.insert(name.clone(), t);
        }
    }
    env.remove(out).unwrap()
}

fn ints(w: &mut HashMap<String, Tensor>, name: &str, v: &[i64]) -> String {
    w.insert(name.into(), Tensor::new(v.iter().map(|&x| x as f32).collect(), vec![v.len()]));
    name.into()
}

fn node(op: OpKind, name: &str, inputs: &[&str], out: &str) -> Node {
    make_node(op, name, inputs.to_vec(), vec![out])
}

fn transpose(name: &str, input: &str, out: &str, perm: &[i64]) -> Node {
    let mut n = node(OpKind::Transpose, name, &[input], out);
    n.attrs.int_lists.insert("perm".into(), perm.to_vec());
    n
}

fn softmax(input: &str, out: &str) -> Node {
    let mut n = node(OpKind::Softmax, "sm", &[input], out);
    n.attrs.ints.insert("axis".into(), -1);
    n
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape);
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// The Gemma 3 grouped-query layout (G KV heads, R query heads each, D=8).
fn gqa_graph(b: usize, s: usize, g: i64, r: i64) -> (Vec<Node>, HashMap<String, Tensor>, Vec<(&'static str, Tensor)>) {
    let d = 8i64;
    let mut w = HashMap::new();
    let q5 = ints(&mut w, "c_q5", &[0, 0, g, r, d]);
    let rows = ints(&mut w, "c_rows", &[0, 0, -1, d]);
    let kv4 = ints(&mut w, "c_kv4", &[0, 0, g, d]);
    let probs = ints(&mut w, "c_probs", &[0, 0, -1, 0]);
    let heads = ints(&mut w, "c_heads", &[0, 0, r, -1, d]);
    let flat = ints(&mut w, "c_flat", &[0, 0, g * r * d]);
    let scores = ints(&mut w, "c_scores", &[b as i64, g, r, s as i64, s as i64]);
    w.insert("scale".into(), Tensor::new(vec![0.25], vec![1]));
    let nodes = vec![
        node(OpKind::Reshape, "q_rs", &["q", &q5], "q5"),
        transpose("q_tr", "q5", "qt", &[0, 2, 3, 1, 4]),
        node(OpKind::Reshape, "q_rows", &["qt", &rows], "qr"),
        node(OpKind::Reshape, "k_rs", &["k", &kv4], "k4"),
        transpose("k_tr", "k4", "kt", &[0, 2, 3, 1]),
        node(OpKind::Reshape, "v_rs", &["v", &kv4], "v4"),
        transpose("v_tr", "v4", "vt", &[0, 2, 1, 3]),
        node(OpKind::MatMul, "qk", &["qr", "kt"], "qk"),
        node(OpKind::Mul, "scale_mul", &["qk", "scale"], "qks"),
        node(OpKind::Reshape, "sc_rs", &["qks", &scores], "sc"),
        node(OpKind::Add, "add_bias", &["sc", "bias"], "scb"),
        softmax("scb", "p"),
        node(OpKind::Reshape, "p_rs", &["p", &probs], "pr"),
        node(OpKind::MatMul, "pv", &["pr", "vt"], "pv"),
        node(OpKind::Reshape, "o_rs", &["pv", &heads], "o5"),
        transpose("o_tr", "o5", "ot", &[0, 3, 1, 2, 4]),
        node(OpKind::Reshape, "o_flat", &["ot", &flat], "out"),
    ];
    let (hq, hkv) = ((g * r * d) as usize, (g * d) as usize);
    let feeds = vec![
        ("q", Tensor::new(rand(b * s * hq, 1), vec![b, s, hq])),
        ("k", Tensor::new(rand(b * s * hkv, 2), vec![b, s, hkv])),
        ("v", Tensor::new(rand(b * s * hkv, 3), vec![b, s, hkv])),
        ("bias", Tensor::new(rand(b * s * s, 4), vec![b, 1, 1, s, s])),
    ];
    (nodes, w, feeds)
}

/// The BERT-style 4-D multi-head layout (H=3, D=4), scale as Div.
fn mha_graph(b: usize, s: usize) -> (Vec<Node>, HashMap<String, Tensor>, Vec<(&'static str, Tensor)>) {
    let (h, d) = (3i64, 4i64);
    let mut w = HashMap::new();
    let split = ints(&mut w, "c_split", &[0, 0, h, d]);
    let flat = ints(&mut w, "c_flat", &[0, 0, h * d]);
    w.insert("sqrt_d".into(), Tensor::new(vec![2.0], vec![1]));
    let nodes = vec![
        node(OpKind::Reshape, "q_rs", &["q", &split], "q4"),
        transpose("q_tr", "q4", "qt", &[0, 2, 1, 3]),
        node(OpKind::Reshape, "k_rs", &["k", &split], "k4"),
        transpose("k_tr", "k4", "kt", &[0, 2, 3, 1]),
        node(OpKind::Reshape, "v_rs", &["v", &split], "v4"),
        transpose("v_tr", "v4", "vt", &[0, 2, 1, 3]),
        node(OpKind::MatMul, "qk", &["qt", "kt"], "qk"),
        node(OpKind::Div, "scale_div", &["qk", "sqrt_d"], "qks"),
        node(OpKind::Add, "add_mask", &["mask", "qks"], "scb"),
        softmax("scb", "p"),
        node(OpKind::MatMul, "pv", &["p", "vt"], "pv"),
        transpose("o_tr", "pv", "ot", &[0, 2, 1, 3]),
        node(OpKind::Reshape, "o_flat", &["ot", &flat], "out"),
    ];
    let hd = (h * d) as usize;
    let feeds = vec![
        ("q", Tensor::new(rand(b * s * hd, 5), vec![b, s, hd])),
        ("k", Tensor::new(rand(b * s * hd, 6), vec![b, s, hd])),
        ("v", Tensor::new(rand(b * s * hd, 7), vec![b, s, hd])),
        ("mask", Tensor::new(rand(b * s, 8), vec![b, 1, 1, s])),
    ];
    (nodes, w, feeds)
}

#[test]
fn fuses_gqa_5d_chain_and_matches_unfused() {
    // G != R catches a swapped head split; G = 1 is Gemma 3's global-attention layer.
    for (g, r) in [(2, 2), (1, 4), (3, 2), (2, 1)] {
        let (nodes, w, feeds) = gqa_graph(2, 5, g, r);
        let fused = fuse_attention(nodes.clone(), &w, &["out".to_string()]);
        assert_eq!(fused.len(), 1, "G={g} R={r}: {:?}", fused.iter().map(|n| &n.name).collect::<Vec<_>>());
        assert_eq!(fused[0].op.as_str(), "OxiAttention");
        let want = run(&nodes, &w, &feeds, "out");
        let got = run(&fused, &w, &feeds, "out");
        assert!(max_diff(&got, &want) < 1e-5, "G={g} R={r}: diff {}", max_diff(&got, &want));
    }
}

#[test]
fn fuses_mha_4d_chain_and_matches_unfused() {
    let (nodes, w, feeds) = mha_graph(2, 6);
    let fused = fuse_attention(nodes.clone(), &w, &["out".to_string()]);
    assert_eq!(fused.len(), 1);
    let want = run(&nodes, &w, &feeds, "out");
    let got = run(&fused, &w, &feeds, "out");
    assert!(max_diff(&got, &want) < 1e-5, "diff {}", max_diff(&got, &want));
}

#[test]
fn declines_when_an_intermediate_is_shared_or_an_output() {
    let (nodes, w, _) = gqa_graph(1, 3, 2, 2);
    // The probabilities are a graph output: nothing may be removed.
    let kept = fuse_attention(nodes.clone(), &w, &["out".to_string(), "p".to_string()]);
    assert_eq!(kept.len(), nodes.len());
    // A second consumer of the head-major K.
    let mut extra = nodes.clone();
    extra.push(node(OpKind::Identity, "peek", &["kt"], "kt_copy"));
    let kept = fuse_attention(extra.clone(), &w, &["out".to_string(), "kt_copy".to_string()]);
    assert_eq!(kept.len(), extra.len());
}

#[test]
fn rejects_wrong_scores_shape_at_run_time() {
    let (nodes, mut w, feeds) = gqa_graph(1, 4, 2, 2);
    let fused = fuse_attention(nodes, &w, &["out".to_string()]);
    // A scores reshape that is not [B, G, R, S, T] must be an error, not a silent mis-read.
    w.insert("c_scores".into(), Tensor::new(vec![1.0, 2.0, 4.0, 2.0, 4.0], vec![5]));
    let registry = oxionnx_ops::default_registry();
    let node = &fused[0];
    let env: HashMap<String, Tensor> = feeds.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let inputs: Vec<Option<&Tensor>> = node.inputs.iter().map(|n| env.get(n).or_else(|| w.get(n))).collect();
    let ctx = OpContext { node, inputs, outer_scope: None, weights: Some(&w), registry: Some(&registry) };
    assert!(registry.get("OxiAttention").unwrap().execute(&ctx).is_err());
}
