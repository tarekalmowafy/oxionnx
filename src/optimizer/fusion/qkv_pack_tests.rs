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

fn node(op: OpKind, name: &str, inputs: &[&str], out: &str) -> Node {
    make_node(op, name, inputs.to_vec(), vec![out])
}

fn t(v: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::new(v, shape.to_vec())
}

const D: usize = 32;
const K: usize = 48;

/// Qwen3-style attention inputs: q (2 heads) and k (1 head) each with a
/// per-head RMSNorm and rotary, v plain; `bias` adds biases to q and v.
fn qwen_block(bias: bool) -> (Vec<Node>, HashMap<String, Tensor>, Vec<String>) {
    let mut w = HashMap::new();
    let widths = [("q", 2 * D), ("k", D), ("v", D)];
    let mut nodes = Vec::new();
    for (i, (p, n)) in widths.iter().enumerate() {
        w.insert(format!("w{p}"), t(rand(K * n, 10 + i as u64), &[K, *n]));
        nodes.push(node(OpKind::MatMul, &format!("{p}_proj"), &["x", &format!("w{p}")], &format!("{p}_mm")));
    }
    w.insert("r_head".into(), t(vec![0.0, -1.0, D as f32], &[3]));
    for (p, n) in [("q", 2 * D), ("k", D)] {
        w.insert(format!("g{p}"), t(rand(D, 20).iter().map(|v| v + 1.0).collect(), &[D]));
        w.insert(format!("r_{p}"), t(vec![0.0, -1.0, n as f32], &[3]));
    }
    let mut q_src = "q_mm".to_string();
    let mut v_out = "v_mm".to_string();
    if bias {
        w.insert("bq".into(), t(rand(2 * D, 30), &[2 * D]));
        w.insert("bv".into(), t(rand(D, 31), &[D]));
        nodes.push(node(OpKind::Add, "q_bias", &["bq", "q_mm"], "q_b"));
        nodes.push(node(OpKind::Add, "v_bias", &["v_mm", "bv"], "v_b"));
        q_src = "q_b".into();
        v_out = "v_b".into();
    }
    w.insert("cos".into(), t(rand(16 * D / 2, 40), &[16, D / 2]));
    w.insert("sin".into(), t(rand(16 * D / 2, 41), &[16, D / 2]));
    w.insert("pos".into(), t(vec![0.0], &[1]));
    for (p, src) in [("q", q_src.as_str()), ("k", "k_mm")] {
        nodes.push(node(OpKind::Reshape, &format!("{p}_r1"), &[src, "r_head"], &format!("{p}_h")));
        let mut sln = node(OpKind::RMSNorm, &format!("{p}_norm"), &[&format!("{p}_h"), &format!("g{p}")], &format!("{p}_n"));
        sln.attrs.floats.insert("epsilon".into(), 1e-6);
        sln.attrs.ints.insert("axis".into(), -1);
        nodes.push(sln);
        nodes.push(node(OpKind::Reshape, &format!("{p}_r2"), &[&format!("{p}_n"), &format!("r_{p}")], &format!("{p}_nf")));
        nodes.push(node(
            OpKind::Unknown("com.microsoft.RotaryEmbedding".into()),
            &format!("{p}_rot"),
            &[&format!("{p}_nf"), "pos", "cos", "sin"],
            &format!("{p}_out"),
        ));
    }
    nodes.push(node(OpKind::Relu, "v_use", &[&v_out], "v_out"));
    (nodes, w, vec!["q_out".into(), "k_out".into(), "v_out".into()])
}

#[test]
fn packs_qwen_projections_into_one_split_projection() {
    for bias in [false, true] {
        let (nodes, w, outs) = qwen_block(bias);
        let feeds = [("x", t(rand(2 * 5 * K, 1), &[2, 5, K]))];
        let want = run(&nodes, &w, &feeds);
        let mut w2 = w.clone();
        let packed = pack_shared_projections(nodes, &mut w2, &outs);
        // The originals are gone; one packed weight remains.
        assert!(!w2.contains_key("wq") && !w2.contains_key("wk") && !w2.contains_key("wv"));
        let fused = crate::optimizer::fusion::fuse_matmul_split(packed, &w2, &outs);
        let ops: Vec<&str> = fused.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, ["OxiMatMulSplit", "Relu"], "bias={bias}");
        let got = run(&fused, &w2, &feeds);
        for o in &outs {
            assert_eq!(got[o].shape, want[o].shape, "{o}");
            let diff = got[o].data.iter().zip(&want[o].data).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(diff < 1e-5, "bias={bias} {o}: {diff}");
        }
    }
}

#[test]
fn leaves_shared_weights_and_mismatched_k_alone() {
    let (mut nodes, mut w, outs) = qwen_block(false);
    // wk is also read elsewhere: only q and v pack.
    nodes.push(node(OpKind::Identity, "peek", &["wk"], "wk_copy"));
    let outs2: Vec<String> = outs.iter().cloned().chain(["wk_copy".to_string()]).collect();
    let packed = pack_shared_projections(nodes.clone(), &mut w, &outs2);
    let split = packed.iter().find(|n| n.op == OpKind::Split).expect("q and v packed");
    assert_eq!(split.outputs, ["q_mm", "v_mm"]);
    assert!(w.contains_key("wk"));
    // A lone projection is left as is.
    let mut w = HashMap::new();
    w.insert("wa".to_string(), t(rand(K * 32, 1), &[K, 32]));
    w.insert("wb".to_string(), t(rand(16 * 32, 2), &[16, 32]));
    let lone = vec![node(OpKind::MatMul, "a", &["x", "wa"], "a"), node(OpKind::MatMul, "b", &["y", "wb"], "b")];
    assert_eq!(pack_shared_projections(lone.clone(), &mut w, &["a".into(), "b".into()]).len(), lone.len());
}
