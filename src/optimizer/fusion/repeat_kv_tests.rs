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
            if !name.is_empty() {
                env.insert(name.clone(), t);
            }
        }
    }
    env
}

fn ints(w: &mut HashMap<String, Tensor>, name: &str, v: &[i64]) -> String {
    w.insert(name.into(), Tensor::new(v.iter().map(|&x| x as f32).collect(), vec![v.len()]));
    name.into()
}

fn node(op: OpKind, name: &str, inputs: &[&str], out: &str) -> Node {
    make_node(op, name, inputs.to_vec(), vec![out])
}

fn transpose(name: &str, input: &str, out: &str) -> Node {
    let mut n = node(OpKind::Transpose, name, &[input], out);
    n.attrs.int_lists.insert("perm".into(), vec![0, 2, 1, 3]);
    n
}

/// How the test graph spells the repeat_kv shapes.
#[derive(Clone, Copy, PartialEq)]
enum Shapes {
    /// `Shape -> Gather -> Unsqueeze -> Concat`, wrapped in `Reshape(.., [-1])` and
    /// `Where(Equal(s, -1), 1, s)` like the onnx-community exports.
    Dynamic,
    /// Folded constants.
    Constant,
    /// The merge reshape folds `G` into the sequence axis instead (must not fuse).
    WrongMerge,
}

/// onnx-community repeat_kv for `name` = "k" | "v": `x [B, Hkv, T, H]` -> `[B, T, Hq*H]`.
fn repeat_kv(nodes: &mut Vec<Node>, w: &mut HashMap<String, Tensor>, x: &str, name: &str, dims: [i64; 4], g: i64, shapes: Shapes) -> String {
    let p = |s: &str| format!("{name}_{s}");
    let [b, hkv, t, h] = dims;
    ints(w, "ax2", &[2]);
    ints(w, "ax0", &[0]);
    ints(w, "flat", &[-1]);
    ints(w, "neg1", &[-1]);
    ints(w, "one", &[1]);
    ints(w, "g", &[g]);
    ints(w, "out3", &[0, 0, hkv * g * h]);
    nodes.push(node(OpKind::Unsqueeze, &p("unsq"), &[x, "ax2"], &p("x5")));
    let (s5, s4) = match shapes {
        Shapes::Constant => (ints(w, &p("s5"), &[b, hkv, g, t, h]), ints(w, &p("s4"), &[b, hkv * g, t, h])),
        Shapes::Dynamic | Shapes::WrongMerge => {
            nodes.push(node(OpKind::Shape, &p("shape"), &[x], &p("shp")));
            let mut dim = Vec::new();
            for i in 0..4 {
                let idx = format!("idx{i}");
                w.insert(idx.clone(), Tensor::new(vec![i as f32], vec![]));
                nodes.push(node(OpKind::Gather, &p(&format!("ga{i}")), &[&p("shp"), &idx], &p(&format!("d{i}"))));
                nodes.push(node(OpKind::Unsqueeze, &p(&format!("un{i}")), &[&p(&format!("d{i}")), "ax0"], &p(&format!("u{i}"))));
                dim.push(p(&format!("u{i}")));
            }
            let mut cat = node(OpKind::Concat, &p("cat5"), &[&dim[0], &dim[1], "g", &dim[2], &dim[3]], &p("c5"));
            cat.attrs.ints.insert("axis".into(), 0);
            nodes.push(cat);
            nodes.push(node(OpKind::Reshape, &p("rs5"), &[&p("c5"), "flat"], &p("r5")));
            nodes.push(node(OpKind::Equal, &p("eq"), &[&p("r5"), "neg1"], &p("eq5")));
            nodes.push(node(OpKind::Where, &p("where"), &[&p("eq5"), "one", &p("r5")], &p("s5")));
            let mut cat4 = if shapes == Shapes::WrongMerge {
                nodes.push(node(OpKind::Mul, &p("mul"), &["g", &dim[2]], &p("gt")));
                node(OpKind::Concat, &p("cat4"), &[&dim[0], &dim[1], &p("gt"), &dim[3]], &p("s4"))
            } else {
                nodes.push(node(OpKind::Mul, &p("mul"), &[&dim[1], "g"], &p("hg")));
                node(OpKind::Concat, &p("cat4"), &[&dim[0], &p("hg"), &dim[2], &dim[3]], &p("s4"))
            };
            cat4.attrs.ints.insert("axis".into(), 0);
            nodes.push(cat4);
            (p("s5"), p("s4"))
        }
    };
    nodes.push(node(OpKind::Expand, &p("expand"), &[&p("x5"), &s5], &p("e")));
    nodes.push(node(OpKind::Reshape, &p("rs4"), &[&p("e"), &s4], &p("r4")));
    nodes.push(transpose(&p("tr2"), &p("r4"), &p("t2")));
    nodes.push(node(OpKind::Reshape, &p("rs3"), &[&p("t2"), "out3"], &p("kr")));
    p("kr")
}

struct Case {
    nodes: Vec<Node>,
    weights: HashMap<String, Tensor>,
    feeds: Vec<(&'static str, Tensor)>,
    outputs: Vec<String>,
}

/// Qwen3-style decoder attention: K/V projections split into `Hkv` heads,
/// optionally concatenated with a past cache (then the 4-D cache is a graph
/// output), repeated to `Hq = Hkv * G` heads, plus a Tile/Expand-broadcast mask.
fn decoder(past: usize, g: i64, shapes: Shapes, mask_tile: bool) -> Case {
    let (b, s, hkv, h) = (2usize, 5usize, 2i64, 4i64);
    let hq = hkv * g;
    let t = past + s;
    let mut w = HashMap::new();
    let mut nodes = Vec::new();
    let mut feeds = vec![
        ("q", Tensor::new(rand(b * s * (hq * h) as usize, 1), vec![b, s, (hq * h) as usize])),
        ("k", Tensor::new(rand(b * s * (hkv * h) as usize, 2), vec![b, s, (hkv * h) as usize])),
        ("v", Tensor::new(rand(b * s * (hkv * h) as usize, 3), vec![b, s, (hkv * h) as usize])),
    ];
    let split = ints(&mut w, "split", &[0, 0, hkv, -1]);
    let mut outputs = vec!["out".to_string()];
    let mut kv = Vec::new();
    for (name, src) in [("k", "k"), ("v", "v")] {
        nodes.push(node(OpKind::Reshape, &format!("{name}_split"), &[src, &split], &format!("{name}4")));
        nodes.push(transpose(&format!("{name}_tr1"), &format!("{name}4"), &format!("{name}t")));
        let x = if past > 0 {
            let pname: &'static str = if name == "k" { "past_k" } else { "past_v" };
            feeds.push((pname, Tensor::new(rand(b * hkv as usize * past * h as usize, 4 + kv.len() as u64), vec![b, hkv as usize, past, h as usize])));
            let mut cat = node(OpKind::Concat, &format!("{name}_cat"), &[pname, &format!("{name}t")], &format!("present_{name}"));
            cat.attrs.ints.insert("axis".into(), 2);
            nodes.push(cat);
            outputs.push(format!("present_{name}"));
            format!("present_{name}")
        } else {
            format!("{name}t")
        };
        kv.push(repeat_kv(&mut nodes, &mut w, &x, name, [b as i64, hkv, t as i64, h], g, shapes));
    }
    // Additive key mask [B, 1, 1, T] broadcast to [B, Hq, S, T].
    let lens = [t, t - 2];
    feeds.push(("mask", Tensor::new((0..b * t).map(|i| if i % t < lens[i / t] { 0.0 } else { -1e9 }).collect(), vec![b, 1, 1, t])));
    let bias = if mask_tile {
        let shape = ints(&mut w, "mask_bst", &[b as i64, 1, s as i64, t as i64]);
        nodes.push(node(OpKind::Expand, "mask_expand_s", &["mask", &shape], "mask_s"));
        let reps = ints(&mut w, "mask_reps", &[1, hq, 1, 1]);
        nodes.push(node(OpKind::Tile, "mask_tile", &["mask_s", &reps], "bias"));
        "bias"
    } else {
        let shape = ints(&mut w, "mask_full", &[b as i64, hq, s as i64, t as i64]);
        nodes.push(node(OpKind::Expand, "mask_expand", &["mask", &shape], "bias"));
        "bias"
    };
    let mut mha = make_node(OpKind::Unknown(MHA.into()), "mha", vec!["q", &kv[0], &kv[1], "", "", bias, "", ""], vec!["out", "", ""]);
    mha.attrs.ints.insert("num_heads".into(), hq);
    mha.attrs.floats.insert("scale".into(), 0.4);
    nodes.push(mha);
    Case { nodes, weights: w, feeds, outputs }
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape, b.shape);
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn fused(c: &Case) -> Vec<Node> {
    fuse_repeat_kv(c.nodes.clone(), &c.weights, &c.outputs)
}

fn mha_of(nodes: &[Node]) -> &Node {
    nodes.iter().find(|n| n.op == OpKind::Unknown(MHA.into())).expect("MHA node")
}

fn check_same(c: &Case, nodes: &[Node]) {
    let want = run(&c.nodes, &c.weights, &c.feeds);
    let got = run(nodes, &c.weights, &c.feeds);
    for o in &c.outputs {
        assert!(max_diff(&want[o], &got[o]) < 1e-5, "{o} differs");
    }
}

#[test]
fn fuses_3d_projections_without_past() {
    for shapes in [Shapes::Dynamic, Shapes::Constant] {
        for mask_tile in [false, true] {
            let c = decoder(0, 2, shapes, mask_tile);
            let nodes = fused(&c);
            let m = mha_of(&nodes);
            assert_eq!(m.attrs.i("kv_num_heads", 0), 2);
            // Straight from the projections, and the mask before its broadcast.
            assert_eq!(&m.inputs[1..3], ["k", "v"]);
            assert_eq!(m.inputs[5], if mask_tile { "mask_s" } else { "mask" });
            assert!(!nodes.iter().any(|n| n.op == OpKind::Expand && n.name != "mask_expand_s"), "repeat_kv copies left");
            assert!(!nodes.iter().any(|n| n.op == OpKind::Tile));
            check_same(&c, &nodes);
        }
    }
}

#[test]
fn fuses_4d_cache_with_past() {
    for g in [2, 3] {
        let c = decoder(3, g, Shapes::Dynamic, true);
        let nodes = fused(&c);
        let m = mha_of(&nodes);
        assert_eq!(m.attrs.i("kv_num_heads", 0), 2);
        assert_eq!(&m.inputs[1..3], ["present_k", "present_v"]);
        assert!(!nodes.iter().any(|n| n.op == OpKind::Expand && n.name != "mask_expand_s"));
        check_same(&c, &nodes);
    }
}

#[test]
fn leaves_other_reshapes_and_attention_inputs_alone() {
    // Merging G into the sequence axis is not repeat_kv.
    let c = decoder(0, 2, Shapes::WrongMerge, false);
    let nodes = fused(&c);
    let m = mha_of(&nodes);
    assert_eq!(m.attrs.i("kv_num_heads", 0), 0);
    assert_eq!(m.inputs[1], "k_kr");
    // A K/V bias or a consumed present output blocks the K/V rewrite (the mask still goes).
    for edit in 0..2 {
        let mut c = decoder(0, 2, Shapes::Dynamic, false);
        let m = c.nodes.iter_mut().find(|n| n.op == OpKind::Unknown(MHA.into())).unwrap();
        if edit == 0 {
            m.inputs[3] = "qkv_bias".into();
        } else {
            m.outputs[1] = "present".into();
        }
        let nodes = fused(&c);
        let m = mha_of(&nodes);
        assert_eq!(m.attrs.i("kv_num_heads", 0), 0);
        assert_eq!(m.inputs[1], "k_kr");
        assert_eq!(m.inputs[5], "mask");
    }
    // A head-axis Tile by anything but num_heads stays.
    let mut c = decoder(0, 2, Shapes::Dynamic, true);
    c.weights.insert("mask_reps".into(), Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![4]));
    assert_eq!(mha_of(&fused(&c)).inputs[5], "bias");
}

#[test]
fn reads_constants_from_constant_nodes() {
    // The onnx-community exports keep shape constants as `Constant` nodes.
    for past in [0, 3] {
        let mut c = decoder(past, 2, Shapes::Dynamic, true);
        let mut consts: Vec<Node> = c
            .weights
            .drain()
            .map(|(name, t)| {
                let mut n = make_node(OpKind::Constant, &format!("const_{name}"), vec![], vec![&name]);
                n.attrs.tensors.insert("value".into(), t);
                n
            })
            .collect();
        consts.append(&mut c.nodes);
        c.nodes = consts;
        let nodes = fused(&c);
        let m = mha_of(&nodes);
        assert_eq!(m.attrs.i("kv_num_heads", 0), 2);
        assert_eq!(m.inputs[5], "mask_s");
        check_same(&c, &nodes);
    }
}

#[test]
fn dead_copy_removal_does_not_need_sorted_nodes() {
    let c = decoder(3, 2, Shapes::Dynamic, true);
    let sorted: HashSet<String> = fused(&c).into_iter().map(|n| n.name).collect();
    let mut rev = c.nodes.clone();
    rev.reverse();
    let unsorted: HashSet<String> = fuse_repeat_kv(rev, &c.weights, &c.outputs).into_iter().map(|n| n.name).collect();
    assert_eq!(sorted, unsorted);
    assert!(sorted.contains("k_cat") && sorted.contains("k_split") && !sorted.contains("k_expand"));
}
