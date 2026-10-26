//! Dynamically quantized MatMuls → `OxiMatMulNBits`.
//!
//! ORT's `quantize_dynamic` (the `model_quantized.onnx` / `model_int8.onnx` /
//! `model_quint8_avx2.onnx` exports) turns each `MatMul` by a constant weight
//! into
//!
//! ```text
//! (xq, xs, xzp) = DynamicQuantizeLinear(x)
//! acc = MatMulInteger(xq, Wq, xzp, wzp)          Wq: [K, N] int8/uint8, wzp: [] or [N]
//! y   = Mul(Cast<float>(acc), Mul(xs, ws))       ws: [] or [N]
//! ```
//!
//! which the generic `MatMulInteger` kernel runs as a scalar integer loop. This
//! pass rewrites the chain into `OxiMatMulNBits(x, packed)` with
//! `W = (Wq - wzp) * ws` packed as an 8-bit block-quantized weight (block 32,
//! one scale and zero point per column), so it runs on the int8 kernels of
//! [`oxionnx_ops::matmul_nbits`] (VNNI where available).
//!
//! The weight is the same; the activations are quantized differently: the
//! kernels quantize each row per K block of 32 (symmetric int8) instead of
//! the whole tensor at once (asymmetric uint8). That is finer, so the result
//! is at least as close to the FP32 model, and no longer depends on the other
//! rows of the batch, but it is not bit-identical to ORT's.
//!
//! Some exports also quantize activation x activation products (attention
//! scores, even the rotary `position x inv_freq`): `MatMulInteger(aq, bq, azp,
//! bzp)` with both sides from `DynamicQuantizeLinear` and the scale
//! `Mul(as, bs)`. Those become the float `MatMul(a, b)` they approximate.
//!
//! A chain is left alone when any intermediate has another consumer or is a
//! graph output, or when the weight, scale or zero point is not a constant of
//! the expected shape.

use crate::graph::{Node, OpKind};
use crate::tensor::Tensor;
use oxionnx_ops::matmul_nbits::{self as nb, Shape, Source, ZeroPoints};
use std::collections::{HashMap, HashSet};

/// K block of the packed weight.
const BLOCK: usize = 32;

/// One matched chain: the node indices of the `MatMulInteger` (replaced by
/// `node`), the `Cast` and the scale `Mul` (dropped), the packed weight (name,
/// data; none for an activation x activation product) and the original weight
/// to free.
struct Rewrite {
    matmul: usize,
    cast: usize,
    mul: usize,
    node: Node,
    packed: Option<(String, Vec<f32>)>,
    free: Option<String>,
}

/// See the module docs.
pub fn rewrite_dynamic_quant_matmul(
    nodes: Vec<Node>,
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
) -> Vec<Node> {
    if !nodes.iter().any(|n| n.op.as_str() == "MatMulInteger") {
        return nodes;
    }
    let mut uses: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for x in n.inputs.iter().filter(|x| !x.is_empty()) {
            uses.entry(x.as_str()).or_default().push(i);
        }
        for o in &n.outputs {
            producer.insert(o.as_str(), i);
        }
    }
    let pinned: HashSet<&str> = output_names.iter().map(String::as_str).collect();
    // The single consumer of `name`, if it has exactly one and is not a graph output.
    let only_use = |name: &str| match uses.get(name).map(Vec::as_slice) {
        Some([i]) if !pinned.contains(name) => Some(*i),
        _ => None,
    };
    let mut rewrites: Vec<Rewrite> = Vec::new();
    for (mi, m) in nodes.iter().enumerate() {
        if m.op.as_str() != "MatMulInteger" || m.inputs.len() < 3 {
            continue;
        }
        let Some(rewrite) = (|| {
            let dq = &nodes[*producer.get(m.inputs[0].as_str())?];
            if dq.op.as_str() != "DynamicQuantizeLinear"
                || dq.outputs.len() < 3
                || m.inputs[2] != dq.outputs[2]
            {
                return None;
            }
            let ci = only_use(&m.outputs[0])?;
            let cast = &nodes[ci];
            if cast.op != OpKind::Cast || cast.attrs.i("to", 0) != 1 {
                return None;
            }
            let ui = only_use(&cast.outputs[0])?;
            let mul = &nodes[ui];
            if mul.op != OpKind::Mul || mul.inputs.len() != 2 {
                return None;
            }
            let scale = if mul.inputs[0] == cast.outputs[0] {
                &mul.inputs[1]
            } else {
                &mul.inputs[0]
            };
            let sm = &nodes[*producer.get(scale.as_str())?];
            if sm.op != OpKind::Mul || sm.inputs.len() != 2 {
                return None;
            }
            let other = match (sm.inputs[0] == dq.outputs[1], sm.inputs[1] == dq.outputs[1]) {
                (true, false) => &sm.inputs[1],
                (false, true) => &sm.inputs[0],
                _ => return None,
            };
            let mut node = m.clone();
            node.outputs = mul.outputs.clone();
            node.attrs = Default::default();
            let Some(w) = weights.get(&m.inputs[1]) else {
                // Activation x activation (attention, rotary): both sides
                // dynamically quantized. The float product of the originals.
                let dqb = &nodes[*producer.get(m.inputs[1].as_str())?];
                if dqb.op.as_str() != "DynamicQuantizeLinear"
                    || dqb.outputs.len() < 3
                    || m.inputs.get(3) != Some(&dqb.outputs[2])
                    || *other != dqb.outputs[1]
                {
                    return None;
                }
                node.op = OpKind::MatMul;
                node.inputs = vec![dq.inputs[0].clone(), dqb.inputs[0].clone()];
                return Some(Rewrite {
                    matmul: mi,
                    cast: ci,
                    mul: ui,
                    node,
                    packed: None,
                    free: None,
                });
            };
            if w.shape.len() != 2 {
                return None;
            }
            let (k, n) = (w.shape[0], w.shape[1]);
            let wzp = match m.inputs.get(3).filter(|z| !z.is_empty()) {
                Some(z) => Some(
                    weights
                        .get(z)
                        .filter(|t| t.data.len() == 1 || t.data.len() == n)?,
                ),
                None => None,
            };
            let ws = weights
                .get(other)
                .filter(|t| t.data.len() == 1 || t.data.len() == n)?;
            let packed = pack(k, n, &w.data, wzp.map(|t| &t.data[..]), &ws.data)?;
            node.op = OpKind::Unknown("OxiMatMulNBits".into());
            let name = format!("{}/oxi_nbits8", m.inputs[1]);
            node.inputs = vec![dq.inputs[0].clone(), name.clone()];
            // The int8 original (held as f32 lanes) goes when nothing else reads it.
            let free = only_use(&m.inputs[1]).map(|_| m.inputs[1].clone());
            Some(Rewrite {
                matmul: mi,
                cast: ci,
                mul: ui,
                node,
                packed: Some((name, packed)),
                free,
            })
        })() else {
            continue;
        };
        rewrites.push(rewrite);
    }
    if rewrites.is_empty() {
        return nodes;
    }
    tracing::debug!(
        count = rewrites.len(),
        "rewrote DynamicQuantizeLinear + MatMulInteger chains into OxiMatMulNBits"
    );
    let mut drop = HashSet::new();
    let mut replace = HashMap::new();
    for Rewrite {
        matmul,
        cast,
        mul,
        node,
        packed,
        free,
    } in rewrites
    {
        if let Some(w) = free {
            weights.remove(&w);
        }
        if let Some((name, data)) = packed {
            let len = data.len();
            weights.insert(
                name,
                Tensor {
                    data,
                    shape: vec![len],
                },
            );
        }
        drop.extend([cast, mul]);
        replace.insert(matmul, node);
    }
    // The scales, zero points, scale products and DynamicQuantizeLinear
    // nodes are left to dead-node elimination.
    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(i, n)| replace.remove(&i).unwrap_or(n))
        .collect()
}

/// `W[k, j] = (q[k, j] - zp[j]) * scale[j]` (`q` int8 or uint8 values, `zp` /
/// `scale` per column or scalar) packed as an 8-bit block-quantized weight.
fn pack(k: usize, n: usize, q: &[f32], zp: Option<&[f32]>, scale: &[f32]) -> Option<Vec<f32>> {
    let col = |v: &[f32], j: usize| if v.len() == 1 { v[0] } else { v[j] };
    let zp_at = |j: usize| zp.map_or(0.0, |z| col(z, j));
    // int8 values (any negative) are shifted into the unsigned range.
    let shift = if q.iter().chain(zp.into_iter().flatten()).any(|&v| v < 0.0) {
        128.0
    } else {
        0.0
    };
    if q.iter()
        .chain(zp.into_iter().flatten())
        .any(|&v| v.fract() != 0.0 || !(0.0..=255.0).contains(&(v + shift)))
    {
        return None;
    }
    let s = Shape::new(n, k, 8, BLOCK).ok()?;
    let kb = s.kb();
    let mut b = vec![0u8; n * kb * BLOCK];
    for p in 0..k {
        for j in 0..n {
            b[j * kb * BLOCK + p] = (q[p * n + j] + shift) as u8;
        }
    }
    // Pad the last block with the zero point, so padding contributes nothing.
    for j in 0..n {
        let z = (zp_at(j) + shift) as u8;
        b[j * kb * BLOCK + k..(j + 1) * kb * BLOCK].fill(z);
    }
    let scales: Vec<f32> = (0..n)
        .flat_map(|j| std::iter::repeat(col(scale, j)).take(kb))
        .collect();
    let zps: Vec<f32> = (0..n)
        .flat_map(|j| std::iter::repeat(zp_at(j) + shift).take(kb))
        .collect();
    nb::pack(
        s,
        &Source {
            b: &b,
            scales: &scales,
            zp: ZeroPoints::Float(&zps),
        },
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attributes, Graph};

    fn node(op: OpKind, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op,
            name: outputs[0].to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: Attributes::default(),
        }
    }

    /// The ORT `quantize_dynamic` chain for `y = x @ W` (K = 40, N = 20), with
    /// int8 or uint8 weights, per-column or scalar scale / zero point.
    fn chain(int8: bool, per_col: bool) -> (Vec<Node>, HashMap<String, Tensor>, Vec<f32>) {
        let (k, n) = (40usize, 20usize);
        let lo = if int8 { -128.0 } else { 0.0 };
        let q: Vec<f32> = (0..k * n)
            .map(|i| lo + ((i * 73 + 11) % 256) as f32)
            .collect();
        let cols = if per_col { n } else { 1 };
        let zp: Vec<f32> = (0..cols).map(|j| lo + 100.0 + (j % 9) as f32).collect();
        let ws: Vec<f32> = (0..cols).map(|j| 0.002 + (j % 5) as f32 * 0.0007).collect();
        let col = |v: &[f32], j: usize| if v.len() == 1 { v[0] } else { v[j] };
        let w: Vec<f32> = (0..k * n)
            .map(|i| (q[i] - col(&zp, i % n)) * col(&ws, i % n))
            .collect();
        let mut cast = node(OpKind::Cast, &["acc"], &["accf"]);
        cast.attrs.ints.insert("to".into(), 1);
        let nodes = vec![
            node(
                OpKind::Unknown("DynamicQuantizeLinear".into()),
                &["x"],
                &["xq", "xs", "xzp"],
            ),
            node(
                OpKind::Unknown("MatMulInteger".into()),
                &["xq", "W", "xzp", "wzp"],
                &["acc"],
            ),
            node(OpKind::Mul, &["xs", "ws"], &["s"]),
            cast,
            node(OpKind::Mul, &["accf", "s"], &["y"]),
        ];
        let shape = if per_col { vec![n] } else { vec![] };
        let weights = HashMap::from([
            ("W".to_string(), Tensor::new(q, vec![k, n])),
            ("wzp".to_string(), Tensor::new(zp, shape.clone())),
            ("ws".to_string(), Tensor::new(ws, shape)),
        ]);
        (nodes, weights, w)
    }

    #[test]
    fn rewrites_the_chain() {
        let (nodes, mut weights, _) = chain(true, true);
        let out = rewrite_dynamic_quant_matmul(nodes, &mut weights, &["y".to_string()]);
        let ops: Vec<&str> = out.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, vec!["DynamicQuantizeLinear", "OxiMatMulNBits", "Mul"]);
        assert_eq!(out[1].inputs, vec!["x", "W/oxi_nbits8"]);
        assert_eq!(out[1].outputs, vec!["y"]);
        assert!(nb::Packed::is_packed(&weights["W/oxi_nbits8"].data));
        assert!(!weights.contains_key("W"));
        // The accumulator also read elsewhere: left alone.
        let (mut nodes, mut weights, _) = chain(true, true);
        nodes.push(node(OpKind::Relu, &["acc"], &["r"]));
        let out =
            rewrite_dynamic_quant_matmul(nodes, &mut weights, &["y".to_string(), "r".to_string()]);
        assert!(out.iter().any(|n| n.op.as_str() == "MatMulInteger"));
    }

    #[test]
    fn activation_products_become_float_matmuls() {
        let mut cast = node(OpKind::Cast, &["acc"], &["accf"]);
        cast.attrs.ints.insert("to".into(), 1);
        let dql = |x: &str| {
            node(
                OpKind::Unknown("DynamicQuantizeLinear".into()),
                &[x],
                &[&format!("{x}q"), &format!("{x}s"), &format!("{x}z")],
            )
        };
        let nodes = vec![
            dql("a"),
            dql("b"),
            node(
                OpKind::Unknown("MatMulInteger".into()),
                &["aq", "bq", "az", "bz"],
                &["acc"],
            ),
            node(OpKind::Mul, &["bs", "as"], &["s"]),
            cast,
            node(OpKind::Mul, &["s", "accf"], &["y"]),
        ];
        let graph = Graph {
            nodes,
            input_names: vec!["a".into(), "b".into()],
            output_names: vec!["y".into()],
            ..Default::default()
        };
        let a = Tensor::new((0..12).map(|i| i as f32 * 0.3 - 1.0).collect(), vec![3, 4]);
        let b = Tensor::new((0..8).map(|i| 0.5 - i as f32 * 0.2).collect(), vec![4, 2]);
        let want = crate::Session::from_graph(
            Graph {
                nodes: vec![node(OpKind::MatMul, &["a", "b"], &["y"])],
                input_names: vec!["a".into(), "b".into()],
                output_names: vec!["y".into()],
                ..Default::default()
            },
            HashMap::new(),
        )
        .unwrap()
        .run(&HashMap::from([("a", a.clone()), ("b", b.clone())]))
        .unwrap()["y"]
            .data
            .clone();
        let mut weights = HashMap::new();
        let out =
            rewrite_dynamic_quant_matmul(graph.nodes.clone(), &mut weights, &["y".to_string()]);
        assert!(out
            .iter()
            .any(|n| n.op == OpKind::MatMul && n.inputs == ["a", "b"] && n.outputs == ["y"]));
        let s = crate::Session::from_graph(graph, HashMap::new()).unwrap();
        assert_eq!(
            s.run(&HashMap::from([("a", a), ("b", b)])).unwrap()["y"].data,
            want
        );
    }

    #[test]
    fn session_matches_the_dequantized_matmul() {
        for (int8, per_col) in [(true, true), (false, true), (true, false), (false, false)] {
            let (nodes, weights, w) = chain(int8, per_col);
            let graph = Graph {
                nodes,
                input_names: vec!["x".into()],
                output_names: vec!["y".into()],
                ..Default::default()
            };
            let (m, k, n) = (5usize, 40usize, 20usize);
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 37) % 19) as f32 / 19.0 - 0.5)
                .collect();
            let session = crate::Session::from_graph(graph, weights).unwrap();
            assert!(
                session.weights().contains_key("W/oxi_nbits8"),
                "int8={int8} per_col={per_col}: not rewritten"
            );
            let y = &session
                .run(&HashMap::from([("x", Tensor::new(x.clone(), vec![m, k]))]))
                .unwrap()["y"];
            assert_eq!(y.shape, vec![m, n]);
            let scale: f32 = w.iter().map(|v| v.abs()).fold(0.0, f32::max) * k as f32;
            for i in 0..m {
                for j in 0..n {
                    let want: f32 = (0..k).map(|p| x[i * k + p] * w[p * n + j]).sum();
                    let got = y.data[i * n + j];
                    assert!(
                        (got - want).abs() < 0.01 * scale,
                        "int8={int8} per_col={per_col} [{i},{j}] {got} vs {want}"
                    );
                }
            }
        }
    }
}
