use super::super::types::OptLevel;
use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use oxionnx_core::{DType, TensorStorage, TypedTensor};
use std::collections::HashMap;

const K: usize = 96;
const N: usize = 70;

fn weight() -> Vec<f32> {
    (0..K * N).map(|i| ((i * 53) % 89) as f32 / 89.0 - 0.5).collect()
}

fn session(bf16: Option<bool>) -> Session {
    session_with(bf16, false)
}

fn session_with(bf16: Option<bool>, keep: bool) -> Session {
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::MatMul,
            name: "mm".into(),
            inputs: vec!["x".into(), "w".into()],
            outputs: vec!["y".into()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".into()],
        output_names: vec!["y".into()],
        ..Default::default()
    };
    let weights = HashMap::from([("w".to_string(), Tensor::new(weight(), vec![K, N]))]);
    let builder = Session::builder().with_optimization_level(OptLevel::None).with_keep_weights(keep);
    let builder = match bf16 {
        Some(on) => builder.with_bf16_weights(on),
        None => builder,
    };
    builder.build_from_graph(graph, weights).expect("build")
}

fn input(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5).collect(), shape.to_vec())
}

fn run(s: &Session, shape: &[usize]) -> Vec<f32> {
    let mut out = s.run(&HashMap::from([("x", input(shape))])).expect("run");
    out.remove("y").expect("output y").data
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 2e-2 * (1.0 + y.abs()), "{x} vs {y}");
    }
}

#[test]
fn bf16_weights_option() {
    let fp32 = session(Some(false));
    let want = run(&fp32, &[2, 5, K]);
    if std::env::var("OXIONNX_BF16_WEIGHTS").is_err() {
        assert_eq!(run(&session(None), &[2, 5, K]), want, "the default must stay FP32");
    }
    let bf16 = session(Some(true));
    let got = run(&bf16, &[2, 5, K]);
    let hw = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16");
    if !hw {
        assert_eq!(got, want, "without AVX-512 BF16 the weights stay FP32");
        return;
    }
    assert_ne!(got, want, "BF16 weights should change the low bits");
    close(&got, &want);
    // Packed at load time and the FP32 copy freed.
    assert!(bf16.weights()["w"].data.is_empty());
    assert_eq!(bf16.weights()["w"].shape, vec![K, N]);
    // Typed runs see the released weight too.
    let x = input(&[3, K]);
    let typed = |s: &Session| {
        let tx = TypedTensor::new(TensorStorage::F32(x.data.clone()), x.shape.clone());
        let mut out = s.run_typed(&HashMap::from([("x", tx)])).expect("run_typed");
        let y = out.remove("y").expect("y");
        assert_eq!(y.dtype(), DType::F32);
        y.storage.to_f32_vec()
    };
    close(&typed(&bf16), &typed(&fp32));
    // The session cache cannot hold freed weights.
    assert!(bf16.to_optimized_bytes().is_err());
    assert!(fp32.to_optimized_bytes().is_ok());
}

/// The packed GEMM is in use, so constant MatMul weights are packed at load.
fn packs_at_load() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[test]
fn fp32_weights_are_packed_at_load_and_rebuilt_exactly_for_the_cache() {
    let s = session(Some(false));
    let want = run(&s, &[2, 5, K]);
    let kept = session_with(Some(false), true);
    assert_eq!(run(&kept, &[2, 5, K]), want, "keeping the weights changes nothing else");
    assert_eq!(kept.weights()["w"].data, weight());
    if !packs_at_load() {
        assert_eq!(s.weights()["w"].data, weight());
        return;
    }
    // The FP32 copy is freed; the shape stays.
    assert!(s.weights()["w"].data.is_empty());
    assert_eq!(s.weights()["w"].shape, vec![K, N]);
    // The session cache holds the exact original weight, and reloads to the same results.
    let bytes = s.to_optimized_bytes().expect("save");
    assert_eq!(bytes, kept.to_optimized_bytes().expect("save kept"), "cache bytes match a session that kept its weights");
    let reloaded = Session::builder().with_keep_weights(true).load_optimized_from_bytes(&bytes).expect("reload");
    assert_eq!(reloaded.weights()["w"].data, weight());
    assert_eq!(run(&reloaded, &[2, 5, K]), want);
}
