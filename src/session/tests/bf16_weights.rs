use super::super::types::OptLevel;
use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

fn run(bf16: Option<bool>) -> Vec<f32> {
    let (k, n) = (96, 70);
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
    let w: Vec<f32> = (0..k * n).map(|i| ((i * 53) % 89) as f32 / 89.0 - 0.5).collect();
    let weights = HashMap::from([("w".to_string(), Tensor::new(w, vec![k, n]))]);
    let builder = Session::builder().with_optimization_level(OptLevel::None);
    let builder = match bf16 {
        Some(on) => builder.with_bf16_weights(on),
        None => builder,
    };
    let session = builder.build_from_graph(graph, weights).expect("build");
    let x: Vec<f32> = (0..2 * 5 * k).map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5).collect();
    let inputs = HashMap::from([("x", Tensor::new(x, vec![2, 5, k]))]);
    let mut out = session.run(&inputs).expect("run");
    let y = out.remove("y").expect("output y");
    assert_eq!(y.shape, vec![2, 5, n]);
    y.data
}

#[test]
fn bf16_weights_option() {
    let fp32 = run(Some(false));
    if std::env::var("OXIONNX_BF16_WEIGHTS").is_err() {
        assert_eq!(run(None), fp32, "the default must stay FP32");
    }
    let bf16 = run(Some(true));
    let hw = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16");
    let gemm = is_x86_feature_detected!("avx512f") || (is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
    if hw && gemm {
        assert_ne!(bf16, fp32, "BF16 weights should change the low bits");
        for (a, b) in bf16.iter().zip(&fp32) {
            assert!((a - b).abs() <= 2e-2 * (1.0 + b.abs()), "{a} vs {b}");
        }
    } else {
        assert_eq!(bf16, fp32, "without AVX-512 BF16 the weights stay FP32");
    }
}
