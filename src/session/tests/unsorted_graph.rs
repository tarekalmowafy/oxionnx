//! Graphs whose nodes are not in topological order (some quantizers append the
//! nodes they rewrite at the end) are sorted before the optimizer runs.

use super::super::Session;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use std::collections::HashMap;

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.into(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs: Attributes::default(),
    }
}

#[test]
fn unsorted_graph_runs_like_the_sorted_one() {
    let (k, n) = (8usize, 6usize);
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 0.5).collect();
    // y = relu(x @ w + b) * x2, with the producers listed after their consumers.
    let sorted = vec![
        node(OpKind::MatMul, "mm", &["x", "w"], &["mm_out"]),
        node(OpKind::Add, "bias", &["mm_out", "b"], &["pre"]),
        node(OpKind::Relu, "act", &["pre"], &["act_out"]),
        node(OpKind::Mul, "scale", &["act_out", "act_out"], &["y"]),
    ];
    let build = |nodes: Vec<Node>| {
        let graph = Graph {
            nodes,
            input_names: vec!["x".into()],
            output_names: vec!["y".into()],
            ..Default::default()
        };
        let weights = HashMap::from([
            ("w".to_string(), Tensor::new(w.clone(), vec![k, n])),
            ("b".to_string(), Tensor::new(b.clone(), vec![n])),
        ]);
        Session::builder()
            .build_from_graph(graph, weights)
            .expect("build")
    };
    let x = Tensor::new(
        (0..3 * k)
            .map(|i| ((i * 5) % 13) as f32 / 13.0 - 0.4)
            .collect(),
        vec![3, k],
    );
    let run = |s: &Session| {
        s.run(&HashMap::from([("x", x.clone())]))
            .expect("run")
            .remove("y")
            .expect("y")
            .data
    };
    let want = run(&build(sorted.clone()));
    let unsorted = vec![
        sorted[3].clone(),
        sorted[1].clone(),
        sorted[2].clone(),
        sorted[0].clone(),
    ];
    assert_eq!(run(&build(unsorted)), want);
}
