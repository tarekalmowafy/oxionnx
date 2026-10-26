//! Dead node elimination and stable topological ordering.

use crate::graph::Node;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Dead node elimination: remove nodes whose outputs are never consumed.
///
/// Walks backward from `output_names` through each tensor's producer, so the
/// result does not depend on the order of `nodes` (some exports, e.g. the
/// onnx-community Qwen3 graphs, list a few consumers before their producers).
/// The surviving nodes keep their relative order.
pub fn dead_node_elimination(nodes: Vec<Node>, output_names: &[String]) -> Vec<Node> {
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for o in node.outputs.iter().filter(|o| !o.is_empty()) {
            producer.entry(o.as_str()).or_insert(i);
        }
    }
    let mut live_nodes = vec![false; nodes.len()];
    let mut seen: HashSet<&str> = HashSet::new();
    let mut stack: Vec<&str> = output_names.iter().map(String::as_str).collect();
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        let Some(&i) = producer.get(t) else { continue };
        if live_nodes[i] {
            continue;
        }
        live_nodes[i] = true;
        stack.extend(nodes[i].inputs.iter().filter(|s| !s.is_empty()).map(String::as_str));
    }
    nodes
        .into_iter()
        .zip(live_nodes)
        .filter(|(_, live)| *live)
        .map(|(n, _)| n)
        .collect()
}

/// Order `nodes` so every producer precedes its consumers, changing the given
/// order as little as possible: among the nodes whose inputs are all
/// available, the one listed first goes next. A list that is already
/// topologically sorted comes back unchanged. Inputs no node produces (graph
/// inputs, weights) count as available. Nodes left on a cycle keep their
/// relative order at the end.
pub fn stable_topological_sort(nodes: Vec<Node>) -> Vec<Node> {
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for o in node.outputs.iter().filter(|o| !o.is_empty()) {
            producer.entry(o.as_str()).or_insert(i);
        }
    }
    let mut pending = vec![0usize; nodes.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        let mut deps: Vec<usize> = node
            .inputs
            .iter()
            .filter_map(|s| producer.get(s.as_str()).copied())
            .filter(|&p| p != i)
            .collect();
        deps.sort_unstable();
        deps.dedup();
        pending[i] = deps.len();
        for p in deps {
            dependents[p].push(i);
        }
    }
    if is_sorted(&nodes, &producer) {
        return nodes;
    }
    let mut ready: BinaryHeap<Reverse<usize>> = (0..nodes.len()).filter(|&i| pending[i] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(nodes.len());
    let mut placed = vec![false; nodes.len()];
    while let Some(Reverse(i)) = ready.pop() {
        order.push(i);
        placed[i] = true;
        for &d in &dependents[i] {
            pending[d] -= 1;
            if pending[d] == 0 {
                ready.push(Reverse(d));
            }
        }
    }
    order.extend((0..nodes.len()).filter(|&i| !placed[i]));
    let mut slots: Vec<Option<Node>> = nodes.into_iter().map(Some).collect();
    order.into_iter().filter_map(|i| slots[i].take()).collect()
}

fn is_sorted(nodes: &[Node], producer: &HashMap<&str, usize>) -> bool {
    nodes
        .iter()
        .enumerate()
        .all(|(i, n)| n.inputs.iter().all(|s| producer.get(s.as_str()).map_or(true, |&p| p <= i)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use crate::optimizer::test_utils::make_node;

    #[test]
    fn test_dead_node_elimination() {
        let nodes = vec![
            make_node(OpKind::MatMul, "matmul1", vec!["x", "w1"], vec!["mm1"]),
            make_node(OpKind::Relu, "relu1", vec!["mm1"], vec!["r1"]),
            make_node(OpKind::MatMul, "matmul2", vec!["x", "w2"], vec!["mm2"]), // dead branch
            make_node(OpKind::Relu, "relu2", vec!["mm2"], vec!["r2"]),          // dead
        ];
        let output_names = vec!["r1".to_string()];
        let result = dead_node_elimination(nodes, &output_names);
        assert_eq!(result.len(), 2); // Only matmul1 + relu1
        assert_eq!(result[0].name, "matmul1");
        assert_eq!(result[1].name, "relu1");
    }

    #[test]
    fn test_dead_node_elimination_empty() {
        let nodes: Vec<Node> = vec![];
        let output_names: Vec<String> = vec![];
        let result = dead_node_elimination(nodes, &output_names);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dead_node_elimination_all_live() {
        let nodes = vec![
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm_out"]),
            make_node(OpKind::Relu, "relu", vec!["mm_out"], vec!["out"]),
        ];
        let output_names = vec!["out".to_string()];
        let result = dead_node_elimination(nodes, &output_names);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn dead_node_elimination_ignores_node_order() {
        // The consumer is listed before its producer.
        let nodes = vec![
            make_node(OpKind::Relu, "relu", vec!["mm"], vec!["out"]),
            make_node(OpKind::MatMul, "mm", vec!["x", "w"], vec!["mm"]),
            make_node(OpKind::Relu, "dead", vec!["x"], vec!["unused"]),
        ];
        let result = dead_node_elimination(nodes, &["out".to_string()]);
        let names: Vec<&str> = result.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, ["relu", "mm"]);
    }

    #[test]
    fn stable_topological_sort_moves_only_what_it_must() {
        let sorted = vec![
            make_node(OpKind::MatMul, "a", vec!["x", "w"], vec!["a"]),
            make_node(OpKind::Relu, "b", vec!["x"], vec!["b"]),
            make_node(OpKind::Add, "c", vec!["a", "b"], vec!["c"]),
        ];
        let names = |v: &[Node]| v.iter().map(|n| n.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&stable_topological_sort(sorted.clone())), ["a", "b", "c"]);
        // `c` and `d` come before their producers; `b` is independent.
        let unsorted = vec![
            make_node(OpKind::Add, "c", vec!["a", "b"], vec!["c"]),
            make_node(OpKind::Relu, "d", vec!["a"], vec!["d"]),
            make_node(OpKind::MatMul, "a", vec!["x", "w"], vec!["a"]),
            make_node(OpKind::Relu, "b", vec!["x"], vec!["b"]),
        ];
        assert_eq!(names(&stable_topological_sort(unsorted)), ["a", "d", "b", "c"]);
        // A cycle keeps its nodes, in their given order, after the rest.
        let cyclic = vec![
            make_node(OpKind::Relu, "p", vec!["q"], vec!["p"]),
            make_node(OpKind::Relu, "q", vec!["p"], vec!["q"]),
            make_node(OpKind::Relu, "r", vec!["x"], vec!["r"]),
        ];
        assert_eq!(names(&stable_topological_sort(cyclic)), ["r", "p", "q"]);
    }
}
