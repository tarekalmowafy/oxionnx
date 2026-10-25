//! Load-time weight prepacking ([`Operator::prepack`]).
//!
//! Called once per session after optimization. Operators may pack their
//! constant inputs now (the BF16-weights option packs every MatMul weight to
//! BF16 here, see `SessionBuilder::with_bf16_weights`) and report which inputs
//! they will never read in FP32 again. An initializer that every consumer
//! reports that way, and that is not a graph output, is then *released*: its
//! data is freed and its shape kept, so the session holds only the packed copy.
//!
//! [`Operator::prepack`]: oxionnx_core::Operator::prepack

use crate::graph::Node;
use crate::tensor::Tensor;
use oxionnx_core::{OpContext, OperatorRegistry};
use std::collections::HashMap;

/// Run every node's prepack hook and release the initializers no consumer
/// needs in FP32 any more. Returns the number of bytes freed.
pub(crate) fn prepack_weights(
    nodes: &[Node],
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
    registry: &OperatorRegistry,
) -> usize {
    // Initializer name -> every consumer so far no longer needs its FP32 data.
    let mut releasable: HashMap<String, bool> = HashMap::new();
    for node in nodes {
        if !node.inputs.iter().any(|n| weights.contains_key(n)) {
            continue;
        }
        let freed = match registry.get(node.op.as_str()) {
            Some(op) => {
                let ctx = OpContext {
                    node,
                    inputs: node.inputs.iter().map(|n| weights.get(n)).collect(),
                    outer_scope: None,
                    weights: Some(&*weights),
                    registry: Some(registry),
                };
                op.prepack(&ctx)
            }
            None => Vec::new(),
        };
        for (i, name) in node.inputs.iter().enumerate() {
            if weights.contains_key(name) {
                let ok = freed.contains(&i);
                releasable.entry(name.clone()).and_modify(|v| *v &= ok).or_insert(ok);
            }
        }
    }
    // Subgraph bodies may read outer initializers by name; keep everything then.
    if nodes.iter().any(|n| !n.attrs.graphs.is_empty()) {
        return 0;
    }
    let mut bytes = 0;
    for (name, ok) in releasable {
        if !ok || output_names.contains(&name) {
            continue;
        }
        if let Some(t) = weights.get_mut(&name) {
            bytes += t.data.len() * std::mem::size_of::<f32>();
            t.data = Vec::new();
        }
    }
    bytes
}

/// Whether `t` is an initializer released by [`prepack_weights`].
pub(crate) fn is_released(t: &Tensor) -> bool {
    t.data.len() != t.shape.iter().product::<usize>()
}
