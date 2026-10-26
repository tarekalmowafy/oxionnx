//! Load-time weight prepacking ([`Operator::prepack`]).
//!
//! Called once per session after optimization. Operators may pack their
//! constant inputs now (the GEMM ops pack their weights here, as BF16 with
//! `SessionBuilder::with_bf16_weights`) and report which inputs they will never
//! read in their original form again. An initializer that every consumer
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
    // Subgraph bodies may read outer initializers by name; keep everything then.
    let release = !registry.keep_weights() && !nodes.iter().any(|n| !n.attrs.graphs.is_empty());
    // Initializer name -> (consumer inputs not yet visited, every visited one
    // no longer needs the data). A weight is freed right after its last
    // consumer packed it, so the session never holds every original and every
    // packed copy at once.
    let mut pending: HashMap<String, (usize, bool)> = HashMap::new();
    for name in nodes.iter().flat_map(|n| &n.inputs) {
        if weights.contains_key(name) {
            pending.entry(name.clone()).or_insert((0, true)).0 += 1;
        }
    }
    let mut bytes = 0;
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
            let Some((left, ok)) = pending.get_mut(name) else { continue };
            *left -= 1;
            *ok &= freed.contains(&i);
            if *left == 0 && *ok && release && !output_names.contains(name) {
                if let Some(t) = weights.get_mut(name) {
                    bytes += t.data.len() * std::mem::size_of::<f32>();
                    t.data = Vec::new();
                }
            }
        }
    }
    if bytes > 0 {
        return_freed_memory();
    }
    bytes
}

/// Decide, on the optimized graph, which tables the loader left in the mapped
/// model file stay there (see [`OperatorRegistry::mapped_weight`]): those whose
/// every consumer reads them from the file ([`Operator::reads_mapped`]) and
/// that are not graph outputs. The others (say an optimizer pass gave one a new
/// consumer, or the session keeps its weights) are decoded into their tensor
/// now, exactly as the loader would have. Returns the bytes left mapped.
///
/// [`OperatorRegistry::mapped_weight`]: oxionnx_core::OperatorRegistry::mapped_weight
/// [`Operator::reads_mapped`]: oxionnx_core::Operator::reads_mapped
pub(crate) fn settle_mapped(
    nodes: &[Node],
    weights: &mut HashMap<String, Tensor>,
    output_names: &[String],
    registry: &OperatorRegistry,
) -> usize {
    let names = registry.mapped_weight_names();
    if names.is_empty() {
        return 0;
    }
    let keep_all = registry.keep_weights() || nodes.iter().any(|n| !n.attrs.graphs.is_empty());
    let mut mapped = 0;
    for name in names {
        let Some(t) = weights.get(&name) else {
            registry.remove_mapped_weight(&name);
            continue;
        };
        let readers_ok = !keep_all
            && !output_names.contains(&name)
            && nodes.iter().all(|node| {
                node.inputs.iter().enumerate().filter(|(_, n)| **n == name).all(|(i, _)| {
                    registry.get(node.op.as_str()).is_some_and(|op| {
                        let ctx = OpContext {
                            node,
                            inputs: node.inputs.iter().map(|n| weights.get(n)).collect(),
                            outer_scope: None,
                            weights: Some(&*weights),
                            registry: Some(registry),
                        };
                        op.reads_mapped(&ctx, i)
                    })
                })
            });
        if readers_ok && is_released(t) {
            if let Some(m) = registry.mapped_weight(&name) {
                mapped += m.bytes().len();
            }
            continue;
        }
        if let Some(m) = registry.remove_mapped_weight(&name) {
            if let Some(t) = weights.get_mut(&name) {
                t.data = m.materialize();
            }
        }
    }
    mapped
}

/// Hand the freed weights back to the OS. glibc keeps them in its heap: after
/// the first large free its mmap threshold rises, so later weight-sized blocks
/// live in the heap rather than in their own mappings, and freeing them does
/// not lower RSS (BF16 sessions kept the size of their FP32 weights resident).
fn return_freed_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        #[allow(unsafe_code)]
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: plain glibc call with no pointers; it only releases free pages.
        #[allow(unsafe_code)]
        unsafe {
            malloc_trim(0);
        }
    }
}

/// The original data of every released weight that some consumer can rebuild
/// exactly ([`Operator::released_weight`]), by name.
///
/// [`Operator::released_weight`]: oxionnx_core::Operator::released_weight
pub(crate) fn rebuild_released(
    nodes: &[Node],
    weights: &HashMap<String, Tensor>,
    registry: &OperatorRegistry,
) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    for node in nodes {
        for (i, name) in node.inputs.iter().enumerate() {
            if out.contains_key(name) || !weights.get(name).is_some_and(is_released) {
                continue;
            }
            // A table left in the mapped model file holds the original bytes.
            if let Some(m) = registry.mapped_weight(name) {
                out.insert(name.clone(), Tensor { data: m.materialize(), shape: weights[name].shape.clone() });
                continue;
            }
            let Some(op) = registry.get(node.op.as_str()) else { continue };
            let ctx = OpContext {
                node,
                inputs: node.inputs.iter().map(|n| weights.get(n)).collect(),
                outer_scope: None,
                weights: Some(weights),
                registry: Some(registry),
            };
            if let Some(t) = op.released_weight(&ctx, i).filter(|t| t.shape == weights[name].shape) {
                out.insert(name.clone(), t);
            }
        }
    }
    out
}

/// Whether `t` is an initializer released by [`prepack_weights`].
pub(crate) fn is_released(t: &Tensor) -> bool {
    t.data.len() != t.shape.iter().product::<usize>()
}
