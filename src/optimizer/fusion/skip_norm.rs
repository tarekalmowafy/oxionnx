//! Residual `Add` + norm fusion → `com.microsoft` `SkipLayerNormalization` /
//! `SkipSimplifiedLayerNormalization`.
//!
//! # Patterns
//!
//! ```text
//! s = Add(a, b)                          [b = Add(t, bias) | Add(bias, t)]   // bias: constant, 1-D
//! y = LayerNormalization(s, gamma[, beta], axis=-1)      ->  SkipLayerNormalization(a, b|t, gamma, beta, [bias])
//! y = SimplifiedLayerNormalization(s, gamma, axis=-1)    ->  SkipSimplifiedLayerNormalization(a, b|t, gamma, [bias])
//! ```
//!
//! ModernBERT / granite exports (pre-norm) keep using `s` for the next residual
//! `Add`, so `s` becomes the fused node's `input_skip_bias_sum` output (output
//! 3) under its original name; BERT-style post-norm exports (arctic) use `s`
//! only for the norm. Gemma 3 (harrier-270m) has `Add -> SimplifiedLayerNormalization`
//! between its post- and pre-norms. Decomposed LayerNorm chains are folded into
//! `LayerNormalization` by `fuse_layer_norm`, which runs first.
//!
//! The fused node takes the `Add`'s place in the node list when `gamma` /
//! `beta` / `bias` are constants (so they are available there), which keeps
//! every other consumer of `s` after its producer; with a computed `beta`
//! (granite multilingual: a `Constant` node) it takes the norm's place, which
//! needs every other consumer of `s` to come after the norm. The kernel handles `a` / `b`
//! in either order and with any broadcast `Add` allows.

use crate::graph::{Node, OpKind};
use crate::optimizer::graph_utils::TensorUsage;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// `true` when `t` is a constant whose shape is `[n]` up to leading 1s.
fn is_vector(weights: &HashMap<String, Tensor>, t: &str) -> bool {
    weights.get(t).is_some_and(|w| w.shape.iter().rev().skip(1).all(|&d| d == 1) && !w.data.is_empty())
}

struct Fused {
    /// Where the fused node goes (the residual `Add` or the norm).
    at: usize,
    /// Nodes removed outright: the norm or the `Add`, and a folded bias `Add`.
    remove: Vec<usize>,
    node: Node,
}

fn try_fuse(
    nodes: &[Node],
    producer: &HashMap<&str, usize>,
    usage: &TensorUsage,
    weights: &HashMap<String, Tensor>,
    ni: usize,
    want_ln: bool,
    want_rms: bool,
) -> Option<Fused> {
    let n = &nodes[ni];
    let rms = match n.op {
        OpKind::LayerNorm if want_ln => false,
        OpKind::RMSNorm if want_rms => true,
        _ => return None,
    };
    if n.attrs.i("axis", -1) != -1 || n.outputs.iter().skip(1).any(|o| !o.is_empty()) || n.outputs.is_empty() {
        return None;
    }
    let gamma = n.inputs.get(1).filter(|g| !g.is_empty())?;
    let beta = n.inputs.get(2).filter(|b| !b.is_empty());
    if rms && beta.is_some() {
        return None;
    }
    // A constant gamma / beta must be a vector; a computed one (e.g. a
    // `Constant` node output) is checked by the kernel at run time.
    if [Some(gamma), beta].into_iter().flatten().any(|t| weights.contains_key(t) && !is_vector(weights, t)) {
        return None;
    }
    let s = n.inputs.first()?;
    let ai = *producer.get(s.as_str())?;
    let add = &nodes[ai];
    if add.op != OpKind::Add || add.inputs.len() != 2 || add.outputs.len() != 1 {
        return None;
    }
    let (a, mut b) = (add.inputs[0].clone(), add.inputs[1].clone());
    if a.is_empty() || b.is_empty() || weights.contains_key(&a) || weights.contains_key(&b) {
        return None;
    }
    // Fold a constant bias Add on either operand (`MatMul + bias` left unfused).
    let mut remove = vec![ni];
    let mut bias = None;
    let mut a = a;
    for swap in [false, true] {
        if bias.is_some() {
            break;
        }
        let operand = if swap { &a } else { &b };
        if !usage.is_fusable_intermediate(operand) {
            continue;
        }
        let Some(&bi) = producer.get(operand.as_str()) else { continue };
        let ba = &nodes[bi];
        if ba.op != OpKind::Add || ba.inputs.len() != 2 || ba.outputs.len() != 1 {
            continue;
        }
        let (c, t) = if is_vector(weights, &ba.inputs[0]) && !weights.contains_key(&ba.inputs[1]) {
            (&ba.inputs[0], &ba.inputs[1])
        } else if is_vector(weights, &ba.inputs[1]) && !weights.contains_key(&ba.inputs[0]) {
            (&ba.inputs[1], &ba.inputs[0])
        } else {
            continue;
        };
        // The bias must broadcast like `[hidden]` against the residual.
        if weights.get(gamma).is_none_or(|g| g.data.len() != weights[c].data.len()) || t.is_empty() {
            continue;
        }
        bias = Some(c.clone());
        let t = t.clone();
        if swap {
            a = t;
        } else {
            b = t;
        }
        remove.push(bi);
    }
    let sum_out = if usage.is_fusable_intermediate(s) { String::new() } else { s.clone() };
    // Where the fused node goes: at the `Add` when gamma / beta are constants
    // (always available there), else at the norm, provided no other consumer
    // of the sum runs before it.
    let consts = [Some(gamma), beta].into_iter().flatten().all(|t| weights.contains_key(t));
    let at = if consts {
        ai
    } else if (0..ni).all(|j| j == ai || !nodes[j].inputs.iter().any(|x| x == s)) {
        ni
    } else {
        return None;
    };
    let mut inputs = vec![a, b, gamma.clone()];
    if rms {
        if let Some(bias) = bias {
            inputs.push(bias);
        }
    } else {
        match (beta, bias) {
            (beta, Some(bias)) => {
                inputs.push(beta.cloned().unwrap_or_default());
                inputs.push(bias);
            }
            (Some(beta), None) => inputs.push(beta.clone()),
            (None, None) => {}
        }
    }
    let mut outputs = vec![n.outputs[0].clone()];
    if !sum_out.is_empty() {
        outputs.extend([String::new(), String::new(), sum_out]);
    }
    let mut attrs = crate::graph::Attributes::default();
    // Our `SimplifiedLayerNormalization` kernel defaults epsilon to 1e-6, the
    // contrib ops to 1e-5: always carry the value the norm actually used.
    attrs.floats.insert("epsilon".into(), n.attrs.f("epsilon", if rms { 1e-6 } else { 1e-5 }));
    let op = if rms { "SkipSimplifiedLayerNormalization" } else { "SkipLayerNormalization" };
    remove.push(ai);
    remove.retain(|&j| j != at);
    Some(Fused {
        at,
        remove,
        node: Node { op: OpKind::Unknown(op.into()), name: format!("{}_skipnorm", n.name), inputs, outputs, attrs },
    })
}

/// Fuse residual `Add` + `LayerNormalization` / `SimplifiedLayerNormalization`
/// pairs (see the module docs). `ln` / `rms` select which norms to fuse (each
/// needs its contrib kernel in the registry).
pub fn fuse_skip_norm(
    nodes: Vec<Node>,
    weights: &HashMap<String, Tensor>,
    output_names: &[String],
    ln: bool,
    rms: bool,
) -> Vec<Node> {
    if !ln && !rms {
        return nodes;
    }
    let fused: Vec<Fused> = {
        let usage = TensorUsage::new(&nodes, output_names);
        let mut producer = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for o in &n.outputs {
                if !o.is_empty() {
                    producer.insert(o.as_str(), i);
                }
            }
        }
        let mut claimed = HashSet::new();
        let mut out = Vec::new();
        for i in 0..nodes.len() {
            if let Some(f) = try_fuse(&nodes, &producer, &usage, weights, i, ln, rms) {
                if f.remove.iter().chain([&f.at]).all(|j| !claimed.contains(j)) {
                    claimed.extend(f.remove.iter().copied());
                    claimed.insert(f.at);
                    out.push(f);
                }
            }
        }
        out
    };
    if fused.is_empty() {
        return nodes;
    }
    tracing::debug!(count = fused.len(), "fused residual Add + norm into Skip[Simplified]LayerNormalization");
    let mut drop = HashSet::new();
    let mut replace = HashMap::new();
    for f in fused {
        drop.extend(f.remove);
        replace.insert(f.at, f.node);
    }
    nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| if let Some(f) = replace.remove(&i) { Some(f) } else { (!drop.contains(&i)).then_some(n) })
        .collect()
}

#[cfg(test)]
#[path = "skip_norm_tests.rs"]
mod tests;
