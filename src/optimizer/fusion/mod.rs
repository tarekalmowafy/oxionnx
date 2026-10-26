//! Fusion optimization passes: MatMul+Add, Conv+BatchNorm, Conv+Relu,
//! Conv+ReLU6, SiLU (Mul+Sigmoid), Div+Sqrt→Rsqrt, standalone BatchNorm
//! folding, LayerNorm pattern, consecutive Transpose/Reshape cancellation,
//! MatMul+Transpose fusion, Add+MatMul→Gemm fusion, Conv+Add+Relu (ResNet),
//! Gather+Gather composition, Dropout elimination, Transpose+Reshape simplification,
//! spatial (InstanceNorm-style) normalization chains, scaled dot-product attention,
//! written-out exact GELU, MatMul + GELU / SiLU (plain and gated), MatMul + Split
//! projections (with head norm and rotary), residual Add + norm
//! (`Skip[Simplified]LayerNormalization`), written-out rotary (`OxiRotateHalf`).
//!
//! All of them are wired into [`crate::optimizer::optimize_with_level`] at
//! `PassLevel::Extended` and above, except `fuse_conv_add_relu`,
//! `fuse_instance_norm`, `fuse_attention`, `fuse_matmul_act`,
//! `fuse_matmul_split`, `fuse_rotate_half` and `fuse_skip_norm`, which emit the optimizer-only `ConvAddRelu` /
//! `OxiInstanceNorm` / `OxiAttention` / `OxiMatMulAct` / `OxiMatMulSplit` ops
//! and therefore run only when the active `OperatorRegistry` provides a kernel
//! for them.
//!
//! Every pass takes the graph's `output_names` and refuses to remove, rename or
//! re-parent a tensor the model declares as an output.

mod attention;
mod conv;
mod gelu_erf;
mod instance_norm;
mod matmul;
mod matmul_act;
mod matmul_split;
mod qkv_pack;
mod repeat_kv;
mod rotate_half;
mod simplify;
mod skip_norm;

pub use attention::fuse_attention;
pub use gelu_erf::fuse_gelu_erf;
pub use instance_norm::fuse_instance_norm;
pub use matmul_act::fuse_matmul_act;
pub use matmul_split::fuse_matmul_split;
pub use qkv_pack::pack_shared_projections;
pub use repeat_kv::fuse_repeat_kv;
pub use rotate_half::fuse_rotate_half;
pub use skip_norm::fuse_skip_norm;

pub use conv::{
    fold_batch_norm_inference, fuse_conv_add_relu, fuse_conv_batchnorm,
    fuse_conv_clip_to_conv_relu6, fuse_conv_relu,
};
pub use matmul::{
    fuse_add_matmul_to_gemm, fuse_layer_norm, fuse_matmul_add, fuse_matmul_transpose,
};
pub use simplify::{
    cancel_consecutive_reshape, cancel_consecutive_transpose, eliminate_dropout_inference,
    fuse_div_sqrt_to_rsqrt, fuse_gather_composition, fuse_mul_sigmoid_to_silu,
    simplify_transpose_reshape,
};
