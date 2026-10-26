//! `com.microsoft::MatMulNBits`, its prepacked form `OxiMatMulNBits`, and
//! `com.microsoft::GatherBlockQuantized`. Kernels live in
//! [`crate::matmul_nbits`].
//!
//! The model loader (`oxionnx_proto::byte_tensor`) keeps the packed uint8
//! initializers of these ops as byte tensors (four bytes per f32 word, each
//! innermost row zero-padded to whole words) and annotates the node with
//! [`ATTR_U8_INPUTS`] / [`ATTR_U8_LAST`]; inputs not listed there hold one
//! byte value per f32 element, as for a hand-built graph. [`ATTR_DTYPES`]
//! records the ONNX element types of initializer inputs, which the f32 runtime
//! tensors no longer carry.
//!
//! `OxiMatMulNBits` (emitted by the optimizer) takes `(A, packed, bias)`, where
//! `packed` is a weight already in the [`crate::matmul_nbits::pack`] layout, so
//! the session holds a single, kernel-ready copy of each quantized weight.
//!
//! `OXIONNX_NBITS_COMPUTE=f32` switches the (unfused) MatMulNBits /
//! OxiMatMulNBits nodes from int8 activations to the f32 packed GEMM on
//! weights dequantized once per session (FP32 memory), for accuracy and speed
//! comparisons. Quantized projections fused into `OxiMatMulAct` /
//! `OxiMatMulSplit` keep the int8 path. The int8 path stays the default on
//! AVX2-only CPUs too: on embeddinggemma (1 and 8 threads) it beat both this
//! mode and the FP32 model on the AVX2 FP32 GEMM, with half the memory.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use oxionnx_core::graph::Attributes;
use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::matmul_nbits::{self as nb, Packed, Shape, Source, Table, ZeroPoints};

/// Mirrors `oxionnx_proto::byte_tensor::ATTR_INPUTS`.
pub const ATTR_U8_INPUTS: &str = "__oxi_u8_inputs";
/// Mirrors `oxionnx_proto::byte_tensor::ATTR_LAST`.
pub const ATTR_U8_LAST: &str = "__oxi_u8_last";
/// Mirrors `oxionnx_proto::byte_tensor::ATTR_DTYPES`.
pub const ATTR_DTYPES: &str = "__oxi_init_dtypes";

const DT_FLOAT: i64 = 1;
const DT_UINT8: i64 = 2;
const DT_FLOAT16: i64 = 10;
const DT_BFLOAT16: i64 = 16;

fn unsupported(op: &str, name: &str, m: impl std::fmt::Display) -> OnnxError {
    OnnxError::Unsupported(format!("{op} '{name}': {m}"))
}

/// Size `slot` to `shape`; its contents are fully overwritten by the caller.
fn shape_slot(slot: &mut Tensor, shape: &[usize]) {
    let n: usize = shape.iter().product();
    if slot.data.len() != n {
        slot.data.resize(n, 0.0);
    }
    slot.shape.clear();
    slot.shape.extend_from_slice(shape);
}

/// ONNX `data_type` of initializer input `idx` (0 when unknown / not one).
fn dtype(attrs: &Attributes, idx: usize) -> i64 {
    attrs.ints(ATTR_DTYPES).get(idx).copied().unwrap_or(0)
}

/// Original innermost size (bytes) of input `idx` when it is a byte tensor.
fn byte_tensor_last(attrs: &Attributes, idx: usize) -> Option<usize> {
    let pos = attrs.ints(ATTR_U8_INPUTS).iter().position(|&i| i == idx as i64)?;
    attrs.ints(ATTR_U8_LAST).get(pos).map(|&l| l.max(0) as usize)
}

/// The uint8 contents of input `idx`, without row padding.
fn input_bytes<'t>(attrs: &Attributes, idx: usize, t: &'t Tensor) -> Result<Cow<'t, [u8]>, String> {
    match byte_tensor_last(attrs, idx) {
        Some(last) => {
            let words = last.div_ceil(4);
            if words == 0 {
                return Ok(Cow::Borrowed(&[]));
            }
            if t.data.len() % words != 0 {
                return Err(format!("byte tensor of {} words does not hold rows of {last} bytes", t.data.len()));
            }
            let mut out = Vec::with_capacity(t.data.len() / words * last);
            for row in t.data.chunks_exact(words) {
                let start = out.len();
                for w in row {
                    out.extend_from_slice(&w.to_bits().to_le_bytes());
                }
                out.truncate(start + last);
            }
            Ok(Cow::Owned(out))
        }
        None => t
            .data
            .iter()
            .map(|&v| if (0.0..=255.0).contains(&v) && v.fract() == 0.0 { Ok(v as u8) } else { Err(v) })
            .collect::<Result<Vec<u8>, f32>>()
            .map(Cow::Owned)
            .map_err(|v| format!("expected uint8 values, found {v}")),
    }
}

/// The geometry from the `K`, `N`, `bits`, `block_size` attributes.
fn nbits_shape(attrs: &Attributes) -> Result<Shape, String> {
    let get = |name: &str, default: i64| -> Result<usize, String> {
        let v = attrs.i(name, default);
        usize::try_from(v).map_err(|_| format!("attribute {name}={v} is invalid"))
    };
    Shape::new(get("N", 0)?, get("K", 0)?, get("bits", 4)?, get("block_size", 0)?)
}

/// Pack a MatMulNBits weight from its ORT inputs (`B`, `scales`, optional
/// `zero_points`), validating dtypes and shapes. `attrs` are the node's.
pub fn pack_matmul_nbits(attrs: &Attributes, b: &Tensor, scales: &Tensor, zp: Option<&Tensor>) -> Result<Vec<f32>, String> {
    let s = nbits_shape(attrs)?;
    match dtype(attrs, 1) {
        0 | DT_UINT8 => {}
        d => return Err(format!("B has ONNX type {d}; only uint8 is supported")),
    }
    match dtype(attrs, 2) {
        0 | DT_FLOAT => {}
        DT_FLOAT16 | DT_BFLOAT16 => {
            return Err("float16/bfloat16 A and scales are not supported; only float32".into())
        }
        d => return Err(format!("scales has ONNX type {d}; only float32 is supported")),
    }
    let bytes = input_bytes(attrs, 1, b)?;
    let zp_bytes;
    let zp = match zp {
        None => ZeroPoints::Default,
        Some(z) => {
            let float = match dtype(attrs, 3) {
                DT_UINT8 => false,
                DT_FLOAT => true,
                0 => {
                    // Not an initializer: the shape tells the two layouts apart
                    // (they coincide for 8 bits, where both mean the same).
                    byte_tensor_last(attrs, 3).is_none() && z.data.len() == s.n * s.kb() && s.zp_row() != s.kb()
                }
                d => return Err(format!("zero_points has ONNX type {d}; only uint8 or float32 are supported")),
            };
            if float {
                ZeroPoints::Float(&z.data)
            } else {
                zp_bytes = input_bytes(attrs, 3, z)?;
                ZeroPoints::Packed(&zp_bytes)
            }
        }
    };
    nb::pack(s, &Source { b: &bytes, scales: &scales.data, zp })
}

fn compute_f32() -> bool {
    static F32: OnceLock<bool> = OnceLock::new();
    *F32.get_or_init(|| std::env::var("OXIONNX_NBITS_COMPUTE").is_ok_and(|v| v.eq_ignore_ascii_case("f32")))
}

/// Whether input `idx` of the node is a session weight (stable for the session).
fn is_weight(ctx: &OpContext<'_>, idx: usize, t: &Tensor) -> bool {
    ctx.weights
        .and_then(|ws| ws.get(ctx.node.inputs.get(idx)?))
        .is_some_and(|w| std::ptr::eq(w, t))
}

/// `com.microsoft::MatMulNBits` (`prepacked = false`) and the optimizer's
/// `OxiMatMulNBits` (`prepacked = true`, inputs `A`, packed weight, `bias`).
pub struct MatMulNBitsOp {
    prepacked: bool,
    /// Packed weights built from ORT-layout session weights, keyed by the
    /// (B, scales, zero_points) data pointers.
    packed: RwLock<HashMap<(usize, usize, usize), Arc<Vec<f32>>>>,
    #[cfg(target_arch = "x86_64")]
    dequantized: RwLock<HashMap<usize, Arc<crate::gemm_packed::PackedB>>>,
}

impl MatMulNBitsOp {
    pub fn new(prepacked: bool) -> Self {
        MatMulNBitsOp {
            prepacked,
            packed: RwLock::default(),
            #[cfg(target_arch = "x86_64")]
            dequantized: RwLock::default(),
        }
    }

    fn name(&self) -> &'static str {
        if self.prepacked { "OxiMatMulNBits" } else { "MatMulNBits" }
    }

    /// The packed weight buffer, plus the index of the bias input.
    fn weight<'t>(&self, ctx: &OpContext<'t>) -> Result<(Words<'t>, usize), OnnxError> {
        let bad = |m: String| unsupported(self.name(), &ctx.node.name, m);
        let attrs = ctx.attrs();
        if self.prepacked {
            return Ok((Words::Borrowed(&ctx.input(1)?.data), 2));
        }
        if ctx.node.inputs.get(4).is_some_and(|n| !n.is_empty()) {
            return Err(bad("the g_idx input (act-order quantization) is not supported".into()));
        }
        let (b, scales, zp) = (ctx.input(1)?, ctx.input(2)?, ctx.optional_input(3));
        let constant = is_weight(ctx, 1, b) && is_weight(ctx, 2, scales) && zp.map_or(true, |z| is_weight(ctx, 3, z));
        if !constant {
            return Ok((Words::Owned(pack_matmul_nbits(attrs, b, scales, zp).map_err(bad)?), 5));
        }
        let key = (b.data.as_ptr() as usize, scales.data.as_ptr() as usize, zp.map_or(0, |z| z.data.as_ptr() as usize));
        if let Some(p) = self.packed.read().ok().and_then(|c| c.get(&key).cloned()) {
            return Ok((Words::Shared(p), 5));
        }
        let p = Arc::new(pack_matmul_nbits(attrs, b, scales, zp).map_err(bad)?);
        if let Ok(mut c) = self.packed.write() {
            c.insert(key, p.clone());
        }
        Ok((Words::Shared(p), 5))
    }

    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let bad = |m: String| OnnxError::ShapeMismatch(format!("{} '{}': {m}", self.name(), ctx.node.name));
        let a = ctx.input(0)?;
        let (words, bias_idx) = self.weight(ctx)?;
        let w = Packed::from_words(words.get()).map_err(bad)?;
        let s = w.s;
        if !self.prepacked {
            let attr = nbits_shape(ctx.attrs()).map_err(bad)?;
            if attr != s {
                return Err(bad(format!("packed weight {s:?} does not match the attributes {attr:?}")));
            }
        }
        if a.ndim() == 0 || a.shape[a.ndim() - 1] != s.k {
            return Err(bad(format!("A {:?} does not end in K={}", a.shape, s.k)));
        }
        let bias = ctx.optional_input(bias_idx);
        if let Some(bt) = bias {
            if bt.data.len() != s.n {
                return Err(bad(format!("bias {:?} is not [N={}]", bt.shape, s.n)));
            }
        }
        let m = a.data.len() / s.k;
        let mut out_shape = a.shape.clone();
        *out_shape.last_mut().expect("rank >= 1") = s.n;
        shape_slot(&mut slots[0], &out_shape);
        let out = &mut slots[0].data;
        let bias = bias.map(|t| &t.data[..]);
        #[cfg(target_arch = "x86_64")]
        if compute_f32() && crate::gemm_packed::available() {
            let key = words.get().as_ptr() as usize;
            let cached = !matches!(words, Words::Owned(_));
            let pb = match self.dequantized.read().ok().and_then(|c| c.get(&key).cloned()).filter(|_| cached) {
                Some(p) => p,
                None => {
                    let p = Arc::new(crate::gemm_packed::PackedB::pack(&w.dequantize(), s.k, s.n));
                    if cached {
                        if let Ok(mut c) = self.dequantized.write() {
                            c.insert(key, p.clone());
                        }
                    }
                    p
                }
            };
            crate::gemm_packed::gemm(m, &a.data, &pb, out, crate::par::threads());
            if let Some(bias) = bias {
                for row in out.chunks_exact_mut(s.n) {
                    row.iter_mut().zip(bias).for_each(|(o, b)| *o += b);
                }
            }
            return Ok(());
        }
        let _ = compute_f32;
        nb::matmul(nb::isa(), m, &a.data, &w, bias, out, crate::par::threads());
        Ok(())
    }
}

/// A packed weight buffer: a session weight, a cached repack, or a one-off.
enum Words<'t> {
    Borrowed(&'t [f32]),
    Shared(Arc<Vec<f32>>),
    Owned(Vec<f32>),
}

impl Words<'_> {
    fn get(&self) -> &[f32] {
        match self {
            Words::Borrowed(w) => w,
            Words::Shared(w) => w,
            Words::Owned(w) => w,
        }
    }
}

impl Operator for MatMulNBitsOp {
    fn op_type(&self) -> &str {
        self.name()
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let mut slots = vec![Tensor { data: vec![], shape: vec![] }];
        self.run(ctx, &mut slots)?;
        Ok(slots)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        self.run(ctx, slots)
    }
}

// ── GatherBlockQuantized ─────────────────────────────────────────────────────

/// `com.microsoft::GatherBlockQuantized` for uint8 data (`bits` 2, 4 or 8,
/// packed low bits first along the last axis): `gather_axis` 0 and
/// `quantize_axis` the last axis, as ORT requires for uint8. Out-of-range
/// indices produce zeros, as in ORT.
#[derive(Default)]
pub struct GatherBlockQuantizedOp {
    /// Byte copies of tables given one f32 per byte, keyed by data pointer.
    bytes: RwLock<HashMap<usize, Arc<Vec<u8>>>>,
}

impl GatherBlockQuantizedOp {
    fn run(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        let name = &ctx.node.name;
        let bad = |m: String| unsupported("GatherBlockQuantized", name, m);
        let (data, indices, scales) = (ctx.input(0)?, ctx.input(1)?, ctx.input(2)?);
        let zp = ctx.optional_input(3);
        let attrs = ctx.attrs();
        let bits = attrs.i("bits", 4);
        let block = attrs.i("block_size", 128);
        match dtype(attrs, 0) {
            0 | DT_UINT8 => {}
            d => return Err(bad(format!("data has ONNX type {d}; only uint8 is supported"))),
        }
        match dtype(attrs, 2) {
            0 | DT_FLOAT => {}
            d => return Err(bad(format!("scales has ONNX type {d}; only float32 is supported"))),
        }
        // Logical data shape (bytes along the last axis).
        let last_bytes = byte_tensor_last(attrs, 0);
        let mut dims = data.shape.clone();
        if let (Some(l), Some(d)) = (last_bytes, dims.last_mut()) {
            *d = l;
        }
        let r = dims.len() as i64;
        let norm = |a: i64| if a < 0 { a + r } else { a };
        if r < 1 || norm(attrs.i("gather_axis", 0)) != 0 || norm(attrs.i("quantize_axis", 1)) != r - 1 {
            return Err(bad(format!(
                "only gather_axis 0 and quantize_axis = last are supported for uint8 data (rank {r}, gather_axis {}, quantize_axis {})",
                attrs.i("gather_axis", 0),
                attrs.i("quantize_axis", 1)
            )));
        }
        if !matches!(bits, 2 | 4 | 8) || block < 16 {
            return Err(bad(format!("bits={bits} / block_size={block} are not supported")));
        }
        let (bits, block) = (bits as usize, block as usize);
        let row_bytes = *dims.last().expect("rank >= 1");
        let cols = row_bytes * (8 / bits);
        let table_rows: usize = dims[..dims.len() - 1].iter().product();
        let per_index: usize = dims[1..dims.len() - 1].iter().product();
        // Rows of the table and of packed zero points, with their strides in bytes.
        let data_rows = self.byte_rows(ctx, 0, data, row_bytes).map_err(bad)?;
        let zp_rows;
        let (zp, zp_stride) = match zp {
            None => (ZeroPoints::Default, 0),
            Some(z) if dtype(attrs, 3) == DT_FLOAT => (ZeroPoints::Float(&z.data), 0),
            Some(z) => {
                let kb = cols.div_ceil(block);
                zp_rows = self.byte_rows(ctx, 3, z, (kb * bits).div_ceil(8)).map_err(bad)?;
                (ZeroPoints::Packed(zp_rows.0.get()), zp_rows.1)
            }
        };
        let table = Table {
            data: data_rows.0.get(),
            stride: data_rows.1,
            rows: table_rows,
            cols,
            bits,
            block,
            scales: &scales.data,
            zp,
            zp_stride,
        };
        table.check().map_err(bad)?;
        let mut out_shape = indices.shape.clone();
        out_shape.extend_from_slice(&dims[1..dims.len() - 1]);
        out_shape.push(cols);
        shape_slot(&mut slots[0], &out_shape);
        let gather_rows = dims[0] as i64;
        let slice = per_index * cols;
        crate::par::for_each_row(&mut slots[0].data, slice.max(1), |i, out| {
            let idx = indices.data[i] as i64;
            let idx = if idx < 0 { idx + gather_rows } else { idx };
            if !(0..gather_rows).contains(&idx) {
                out.fill(0.0);
                return;
            }
            for (j, row) in out.chunks_exact_mut(cols).enumerate() {
                table.row(idx as usize * per_index + j, row);
            }
        });
        Ok(())
    }
}

/// Byte rows borrowed from a loader byte tensor, or a (cached) copy.
enum Bytes<'t> {
    Borrowed(&'t [u8]),
    Shared(Arc<Vec<u8>>),
}

impl Bytes<'_> {
    fn get(&self) -> &[u8] {
        match self {
            Bytes::Borrowed(b) => b,
            Bytes::Shared(b) => b,
        }
    }
}

impl GatherBlockQuantizedOp {
    /// The uint8 rows of input `idx` (`row` bytes each) and their stride:
    /// borrowed from a loader byte tensor, else converted once per session
    /// weight (or per call for a dynamic input).
    fn byte_rows<'t>(&self, ctx: &OpContext<'t>, idx: usize, t: &'t Tensor, row: usize) -> Result<(Bytes<'t>, usize), String> {
        let attrs = ctx.attrs();
        if let Some(last) = byte_tensor_last(attrs, idx) {
            if last != row {
                return Err(format!("input {idx} has rows of {last} bytes, expected {row}"));
            }
            return Ok((Bytes::Borrowed(words_as_bytes(&t.data)), last.div_ceil(4) * 4));
        }
        let key = t.data.as_ptr() as usize;
        let cached = is_weight(ctx, idx, t);
        if let Some(b) = self.bytes.read().ok().and_then(|c| c.get(&key).cloned()).filter(|_| cached) {
            return Ok((Bytes::Shared(b), row));
        }
        let b = Arc::new(input_bytes(attrs, idx, t)?.into_owned());
        if cached {
            if let Ok(mut c) = self.bytes.write() {
                c.insert(key, b.clone());
            }
        }
        Ok((Bytes::Shared(b), row))
    }
}

/// The bytes of a loader byte tensor (little-endian words).
fn words_as_bytes(w: &[f32]) -> &[u8] {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 memory is valid as bytes (alignment 1); on a
        // little-endian target the in-memory byte order is the file's.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(w.as_ptr().cast::<u8>(), std::mem::size_of_val(w))
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        compile_error!("byte tensors assume a little-endian target")
    }
}

impl Operator for GatherBlockQuantizedOp {
    fn op_type(&self) -> &str {
        "GatherBlockQuantized"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let mut slots = vec![Tensor { data: vec![], shape: vec![] }];
        self.run(ctx, &mut slots)?;
        Ok(slots)
    }
    fn supports_output_slots(&self) -> bool {
        true
    }
    fn execute_into_slots(&self, ctx: &OpContext<'_>, slots: &mut [Tensor]) -> Result<(), OnnxError> {
        self.run(ctx, slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxionnx_core::graph::{Node, OpKind};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn f(&mut self) -> f32 {
            self.next() as f32 / (1u64 << 31) as f32 - 0.5
        }
    }

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor { data, shape: shape.to_vec() }
    }

    /// As the loader stores a uint8 initializer (see the module docs).
    fn byte_tensor(bytes: &[u8], shape: &[usize]) -> Tensor {
        let last = *shape.last().unwrap();
        let words = last.div_ceil(4);
        let mut data = vec![];
        for row in bytes.chunks(last) {
            for w in row.chunks(4) {
                let mut b = [0u8; 4];
                b[..w.len()].copy_from_slice(w);
                data.push(f32::from_bits(u32::from_le_bytes(b)));
            }
        }
        let mut s = shape.to_vec();
        *s.last_mut().unwrap() = words;
        t(data, &s)
    }

    fn node(op: &str, inputs: &[&str], ints: &[(&str, i64)], lists: &[(&str, Vec<i64>)]) -> Node {
        let mut attrs = Attributes::default();
        for (k, v) in ints {
            attrs.ints.insert(k.to_string(), *v);
        }
        for (k, v) in lists {
            attrs.int_lists.insert(k.to_string(), v.clone());
        }
        Node {
            op: OpKind::Unknown(op.into()),
            name: "n".into(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec!["y".into()],
            attrs,
        }
    }

    fn run(op: &dyn Operator, node: &Node, weights: &HashMap<String, Tensor>, inputs: &[Option<&Tensor>]) -> Result<Tensor, OnnxError> {
        let ctx = OpContext { node, inputs: inputs.to_vec(), outer_scope: None, weights: Some(weights), registry: None };
        let mut out = op.execute(&ctx)?;
        Ok(out.remove(0))
    }

    struct Case {
        k: usize,
        n: usize,
        bits: usize,
        block: usize,
        b: Vec<u8>,
        scales: Vec<f32>,
        zp: Vec<u8>,
    }

    fn case(rng: &mut Lcg, k: usize, n: usize, bits: usize, block: usize) -> Case {
        let kb = k.div_ceil(block);
        let b = (0..n * kb * block * bits / 8).map(|_| rng.next() as u8).collect();
        let scales = (0..n * kb).map(|_| 0.01 + 0.04 * rng.f().abs()).collect();
        let zp = (0..n * (kb * bits).div_ceil(8)).map(|_| rng.next() as u8).collect();
        Case { k, n, bits, block, b, scales, zp }
    }

    /// `A @ W` with W dequantized per the ORT definition, in f64.
    fn reference(c: &Case, a: &[f32], m: usize, zp_float: Option<&[f32]>) -> Vec<f64> {
        let kb = c.k.div_ceil(c.block);
        let blob = c.block * c.bits / 8;
        let w = |n: usize, k: usize| -> f64 {
            let b = k / c.block;
            let byte = c.b[n * kb * blob + if c.bits == 8 { k } else { k / 2 }];
            let q = if c.bits == 8 { byte } else if k % 2 == 0 { byte & 15 } else { byte >> 4 };
            let zrow = (kb * c.bits).div_ceil(8);
            let z = match zp_float {
                Some(z) => f64::from(z[n * kb + b]),
                None if c.bits == 8 => f64::from(c.zp[n * zrow + b]),
                None => f64::from((c.zp[n * zrow + b / 2] >> (4 * (b % 2))) & 15),
            };
            (f64::from(q) - z) * f64::from(c.scales[n * kb + b])
        };
        let mut out = vec![0f64; m * c.n];
        for i in 0..m {
            for n in 0..c.n {
                out[i * c.n + n] = (0..c.k).map(|k| f64::from(a[i * c.k + k]) * w(n, k)).sum();
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f64], what: &str) {
        let norm = (want.iter().map(|x| x * x).sum::<f64>() / want.len() as f64).sqrt();
        let err = (got.iter().zip(want).map(|(&g, w)| (f64::from(g) - w).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        assert!(err <= 0.02 * norm, "{what}: RMS error {err} vs RMS {norm}");
    }

    #[test]
    fn matmul_nbits_matches_the_dequantized_reference() {
        let mut rng = Lcg(3);
        for &(k, n, bits, block) in &[(100, 37, 4, 32), (64, 16, 8, 32), (40, 9, 8, 16), (300, 70, 4, 64), (33, 5, 4, 16)] {
            let c = case(&mut rng, k, n, bits, block);
            let kb = k.div_ceil(block);
            let blob = block * bits / 8;
            let zrow = (kb * bits).div_ceil(8);
            let (m0, m1) = (2, 3);
            let a = t((0..m0 * m1 * k).map(|_| rng.f()).collect(), &[m0, m1, k]);
            let bias = t((0..n).map(|_| rng.f()).collect(), &[n]);
            let want = reference(&c, &a.data, m0 * m1, None);
            let want_bias: Vec<f64> = want.iter().enumerate().map(|(i, w)| w + f64::from(bias.data[i % n])).collect();
            let attrs = [("K", k as i64), ("N", n as i64), ("bits", bits as i64), ("block_size", block as i64)];
            let scales = t(c.scales.clone(), &[n, kb]);
            let op = MatMulNBitsOp::new(false);
            // One f32 per byte (a hand-built graph) and loader byte tensors.
            for loader in [false, true] {
                let (b, zp) = if loader {
                    (byte_tensor(&c.b, &[n, kb, blob]), byte_tensor(&c.zp, &[n, zrow]))
                } else {
                    let f = |v: &[u8]| v.iter().map(|&x| f32::from(x)).collect();
                    (t(f(&c.b), &[n, kb, blob]), t(f(&c.zp), &[n, zrow]))
                };
                let lists: Vec<(&str, Vec<i64>)> = if loader {
                    vec![
                        (ATTR_U8_INPUTS, vec![1, 3]),
                        (ATTR_U8_LAST, vec![blob as i64, zrow as i64]),
                        (ATTR_DTYPES, vec![0, 2, 1, 2, 0, 1]),
                    ]
                } else {
                    vec![]
                };
                let nd = node("MatMulNBits", &["a", "b", "s", "z", "", "bias"], &attrs, &lists);
                let weights: HashMap<String, Tensor> =
                    [("b", b), ("s", scales.clone()), ("z", zp)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
                let (b, s, z) = (&weights["b"], &weights["s"], &weights["z"]);
                let what = format!("k={k} n={n} bits={bits} block={block} loader={loader}");
                // Twice: the second run uses the cached repack.
                for _ in 0..2 {
                    let y = run(&op, &nd, &weights, &[Some(&a), Some(b), Some(s), Some(z), None, Some(&bias)]).unwrap();
                    assert_eq!(y.shape, vec![m0, m1, n]);
                    assert_close(&y.data, &want_bias, &what);
                }
                let y_nobias = run(&op, &nd, &weights, &[Some(&a), Some(b), Some(s), Some(z), None, None]).unwrap();
                assert_close(&y_nobias.data, &want, &what);
                // The optimizer's prepacked form gives the same bits.
                let packed = pack_matmul_nbits(&nd.attrs, b, s, Some(z)).unwrap();
                let pw: HashMap<String, Tensor> = [("p".to_string(), t(packed.clone(), &[packed.len()]))].into();
                let nd2 = node("OxiMatMulNBits", &["a", "p", "bias"], &[], &[]);
                let y2 = run(&MatMulNBitsOp::new(true), &nd2, &pw, &[Some(&a), Some(&pw["p"]), Some(&bias)]).unwrap();
                let y1 = run(&op, &nd, &weights, &[Some(&a), Some(b), Some(s), Some(z), None, Some(&bias)]).unwrap();
                assert_eq!(y1.data, y2.data, "{what}");
            }
        }
    }

    #[test]
    fn matmul_nbits_default_and_float_zero_points() {
        let mut rng = Lcg(5);
        let c = case(&mut rng, 70, 20, 4, 32);
        let kb = 3;
        let a = t((0..4 * 70).map(|_| rng.f()).collect(), &[4, 70]);
        let attrs = [("K", 70), ("N", 20), ("bits", 4), ("block_size", 32)];
        let b = t(c.b.iter().map(|&x| f32::from(x)).collect(), &[20, kb, 16]);
        let s = t(c.scales.clone(), &[20, kb]);
        let op = MatMulNBitsOp::new(false);
        let no_weights = HashMap::new();
        // Default zero point 8.
        let nd = node("MatMulNBits", &["a", "b", "s"], &attrs, &[]);
        let y = run(&op, &nd, &no_weights, &[Some(&a), Some(&b), Some(&s)]).unwrap();
        assert_close(&y.data, &reference(&c, &a.data, 4, Some(&[8.0; 60])), "default zp");
        // Float zero points, by declared dtype and by shape alone.
        let zf: Vec<f32> = (0..20 * kb).map(|i| (i % 16) as f32 + 0.25).collect();
        let z = t(zf.clone(), &[20, kb]);
        for lists in [vec![(ATTR_DTYPES, vec![0, 2, 1, 1])], vec![]] {
            let nd = node("MatMulNBits", &["a", "b", "s", "z"], &attrs, &lists);
            let y = run(&op, &nd, &no_weights, &[Some(&a), Some(&b), Some(&s), Some(&z)]).unwrap();
            assert_close(&y.data, &reference(&c, &a.data, 4, Some(&zf)), "float zp");
        }
    }

    #[test]
    fn matmul_nbits_rejects_unsupported_inputs() {
        let a = t(vec![0.0; 32], &[1, 32]);
        let b = t(vec![0.0; 4 * 16], &[4, 1, 16]);
        let s = t(vec![1.0; 4], &[4, 1]);
        let g = t(vec![0.0; 32], &[32]);
        let op = MatMulNBitsOp::new(false);
        let w = HashMap::new();
        let attrs = [("K", 32), ("N", 4), ("bits", 4), ("block_size", 32)];
        let err = |nd: &Node, inputs: &[Option<&Tensor>]| run(&op, nd, &w, inputs).unwrap_err().to_string();
        let nd = node("MatMulNBits", &["a", "b", "s", "", "g"], &attrs, &[]);
        assert!(err(&nd, &[Some(&a), Some(&b), Some(&s), None, Some(&g)]).contains("g_idx"));
        let nd = node("MatMulNBits", &["a", "b", "s"], &attrs, &[(ATTR_DTYPES, vec![0, 2, 10])]);
        assert!(err(&nd, &[Some(&a), Some(&b), Some(&s)]).contains("float16"));
        let nd = node("MatMulNBits", &["a", "b", "s"], &attrs, &[(ATTR_DTYPES, vec![0, 3, 1])]);
        assert!(err(&nd, &[Some(&a), Some(&b), Some(&s)]).contains("uint8"));
        let nd = node("MatMulNBits", &["a", "b", "s"], &[("K", 32), ("N", 4), ("bits", 2), ("block_size", 32)], &[]);
        assert!(err(&nd, &[Some(&a), Some(&b), Some(&s)]).contains("bits=2"));
        let nd = node("MatMulNBits", &["a", "b", "s"], &attrs, &[]);
        let a_bad = t(vec![0.0; 31], &[1, 31]);
        assert!(err(&nd, &[Some(&a_bad), Some(&b), Some(&s)]).contains("K=32"));
        let b_bad = t(vec![300.0; 4 * 16], &[4, 1, 16]);
        assert!(err(&nd, &[Some(&a), Some(&b_bad), Some(&s)]).contains("uint8"));
    }

    #[test]
    fn gather_block_quantized_dequantizes_rows() {
        let mut rng = Lcg(9);
        // 5 rows x 40 4-bit values (20 bytes), blocks of 16 (3 per row, the last partial).
        let (rows, cols, block) = (5usize, 40usize, 16usize);
        let data: Vec<u8> = (0..rows * 20).map(|_| rng.next() as u8).collect();
        let scales: Vec<f32> = (0..rows * 3).map(|_| rng.f()).collect();
        let zp: Vec<u8> = (0..rows * 2).map(|_| rng.next() as u8).collect();
        let want_row = |r: usize| -> Vec<f32> {
            (0..cols)
                .map(|i| {
                    let byte = data[r * 20 + i / 2];
                    let q = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let b = i / block;
                    let zb = zp[r * 2 + b / 2];
                    let z = if b % 2 == 0 { zb & 15 } else { zb >> 4 };
                    (f32::from(q) - f32::from(z)) * scales[r * 3 + b]
                })
                .collect()
        };
        let idx = t(vec![1.0, 4.0, -1.0, 7.0], &[2, 2]);
        let attrs = [("bits", 4), ("block_size", 16), ("gather_axis", 0), ("quantize_axis", 1)];
        let op = GatherBlockQuantizedOp::default();
        let s = t(scales.clone(), &[rows, 3]);
        for loader in [false, true] {
            let (d, z, lists) = if loader {
                (
                    byte_tensor(&data, &[rows, 20]),
                    byte_tensor(&zp, &[rows, 2]),
                    vec![(ATTR_U8_INPUTS, vec![0, 3]), (ATTR_U8_LAST, vec![20, 2]), (ATTR_DTYPES, vec![2, 0, 1, 2])],
                )
            } else {
                let f = |v: &[u8]| v.iter().map(|&x| f32::from(x)).collect();
                (t(f(&data), &[rows, 20]), t(f(&zp), &[rows, 2]), vec![])
            };
            let nd = node("GatherBlockQuantized", &["d", "i", "s", "z"], &attrs, &lists);
            let w: HashMap<String, Tensor> = [("d".to_string(), d)].into();
            let y = run(&op, &nd, &w, &[Some(&w["d"]), Some(&idx), Some(&s), Some(&z)]).unwrap();
            assert_eq!(y.shape, vec![2, 2, cols]);
            let mut want = want_row(1);
            want.extend(want_row(4));
            want.extend(want_row(4));
            want.extend(vec![0.0; cols]);
            assert_eq!(y.data, want, "loader={loader}");
        }
        // 8-bit with the default zero point 128.
        let d8: Vec<u8> = (0..2 * 32).map(|_| rng.next() as u8).collect();
        let s8 = t(vec![0.5, 2.0], &[2, 1]);
        let nd = node("GatherBlockQuantized", &["d", "i", "s"], &[("bits", 8), ("block_size", 32)], &[]);
        let d = t(d8.iter().map(|&x| f32::from(x)).collect(), &[2, 32]);
        let y = run(&op, &nd, &HashMap::new(), &[Some(&d), Some(&t(vec![1.0], &[1])), Some(&s8)]).unwrap();
        let want: Vec<f32> = d8[32..].iter().map(|&q| (f32::from(q) - 128.0) * 2.0).collect();
        assert_eq!(y.data, want);
        let nd = node("GatherBlockQuantized", &["d", "i", "s"], &[("bits", 8), ("block_size", 32), ("gather_axis", 1)], &[]);
        assert!(run(&op, &nd, &HashMap::new(), &[Some(&d), Some(&t(vec![1.0], &[1])), Some(&s8)]).is_err());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fused_ops_take_quantized_weights() {
        use crate::registry::contrib_ops::{OxiMatMulActOp, OxiMatMulSplitOp};
        let mut rng = Lcg(21);
        let gelu = |x: f32| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh());
        // K = 80 (a partial block), N = 2 x 64 (gated) and 96 (split 32 + 64).
        for &(n, bits) in &[(128usize, 4usize), (128, 8), (96, 8)] {
            let (k, block, m) = (80usize, 32usize, 7usize);
            let c = case(&mut rng, k, n, bits, block);
            let kb = k.div_ceil(block);
            let attrs = [("K", k as i64), ("N", n as i64), ("bits", bits as i64), ("block_size", block as i64)];
            let nd = node("MatMulNBits", &["a", "b", "s", "z"], &attrs, &[]);
            let f = |v: &[u8]| v.iter().map(|&x| f32::from(x)).collect();
            let b = t(f(&c.b), &[n, kb, block * bits / 8]);
            let z = t(f(&c.zp), &[n, (kb * bits).div_ceil(8)]);
            let packed = pack_matmul_nbits(&nd.attrs, &b, &t(c.scales.clone(), &[n, kb]), Some(&z)).unwrap();
            let pw: HashMap<String, Tensor> = [("p".to_string(), t(packed.clone(), &[packed.len()]))].into();
            let a = t((0..m * k).map(|_| rng.f()).collect(), &[1, m, k]);
            let p = &pw["p"];
            let y = run(&MatMulNBitsOp::new(true), &node("OxiMatMulNBits", &["a", "p"], &[], &[]), &pw, &[Some(&a), Some(p)]).unwrap();
            let close = |got: &[f32], want: &[f32], what: &str| {
                assert_eq!(got.len(), want.len(), "{what}");
                for (g, w) in got.iter().zip(want) {
                    assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "{what}: {g} vs {w}");
                }
            };
            if n == 128 {
                let silu = |x: f32| x / (1.0 + (-x).exp());
                for (gate, activation) in [(0i64, "gelu"), (1, "gelu"), (0, "silu"), (1, "silu")] {
                    let gelu = |x: f32| if activation == "silu" { silu(x) } else { gelu(x) };
                    let mut nd = node("OxiMatMulAct", &["a", "p"], &[("gate", gate), ("nbits", 1)], &[]);
                    nd.attrs.strings.insert("activation".into(), activation.into());
                    nd.attrs.strings.insert("approximate".into(), "tanh".into());
                    let got = run(&OxiMatMulActOp::default(), &nd, &pw, &[Some(&a), Some(p)]).unwrap();
                    assert_eq!(got.shape, vec![1, m, n / 2]);
                    let h = n / 2;
                    let want: Vec<f32> = (0..m * h)
                        .map(|i| {
                            let (r, j) = (i / h, i % h);
                            let (lo, hi) = (y.data[r * n + j], y.data[r * n + h + j]);
                            if gate == 0 { gelu(lo) * hi } else { gelu(hi) * lo }
                        })
                        .collect();
                    close(&got.data, &want, &format!("gated {activation} gate={gate} bits={bits}"));
                }
            } else {
                let mut nd = node("OxiMatMulSplit", &["a", "p"], &[("nbits", 1)], &[("split", vec![32, 64])]);
                nd.outputs = vec!["y0".into(), "y1".into()];
                let ctx = OpContext { node: &nd, inputs: vec![Some(&a), Some(p)], outer_scope: None, weights: Some(&pw), registry: None };
                let outs = OxiMatMulSplitOp::default().execute(&ctx).unwrap();
                assert_eq!((outs[0].shape.clone(), outs[1].shape.clone()), (vec![1, m, 32], vec![1, m, 64]));
                let col = |lo: usize, w: usize| -> Vec<f32> { (0..m).flat_map(|r| y.data[r * n + lo..r * n + lo + w].to_vec()).collect() };
                close(&outs[0].data, &col(0, 32), "split 0");
                close(&outs[1].data, &col(32, 64), "split 1");
            }
        }
    }
}
