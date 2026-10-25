//! Prototype (onnx_spike): `MatMul` by a constant weight with a fused
//! elementwise epilogue (see `OxiMatMulAct` in `registry::contrib_ops`):
//!
//! * plain: `out = act(x @ w + bias) [* mul]`, `out` is `[m, n]`;
//! * gated (`gate = Some(h)`): `w` is `[k, 2H]` whose columns split into halves
//!   `y0 = (x @ w + bias)[:, ..H]` and `y1 = ...[:, H..]`, and
//!   `out = act(y_h) * y_(1-h)`, `out` is `[m, H]` — the
//!   `Split -> Gelu -> Mul` of gated MLPs (GeGLU).
//!
//! With the packed GEMM the epilogue runs on each `rows x width` block of the
//! product while it is in L1, so the full `[m, n]` product is never written.
//! For the gated form the weight is packed with the halves interleaved in
//! blocks, so every block holds matching columns of both halves.
//!
//! [`SplitSpec`] is the projection form (`OxiMatMulSplit`): `x @ w + bias`, an
//! optional per-head RMSNorm and column scale, split along the columns into
//! several outputs, some of them rotated (`com.microsoft::RotaryEmbedding`) —
//! the fused QKV projection of Gemma-style attention, or plain `MatMul -> Split`.

#![allow(unsafe_code)]

use crate::gelu;

/// The activation applied by the epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// tanh-approximation GELU.
    GeluTanh,
    /// Exact (erf) GELU.
    GeluErf,
}

/// What follows the MatMul.
#[derive(Clone, Copy)]
pub struct Spec<'a> {
    pub act: Act,
    /// Gated form: index (0 or 1) of the half the activation applies to.
    pub gate: Option<usize>,
    /// `[n]` bias in the weight's column order.
    pub bias: Option<&'a [f32]>,
    /// Plain form only: elementwise multiplier with the output's shape.
    pub mul: Option<&'a [f32]>,
}

/// Columns per epilogue block: for the gated form the half-block of each half
/// (the block is twice as wide), else the block. `None` if `n` has no suitable
/// divisor (callers then use [`matmul_act_unfused`]).
pub fn block_cols(n: usize, gated: bool) -> Option<usize> {
    if gated {
        if n % 2 != 0 {
            return None;
        }
        [128, 64, 32, 16].into_iter().find(|&hb| (n / 2) % hb == 0)
    } else {
        [256, 128, 64, 32].into_iter().find(|&w| n % w == 0)
    }
}

/// Packed-column order for the gated form: block `s` holds half-block `s` of
/// the activated half, then half-block `s` of the other half.
pub fn gate_columns(n: usize, act_half: usize, hb: usize) -> Vec<usize> {
    let h = n / 2;
    let (a0, b0) = if act_half == 0 { (0, h) } else { (h, 0) };
    (0..h / hb).flat_map(|s| (a0 + s * hb..a0 + (s + 1) * hb).chain(b0 + s * hb..b0 + (s + 1) * hb)).collect()
}

/// `out[j] = act(x[j] + xb[j]) * (g[j] + gb[j])` (each bias and the multiplier
/// optional). `x` and `g` are scratch and may be modified.
fn finish(act: Act, x: &mut [f32], xb: Option<&[f32]>, g: Option<&mut [f32]>, gb: Option<&[f32]>, out: &mut [f32]) {
    if let Some(b) = xb {
        x.iter_mut().zip(b).for_each(|(v, b)| *v += b);
    }
    let g = g.map(|g| {
        if let Some(b) = gb {
            g.iter_mut().zip(b).for_each(|(v, b)| *v += b);
        }
        &*g
    });
    match act {
        Act::GeluTanh => gelu::gelu_mul(x, g, out),
        Act::GeluErf => {
            for (j, o) in out.iter_mut().enumerate() {
                let a = crate::registry::nn_ops::gelu_exact(x[j]);
                *o = g.map_or(a, |g| a * g[j]);
            }
        }
    }
}

/// Reference path: the full product, then the epilogue row by row.
/// `product` is `x @ w` (`[m, n]`, consumed as scratch).
pub fn matmul_act_unfused(m: usize, n: usize, product: &mut [f32], spec: &Spec<'_>, out: &mut [f32]) {
    let n_out = if spec.gate.is_some() { n / 2 } else { n };
    assert_eq!(product.len(), m * n);
    assert_eq!(out.len(), m * n_out);
    let rows = product.chunks_mut(n).zip(out.chunks_mut(n_out.max(1)));
    for (i, (row, o)) in rows.enumerate() {
        match spec.gate {
            Some(ah) => {
                let h = n / 2;
                let (lo, hi) = row.split_at_mut(h);
                let (x, g) = if ah == 0 { (lo, hi) } else { (hi, lo) };
                let (xb, gb) = match spec.bias {
                    Some(b) if ah == 0 => (Some(&b[..h]), Some(&b[h..])),
                    Some(b) => (Some(&b[h..]), Some(&b[..h])),
                    None => (None, None),
                };
                finish(spec.act, x, xb, Some(g), gb, o);
            }
            None => {
                let mut mul = spec.mul.map(|mul| mul[i * n..(i + 1) * n].to_vec());
                finish(spec.act, row, spec.bias, mul.as_deref_mut(), None, o);
            }
        }
    }
}

/// Per-head RMSNorm of the split form: `x * gamma / sqrt(mean(x^2) + eps)` over
/// each `gamma.len()`-wide head.
#[derive(Clone, Copy)]
pub struct HeadNorm<'a> {
    pub gamma: &'a [f32],
    pub eps: f32,
}

/// Rotary embedding of the split form's rotated outputs.
#[derive(Clone, Copy)]
pub struct SplitRotary<'a> {
    pub cache: crate::contrib::RotaryCache<'a>,
    /// Tokens per batch item (rows of the product are `(b, s)` pairs).
    pub seq: usize,
    pub rotary_dim: usize,
    pub interleaved: bool,
}

/// The split (projection) form; see the module docs.
pub struct SplitSpec<'a> {
    /// Output column counts, summing to the weight's `n`.
    pub splits: Vec<usize>,
    pub bias: Option<&'a [f32]>,
    pub norm: Option<HeadNorm<'a>>,
    pub col_scale: Option<&'a [f32]>,
    pub rotary: Option<SplitRotary<'a>>,
    /// Per output: apply `rotary` (its heads are then `rotary`'s head size wide).
    pub rotate: Vec<bool>,
}

impl SplitSpec<'_> {
    /// Column block that never straddles an output or a head: the head width
    /// when there is a norm or rotary, else the largest common divisor of the
    /// splits. `head` is the rotary head size, if any.
    pub fn block(&self, head: Option<usize>) -> Option<usize> {
        let d = self.norm.map(|n| n.gamma.len()).or(head);
        let gcd = |mut a: usize, mut b: usize| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let g = self.splits.iter().fold(0, |a, &b| gcd(a, b));
        match d {
            Some(d) if d > 0 && g % d == 0 && head.is_none_or(|h| h == d) => Some(d),
            Some(_) => None,
            None => (g > 0).then_some(g),
        }
    }

    /// (output, first column of the output) holding global column `j`.
    fn output_of(&self, j: usize) -> (usize, usize) {
        let mut start = 0;
        for (o, &w) in self.splits.iter().enumerate() {
            if j < start + w {
                return (o, start);
            }
            start += w;
        }
        unreachable!("column {j} outside the splits");
    }

    /// Finish columns `j0..j0 + v.len()` of product row `i` (one block; `v` is
    /// scratch) and write them to `out`, the matching slice of their output.
    fn finish(&self, i: usize, j0: usize, rotate: bool, v: &mut [f32], out: &mut [f32]) {
        let w = v.len();
        if let Some(b) = self.bias {
            v.iter_mut().zip(&b[j0..j0 + w]).for_each(|(x, b)| *x += b);
        }
        if let Some(HeadNorm { gamma, eps }) = self.norm {
            let mut lanes = [0.0f32; 8];
            let mut chunks = v.chunks_exact(8);
            for c in &mut chunks {
                for (l, &x) in lanes.iter_mut().zip(c) {
                    *l += x * x;
                }
            }
            let tail: f32 = chunks.remainder().iter().map(|&x| x * x).sum();
            let inv = ((lanes.iter().sum::<f32>() + tail) / w as f32 + eps).sqrt().recip();
            v.iter_mut().zip(gamma).for_each(|(x, g)| *x = *x * inv * g);
        }
        if let Some(cs) = self.col_scale {
            v.iter_mut().zip(&cs[j0..j0 + w]).for_each(|(x, c)| *x *= c);
        }
        match self.rotary.filter(|_| rotate) {
            Some(r) => {
                let (c, sn) = r.cache.row(i / r.seq, i % r.seq);
                crate::contrib::rotate_head(v, c, sn, r.rotary_dim, r.interleaved, out);
            }
            None => out.copy_from_slice(v),
        }
    }
}

/// Reference path of the split form: `product` is `x @ w` (`[m, n]`, used as
/// scratch); `outs[o]` is `[m, splits[o]]`; `block` from [`SplitSpec::block`].
pub fn matmul_split_unfused(m: usize, product: &mut [f32], spec: &SplitSpec<'_>, block: usize, outs: &mut [&mut [f32]]) {
    let n: usize = spec.splits.iter().sum();
    assert_eq!(product.len(), m * n);
    for (i, row) in product.chunks_mut(n.max(1)).enumerate().take(m) {
        for (bi, v) in row.chunks_mut(block).enumerate() {
            let j0 = bi * block;
            let (o, start) = spec.output_of(j0);
            let w = spec.splits[o];
            let dst = &mut outs[o][i * w + j0 - start..i * w + j0 - start + block];
            spec.finish(i, j0, spec.rotate[o], v, dst);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use packed::{matmul_act_nbits, matmul_act_packed, matmul_split_nbits, matmul_split_packed};

#[cfg(target_arch = "x86_64")]
mod packed {
    use super::{finish, Spec, SplitSpec};
    use crate::gemm_packed::{gemm_epilogue, Epilogue, PackedB};
    use crate::matmul_nbits::{isa as nbits_isa, matmul_epilogue, Packed};

    struct Out(*mut f32);
    unsafe impl Send for Out {}
    unsafe impl Sync for Out {}

    struct ActEpilogue<'a> {
        spec: Spec<'a>,
        n: usize,
        n_out: usize,
        width: usize,
        out: Out,
    }

    impl Epilogue for ActEpilogue<'_> {
        fn width(&self) -> usize {
            self.width
        }

        unsafe fn apply(&self, i0: usize, rows: usize, j0: usize, block: &mut [f32]) {
            let w = self.width;
            for (r, row) in block.chunks_mut(w).take(rows).enumerate() {
                let i = i0 + r;
                match self.spec.gate {
                    Some(ah) => {
                        let (hb, h, s) = (w / 2, self.n / 2, j0 / w);
                        let (x, g) = row.split_at_mut(hb);
                        let (xc, gc) = if ah == 0 { (s * hb, h + s * hb) } else { (h + s * hb, s * hb) };
                        let xb = self.spec.bias.map(|b| &b[xc..xc + hb]);
                        let gb = self.spec.bias.map(|b| &b[gc..gc + hb]);
                        // SAFETY: row i, columns s*hb..(s+1)*hb of the [m, H] output belong to this block only.
                        let o = std::slice::from_raw_parts_mut(self.out.0.add(i * self.n_out + s * hb), hb);
                        finish(self.spec.act, x, xb, Some(g), gb, o);
                    }
                    None => {
                        let xb = self.spec.bias.map(|b| &b[j0..j0 + w]);
                        let mut mul: [f32; 256] = [0.0; 256];
                        let g = self.spec.mul.map(|m| {
                            mul[..w].copy_from_slice(&m[i * self.n + j0..i * self.n + j0 + w]);
                            &mut mul[..w]
                        });
                        // SAFETY: as above, for columns j0..j0+w of the [m, n] output.
                        let o = std::slice::from_raw_parts_mut(self.out.0.add(i * self.n_out + j0), w);
                        finish(self.spec.act, row, xb, g, None, o);
                    }
                }
            }
        }
    }

    /// `x` (`[m, k]`) @ the packed weight with the epilogue `spec`, into `out`.
    /// `b` must be packed with [`super::gate_columns`] for the gated form, and
    /// `width` must come from [`super::block_cols`] (doubled when gated).
    pub(crate) fn matmul_act_packed(m: usize, x: &[f32], b: &PackedB, width: usize, spec: &Spec<'_>, out: &mut [f32], threads: usize) {
        let n_out = if spec.gate.is_some() { b.n / 2 } else { b.n };
        assert_eq!(out.len(), m * n_out);
        assert!(width <= 256 && spec.mul.map_or(true, |mul| mul.len() == m * b.n));
        let epi = ActEpilogue { spec: *spec, n: b.n, n_out, width, out: Out(out.as_mut_ptr()) };
        gemm_epilogue(m, x, b, &epi, threads);
    }

    /// [`matmul_act_packed`] for a block-quantized weight (unpermuted: the
    /// gated halves are addressed per block instead of interleaved).
    pub(crate) fn matmul_act_nbits(m: usize, x: &[f32], w: &Packed<'_>, width: usize, spec: &Spec<'_>, out: &mut [f32], threads: usize) {
        let n = w.s.n;
        let n_out = if spec.gate.is_some() { n / 2 } else { n };
        assert_eq!(out.len(), m * n_out);
        assert!(width <= 256 && spec.mul.map_or(true, |mul| mul.len() == m * n));
        let epi = ActEpilogue { spec: *spec, n, n_out, width, out: Out(out.as_mut_ptr()) };
        let gate = spec.gate;
        let segs = move |u: usize| match gate {
            Some(ah) => {
                let (hb, h) = (width / 2, n / 2);
                let (a0, b0) = if ah == 0 { (0, h) } else { (h, 0) };
                [(a0 + u * hb, hb), (b0 + u * hb, hb)]
            }
            None => [(u * width, width), (0, 0)],
        };
        matmul_epilogue(nbits_isa(), m, x, w, &epi, segs, threads);
    }

    /// [`matmul_split_packed`] for a block-quantized weight; `block` a multiple of 16.
    pub(crate) fn matmul_split_nbits(m: usize, x: &[f32], w: &Packed<'_>, spec: &SplitSpec<'_>, block: usize, outs: &mut [&mut [f32]], threads: usize) {
        assert_eq!(spec.splits.iter().sum::<usize>(), w.s.n);
        assert!(outs.iter().zip(&spec.splits).all(|(o, &ow)| o.len() == m * ow));
        let epi = SplitEpilogue { spec, block, outs: outs.iter_mut().map(|o| Out(o.as_mut_ptr())).collect() };
        matmul_epilogue(nbits_isa(), m, x, w, &epi, |u| [(u * block, block), (0, 0)], threads);
    }

    struct SplitEpilogue<'a, 'b> {
        spec: &'a SplitSpec<'b>,
        block: usize,
        outs: Vec<Out>,
    }

    impl Epilogue for SplitEpilogue<'_, '_> {
        fn width(&self) -> usize {
            self.block
        }

        unsafe fn apply(&self, i0: usize, rows: usize, j0: usize, block: &mut [f32]) {
            let (o, start) = self.spec.output_of(j0);
            let w = self.spec.splits[o];
            for (r, v) in block.chunks_mut(self.block).take(rows).enumerate() {
                let i = i0 + r;
                // SAFETY: row i, columns j0-start.. of output o belong to this block only.
                let dst = std::slice::from_raw_parts_mut(self.outs[o].0.add(i * w + j0 - start), self.block);
                self.spec.finish(i, j0, self.spec.rotate[o], v, dst);
            }
        }
    }

    /// The split form with the packed GEMM; `block` from [`SplitSpec::block`],
    /// a multiple of 32.
    pub(crate) fn matmul_split_packed(m: usize, x: &[f32], b: &PackedB, spec: &SplitSpec<'_>, block: usize, outs: &mut [&mut [f32]], threads: usize) {
        assert_eq!(spec.splits.iter().sum::<usize>(), b.n);
        assert!(outs.iter().zip(&spec.splits).all(|(o, &w)| o.len() == m * w));
        let epi = SplitEpilogue { spec, block, outs: outs.iter_mut().map(|o| Out(o.as_mut_ptr())).collect() };
        gemm_epilogue(m, x, b, &epi, threads);
    }
}


#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::gemm_bf16::tests::round;
    use crate::gemm_packed::{self, Isa, PackedB};

    /// The packings to test: the CPU's FP32 kernel, then BF16 weights (its
    /// reference product uses BF16-rounded inputs).
    fn isas() -> [Isa; 2] {
        [gemm_packed::isa().unwrap(), Isa::Avx512Bf16]
    }

    fn rounded(isa: Isa, v: &[f32]) -> Vec<f32> {
        if isa == Isa::Avx512Bf16 { round(v) } else { v.to_vec() }
    }

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn packed_epilogue_matches_unfused() {
        if !gemm_packed::available() {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for &(m, k, n) in &[(1, 8, 64), (13, 300, 96), (40, 520, 256), (25, 64, 32)] {
            let x = rand(m * k, 1);
            let w = rand(k * n, 2);
            let bias = rand(n, 3);
            for act in [Act::GeluTanh, Act::GeluErf] {
                for gate in [None, Some(0), Some(1)] {
                    for with_bias in [false, true] {
                        let n_out = if gate.is_some() { n / 2 } else { n };
                        let mul = rand(m * n, 4);
                        let spec = Spec {
                            act,
                            gate,
                            bias: with_bias.then_some(&bias[..]),
                            mul: (gate.is_none() && with_bias).then_some(&mul[..]),
                        };
                        for isa in isas() {
                            let mut want = vec![0f32; m * n_out];
                            let product = &mut naive(m, k, n, &rounded(isa, &x), &rounded(isa, &w));
                            matmul_act_unfused(m, n, product, &spec, &mut want);
                            let hb = block_cols(n, gate.is_some()).unwrap();
                            let (b, width) = match gate {
                                Some(ah) => (PackedB::pack_cols_for(isa, &w, k, n, &gate_columns(n, ah, hb)), 2 * hb),
                                None => (PackedB::pack_for(isa, &w, k, n), hb),
                            };
                            for threads in [1, 4] {
                                let mut got = vec![f32::NAN; m * n_out];
                                pool.install(|| matmul_act_packed(m, &x, &b, width, &spec, &mut got, threads));
                                for (g, w) in got.iter().zip(&want) {
                                    assert!(
                                        (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                                        "{isa:?} m={m} k={k} n={n} {act:?} gate={gate:?}: {g} vs {w}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_split_epilogue_matches_unfused() {
        use crate::contrib::RotaryCache;
        if !gemm_packed::available() {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        // (batch, seq, k, head D, splits in heads)
        for &(b, s, k, d, heads) in &[(2, 5, 40, 32, &[2usize, 1, 1][..]), (1, 13, 520, 64, &[4, 2, 2][..]), (3, 2, 16, 96, &[1, 1][..])] {
            let m = b * s;
            let splits: Vec<usize> = heads.iter().map(|h| h * d).collect();
            let n: usize = splits.iter().sum();
            let (x, w, bias, gamma, cs) = (rand(m * k, 1), rand(k * n, 2), rand(n, 3), rand(d, 4), rand(n, 5));
            let (cos, sin) = (rand(s * d / 2, 6), rand(s * d / 2, 7));
            let pos = [0.0f32];
            for variant in 0..4 {
                let rotary = (variant >= 2).then(|| SplitRotary {
                    cache: RotaryCache::new(&pos, &cos, &sin, false, b, s, d / 2).unwrap(),
                    seq: s,
                    rotary_dim: d,
                    interleaved: variant == 3,
                });
                let spec = SplitSpec {
                    splits: splits.clone(),
                    bias: (variant % 2 == 1).then_some(&bias[..]),
                    norm: (variant >= 1).then_some(HeadNorm { gamma: &gamma, eps: 1e-6 }),
                    col_scale: (variant >= 1).then_some(&cs[..]),
                    rotary,
                    rotate: (0..splits.len()).map(|o| o < 2).collect(),
                };
                let block = spec.block(rotary.map(|_| d)).unwrap();
                for isa in isas() {
                    let mut want: Vec<Vec<f32>> = splits.iter().map(|&w| vec![0f32; m * w]).collect();
                    {
                        let mut outs: Vec<&mut [f32]> = want.iter_mut().map(|v| &mut v[..]).collect();
                        let product = &mut naive(m, k, n, &rounded(isa, &x), &rounded(isa, &w));
                        matmul_split_unfused(m, product, &spec, block, &mut outs);
                    }
                    let pb = PackedB::pack_for(isa, &w, k, n);
                    for threads in [1, 4] {
                        let mut got: Vec<Vec<f32>> = splits.iter().map(|&w| vec![f32::NAN; m * w]).collect();
                        {
                            let mut outs: Vec<&mut [f32]> = got.iter_mut().map(|v| &mut v[..]).collect();
                            pool.install(|| matmul_split_packed(m, &x, &pb, &spec, block, &mut outs, threads));
                        }
                        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
                            assert!(
                                (g - w).abs() <= 2e-4 * (1.0 + w.abs()),
                                "{isa:?} b={b} s={s} d={d} variant={variant}: {g} vs {w}"
                            );
                        }
                    }
                }
            }
        }
    }
}
