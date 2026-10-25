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

#[cfg(target_arch = "x86_64")]
pub(crate) use packed::matmul_act_packed;

#[cfg(target_arch = "x86_64")]
mod packed {
    use super::{finish, Spec};
    use crate::gemm_packed::{gemm_epilogue, Epilogue, PackedB};

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
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::gemm_packed::{self, PackedB};

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
                        let mut want = vec![0f32; m * n_out];
                        matmul_act_unfused(m, n, &mut naive(m, k, n, &x, &w), &spec, &mut want);
                        let hb = block_cols(n, gate.is_some()).unwrap();
                        let (b, width) = match gate {
                            Some(ah) => (PackedB::pack_cols(&w, k, n, &gate_columns(n, ah, hb)), 2 * hb),
                            None => (PackedB::pack(&w, k, n), hb),
                        };
                        for threads in [1, 4] {
                            let mut got = vec![f32::NAN; m * n_out];
                            pool.install(|| matmul_act_packed(m, &x, &b, width, &spec, &mut got, threads));
                            for (g, w) in got.iter().zip(&want) {
                                assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "m={m} k={k} n={n} {act:?} gate={gate:?}: {g} vs {w}");
                            }
                        }
                    }
                }
            }
        }
    }
}
