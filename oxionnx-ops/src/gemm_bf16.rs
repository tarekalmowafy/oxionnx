//! Opt-in BF16 weights for the packed GEMM ([`crate::gemm_packed`]).
//!
//! With `SessionBuilder::with_bf16_weights(true)` (or `OXIONNX_BF16_WEIGHTS=1`)
//! the constant FP32 MatMul weights are packed as BF16 instead of FP32, and the
//! GEMM runs on AVX-512 BF16 dot products (`vdpbf16ps`) with FP32 accumulation:
//! the weights take half the memory and half the bandwidth, and each
//! instruction does two multiply-adds per lane.
//!
//! Layout ([`crate::gemm_packed::Isa::Avx512Bf16`]): the same KC-row blocks of
//! 32-column panels as the FP32 packing, but each 32-bit slot holds a K pair,
//! `B[2p, j]` in the low half and `B[2p + 1, j]` in the high half, as
//! `vdpbf16ps` consumes it (an odd last row is paired with a zero). The
//! activations are converted to BF16 on the fly (`vcvtne2ps2bf16`, round to
//! nearest even); a row-major BF16 row is already K-pair interleaved, so each
//! kernel step broadcasts one 32-bit pair of A per row. This is what llama.cpp
//! does for its BF16 matmuls.
//!
//! The BF16 path is used only when the CPU has `avx512f` + `avx512bf16`
//! ([`hw_available`]); elsewhere the weights stay FP32, bit-identical to the
//! default (a widen-to-FP32 BF16 kernel would save bandwidth on AVX2 machines
//! but cost a convert per FMA, and the doc-batch GEMMs are compute bound). The
//! scalar kernel here only backs the unit tests on such machines.

#![allow(unsafe_code)]

use crate::gemm_packed::{Kernel, KC};
use oxionnx_core::OperatorRegistry;
use std::sync::OnceLock;

/// Environment variable that enables BF16 weights for sessions that do not set
/// the option explicitly.
pub const ENV_VAR: &str = "OXIONNX_BF16_WEIGHTS";

/// `OXIONNX_BF16_WEIGHTS` is set to `1` / `true` / `on` / `yes`.
pub fn env_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var(ENV_VAR).is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
    })
}

/// The CPU runs the AVX-512 BF16 kernel.
pub fn hw_available() -> bool {
    static HW: OnceLock<bool> = OnceLock::new();
    *HW.get_or_init(|| is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16"))
}

/// Whether the session bound to `registry` packs its weights as BF16: the
/// session option if set, else [`ENV_VAR`]; always `false` without
/// [`hw_available`].
pub fn weights_enabled(registry: Option<&OperatorRegistry>) -> bool {
    registry.and_then(OperatorRegistry::bf16_weights).unwrap_or_else(env_enabled) && hw_available()
}

/// FP32 -> BF16, round to nearest even (NaNs stay quiet NaNs).
#[inline]
pub fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x40;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[inline]
pub fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn pair(lo: f32, hi: f32) -> u32 {
    to_bf16(lo) as u32 | (to_bf16(hi) as u32) << 16
}

/// Pack the `k x n` matrix `B[p, j] = src(p, j)` in the K-pair layout, `nr`
/// columns per panel. Returns the buffer (u32 pairs stored as f32 bits) and
/// the offset of its 64-byte aligned start.
pub(crate) fn pack(k: usize, n: usize, nr: usize, src: impl Fn(usize, usize) -> f32) -> (Vec<f32>, usize) {
    let panels = n.div_ceil(nr);
    let len = pairs_total(k) * panels * nr;
    let mut buf = vec![0f32; len + 16];
    let off = buf.as_ptr().align_offset(64).min(16);
    // SAFETY: f32 and u32 have the same size and alignment.
    let dst: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().add(off).cast::<u32>(), len) };
    let mut base = 0;
    for k0 in (0..k).step_by(KC) {
        let kc = KC.min(k - k0);
        let kp = kc.div_ceil(2);
        for j in 0..panels {
            let j0 = j * nr;
            let nb = nr.min(n - j0);
            let panel = &mut dst[base + j * kp * nr..base + (j + 1) * kp * nr];
            for pp in 0..kp {
                let p = k0 + 2 * pp;
                let odd = p + 1 == k0 + kc;
                for jj in 0..nb {
                    let hi = if odd { 0.0 } else { src(p + 1, j0 + jj) };
                    panel[pp * nr + jj] = pair(src(p, j0 + jj), hi);
                }
            }
        }
        base += kp * panels * nr;
    }
    (buf, off)
}

/// Packed K pairs over all KC blocks of a K-row matrix (only the last block can be odd).
fn pairs_total(k: usize) -> usize {
    if k == 0 {
        0
    } else {
        (k - 1) / KC * (KC / 2) + (k - (k - 1) / KC * KC).div_ceil(2)
    }
}

/// Convert `rows` rows of `a` (row stride `lda`, `k` used columns) to BF16 pairs
/// in `dst` (row stride `k.div_ceil(2)`, an odd row end padded with zero).
fn convert_rows(a: *const f32, lda: usize, rows: usize, k: usize, dst: &mut [u32]) {
    let kp = k.div_ceil(2);
    debug_assert!(dst.len() >= rows * kp);
    if hw_available() {
        // SAFETY: the CPU has AVX-512 BF16; the caller guarantees `a` covers the rows.
        unsafe { avx512::convert_rows(a, lda, rows, k, dst.as_mut_ptr()) };
        return;
    }
    for r in 0..rows {
        // SAFETY: row r of `a` has `k` readable elements (caller contract).
        let src = unsafe { std::slice::from_raw_parts(a.add(r * lda), k) };
        let out = &mut dst[r * kp..(r + 1) * kp];
        for (o, c) in out.iter_mut().zip(src.chunks(2)) {
            *o = pair(c[0], c.get(1).copied().unwrap_or(0.0));
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::Cell<Vec<u32>> = const { std::cell::Cell::new(Vec::new()) };
}

/// Runs `f(a_bf16, row stride in pairs)` on `a` (`m x k`, row stride `lda`)
/// converted to BF16 pairs, in a per-thread scratch buffer (taken out of the
/// slot while in use, so re-entrant calls allocate their own).
///
/// # Safety
/// `a` must be valid for `m` rows of `k` elements at stride `lda`.
pub(crate) unsafe fn with_a<R>(m: usize, a: *const f32, lda: usize, k: usize, threads: usize, f: impl FnOnce(*const f32, usize) -> R) -> R {
    use rayon::prelude::*;
    let kp = k.div_ceil(2);
    let mut buf = SCRATCH.with(|s| s.take());
    if buf.len() < m * kp {
        buf.resize(m * kp, 0);
    }
    const PAR_MIN: usize = 1 << 17;
    if threads > 1 && m > 1 && m * k >= PAR_MIN {
        let rows = m.div_ceil(threads).max(PAR_MIN / k.max(1)).max(1);
        let ap = a as usize;
        buf[..m * kp].par_chunks_mut(rows * kp).enumerate().for_each(|(t, chunk)| {
            let i0 = t * rows;
            convert_rows((ap as *const f32).wrapping_add(i0 * lda), lda, chunk.len() / kp.max(1), k, chunk);
        });
    } else {
        convert_rows(a, lda, m, k, &mut buf[..m * kp]);
    }
    let r = f(buf.as_ptr().cast::<f32>(), kp);
    SCRATCH.with(|s| s.set(buf));
    r
}

/// One micro-kernel call on BF16 data: `k.a` points at BF16 pairs (row stride
/// `k.rs` pairs) and `k.bp` at a K-pair panel; `k.kc` counts K elements.
///
/// # Safety
/// As for the FP32 kernels.
pub(crate) unsafe fn dispatch(r: usize, k: &Kernel) {
    if hw_available() {
        avx512::dispatch(r, k)
    } else {
        scalar(r, k)
    }
}

/// Reference kernel (tests on CPUs without AVX-512 BF16).
unsafe fn scalar(r: usize, k: &Kernel) {
    let Kernel { kc, a, rs, bp, c, ldc, nb, accumulate } = *k;
    let (a, bp) = (a.cast::<u32>(), bp.cast::<u32>());
    let split = |v: u32| (from_bf16(v as u16), from_bf16((v >> 16) as u16));
    for i in 0..r {
        for j in 0..nb {
            let mut acc = 0f32;
            for p in 0..kc.div_ceil(2) {
                let (a0, a1) = split(*a.add(i * rs + p));
                let (b0, b1) = split(*bp.add(p * 32 + j));
                acc += a0 * b0 + a1 * b1;
            }
            let cij = c.add(i * ldc + j);
            *cij = if accumulate { *cij + acc } else { acc };
        }
    }
}

mod avx512 {
    use super::Kernel;
    use std::arch::x86_64::*;

    pub unsafe fn dispatch(r: usize, k: &Kernel) {
        match r {
            12 => kernel::<12>(k),
            11 => kernel::<11>(k),
            10 => kernel::<10>(k),
            9 => kernel::<9>(k),
            8 => kernel::<8>(k),
            7 => kernel::<7>(k),
            6 => kernel::<6>(k),
            5 => kernel::<5>(k),
            4 => kernel::<4>(k),
            3 => kernel::<3>(k),
            2 => kernel::<2>(k),
            _ => kernel::<1>(k),
        }
    }

    /// `C[R, nb] (+)= A[R, kc] @ B[kc, 32]`: per K pair, two 16-column B
    /// vectors and one broadcast A pair per row, 2R `vdpbf16ps`.
    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn kernel<const R: usize>(k: &Kernel) {
        let Kernel { kc, a, rs, bp, c, ldc, nb, accumulate } = *k;
        let a = a.cast::<i32>();
        let bp = bp.cast::<__m512i>();
        let mut lo = [_mm512_setzero_ps(); R];
        let mut hi = [_mm512_setzero_ps(); R];
        for p in 0..kc.div_ceil(2) {
            let b0: __m512bh = std::mem::transmute(_mm512_load_si512(bp.add(2 * p)));
            let b1: __m512bh = std::mem::transmute(_mm512_load_si512(bp.add(2 * p + 1)));
            for r in 0..R {
                let av: __m512bh = std::mem::transmute(_mm512_set1_epi32(*a.add(r * rs + p)));
                lo[r] = _mm512_dpbf16_ps(lo[r], av, b0);
                hi[r] = _mm512_dpbf16_ps(hi[r], av, b1);
            }
        }
        let m0: __mmask16 = if nb >= 16 { 0xffff } else { (1u16 << nb) - 1 };
        let m1: __mmask16 = if nb >= 32 { 0xffff } else if nb > 16 { (1u16 << (nb - 16)) - 1 } else { 0 };
        for r in 0..R {
            let cr = c.add(r * ldc);
            let (mut x0, mut x1) = (lo[r], hi[r]);
            if accumulate {
                x0 = _mm512_add_ps(x0, _mm512_maskz_loadu_ps(m0, cr));
                x1 = _mm512_add_ps(x1, _mm512_maskz_loadu_ps(m1, cr.add(16)));
            }
            _mm512_mask_storeu_ps(cr, m0, x0);
            _mm512_mask_storeu_ps(cr.add(16), m1, x1);
        }
    }

    /// See [`super::convert_rows`].
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub unsafe fn convert_rows(a: *const f32, lda: usize, rows: usize, k: usize, dst: *mut u32) {
        let kp = k.div_ceil(2);
        for r in 0..rows {
            let (src, out) = (a.add(r * lda), dst.add(r * kp));
            let mut j = 0;
            while j + 32 <= k {
                let v = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(src.add(j + 16)), _mm512_loadu_ps(src.add(j)));
                _mm512_storeu_si512(out.add(j / 2).cast(), std::mem::transmute::<__m512bh, __m512i>(v));
                j += 32;
            }
            let rem = k - j;
            if rem > 0 {
                let m0: __mmask16 = if rem >= 16 { 0xffff } else { (1u16 << rem) - 1 };
                let m1: __mmask16 = if rem > 16 { (1u16 << (rem - 16)) - 1 } else { 0 };
                let v = _mm512_cvtne2ps_pbh(_mm512_maskz_loadu_ps(m1, src.add(j + 16)), _mm512_maskz_loadu_ps(m0, src.add(j)));
                let ms: __mmask16 = ((1u32 << rem.div_ceil(2)) - 1) as u16;
                _mm512_mask_storeu_epi32(out.add(j / 2).cast(), ms, std::mem::transmute::<__m512bh, __m512i>(v));
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn rounding_is_nearest_even() {
        assert_eq!(to_bf16(1.0), 0x3f80);
        // 1 + 2^-8 is halfway between two BF16 values: ties to even (down).
        assert_eq!(to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(to_bf16(-2.5), 0xc020);
        assert!(from_bf16(to_bf16(f32::NAN)).is_nan());
        assert_eq!(to_bf16(f32::INFINITY), 0x7f80);
    }

    #[test]
    fn pair_counts() {
        for k in [0, 1, 2, 511, 512, 513, 1024, 1025] {
            let want: usize = (0..k).step_by(KC).map(|k0| KC.min(k - k0).div_ceil(2)).sum();
            assert_eq!(pairs_total(k), want, "k={k}");
        }
    }

    #[test]
    fn conversion_matches_scalar() {
        for k in [1usize, 2, 15, 16, 17, 31, 32, 33, 63, 100] {
            let rows = 3;
            let lda = k + 5;
            let a: Vec<f32> = (0..rows * lda).map(|i| (i as f32 * 0.377).sin() * 3.0).collect();
            let kp = k.div_ceil(2);
            let mut got = vec![u32::MAX; rows * kp];
            convert_rows(a.as_ptr(), lda, rows, k, &mut got);
            for r in 0..rows {
                for p in 0..kp {
                    let hi = if 2 * p + 1 < k { a[r * lda + 2 * p + 1] } else { 0.0 };
                    assert_eq!(got[r * kp + p], pair(a[r * lda + 2 * p], hi), "k={k} r={r} p={p}");
                }
            }
        }
    }

    pub(crate) fn round(v: &[f32]) -> Vec<f32> {
        v.iter().map(|&x| from_bf16(to_bf16(x))).collect()
    }

    pub(crate) fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// `(C, bound)`: the f64 product of the BF16-rounded inputs and, per element,
    /// a tolerance for FP32 accumulation in any order.
    pub(crate) fn reference(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (a, b) = (round(a), round(b));
        let mut c = vec![0f32; m * n];
        let mut bound = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let (mut acc, mut mag) = (0f64, 0f64);
                for p in 0..k {
                    let t = a[i * k + p] as f64 * b[p * n + j] as f64;
                    acc += t;
                    mag += t.abs();
                }
                c[i * n + j] = acc as f32;
                bound[i * n + j] = (mag * 1e-6 * (k as f64).sqrt().max(1.0) + 1e-6) as f32;
            }
        }
        (c, bound)
    }

    #[test]
    fn bf16_gemm_matches_rounded_reference() {
        use crate::gemm_packed::{gemm, gemm_strided_serial, Isa, PackedB};
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for &(m, k, n) in &[(1, 1, 1), (5, 7, 3), (12, 2, 32), (13, 300, 33), (97, 513, 70), (25, 1025, 64), (200, 64, 260), (12, 1024, 512)] {
            let a = rand(m * k, 11);
            let b = rand(k * n, 12);
            let (want, bound) = reference(m, k, n, &a, &b);
            let pb = PackedB::pack_for(Isa::Avx512Bf16, &b, k, n);
            assert_eq!(pb.isa(), Isa::Avx512Bf16);
            let check = |c: &[f32], what: &str| {
                for (idx, ((x, y), e)) in c.iter().zip(&want).zip(&bound).enumerate() {
                    assert!((x - y).abs() <= *e, "{what} m={m} k={k} n={n} [{idx}]: {x} vs {y} (tol {e})");
                }
            };
            for threads in [1, 4] {
                let mut c = vec![f32::NAN; m * n];
                pool.install(|| gemm(m, &a, &pb, &mut c, threads));
                check(&c, &format!("gemm t={threads}"));
            }
            // Strided A and C.
            let (lda, ldc) = (k + 3, n + 5);
            let mut a_s = vec![f32::NAN; m * lda];
            for i in 0..m {
                a_s[i * lda..i * lda + k].copy_from_slice(&a[i * k..(i + 1) * k]);
            }
            let mut c_s = vec![f32::NAN; m * ldc];
            unsafe { gemm_strided_serial(m, a_s.as_ptr(), lda, &pb, c_s.as_mut_ptr(), ldc) };
            let c: Vec<f32> = (0..m).flat_map(|i| c_s[i * ldc..i * ldc + n].to_vec()).collect();
            check(&c, "strided");
            // Transposed source packing agrees with the row-major one.
            let bt: Vec<f32> = (0..n * k).map(|idx| b[(idx % k) * n + idx / k]).collect();
            let pt = PackedB::pack_strided_for(Isa::Avx512Bf16, &bt, k, n, 1, k);
            let mut c = vec![f32::NAN; m * n];
            gemm(m, &a, &pt, &mut c, 1);
            check(&c, "transposed pack");
        }
    }

    #[test]
    fn bf16_error_vs_fp32_is_small() {
        use crate::gemm_packed::{gemm, Isa, PackedB};
        let (m, k, n) = (16, 1024, 96);
        let (a, b) = (rand(m * k, 21), rand(k * n, 22));
        let pb = PackedB::pack_for(Isa::Avx512Bf16, &b, k, n);
        let mut c = vec![0f32; m * n];
        gemm(m, &a, &pb, &mut c, 1);
        let mut exact = vec![0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    exact[i * n + j] += a[i * k + p] as f64 * b[p * n + j] as f64;
                }
            }
        }
        let dot: f64 = c.iter().zip(&exact).map(|(&x, &y)| x as f64 * y).sum();
        let nc: f64 = c.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let ne: f64 = exact.iter().map(|y| y * y).sum::<f64>().sqrt();
        let cos = dot / (nc * ne);
        assert!(cos > 0.9999, "cosine {cos}");
    }

    /// `cargo test --release -p oxionnx-ops --lib bench_bf16 -- --ignored --nocapture`
    /// (`BENCH_THREADS=1,8`): FP32 vs BF16 packed GEMM on the EmbeddingGemma shapes.
    #[test]
    #[ignore]
    fn bench_bf16_shapes() {
        use crate::gemm_packed::{gemm, isa, Isa, PackedB};
        use std::time::Instant;
        if !hw_available() {
            return;
        }
        let threads: Vec<usize> =
            std::env::var("BENCH_THREADS").unwrap_or("1".into()).split(',').map(|t| t.parse().unwrap()).collect();
        for &th in &threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
            for &m in &[32usize, 256, 2048] {
                for &(k, n) in &[(512usize, 512usize), (512, 2048), (2048, 512), (512, 4096), (1024, 512), (512, 3072)] {
                    let (a, b) = (rand(m * k, 1), rand(k * n, 2));
                    // Enough weight copies to spill L3 when M is small, as in a model run.
                    let copies = if m <= 256 { ((256usize << 20) / (k * n * 4)).max(1) } else { 1 };
                    let mut c = vec![0f32; m * n];
                    let mut res = vec![];
                    for which in [isa().unwrap(), Isa::Avx512Bf16] {
                        let pbs: Vec<PackedB> = (0..copies).map(|_| PackedB::pack_for(which, &b, k, n)).collect();
                        let flops = 2.0 * (m * n * k) as f64;
                        let iters = ((4e9 / flops) as usize).clamp(copies.max(5), 4000);
                        pool.install(|| {
                            for pb in pbs.iter().take(3) {
                                gemm(m, &a, pb, &mut c, th);
                            }
                            let mut best = f64::MAX;
                            for _ in 0..3 {
                                let t = Instant::now();
                                for it in 0..iters {
                                    gemm(m, &a, &pbs[it % copies], &mut c, th);
                                }
                                best = best.min(t.elapsed().as_secs_f64() / iters as f64);
                            }
                            res.push((best * 1e6, flops / best / 1e9));
                        });
                    }
                    println!(
                        "T={th:>2} M={m:>5} K={k:>5} N={n:>5}  fp32 {:>9.1} us {:>6.1} GF/s   bf16 {:>9.1} us {:>6.1} GF/s   x{:.2}",
                        res[0].0, res[0].1, res[1].0, res[1].1, res[0].0 / res[1].0
                    );
                }
            }
        }
    }
}
