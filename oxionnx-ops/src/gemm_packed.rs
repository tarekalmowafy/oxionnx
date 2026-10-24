//! Prototype (onnx_spike): row-major SGEMM `C[m,n] = A[m,k] @ B[k,n]` with B packed once.
//!
//! B is stored as KC-row blocks of NR-column panels (zero-padded on the right),
//! so the micro-kernel streams each panel contiguously. A is read in place
//! (MR row streams, broadcast loads) for small M, as in MLAS's SGEMM, and packed
//! into MR-row strips per tile for large M. Two micro-kernels, picked once per
//! process from the CPU features (MLAS does the same):
//! * AVX-512F: 12x32, 24 zmm accumulators;
//! * AVX2+FMA: 6x16, 12 ymm accumulators.
//!
//! Row tails use the same kernel with fewer rows; the last partial panel goes
//! through a scratch tile (AVX2) or masked stores (AVX-512).
//! `OXIONNX_GEMM_ISA=avx2` forces the AVX2 kernel, for A/B comparisons.

#![allow(unsafe_code)]

use rayon::prelude::*;
use std::sync::OnceLock;

const KC: usize = 256;
const MC: usize = 96; // multiple of both MRs
const MIN_MACS_PER_TASK: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Avx2,
    Avx512,
}

impl Isa {
    /// Micro-tile `(rows, cols)`.
    const fn tile(self) -> (usize, usize) {
        match self {
            Isa::Avx2 => (6, 16),
            Isa::Avx512 => (12, 32),
        }
    }
}

/// The kernel this process uses, or `None` if the CPU has neither AVX2+FMA nor AVX-512F.
pub fn isa() -> Option<Isa> {
    static ISA: OnceLock<Option<Isa>> = OnceLock::new();
    *ISA.get_or_init(|| {
        let force_avx2 = std::env::var("OXIONNX_GEMM_ISA").is_ok_and(|v| v.eq_ignore_ascii_case("avx2"));
        if is_x86_feature_detected!("avx512f") && !force_avx2 {
            Some(Isa::Avx512)
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Some(Isa::Avx2)
        } else {
            None
        }
    })
}

pub fn available() -> bool {
    isa().is_some()
}

pub struct PackedB {
    pub k: usize,
    pub n: usize,
    isa: Isa,
    nr: usize,
    panels: usize,
    buf: Vec<f32>,
    off: usize, // offset of the 64-byte aligned start in `buf`
}

impl PackedB {
    /// Pack `b` (k x n, row-major) for the current [`isa`]. Panics if [`available`] is false.
    pub fn pack(b: &[f32], k: usize, n: usize) -> Self {
        Self::pack_for(isa().expect("gemm_packed: no AVX2/AVX-512"), b, k, n)
    }

    pub fn pack_for(isa: Isa, b: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(b.len(), k * n);
        Self::pack_strided_for(isa, b, k, n, n, 1)
    }

    /// Pack the `k x n` matrix with `B[p, j] = b[p * rs + j * cs]` (e.g. `rs = 1,
    /// cs = ld` packs the transpose of a row-major `n x k` matrix with row stride `ld`).
    pub fn pack_strided(b: &[f32], k: usize, n: usize, rs: usize, cs: usize) -> Self {
        Self::pack_strided_for(isa().expect("gemm_packed: no AVX2/AVX-512"), b, k, n, rs, cs)
    }

    pub fn pack_strided_for(isa: Isa, b: &[f32], k: usize, n: usize, rs: usize, cs: usize) -> Self {
        assert!(k == 0 || n == 0 || (k - 1) * rs + (n - 1) * cs < b.len());
        let nr = isa.tile().1;
        let panels = n.div_ceil(nr);
        let len = k * panels * nr;
        let mut buf = vec![0f32; len + 16];
        let off = buf.as_ptr().align_offset(64).min(16);
        let dst = &mut buf[off..off + len];
        let mut base = 0;
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);
            for j in 0..panels {
                let j0 = j * nr;
                let nb = nr.min(n - j0);
                let panel = &mut dst[base + j * kc * nr..base + (j + 1) * kc * nr];
                if cs == 1 {
                    for p in 0..kc {
                        let src = &b[(k0 + p) * rs + j0..(k0 + p) * rs + j0 + nb];
                        panel[p * nr..p * nr + nb].copy_from_slice(src);
                    }
                } else {
                    // Transposed source: walk each source column contiguously.
                    for jj in 0..nb {
                        let col = (j0 + jj) * cs;
                        for p in 0..kc {
                            panel[p * nr + jj] = b[(k0 + p) * rs + col];
                        }
                    }
                }
            }
            base += kc * panels * nr;
        }
        PackedB { k, n, isa, nr, panels, buf, off }
    }

    /// Start of panel `j` in the k-block starting at row `k0`.
    #[inline]
    fn panel(&self, k0: usize, j: usize) -> *const f32 {
        let kc = KC.min(self.k - k0);
        // SAFETY: k0 < k and j < panels, so the offset is inside the packed data.
        unsafe { self.buf.as_ptr().add(self.off + k0 * self.panels * self.nr + j * kc * self.nr) }
    }
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// `c` (m x n, row-major) = `a` (m x k, row-major) @ `b`. Uses up to `threads` rayon tasks.
pub fn gemm(m: usize, a: &[f32], b: &PackedB, c: &mut [f32], threads: usize) {
    let (k, n) = (b.k, b.n);
    assert_eq!(a.len(), m * k);
    assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.0);
        return;
    }
    let mr = b.isa.tile().0;
    let tasks = (m * n * k / MIN_MACS_PER_TASK).clamp(1, threads.max(1));
    // Tile grid: MC-row blocks x groups of panels, enough tiles for ~2 per task.
    let mc = MC.min(m.div_ceil(mr) * mr);
    let mtiles = m.div_ceil(mc);
    let ntiles = if tasks == 1 { 1 } else { (2 * tasks).div_ceil(mtiles).clamp(1, b.panels) };
    let panels_per_tile = b.panels.div_ceil(ntiles);
    let ntiles = b.panels.div_ceil(panels_per_tile);
    let cp = SendPtr(c.as_mut_ptr());
    let cp = &cp;
    let run = |t: usize| {
        let (ti, tj) = (t / ntiles, t % ntiles);
        let i0 = ti * mc;
        let mb = mc.min(m - i0);
        let p0 = tj * panels_per_tile;
        let p1 = (p0 + panels_per_tile).min(b.panels);
        // SAFETY: tiles cover disjoint (row, column) ranges of C; all pointer
        // offsets stay within a/b/c bounds (checked by the asserts above), and
        // `b.isa` was detected on this CPU.
        unsafe { tile(a.as_ptr().add(i0 * k), k, mb, b, p0, p1, cp.0.add(i0 * n), n) };
    };
    let total = mtiles * ntiles;
    if tasks == 1 || total == 1 {
        (0..total).for_each(run);
    } else {
        (0..total).into_par_iter().for_each(run);
    }
}

/// Single-threaded `C[m, n] = A[m, k] @ B` with row strides `lda` / `ldc`.
///
/// # Safety
/// `a` must be valid for `m` rows of `b.k` elements at stride `lda`, `c` for `m`
/// rows of `b.n` elements at stride `ldc`, and `b` packed for this CPU.
pub unsafe fn gemm_strided_serial(m: usize, a: *const f32, lda: usize, b: &PackedB, c: *mut f32, ldc: usize) {
    if m == 0 || b.n == 0 {
        return;
    }
    if b.k == 0 {
        for i in 0..m {
            std::slice::from_raw_parts_mut(c.add(i * ldc), b.n).fill(0.0);
        }
        return;
    }
    let mut i0 = 0;
    while i0 < m {
        let mb = MC.min(m - i0);
        tile(a.add(i0 * lda), lda, mb, b, 0, b.panels, c.add(i0 * ldc), ldc);
        i0 += mb;
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn tile(a: *const f32, lda: usize, mb: usize, b: &PackedB, p0: usize, p1: usize, c: *mut f32, ldc: usize) {
    let (k, n, nr) = (b.k, b.n, b.nr);
    let mr = b.isa.tile().0;
    // Pack A into MR-row strips (p-major) when it is reused across enough panels;
    // for small M the MR row streams of A stay in L1 anyway.
    let pack_a = p1 - p0 >= 4 && mb > mr;
    let strips = mb.div_ceil(mr);
    let mut abuf = if pack_a { vec![0f32; strips * mr * KC.min(k)] } else { Vec::new() };
    for k0 in (0..k).step_by(KC) {
        let kc = KC.min(k - k0);
        let acc = k0 > 0;
        if pack_a {
            for (s, strip) in abuf.chunks_mut(mr * kc).take(strips).enumerate() {
                for r in 0..mr.min(mb - s * mr) {
                    let src = a.add((s * mr + r) * lda + k0);
                    for p in 0..kc {
                        *strip.get_unchecked_mut(p * mr + r) = *src.add(p);
                    }
                }
            }
        }
        for j in p0..p1 {
            let bp = b.panel(k0, j);
            let j0 = j * nr;
            let nb = nr.min(n - j0);
            let mut i = 0;
            while i < mb {
                let r = mr.min(mb - i);
                let (ap, rs, ps) = if pack_a { (abuf.as_ptr().add(i * kc), 1, mr) } else { (a.add(i * lda + k0), lda, 1) };
                let k = Kernel { kc, a: ap, rs, ps, bp, c: c.add(i * ldc + j0), ldc, nb, accumulate: acc };
                match b.isa {
                    Isa::Avx512 => avx512::dispatch(r, &k),
                    Isa::Avx2 => avx2::dispatch(r, &k),
                }
                i += r;
            }
        }
    }
}

/// One micro-kernel call: `C[R, nb] (+)= A[R, kc] @ Bpanel[kc, NR]`, with
/// `A[r, p] = a[r * rs + p * ps]`.
#[derive(Clone, Copy)]
struct Kernel {
    kc: usize,
    a: *const f32,
    rs: usize,
    ps: usize,
    bp: *const f32,
    c: *mut f32,
    ldc: usize,
    nb: usize,
    accumulate: bool,
}

mod avx2 {
    use super::Kernel;
    use std::arch::x86_64::*;

    pub unsafe fn dispatch(r: usize, k: &Kernel) {
        match r {
            6 => kernel::<6>(k),
            5 => kernel::<5>(k),
            4 => kernel::<4>(k),
            3 => kernel::<3>(k),
            2 => kernel::<2>(k),
            _ => kernel::<1>(k),
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn kernel<const R: usize>(k: &Kernel) {
        let Kernel { kc, a, rs, ps, bp, c, ldc, nb, accumulate } = *k;
        let mut lo = [_mm256_setzero_ps(); R];
        let mut hi = [_mm256_setzero_ps(); R];
        for p in 0..kc {
            let b0 = _mm256_load_ps(bp.add(p * 16));
            let b1 = _mm256_load_ps(bp.add(p * 16 + 8));
            for r in 0..R {
                let av = _mm256_broadcast_ss(&*a.add(r * rs + p * ps));
                lo[r] = _mm256_fmadd_ps(av, b0, lo[r]);
                hi[r] = _mm256_fmadd_ps(av, b1, hi[r]);
            }
        }
        if nb == 16 {
            for r in 0..R {
                let cr = c.add(r * ldc);
                let (mut x0, mut x1) = (lo[r], hi[r]);
                if accumulate {
                    x0 = _mm256_add_ps(x0, _mm256_loadu_ps(cr));
                    x1 = _mm256_add_ps(x1, _mm256_loadu_ps(cr.add(8)));
                }
                _mm256_storeu_ps(cr, x0);
                _mm256_storeu_ps(cr.add(8), x1);
            }
        } else {
            let mut tmp = [0f32; 16];
            for r in 0..R {
                _mm256_storeu_ps(tmp.as_mut_ptr(), lo[r]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), hi[r]);
                let cr = c.add(r * ldc);
                for (j, &v) in tmp.iter().enumerate().take(nb) {
                    *cr.add(j) = if accumulate { *cr.add(j) + v } else { v };
                }
            }
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

    #[target_feature(enable = "avx512f")]
    unsafe fn kernel<const R: usize>(k: &Kernel) {
        let Kernel { kc, a, rs, ps, bp, c, ldc, nb, accumulate } = *k;
        let mut lo = [_mm512_setzero_ps(); R];
        let mut hi = [_mm512_setzero_ps(); R];
        for p in 0..kc {
            let b0 = _mm512_load_ps(bp.add(p * 32));
            let b1 = _mm512_load_ps(bp.add(p * 32 + 16));
            for r in 0..R {
                let av = _mm512_set1_ps(*a.add(r * rs + p * ps));
                lo[r] = _mm512_fmadd_ps(av, b0, lo[r]);
                hi[r] = _mm512_fmadd_ps(av, b1, hi[r]);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn matches_naive_on_tails_and_k_blocks() {
        let mut isas = vec![];
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            isas.push(Isa::Avx2);
        }
        if is_x86_feature_detected!("avx512f") {
            isas.push(Isa::Avx512);
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for isa in isas {
            for &(m, k, n) in &[(1, 1, 1), (5, 7, 3), (6, 16, 16), (13, 300, 33), (97, 513, 70), (200, 64, 260), (12, 1024, 512)] {
                let a: Vec<f32> = (0..m * k).map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5).collect();
                let b: Vec<f32> = (0..k * n).map(|i| ((i * 53) % 89) as f32 / 89.0 - 0.5).collect();
                let want = naive(m, k, n, &a, &b);
                let pb = PackedB::pack_for(isa, &b, k, n);
                for threads in [1, 4] {
                    let mut c = vec![f32::NAN; m * n];
                    pool.install(|| gemm(m, &a, &pb, &mut c, threads));
                    for (x, y) in c.iter().zip(&want) {
                        assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()), "{isa:?} m={m} k={k} n={n} t={threads}: {x} vs {y}");
                    }
                }
            }
        }
    }
}
