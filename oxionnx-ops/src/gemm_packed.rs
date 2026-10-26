//! Prototype (onnx_spike): row-major SGEMM `C[m,n] = A[m,k] @ B[k,n]` with B packed once.
//!
//! B is stored as KC-row blocks of NR-column panels (zero-padded on the right),
//! so the micro-kernel streams each panel contiguously. A is read in place (MR
//! row streams, broadcast loads). The loop order is MLAS's: slices of
//! `SLICE_PANELS` panels (a KC x 256 block of B that stays in L2), then k
//! blocks, then every MR-row strip of the task's rows. Two micro-kernels, picked
//! once per process from the CPU features (MLAS does the same):
//! * AVX-512F: 12x32, 24 zmm accumulators;
//! * AVX2+FMA: 6x16, 12 ymm accumulators.
//!
//! Row tails use the same kernel with fewer rows; the last partial panel goes
//! through a scratch tile (AVX2) or masked stores (AVX-512).
//! `OXIONNX_GEMM_ISA=avx2` forces the AVX2 kernel, for A/B comparisons.
//!
//! Opt-in BF16 weights ([`Isa::Avx512Bf16`], see [`crate::gemm_bf16`]) use the
//! same blocking and task grid with K-pair panels, A converted to BF16 per call.

#![allow(unsafe_code)]

use rayon::prelude::*;
use std::sync::OnceLock;

pub(crate) const KC: usize = 512;
/// Panels per B slice: 256 columns with the AVX-512 kernel, 128 with AVX2.
const SLICE_PANELS: usize = 8;
/// Fewest rows a task tile is split down to along M.
const MIN_ROWS_PER_TILE: usize = 96;
const MIN_MACS_PER_TASK: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Avx2,
    Avx512,
    /// BF16 weights and activations (one or two BF16 terms each), FP32 accumulation (`vdpbf16ps`); see [`crate::gemm_bf16`].
    Avx512Bf16,
}

impl Isa {
    /// Micro-tile `(rows, cols)`.
    const fn tile(self) -> (usize, usize) {
        match self {
            Isa::Avx2 => (6, 16),
            Isa::Avx512 | Isa::Avx512Bf16 => (12, 32),
        }
    }

    /// K elements per packed 32-bit slot (BF16 packs K pairs).
    const fn k_per_slot(self) -> usize {
        match self {
            Isa::Avx512Bf16 => 2,
            _ => 1,
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

/// The packing for a constant weight: [`Isa::Avx512Bf16`] when `bf16` is
/// requested and the CPU supports it, else [`isa`].
pub fn weight_isa(bf16: bool) -> Option<Isa> {
    if bf16 && crate::gemm_bf16::hw_available() {
        Some(Isa::Avx512Bf16)
    } else {
        isa()
    }
}

pub struct PackedB {
    pub k: usize,
    pub n: usize,
    isa: Isa,
    nr: usize,
    panels: usize,
    buf: Vec<f32>,
    off: usize, // offset of the 64-byte aligned start in `buf`
    /// BF16 only: split each activation into two BF16 terms (see [`PackedB::with_split_activations`]).
    split: bool,
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
        if isa == Isa::Avx512Bf16 {
            let (buf, off) = crate::gemm_bf16::pack(k, n, nr, |p, j| b[p * rs + j * cs]);
            return PackedB { k, n, isa, nr, panels, buf, off, split: false };
        }
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
        PackedB { k, n, isa, nr, panels, buf, off, split: false }
    }

    /// Pack a column selection of the row-major `k x ld` matrix `b`: packed
    /// column `j` is `b[:, cols[j]]` (e.g. to interleave the halves of a fused
    /// gate/up projection for [`gemm_epilogue`]).
    #[allow(dead_code)]
    pub fn pack_cols(b: &[f32], k: usize, ld: usize, cols: &[usize]) -> Self {
        Self::pack_cols_for(isa().expect("gemm_packed: no AVX2/AVX-512"), b, k, ld, cols)
    }

    pub fn pack_cols_for(isa: Isa, b: &[f32], k: usize, ld: usize, cols: &[usize]) -> Self {
        assert!(cols.iter().all(|&c| c < ld) && b.len() >= k * ld);
        let (nr, n) = (isa.tile().1, cols.len());
        let panels = n.div_ceil(nr);
        if isa == Isa::Avx512Bf16 {
            let (buf, off) = crate::gemm_bf16::pack(k, n, nr, |p, j| b[p * ld + cols[j]]);
            return PackedB { k, n, isa, nr, panels, buf, off, split: false };
        }
        let len = k * panels * nr;
        let mut buf = vec![0f32; len + 16];
        let off = buf.as_ptr().align_offset(64).min(16);
        let dst = &mut buf[off..off + len];
        let mut base = 0;
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);
            for (j, panel_cols) in cols.chunks(nr).enumerate() {
                let panel = &mut dst[base + j * kc * nr..base + (j + 1) * kc * nr];
                for p in 0..kc {
                    let row = &b[(k0 + p) * ld..(k0 + p + 1) * ld];
                    for (jj, &c) in panel_cols.iter().enumerate() {
                        panel[p * nr + jj] = row[c];
                    }
                }
            }
            base += kc * panels * nr;
        }
        PackedB { k, n, isa, nr, panels, buf, off, split: false }
    }

    /// For a BF16 packing: multiply by activations split into two BF16 terms,
    /// `a = hi + lo` with `hi = bf16(a)`, `lo = bf16(a - hi)` (16 significant
    /// bits instead of 8), at twice the dot-product count. See
    /// [`crate::gemm_bf16`]. No effect on FP32 packings.
    pub fn with_split_activations(mut self, on: bool) -> Self {
        self.split = on && self.isa == Isa::Avx512Bf16;
        self
    }

    /// Whether the activations are split into two BF16 terms.
    #[allow(dead_code)]
    pub fn split_activations(&self) -> bool {
        self.split
    }

    /// Pair offset of the `lo` activation terms from the start of each
    /// converted A row (0 = no split), as [`crate::gemm_bf16::with_a`] lays them out.
    fn lo_offset(&self) -> usize {
        if self.split {
            self.k.div_ceil(2)
        } else {
            0
        }
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// The packed matrix back as row-major `k x n` FP32 (in packed column
    /// order; BF16 values widened), for fallbacks once the source is freed.
    pub fn unpack(&self) -> Vec<f32> {
        let (k, n, nr) = (self.k, self.n, self.nr);
        let mut out = vec![0f32; k * n];
        for k0 in (0..k).step_by(KC) {
            for j in 0..self.panels {
                let panel = self.panel(k0, j);
                for p in k0..KC.min(k - k0) + k0 {
                    for jj in 0..nr.min(n - j * nr) {
                        // SAFETY: (p - k0, jj) lies inside panel j of this k block.
                        let v = unsafe {
                            match self.isa {
                                Isa::Avx512Bf16 => {
                                    let pair = *panel.cast::<u32>().add((p - k0) / 2 * nr + jj);
                                    crate::gemm_bf16::from_bf16((pair >> (16 * ((p - k0) % 2))) as u16)
                                }
                                _ => *panel.add((p - k0) * nr + jj),
                            }
                        };
                        out[p * n + j * nr + jj] = v;
                    }
                }
            }
        }
        out
    }

    /// Start of panel `j` in the k-block starting at row `k0`.
    #[inline]
    fn panel(&self, k0: usize, j: usize) -> *const f32 {
        let s = self.isa.k_per_slot();
        let kc = KC.min(self.k - k0).div_ceil(s);
        // SAFETY: k0 < k and j < panels, so the offset is inside the packed data
        // (k0 is a multiple of the even KC, so k0 / s slots precede its block).
        unsafe { self.buf.as_ptr().add(self.off + k0 / s * self.panels * self.nr + j * kc * self.nr) }
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
    let tasks = (m * n * k / MIN_MACS_PER_TASK).clamp(1, threads.max(1));
    let slices = b.panels.div_ceil(SLICE_PANELS);
    // Whole slices per tile when there are enough of them, else single panels.
    let splits = Grid::col_splits(m, b.isa.tile().0, tasks);
    let unit = if slices % splits == 0 || slices >= 4 * splits { SLICE_PANELS } else { 1 };
    let grid = Grid::new(m, b.isa.tile().0, b.panels.div_ceil(unit), tasks);
    let cp = SendPtr(c.as_mut_ptr());
    let cp = &cp;
    let run = |ap: *const f32, lda: usize| {
        let ap = ap as usize;
        grid.run(tasks, |i0, mb, u0, u1| {
            let (p0, p1) = (u0 * unit, (u1 * unit).min(b.panels));
            // SAFETY: tiles cover disjoint (row, column) ranges of C; all pointer
            // offsets stay within a/b/c bounds (checked by the asserts above), and
            // `b.isa` was detected on this CPU.
            unsafe { tile((ap as *const f32).add(i0 * lda), lda, mb, b, p0, p1, cp.0.add(i0 * n), n) };
        });
    };
    if b.isa == Isa::Avx512Bf16 {
        // SAFETY: `a` is m x k (asserted above).
        unsafe { crate::gemm_bf16::with_a(m, a.as_ptr(), k, k, threads, b.split, run) };
    } else {
        run(a.as_ptr(), k);
    }
}

/// Receives each finished block of C while it is still in L1, instead of C
/// being written out (see [`gemm_epilogue`]).
pub trait Epilogue: Sync {
    /// Columns per block: a multiple of 32 (both kernels' NR) that divides `n`.
    fn width(&self) -> usize;
    /// `block` holds rows `i0..i0 + rows`, columns `j0..j0 + width()` of C,
    /// row-major with row stride `width()`.
    ///
    /// # Safety
    /// Called concurrently from several threads, for disjoint blocks.
    unsafe fn apply(&self, i0: usize, rows: usize, j0: usize, block: &mut [f32]);
}

/// `epi` receives every block of `a` (m x k, row-major) @ `b`. Uses up to `threads` rayon tasks.
pub fn gemm_epilogue<E: Epilogue>(m: usize, a: &[f32], b: &PackedB, epi: &E, threads: usize) {
    let (k, n, w) = (b.k, b.n, epi.width());
    assert_eq!(a.len(), m * k);
    assert!(w > 0 && w % 32 == 0 && n % w == 0, "gemm_epilogue: width {w} must be a multiple of 32 dividing n={n}");
    if m == 0 || n == 0 {
        return;
    }
    let (mr, nr) = b.isa.tile();
    let tasks = (m * n * k / MIN_MACS_PER_TASK).clamp(1, threads.max(1));
    let grid = Grid::new(m, mr, n / w, tasks);
    let s = b.isa.k_per_slot();
    let run = |ap: *const f32, lda: usize| {
        let ap = ap as usize;
        grid.run(tasks, |i0, mb, u0, u1| {
            let ap = ap as *const f32;
            let mut block = vec![0f32; mr * w];
            for u in u0..u1 {
                let mut i = 0;
                while i < mb {
                    let r = mr.min(mb - i);
                    if k == 0 {
                        block.fill(0.0);
                    }
                    for k0 in (0..k).step_by(KC) {
                        let kc = KC.min(k - k0);
                        for jj in 0..w / nr {
                            let kernel = Kernel {
                                kc,
                                a: unsafe { ap.add((i0 + i) * lda + k0 / s) },
                                rs: lda,
                                bp: b.panel(k0, u * w / nr + jj),
                                c: unsafe { block.as_mut_ptr().add(jj * nr) },
                                ldc: w,
                                nb: nr,
                                accumulate: k0 > 0,
                                lo: b.lo_offset(),
                            };
                            // SAFETY: rows i0+i..i0+i+r of `a` and panel u*w/nr+jj of `b` exist
                            // (n % w == 0), `block` holds mr x w, and `b.isa` was detected here.
                            unsafe {
                                match b.isa {
                                    Isa::Avx512 => avx512::dispatch(r, &kernel),
                                    Isa::Avx2 => avx2::dispatch(r, &kernel),
                                    Isa::Avx512Bf16 => crate::gemm_bf16::dispatch(r, &kernel),
                                }
                            }
                        }
                    }
                    // SAFETY: each (row strip, column block) is visited by exactly one task.
                    unsafe { epi.apply(i0 + i, r, u * w, &mut block[..r * w]) };
                    i += r;
                }
            }
        });
    };
    if b.isa == Isa::Avx512Bf16 {
        // SAFETY: `a` is m x k (asserted above).
        unsafe { crate::gemm_bf16::with_a(m, a.as_ptr(), k, k, threads, b.split, run) };
    } else {
        run(a.as_ptr(), k);
    }
}

/// Task tiles: one per task (equal tiles balance better than more, smaller
/// ones), split along M first (each tile then reuses its B columns over as
/// many rows as possible), then along `units` column groups.
struct Grid {
    m: usize,
    rows_per_tile: usize,
    units: usize,
    units_per_tile: usize,
    ntiles: usize,
    total: usize,
}

impl Grid {
    /// `(rows per tile, row tiles, column splits)`.
    fn rows(m: usize, mr: usize, tasks: usize) -> (usize, usize, usize) {
        let target = tasks;
        let strips = m.div_ceil(mr).max(1);
        // Every row tile streams all of its B columns, so small M (where B is most
        // of the traffic) is split along N only.
        let max_mtiles = (m / MIN_ROWS_PER_TILE).clamp(1, strips);
        // The largest divisor of `target` that fits, so the column splits come out whole.
        let mt = (1..=max_mtiles.min(target)).rev().find(|d| target % d == 0).unwrap_or(1);
        let rows_per_tile = strips.div_ceil(mt) * mr;
        (rows_per_tile, m.div_ceil(rows_per_tile).max(1), target / mt)
    }

    fn col_splits(m: usize, mr: usize, tasks: usize) -> usize {
        Self::rows(m, mr, tasks).2
    }

    fn new(m: usize, mr: usize, units: usize, tasks: usize) -> Self {
        let (rows_per_tile, mtiles, splits) = Self::rows(m, mr, tasks);
        let units_per_tile = units.div_ceil(splits.clamp(1, units.max(1))).max(1);
        let ntiles = units.div_ceil(units_per_tile).max(1);
        Grid { m, rows_per_tile, units, units_per_tile, ntiles, total: mtiles * ntiles }
    }

    /// Calls `f(first row, rows, first unit, end unit)` for every tile.
    fn run(&self, tasks: usize, f: impl Fn(usize, usize, usize, usize) + Sync) {
        let one = |t: usize| {
            let (ti, tj) = (t / self.ntiles, t % self.ntiles);
            let i0 = ti * self.rows_per_tile;
            let u0 = tj * self.units_per_tile;
            f(i0, self.rows_per_tile.min(self.m - i0), u0, (u0 + self.units_per_tile).min(self.units));
        };
        if tasks == 1 || self.total == 1 {
            (0..self.total).for_each(one);
        } else {
            (0..self.total).into_par_iter().for_each(one);
        }
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
    if b.isa == Isa::Avx512Bf16 {
        crate::gemm_bf16::with_a(m, a, lda, b.k, 1, b.split, |ap, lda| tile(ap, lda, m, b, 0, b.panels, c, ldc));
    } else {
        tile(a, lda, m, b, 0, b.panels, c, ldc);
    }
}

#[allow(clippy::too_many_arguments)]
/// `a` is FP32, or BF16 pairs (row stride `lda` pairs, `lo` terms after the
/// `hi` ones when split) for [`Isa::Avx512Bf16`].
unsafe fn tile(a: *const f32, lda: usize, mb: usize, b: &PackedB, p0: usize, p1: usize, c: *mut f32, ldc: usize) {
    let (k, n, nr) = (b.k, b.n, b.nr);
    let mr = b.isa.tile().0;
    let s = b.isa.k_per_slot();
    let mut s0 = p0;
    while s0 < p1 {
        let s1 = (s0 + SLICE_PANELS).min(p1);
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);
            let mut i = 0;
            while i < mb {
                let r = mr.min(mb - i);
                for j in s0..s1 {
                    let j0 = j * nr;
                    let kernel = Kernel {
                        kc,
                        a: a.add(i * lda + k0 / s),
                        rs: lda,
                        bp: b.panel(k0, j),
                        c: c.add(i * ldc + j0),
                        ldc,
                        nb: nr.min(n - j0),
                        accumulate: k0 > 0,
                        lo: b.lo_offset(),
                    };
                    match b.isa {
                        Isa::Avx512 => avx512::dispatch(r, &kernel),
                        Isa::Avx2 => avx2::dispatch(r, &kernel),
                        Isa::Avx512Bf16 => crate::gemm_bf16::dispatch(r, &kernel),
                    }
                }
                i += r;
            }
        }
        s0 = s1;
    }
}

/// One micro-kernel call: `C[R, nb] (+)= A[R, kc] @ Bpanel[kc, NR]`, with
/// `A[r, p] = a[r * rs + p]` (BF16: `a` holds K pairs, `rs` counts pairs).
#[derive(Clone, Copy)]
pub(crate) struct Kernel {
    pub(crate) kc: usize,
    pub(crate) a: *const f32,
    pub(crate) rs: usize,
    pub(crate) bp: *const f32,
    pub(crate) c: *mut f32,
    pub(crate) ldc: usize,
    pub(crate) nb: usize,
    pub(crate) accumulate: bool,
    /// BF16 split activations: the `lo` terms of row `r` start at
    /// `a + r * rs + lo` pairs (0 = not split). Unused by the FP32 kernels.
    pub(crate) lo: usize,
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
        let Kernel { kc, a, rs, bp, c, ldc, nb, accumulate, .. } = *k;
        let mut lo = [_mm256_setzero_ps(); R];
        let mut hi = [_mm256_setzero_ps(); R];
        for p in 0..kc {
            let b0 = _mm256_load_ps(bp.add(p * 16));
            let b1 = _mm256_load_ps(bp.add(p * 16 + 8));
            for r in 0..R {
                let av = _mm256_broadcast_ss(&*a.add(r * rs + p));
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
        let Kernel { kc, a, rs, bp, c, ldc, nb, accumulate, .. } = *k;
        let mut lo = [_mm512_setzero_ps(); R];
        let mut hi = [_mm512_setzero_ps(); R];
        for p in 0..kc {
            let b0 = _mm512_load_ps(bp.add(p * 32));
            let b1 = _mm512_load_ps(bp.add(p * 32 + 16));
            for r in 0..R {
                let av = _mm512_set1_ps(*a.add(r * rs + p));
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
