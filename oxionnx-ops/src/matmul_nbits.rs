//! Block-quantized weights: `com.microsoft::MatMulNBits` and
//! `com.microsoft::GatherBlockQuantized` kernels.
//!
//! # Weights
//! ORT stores a `bits`-bit weight `W[K, N]` transposed, block-quantized along
//! K: `B[N, k_blocks, blob]` (`blob = block_size * bits / 8`, low nibble
//! first for 4 bits), `scales[N, k_blocks]` and optional `zero_points`
//! (packed uint8 like `B`, or float `[N, k_blocks]`; default `2^(bits-1)`),
//! with `W[k, n] = (q[n, k] - zp[n, b]) * scale[n, b]` for `b = k / block_size`.
//!
//! [`pack`] turns that into a self-contained word buffer (so it can live in a
//! session weight) holding, per 16-column panel and K block:
//! * the quantized values, in groups of 4 consecutive k per column — 64 bytes
//!   are `[col 0..16][k 0..4]`, one `vpdpbusd` operand. 4-bit values keep two
//!   groups per 64 bytes: the low nibbles hold k `0..4`, the high nibbles k
//!   `4..8`;
//! * `sb[n] = scale[n, b]` and `zb[n] = scale[n, b] * zp[n, b]`, 16 floats each.
//!
//! # Compute
//! Activations are quantized on the fly per row and K block to int8
//! (`a ≈ sa * qa`, symmetric, `|qa| <= 127`, as ORT's `accuracy_level` 4), and
//!
//! ```text
//! C[m, n] = Σ_b sa[m,b] * sb[n,b] * Σ_{k∈b} qa[m,k] * q[n,k]  -  Σ_b sa[m,b] * Σ_{k∈b} qa[m,k] * zb[n,b]
//! ```
//!
//! with the integer dot products done by `vpdpbusd` (AVX-512 VNNI: unsigned
//! weights × signed activations, 6 rows × 32 columns per micro-tile), by
//! `vpmaddubsw` (AVX2: 4 rows × 8 columns; 8-bit weights are recentred to
//! signed and multiplied through the sign trick so pair sums cannot saturate),
//! or by plain integer loops. All three quantize the activations identically.
//!
//! Tasks split rows and panel pairs like [`crate::gemm_packed`]'s grid; each
//! task walks slices of [`SLICE_PANELS`] panels (L2-resident) and, inside, MR
//! row strips whose quantized activations stay in L1.
//!
//! `OXIONNX_NBITS_ISA=scalar|avx2` caps the kernel ISA, for A/B comparisons.

#![allow(unsafe_code)]

use std::sync::OnceLock;

/// Columns per panel.
pub const NP: usize = 16;
/// Panels per L2 slice of a task.
const SLICE_PANELS: usize = 8;
/// Fewest rows a task tile is split down to along M.
const MIN_ROWS_PER_TILE: usize = 96;
const MIN_MACS_PER_TASK: usize = 128 * 1024;
const MAGIC: u32 = 0x4F58_4E42; // "OXNB"
const VERSION: u32 = 1;
/// Header words of a packed buffer.
const HEADER: usize = 16;

/// The geometry of one quantized weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub n: usize,
    pub k: usize,
    pub bits: usize,
    pub block: usize,
}

impl Shape {
    pub fn new(n: usize, k: usize, bits: usize, block: usize) -> Result<Self, String> {
        if bits != 4 && bits != 8 {
            return Err(format!("bits={bits} is not supported (only 4 and 8)"));
        }
        if block < 16 || block % 16 != 0 {
            return Err(format!("block_size={block} must be a multiple of 16 (ORT: a power of two >= 16)"));
        }
        if n == 0 || k == 0 {
            return Err(format!("empty weight K={k} N={n}"));
        }
        Ok(Shape { n, k, bits, block })
    }
    /// K blocks.
    pub fn kb(&self) -> usize {
        self.k.div_ceil(self.block)
    }
    /// Bytes per (column, K block) in ORT's `B`.
    pub fn blob(&self) -> usize {
        self.block * self.bits / 8
    }
    pub fn panels(&self) -> usize {
        self.n.div_ceil(NP)
    }
    /// Bytes per (panel, K block) in the packed layout.
    fn qblock(&self) -> usize {
        self.block * NP * self.bits / 8
    }
    /// Bytes per row of ORT's packed uint8 zero points.
    pub fn zp_row(&self) -> usize {
        (self.kb() * self.bits).div_ceil(8)
    }
    fn default_zp(&self) -> f32 {
        (1u32 << (self.bits - 1)) as f32
    }
}

/// Zero points of an ORT-layout weight.
#[derive(Clone, Copy, Debug)]
pub enum ZeroPoints<'a> {
    /// `2^(bits - 1)`.
    Default,
    /// uint8 `[N, ceil(k_blocks * bits / 8)]`, packed like `B`.
    Packed(&'a [u8]),
    /// float `[N, k_blocks]`.
    Float(&'a [f32]),
}

/// An ORT-layout quantized weight.
#[derive(Clone, Copy, Debug)]
pub struct Source<'a> {
    /// `[N, k_blocks, blob]`.
    pub b: &'a [u8],
    /// `[N, k_blocks]`.
    pub scales: &'a [f32],
    pub zp: ZeroPoints<'a>,
}

impl Source<'_> {
    pub fn check(&self, s: Shape) -> Result<(), String> {
        let (n, kb) = (s.n, s.kb());
        if self.b.len() != n * kb * s.blob() {
            return Err(format!("B has {} bytes, expected N*k_blocks*blob = {n}*{kb}*{}", self.b.len(), s.blob()));
        }
        if self.scales.len() != n * kb {
            return Err(format!("scales has {} elements, expected N*k_blocks = {n}*{kb}", self.scales.len()));
        }
        match self.zp {
            ZeroPoints::Default => {}
            ZeroPoints::Packed(z) if z.len() != n * s.zp_row() => {
                return Err(format!("uint8 zero_points has {} bytes, expected {n}*{}", z.len(), s.zp_row()))
            }
            ZeroPoints::Float(z) if z.len() != n * kb => {
                return Err(format!("float zero_points has {} elements, expected {n}*{kb}", z.len()))
            }
            _ => {}
        }
        Ok(())
    }

    /// Quantized value at column `n`, row `k < K`.
    fn q(&self, s: Shape, n: usize, k: usize) -> u8 {
        let base = n * s.kb() * s.blob();
        match s.bits {
            8 => self.b[base + k],
            _ => (self.b[base + k / 2] >> (4 * (k % 2))) & 0x0f,
        }
    }

    fn zero_point(&self, s: Shape, n: usize, b: usize) -> f32 {
        match self.zp {
            ZeroPoints::Default => s.default_zp(),
            ZeroPoints::Float(z) => z[n * s.kb() + b],
            ZeroPoints::Packed(z) => {
                let row = &z[n * s.zp_row()..];
                match s.bits {
                    8 => f32::from(row[b]),
                    _ => f32::from((row[b / 2] >> (4 * (b % 2))) & 0x0f),
                }
            }
        }
    }
}

/// The dequantized weight `W[K, N]`, row-major (the reference semantics).
pub fn dequantize(s: Shape, src: &Source<'_>) -> Vec<f32> {
    let mut w = vec![0f32; s.k * s.n];
    for n in 0..s.n {
        for b in 0..s.kb() {
            let (scale, zp) = (src.scales[n * s.kb() + b], src.zero_point(s, n, b));
            for k in b * s.block..((b + 1) * s.block).min(s.k) {
                w[k * s.n + n] = (f32::from(src.q(s, n, k)) - zp) * scale;
            }
        }
    }
    w
}

fn as_bytes(w: &[f32]) -> &[u8] {
    // SAFETY: any initialized f32 memory is valid as bytes; alignment 1.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<u8>(), std::mem::size_of_val(w)) }
}

fn as_bytes_mut(w: &mut [f32]) -> &mut [u8] {
    // SAFETY: as above; every byte pattern is a valid f32 bit pattern.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(w)) }
}

/// Word offsets of the three sections of a packed buffer.
fn sections(s: Shape) -> (usize, usize, usize, usize) {
    let qwords = (s.panels() * s.kb() * s.qblock()).div_ceil(4);
    let scales = s.panels() * s.kb() * NP;
    let q0 = HEADER;
    let sb0 = q0 + qwords;
    let zb0 = sb0 + scales;
    (q0, sb0, zb0, zb0 + scales)
}

/// Repack an ORT-layout weight for the kernels (see the module docs).
pub fn pack(s: Shape, src: &Source<'_>) -> Result<Vec<f32>, String> {
    src.check(s)?;
    let (q0, sb0, zb0, end) = sections(s);
    let mut w = vec![0f32; end];
    let header = [MAGIC, VERSION, s.n as u32, s.k as u32, s.bits as u32, s.block as u32];
    if [s.n, s.k].iter().any(|&d| d > u32::MAX as usize) {
        return Err("weight dimensions exceed u32".into());
    }
    for (i, h) in header.iter().enumerate() {
        w[i] = f32::from_bits(*h);
    }
    let (kb, block, qblock) = (s.kb(), s.block, s.qblock());
    let (head, scales) = w.split_at_mut(sb0);
    let q = &mut as_bytes_mut(&mut head[q0..])[..s.panels() * kb * qblock];
    let qv = |n: usize, k: usize| if n < s.n && k < s.k { src.q(s, n, k) } else { 0 };
    for p in 0..s.panels() {
        for b in 0..kb {
            let dst = &mut q[(p * kb + b) * qblock..(p * kb + b + 1) * qblock];
            let k0 = b * block;
            for (g, chunk) in dst.chunks_exact_mut(64).enumerate() {
                for c in 0..NP {
                    let n = p * NP + c;
                    for j in 0..4 {
                        chunk[c * 4 + j] = if s.bits == 8 {
                            qv(n, k0 + g * 4 + j)
                        } else {
                            qv(n, k0 + g * 8 + j) | (qv(n, k0 + g * 8 + 4 + j) << 4)
                        };
                    }
                }
            }
            for c in 0..NP {
                let n = p * NP + c;
                if n < s.n {
                    let scale = src.scales[n * kb + b];
                    let i = (p * kb + b) * NP + c;
                    scales[i] = scale;
                    scales[zb0 - sb0 + i] = scale * src.zero_point(s, n, b);
                }
            }
        }
    }
    Ok(w)
}

/// A packed weight, borrowed from its word buffer.
#[derive(Clone, Copy)]
pub struct Packed<'a> {
    pub s: Shape,
    q: &'a [u8],
    sb: &'a [f32],
    zb: &'a [f32],
}

impl<'a> Packed<'a> {
    /// Whether `w` starts with a packed-buffer header.
    pub fn is_packed(w: &[f32]) -> bool {
        w.len() >= HEADER && w[0].to_bits() == MAGIC
    }

    pub fn from_words(w: &'a [f32]) -> Result<Self, String> {
        if !Self::is_packed(w) || w[1].to_bits() != VERSION {
            return Err("not a packed MatMulNBits weight".into());
        }
        let h = |i: usize| w[i].to_bits() as usize;
        let s = Shape::new(h(2), h(3), h(4), h(5))?;
        let (q0, sb0, zb0, end) = sections(s);
        if w.len() != end {
            return Err(format!("packed MatMulNBits weight has {} words, expected {end}", w.len()));
        }
        let q = &as_bytes(&w[q0..sb0])[..s.panels() * s.kb() * s.qblock()];
        Ok(Packed { s, q, sb: &w[sb0..zb0], zb: &w[zb0..end] })
    }

    /// The dequantized weight `W[K, N]` (for the f32 path and tests).
    pub fn dequantize(&self) -> Vec<f32> {
        let s = self.s;
        let (kb, qblock) = (s.kb(), s.qblock());
        let mut w = vec![0f32; s.k * s.n];
        for n in 0..s.n {
            let (p, c) = (n / NP, n % NP);
            for k in 0..s.k {
                let b = k / s.block;
                let blk = &self.q[(p * kb + b) * qblock..];
                let kk = k % s.block;
                let q = if s.bits == 8 {
                    blk[(kk / 4) * 64 + c * 4 + kk % 4]
                } else {
                    let byte = blk[(kk / 8) * 64 + c * 4 + kk % 4];
                    if kk % 8 < 4 { byte & 0x0f } else { byte >> 4 }
                };
                let i = (p * kb + b) * NP + c;
                w[k * s.n + n] = f32::from(q) * self.sb[i] - self.zb[i];
            }
        }
        w
    }
}

// ── ISA selection ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
}

impl Isa {
    /// Rows per micro-tile.
    fn mr(self) -> usize {
        match self {
            Isa::Scalar => 4,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => 4,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => 6,
        }
    }
}

/// The best kernel this CPU supports (capped by `OXIONNX_NBITS_ISA`).
pub fn isa() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        let cap = std::env::var("OXIONNX_NBITS_ISA").unwrap_or_default().to_ascii_lowercase();
        detect(&cap)
    })
}

#[cfg(target_arch = "x86_64")]
fn detect(cap: &str) -> Isa {
    let vnni = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni");
    let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    match cap {
        "scalar" => Isa::Scalar,
        "avx2" if avx2 => Isa::Avx2,
        _ if vnni && cap != "avx2" => Isa::Avx512Vnni,
        _ if avx2 => Isa::Avx2,
        _ => Isa::Scalar,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect(_: &str) -> Isa {
    Isa::Scalar
}

/// Every ISA this CPU can run (for tests).
pub fn supported_isas() -> Vec<Isa> {
    #[allow(unused_mut)]
    let mut v = vec![Isa::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if detect("avx2") == Isa::Avx2 {
            v.push(Isa::Avx2);
        }
        if detect("") == Isa::Avx512Vnni {
            v.push(Isa::Avx512Vnni);
        }
    }
    v
}

// ── Activation quantization ──────────────────────────────────────────────────

/// Per-(row, block) int8 activations.
#[derive(Default)]
pub struct QuantA {
    m: usize,
    /// Row stride of `aq` (K rounded up to whole blocks).
    kp: usize,
    kb: usize,
    aq: Vec<i8>,
    /// Scale per row and block.
    sa: Vec<f32>,
    /// `sa * Σ qa` per row and block.
    ss: Vec<f32>,
}

impl QuantA {
    /// Quantize `a` (`m x k`, row-major) with blocks of `block` along K.
    pub fn quantize(&mut self, isa: Isa, a: &[f32], m: usize, k: usize, block: usize, threads: usize) {
        let kb = k.div_ceil(block);
        let kp = kb * block;
        self.m = m;
        self.kp = kp;
        self.kb = kb;
        self.aq.resize(m * kp, 0);
        self.sa.resize(m * kb, 0.0);
        self.ss.resize(m * kb, 0.0);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if threads > 1 && m * k >= 64 * 1024 {
            use rayon::prelude::*;
            let rows_per_task = (32 * 1024 / k.max(1)).max(1);
            self.aq
                .par_chunks_mut(kp * rows_per_task)
                .zip(self.sa.par_chunks_mut(kb * rows_per_task))
                .zip(self.ss.par_chunks_mut(kb * rows_per_task))
                .enumerate()
                .for_each(|(t, ((aq, sa), ss))| {
                    let i0 = t * rows_per_task;
                    let rows = aq.chunks_mut(kp).zip(sa.chunks_mut(kb)).zip(ss.chunks_mut(kb));
                    for (j, ((aq, sa), ss)) in rows.enumerate() {
                        let i = i0 + j;
                        quantize_row(isa, &a[i * k..(i + 1) * k], block, aq, sa, ss);
                    }
                });
            return;
        }
        let _ = threads;
        let rows = self.aq.chunks_mut(kp).zip(self.sa.chunks_mut(kb)).zip(self.ss.chunks_mut(kb));
        for (i, ((aq, sa), ss)) in rows.enumerate() {
            quantize_row(isa, &a[i * k..(i + 1) * k], block, aq, sa, ss);
        }
    }
}

/// Quantize one row; `aq` has whole blocks (the tail past `a.len()` is zeroed).
fn quantize_row(isa: Isa, a: &[f32], block: usize, aq: &mut [i8], sa: &mut [f32], ss: &mut [f32]) {
    for (b, (q, (sa, ss))) in aq.chunks_exact_mut(block).zip(sa.iter_mut().zip(ss.iter_mut())).enumerate() {
        let src = &a[(b * block).min(a.len())..((b + 1) * block).min(a.len())];
        let amax = src.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        *sa = amax / 127.0;
        let sum = match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 | Isa::Avx512Vnni if src.len() == block && block % 32 == 0 => {
                // SAFETY: AVX2 is implied by both ISAs (checked at detection).
                unsafe { quantize_block_avx2(src, inv, q) }
            }
            _ => {
                let mut sum = 0i32;
                for (i, d) in q.iter_mut().enumerate() {
                    let v = src.get(i).map_or(0, |&x| round_even(x * inv) as i32);
                    *d = v as i8;
                    sum += v;
                }
                sum
            }
        };
        *ss = *sa * sum as f32;
    }
}

/// Round half to even (as `cvtps2dq` in the default MXCSR mode), for `|x| < 2^22`.
#[inline]
fn round_even(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23
    (x + MAGIC) - MAGIC
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_block_avx2(src: &[f32], inv: f32, q: &mut [i8]) -> i32 {
    use std::arch::x86_64::*;
    let vinv = _mm256_set1_ps(inv);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let mut sum = _mm256_setzero_si256();
    for (s, d) in src.chunks_exact(32).zip(q.chunks_exact_mut(32)) {
        let p = s.as_ptr();
        // Round to nearest even (the default MXCSR mode), as `round_even`.
        let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p), vinv));
        let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(8)), vinv));
        let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(16)), vinv));
        let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(24)), vinv));
        sum = _mm256_add_epi32(sum, _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        let w0 = _mm256_packs_epi32(i0, i1);
        let w1 = _mm256_packs_epi32(i2, i3);
        let b = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(w0, w1), perm);
        _mm256_storeu_si256(d.as_mut_ptr().cast(), b);
    }
    let s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

// ── GEMM ─────────────────────────────────────────────────────────────────────

/// One micro-tile: `C[R, nb] = A[R, :] @ W[:, panels q[0], q[1]]` (+ bias).
#[derive(Clone, Copy)]
struct Micro {
    kb: usize,
    block: usize,
    /// Row `r`, block `b` at `a + r * lda + b * block`.
    a: *const i8,
    lda: usize,
    /// Row `r`, block `b` at `+ r * kb + b`.
    sa: *const f32,
    ss: *const f32,
    /// Panel `v`, block `b` at `q[v] + b * qblock`.
    q: [*const u8; 2],
    qblock: usize,
    /// Panel `v`, block `b` at `+ b * NP`.
    sb: [*const f32; 2],
    zb: [*const f32; 2],
    c: *mut f32,
    ldc: usize,
    /// Valid columns (`<= NP * panels`).
    nb: usize,
    /// Null, or the bias of the first column.
    bias: *const f32,
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

thread_local! {
    static SCRATCH: std::cell::Cell<QuantA> = std::cell::Cell::new(QuantA::default());
}

/// `c` (`m x N`, row-major) = `a` (`m x K`, row-major) @ `W` (+ `bias`), with
/// int8-quantized activations. Uses up to `threads` rayon tasks.
pub fn matmul(isa: Isa, m: usize, a: &[f32], w: &Packed<'_>, bias: Option<&[f32]>, c: &mut [f32], threads: usize) {
    let s = w.s;
    assert_eq!(a.len(), m * s.k, "MatMulNBits: A is not m x K");
    assert_eq!(c.len(), m * s.n, "MatMulNBits: C is not m x N");
    assert!(bias.map_or(true, |b| b.len() == s.n), "MatMulNBits: bias is not [N]");
    if m == 0 {
        return;
    }
    // Taken out rather than borrowed: while this thread waits on its rayon
    // tasks it may run another MatMulNBits job, which then gets a fresh buffer.
    let mut qa = SCRATCH.with(|q| q.take());
    qa.quantize(isa, a, m, s.k, s.block, threads);
    gemm_quantized(isa, &qa, w, bias, c, threads);
    SCRATCH.with(|q| q.replace(qa));
}

fn gemm_quantized(isa: Isa, qa: &QuantA, w: &Packed<'_>, bias: Option<&[f32]>, c: &mut [f32], threads: usize) {
    let (s, m) = (w.s, qa.m);
    let mr = isa.mr();
    let tasks = (m * s.n * s.k / MIN_MACS_PER_TASK).clamp(1, threads.max(1));
    // Units of two panels, so the AVX-512 kernel always gets whole pairs.
    let units = s.panels().div_ceil(2);
    let strips = m.div_ceil(mr);
    // Split along M only when there are many rows (each row tile streams all of
    // its weight columns), and only by a divisor of the task count.
    let max_mtiles = (m / MIN_ROWS_PER_TILE).clamp(1, strips);
    let mt = (1..=max_mtiles.min(tasks)).rev().find(|d| tasks % d == 0).unwrap_or(1);
    let rows_per_tile = strips.div_ceil(mt) * mr;
    let mtiles = m.div_ceil(rows_per_tile);
    let splits = (tasks / mt).clamp(1, units);
    let units_per_tile = units.div_ceil(splits);
    let ntiles = units.div_ceil(units_per_tile);
    let cp = SendPtr(c.as_mut_ptr());
    let cp = &cp;
    let one = |t: usize| {
        let (ti, tj) = (t / ntiles, t % ntiles);
        let i0 = ti * rows_per_tile;
        let mb = rows_per_tile.min(m - i0);
        let p0 = tj * units_per_tile * 2;
        let p1 = ((tj + 1) * units_per_tile * 2).min(s.panels());
        // SAFETY: tiles cover disjoint (row, panel) ranges of C, inside its m x N
        // bounds; `isa` was detected on this CPU.
        unsafe { tile(isa, qa, w, bias, i0, mb, p0, p1, cp.0) };
    };
    let total = mtiles * ntiles;
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if tasks > 1 && total > 1 {
        use rayon::prelude::*;
        (0..total).into_par_iter().for_each(one);
        return;
    }
    (0..total).for_each(one);
}

#[allow(clippy::too_many_arguments)]
unsafe fn tile(isa: Isa, qa: &QuantA, w: &Packed<'_>, bias: Option<&[f32]>, i0: usize, mb: usize, p0: usize, p1: usize, c: *mut f32) {
    let s = w.s;
    let (kb, qblock, mr) = (s.kb(), s.qblock(), isa.mr());
    // Only the AVX-512 kernel takes two panels at once.
    #[cfg(target_arch = "x86_64")]
    let wide = isa == Isa::Avx512Vnni;
    #[cfg(not(target_arch = "x86_64"))]
    let wide = false;
    let mut s0 = p0;
    while s0 < p1 {
        let s1 = (s0 + SLICE_PANELS).min(p1);
        let mut i = 0;
        while i < mb {
            let r = mr.min(mb - i);
            let row = i0 + i;
            let mut p = s0;
            while p < s1 {
                let nv = if wide && p + 1 < s1 { 2 } else { 1 };
                let pv = [p, p + nv - 1];
                let micro = Micro {
                    kb,
                    block: s.block,
                    a: qa.aq.as_ptr().add(row * qa.kp),
                    lda: qa.kp,
                    sa: qa.sa.as_ptr().add(row * kb),
                    ss: qa.ss.as_ptr().add(row * kb),
                    q: pv.map(|p| w.q.as_ptr().add(p * kb * qblock)),
                    qblock,
                    sb: pv.map(|p| w.sb.as_ptr().add(p * kb * NP)),
                    zb: pv.map(|p| w.zb.as_ptr().add(p * kb * NP)),
                    c: c.add(row * s.n + p * NP),
                    ldc: s.n,
                    nb: (nv * NP).min(s.n - p * NP),
                    bias: bias.map_or(std::ptr::null(), |b| b.as_ptr().add(p * NP)),
                };
                match isa {
                    Isa::Scalar => scalar::kernel(r, s.bits, &micro),
                    #[cfg(target_arch = "x86_64")]
                    Isa::Avx2 => avx2::dispatch(r, s.bits, &micro),
                    #[cfg(target_arch = "x86_64")]
                    Isa::Avx512Vnni => avx512::dispatch(r, nv, s.bits, &micro),
                }
                p += nv;
            }
            i += r;
        }
        s0 = s1;
    }
}

#[allow(clippy::needless_range_loop)]
mod scalar {
    use super::{Micro, NP};

    pub unsafe fn kernel(rows: usize, bits: usize, k: &Micro) {
        let mut acc = [0f32; NP];
        for r in 0..rows {
            for c in 0..k.nb {
                let mut a = 0f32;
                for b in 0..k.kb {
                    let ar = k.a.add(r * k.lda + b * k.block);
                    let blk = k.q[0].add(b * k.qblock);
                    let mut isum = 0i32;
                    for kk in 0..k.block {
                        let q = if bits == 8 {
                            *blk.add((kk / 4) * 64 + c * 4 + kk % 4)
                        } else {
                            let byte = *blk.add((kk / 8) * 64 + c * 4 + kk % 4);
                            if kk % 8 < 4 { byte & 0x0f } else { byte >> 4 }
                        };
                        isum += i32::from(*ar.add(kk)) * i32::from(q);
                    }
                    let (sa, ss) = (*k.sa.add(r * k.kb + b), *k.ss.add(r * k.kb + b));
                    let (sb, zb) = (*k.sb[0].add(b * NP + c), *k.zb[0].add(b * NP + c));
                    a = (isum as f32).mul_add(sb * sa, a);
                    a = (-ss).mul_add(zb, a);
                }
                acc[c] = a;
            }
            let cr = k.c.add(r * k.ldc);
            for (c, &v) in acc.iter().enumerate().take(k.nb) {
                *cr.add(c) = if k.bias.is_null() { v } else { v + *k.bias.add(c) };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::needless_range_loop)]
mod avx2 {
    use super::{Micro, NP};
    use std::arch::x86_64::*;

    pub unsafe fn dispatch(r: usize, bits: usize, k: &Micro) {
        match (r, bits) {
            (4, 8) => kernel::<4, false>(k),
            (3, 8) => kernel::<3, false>(k),
            (2, 8) => kernel::<2, false>(k),
            (_, 8) => kernel::<1, false>(k),
            (4, _) => kernel::<4, true>(k),
            (3, _) => kernel::<3, true>(k),
            (2, _) => kernel::<2, true>(k),
            _ => kernel::<1, true>(k),
        }
    }

    /// One panel, as two 8-column halves, R rows at a time.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn kernel<const R: usize, const B4: bool>(k: &Micro) {
        let ones = _mm256_set1_epi16(1);
        let flip = _mm256_set1_epi8(-128);
        let low = _mm256_set1_epi8(0x0f);
        let mut tmp = [0f32; NP];
        let mut out = [[0f32; NP]; R];
        for h in 0..2 {
            if h * 8 >= k.nb {
                break;
            }
            let mut acc = [_mm256_setzero_ps(); R];
            for b in 0..k.kb {
                let mut isum = [_mm256_setzero_si256(); R];
                let qb = k.q[0].add(b * k.qblock + h * 32);
                let ab = k.a.add(b * k.block);
                if B4 {
                    for g in 0..k.block / 8 {
                        let raw = _mm256_loadu_si256(qb.add(g * 64).cast());
                        let lo = _mm256_and_si256(raw, low);
                        let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), low);
                        for r in 0..R {
                            let ap = ab.add(r * k.lda + g * 8);
                            let a0 = _mm256_set1_epi32(std::ptr::read_unaligned(ap.cast::<i32>()));
                            let a1 = _mm256_set1_epi32(std::ptr::read_unaligned(ap.add(4).cast::<i32>()));
                            // |q| <= 15, so the u8 x s8 pair sums cannot saturate.
                            let p = _mm256_add_epi16(_mm256_maddubs_epi16(lo, a0), _mm256_maddubs_epi16(hi, a1));
                            isum[r] = _mm256_add_epi32(isum[r], _mm256_madd_epi16(p, ones));
                        }
                    }
                } else {
                    for g in 0..k.block / 4 {
                        // q - 128 as s8; multiply |q - 128| (u8 <= 128) by the
                        // activation carrying its sign (|a| <= 127): no saturation.
                        let qs = _mm256_xor_si256(_mm256_loadu_si256(qb.add(g * 64).cast()), flip);
                        let qa = _mm256_abs_epi8(qs);
                        for r in 0..R {
                            let ap = ab.add(r * k.lda + g * 4);
                            let av = _mm256_set1_epi32(std::ptr::read_unaligned(ap.cast::<i32>()));
                            let p = _mm256_maddubs_epi16(qa, _mm256_sign_epi8(av, qs));
                            isum[r] = _mm256_add_epi32(isum[r], _mm256_madd_epi16(p, ones));
                        }
                    }
                }
                let sb = _mm256_loadu_ps(k.sb[0].add(b * NP + h * 8));
                let mut zb = _mm256_loadu_ps(k.zb[0].add(b * NP + h * 8));
                if !B4 {
                    // Σ a (q - zp) = Σ a (q - 128) - (zp - 128) Σ a.
                    zb = _mm256_fnmadd_ps(_mm256_set1_ps(128.0), sb, zb);
                }
                for r in 0..R {
                    let sa = _mm256_set1_ps(*k.sa.add(r * k.kb + b));
                    let ss = _mm256_set1_ps(-*k.ss.add(r * k.kb + b));
                    acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[r]), _mm256_mul_ps(sb, sa), acc[r]);
                    acc[r] = _mm256_fmadd_ps(ss, zb, acc[r]);
                }
            }
            for r in 0..R {
                _mm256_storeu_ps(out[r].as_mut_ptr().add(h * 8), acc[r]);
            }
        }
        for (r, row) in out.iter().enumerate() {
            let cr = k.c.add(r * k.ldc);
            if k.nb == NP {
                let mut x0 = _mm256_loadu_ps(row.as_ptr());
                let mut x1 = _mm256_loadu_ps(row.as_ptr().add(8));
                if !k.bias.is_null() {
                    x0 = _mm256_add_ps(x0, _mm256_loadu_ps(k.bias));
                    x1 = _mm256_add_ps(x1, _mm256_loadu_ps(k.bias.add(8)));
                }
                _mm256_storeu_ps(cr, x0);
                _mm256_storeu_ps(cr.add(8), x1);
            } else {
                tmp[..k.nb].copy_from_slice(&row[..k.nb]);
                for (c, &v) in tmp.iter().enumerate().take(k.nb) {
                    *cr.add(c) = if k.bias.is_null() { v } else { v + *k.bias.add(c) };
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::needless_range_loop)]
mod avx512 {
    use super::{Micro, NP};
    use std::arch::x86_64::*;

    pub unsafe fn dispatch(r: usize, nv: usize, bits: usize, k: &Micro) {
        macro_rules! go {
            ($nv:literal, $b4:literal) => {
                match r {
                    6 => kernel::<6, $nv, $b4>(k),
                    5 => kernel::<5, $nv, $b4>(k),
                    4 => kernel::<4, $nv, $b4>(k),
                    3 => kernel::<3, $nv, $b4>(k),
                    2 => kernel::<2, $nv, $b4>(k),
                    _ => kernel::<1, $nv, $b4>(k),
                }
            };
        }
        match (nv, bits) {
            (2, 8) => go!(2, false),
            (_, 8) => go!(1, false),
            (2, _) => go!(2, true),
            _ => go!(1, true),
        }
    }

    #[inline(always)]
    unsafe fn bcast(p: *const i8) -> __m512i {
        _mm512_set1_epi32(std::ptr::read_unaligned(p.cast::<i32>()))
    }

    /// R rows x NV panels of 16 columns.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn kernel<const R: usize, const NV: usize, const B4: bool>(k: &Micro) {
        let low = _mm512_set1_epi8(0x0f);
        let mut acc = [[_mm512_setzero_ps(); NV]; R];
        for b in 0..k.kb {
            let mut isum = [[_mm512_setzero_si512(); NV]; R];
            let ab = k.a.add(b * k.block);
            let qoff = b * k.qblock;
            if B4 {
                for g in 0..k.block / 8 {
                    let mut lo = [_mm512_setzero_si512(); NV];
                    let mut hi = [_mm512_setzero_si512(); NV];
                    for v in 0..NV {
                        let raw = _mm512_loadu_si512(k.q[v].add(qoff + g * 64).cast());
                        lo[v] = _mm512_and_si512(raw, low);
                        hi[v] = _mm512_and_si512(_mm512_srli_epi16(raw, 4), low);
                    }
                    for r in 0..R {
                        let ap = ab.add(r * k.lda + g * 8);
                        let (a0, a1) = (bcast(ap), bcast(ap.add(4)));
                        for v in 0..NV {
                            isum[r][v] = _mm512_dpbusd_epi32(isum[r][v], lo[v], a0);
                            isum[r][v] = _mm512_dpbusd_epi32(isum[r][v], hi[v], a1);
                        }
                    }
                }
            } else {
                for g in 0..k.block / 4 {
                    let mut qv = [_mm512_setzero_si512(); NV];
                    for v in 0..NV {
                        qv[v] = _mm512_loadu_si512(k.q[v].add(qoff + g * 64).cast());
                    }
                    for r in 0..R {
                        let av = bcast(ab.add(r * k.lda + g * 4));
                        for v in 0..NV {
                            isum[r][v] = _mm512_dpbusd_epi32(isum[r][v], qv[v], av);
                        }
                    }
                }
            }
            for v in 0..NV {
                let sb = _mm512_loadu_ps(k.sb[v].add(b * NP));
                let zb = _mm512_loadu_ps(k.zb[v].add(b * NP));
                for r in 0..R {
                    let sa = _mm512_set1_ps(*k.sa.add(r * k.kb + b));
                    let ss = _mm512_set1_ps(-*k.ss.add(r * k.kb + b));
                    acc[r][v] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[r][v]), _mm512_mul_ps(sb, sa), acc[r][v]);
                    acc[r][v] = _mm512_fmadd_ps(ss, zb, acc[r][v]);
                }
            }
        }
        let mask = |v: usize| -> __mmask16 {
            let n = k.nb.saturating_sub(v * NP).min(NP);
            if n == NP { 0xffff } else { (1u16 << n) - 1 }
        };
        for v in 0..NV {
            let m = mask(v);
            let bias = if k.bias.is_null() { _mm512_setzero_ps() } else { _mm512_maskz_loadu_ps(m, k.bias.add(v * NP)) };
            for r in 0..R {
                _mm512_mask_storeu_ps(k.c.add(r * k.ldc + v * NP), m, _mm512_add_ps(acc[r][v], bias));
            }
        }
    }
}

// ── GatherBlockQuantized ─────────────────────────────────────────────────────

/// A uint8 block-quantized table, quantized along its rows.
pub struct Table<'a> {
    /// Row `r` starts at `data[r * stride]`.
    pub data: &'a [u8],
    pub stride: usize,
    pub rows: usize,
    /// Elements per row.
    pub cols: usize,
    pub bits: usize,
    pub block: usize,
    /// `[rows, ceil(cols / block)]`.
    pub scales: &'a [f32],
    /// Packed uint8 (`[rows, ceil(blocks * bits / 8)]`) or float `[rows, blocks]`.
    pub zp: ZeroPoints<'a>,
}

impl Table<'_> {
    pub fn check(&self) -> Result<(), String> {
        if !matches!(self.bits, 2 | 4 | 8) {
            return Err(format!("bits={} is not supported (2, 4 or 8)", self.bits));
        }
        if self.block == 0 {
            return Err("block_size must be positive".into());
        }
        let row_bytes = (self.cols * self.bits).div_ceil(8);
        let kb = self.cols.div_ceil(self.block);
        if self.rows > 0 && (self.stride < row_bytes || self.data.len() < (self.rows - 1) * self.stride + row_bytes) {
            return Err(format!("data is too short for {} rows of {row_bytes} bytes", self.rows));
        }
        if self.scales.len() != self.rows * kb {
            return Err(format!("scales has {} elements, expected {}*{kb}", self.scales.len(), self.rows));
        }
        match self.zp {
            ZeroPoints::Packed(z) if z.len() != self.rows * (kb * self.bits).div_ceil(8) => {
                Err(format!("zero_points has {} bytes, expected {}*{}", z.len(), self.rows, (kb * self.bits).div_ceil(8)))
            }
            ZeroPoints::Float(z) if z.len() != self.rows * kb => {
                Err(format!("zero_points has {} elements, expected {}*{kb}", z.len(), self.rows))
            }
            _ => Ok(()),
        }
    }

    /// Dequantize row `r` into `out` (`cols` elements).
    pub fn row(&self, r: usize, out: &mut [f32]) {
        let (bits, kb) = (self.bits, self.cols.div_ceil(self.block));
        let data = &self.data[r * self.stride..];
        let per = 8 / bits;
        let mask = ((1u16 << bits) - 1) as u8;
        for b in 0..kb {
            let scale = self.scales[r * kb + b];
            let zp = match self.zp {
                ZeroPoints::Default => (1u32 << (bits - 1)) as f32,
                ZeroPoints::Float(z) => z[r * kb + b],
                ZeroPoints::Packed(z) => {
                    let zr = &z[r * (kb * bits).div_ceil(8)..];
                    f32::from((zr[b / per] >> (bits * (b % per))) & mask)
                }
            };
            for i in b * self.block..((b + 1) * self.block).min(self.cols) {
                let q = (data[i / per] >> (bits * (i % per))) & mask;
                out[i] = (f32::from(q) - zp) * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn weight(s: Shape, zp_kind: u8, rng: &mut Lcg) -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>) {
        let b: Vec<u8> = (0..s.n * s.kb() * s.blob()).map(|_| rng.next() as u8).collect();
        let scales: Vec<f32> = (0..s.n * s.kb()).map(|_| 0.01 + rng.f().abs() * 0.05).collect();
        let zpu: Vec<u8> = if zp_kind == 1 { (0..s.n * s.zp_row()).map(|_| rng.next() as u8).collect() } else { vec![] };
        let zpf: Vec<f32> = if zp_kind == 2 { (0..s.n * s.kb()).map(|_| (rng.next() % 256) as f32 * 0.37).collect() } else { vec![] };
        (b, scales, zpu, zpf)
    }

    fn naive(m: usize, k: usize, n: usize, a: &[f32], w: &[f32]) -> Vec<f64> {
        let mut c = vec![0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += f64::from(a[i * k + p]) * f64::from(w[p * n + j]);
                }
            }
        }
        c
    }

    /// Exact reference for the quantized-activation math.
    fn quantized_reference(a: &[f32], m: usize, s: Shape, w: &[f32]) -> Vec<f64> {
        let mut qa = QuantA::default();
        qa.quantize(Isa::Scalar, a, m, s.k, s.block, 1);
        let mut deq = vec![0f32; m * s.k];
        for i in 0..m {
            for p in 0..s.k {
                deq[i * s.k + p] = f32::from(qa.aq[i * qa.kp + p]) * qa.sa[i * qa.kb + p / s.block];
            }
        }
        naive(m, s.k, s.n, &deq, w)
    }

    #[test]
    fn pack_round_trips_to_the_reference_dequantization() {
        let mut rng = Lcg(7);
        for &(n, k, bits, block) in &[(1, 16, 4, 16), (17, 40, 4, 16), (33, 100, 8, 32), (16, 64, 8, 64), (5, 7, 4, 32)] {
            let s = Shape::new(n, k, bits, block).unwrap();
            for zp_kind in 0..3 {
                let (b, scales, zpu, zpf) = weight(s, zp_kind, &mut rng);
                let zp = match zp_kind {
                    1 => ZeroPoints::Packed(&zpu),
                    2 => ZeroPoints::Float(&zpf),
                    _ => ZeroPoints::Default,
                };
                let src = Source { b: &b, scales: &scales, zp };
                let want = dequantize(s, &src);
                let words = pack(s, &src).unwrap();
                let got = Packed::from_words(&words).unwrap().dequantize();
                for (x, y) in got.iter().zip(&want) {
                    assert!((x - y).abs() <= 1e-5 * (1.0 + y.abs()), "{s:?} zp {zp_kind}: {x} vs {y}");
                }
            }
        }
    }

    #[test]
    fn kernels_match_the_references() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut rng = Lcg(11);
        let cases = [
            (1, 16, 4, 16, 1),
            (3, 40, 8, 16, 1),
            (7, 100, 4, 32, 33),
            (13, 64, 8, 32, 47),
            (6, 512, 8, 32, 64),
            (19, 513, 4, 32, 70),
            (200, 96, 8, 32, 130),
            (2, 256, 4, 128, 16),
            (5, 72, 8, 64, 31),
        ];
        for &(m, k, bits, block, n) in &cases {
            let s = Shape::new(n, k, bits, block).unwrap();
            for zp_kind in 0..3 {
                let (b, scales, zpu, zpf) = weight(s, zp_kind, &mut rng);
                let zp = match zp_kind {
                    1 => ZeroPoints::Packed(&zpu),
                    2 => ZeroPoints::Float(&zpf),
                    _ => ZeroPoints::Default,
                };
                let src = Source { b: &b, scales: &scales, zp };
                let w = dequantize(s, &src);
                let words = pack(s, &src).unwrap();
                let packed = Packed::from_words(&words).unwrap();
                let a: Vec<f32> = (0..m * k).map(|_| rng.f() * 2.0).collect();
                let bias: Vec<f32> = (0..n).map(|_| rng.f()).collect();
                let exact = naive(m, k, n, &a, &w);
                let qref = quantized_reference(&a, m, s, &w);
                // Scale for the tolerances: the typical magnitude of an output.
                let norm = (exact.iter().map(|x| x * x).sum::<f64>() / exact.len() as f64).sqrt().max(1e-3);
                for isa in supported_isas() {
                    for threads in [1, 4] {
                        for with_bias in [false, true] {
                            let mut c = vec![f32::NAN; m * n];
                            let bias = with_bias.then_some(&bias[..]);
                            pool.install(|| matmul(isa, m, &a, &packed, bias, &mut c, threads));
                            let mut err2 = 0f64;
                            for i in 0..m * n {
                                let add = bias.map_or(0.0, |b| f64::from(b[i % n]));
                                let got = f64::from(c[i]);
                                assert!(
                                    (got - qref[i] - add).abs() <= 2e-5 * norm * (k as f64).sqrt(),
                                    "{isa:?} {s:?} m={m} zp={zp_kind} t={threads} [{i}]: {got} vs quantized ref {}",
                                    qref[i] + add
                                );
                                err2 += (got - exact[i] - add).powi(2);
                            }
                            // int8 activations: within ~1% (RMS) of the exact product.
                            let rel = (err2 / (m * n) as f64).sqrt() / norm;
                            assert!(rel < 0.02, "{isa:?} {s:?} m={m} zp={zp_kind}: RMS error {rel} vs exact");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(Shape::new(4, 16, 3, 16).is_err());
        assert!(Shape::new(4, 16, 4, 24).is_err());
        let s = Shape::new(4, 32, 4, 16).unwrap();
        let b = vec![0u8; 4 * 2 * 8];
        let scales = vec![1f32; 7];
        assert!(pack(s, &Source { b: &b, scales: &scales, zp: ZeroPoints::Default }).is_err());
        assert!(Packed::from_words(&[0.0; 20]).is_err());
    }

    #[test]
    fn table_rows_dequantize() {
        // 2 rows of 40 4-bit values, blocks of 16 (3 blocks, the last partial).
        let data: Vec<u8> = (0..2 * 20).map(|i| (i * 37 % 256) as u8).collect();
        let scales: Vec<f32> = (0..6).map(|i| 0.5 + i as f32).collect();
        let zp: Vec<u8> = vec![0x21, 0x03, 0x54, 0x06];
        let t = Table { data: &data, stride: 20, rows: 2, cols: 40, bits: 4, block: 16, scales: &scales, zp: ZeroPoints::Packed(&zp) };
        t.check().unwrap();
        let mut out = vec![0f32; 40];
        t.row(1, &mut out);
        for (i, &o) in out.iter().enumerate() {
            let byte = data[20 + i / 2];
            let q = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
            let b = i / 16;
            let z = [4u8, 5, 6][b];
            assert_eq!(o, (f32::from(q) - f32::from(z)) * scales[3 + b]);
        }
    }
}
