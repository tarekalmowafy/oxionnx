//! Elementwise activation kernels: GELU (tanh approximation and exact erf
//! form), SiLU and erf, with an optional elementwise multiplier (the gated
//! MLP epilogues of `fused_matmul`).
//!
//! * tanh GELU `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` is evaluated
//!   as `x * sigmoid(z)` with `z = x (A + B x^2)`, `A = 2 sqrt(2/pi)`,
//!   `B = 0.044715 A` (the same function: `0.5 (1 + tanh(t)) = sigmoid(2t)`);
//! * SiLU `x * sigmoid(x)` is the same form with `z = x`;
//! * exact GELU `0.5 x (1 + erf(x / sqrt 2))` and `erf` use
//!   `erfc(u) = t exp(-u^2 + P(t))`, `t = 1 / (1 + u / 2)` (Numerical Recipes'
//!   `erfcc`, fractional error < 1.2e-7 for all `u >= 0`), and take
//!   `1 + erf(x) = erfc(-x)` directly for negative `x`, so the tail does not
//!   cancel.
//!
//! The exp is a degree-6 polynomial after range reduction and the sigmoid
//! uses a refined reciprocal: |error| <= ~3e-7 relative to f64 for the
//! sigmoid forms, ~5e-7 for the erf forms. AVX-512F and AVX2+FMA kernels are
//! picked once per process; other targets use the scalar formulas, which the
//! vector kernels follow lane by lane.

#![allow(unsafe_code)]

const A: f32 = 1.595_769_1; // 2 * sqrt(2 / pi)
const B: f32 = 0.071_354_816; // 0.044715 * A
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const C: [f32; 6] = [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0];
/// `erfcc` polynomial in `t`, highest degree first (then `-1.26551223` at degree 0).
const ERFC: [f32; 10] = [
    0.170_872_77,
    -0.822_152_23,
    1.488_515_9,
    -1.135_204,
    0.278_868_07,
    -0.186_288_06,
    0.096_784_18,
    0.374_091_96,
    1.000_023_7,
    -1.265_512_2,
];

/// The activation a kernel applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// tanh-approximation GELU.
    GeluTanh,
    /// Exact (erf) GELU.
    GeluErf,
    /// `x * sigmoid(x)`.
    Silu,
    /// The error function.
    Erf,
}

impl Kind {
    const fn id(self) -> u8 {
        match self {
            Kind::GeluTanh => 0,
            Kind::GeluErf => 1,
            Kind::Silu => 2,
            Kind::Erf => 3,
        }
    }
}

/// `exp(t)` for `t` in `[-88, 88]` (the kernels' polynomial).
#[inline]
fn exp_poly(t: f32) -> f32 {
    let n = (t * std::f32::consts::LOG2_E).round();
    let r = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, t));
    let mut p = C[0];
    for &c in &C[1..] {
        p = p.mul_add(r, c);
    }
    p.mul_add(r, 1.0) * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// `x * sigmoid(z)`.
#[inline]
fn x_sigmoid(x: f32, z: f32) -> f32 {
    x / (1.0 + exp_poly((-z).clamp(-88.0, 88.0)))
}

/// `erfc(u)` for `u >= 0`.
#[inline]
fn erfc_pos(u: f32) -> f32 {
    let t = 1.0 / 0.5f32.mul_add(u, 1.0);
    let mut p = ERFC[0];
    for &c in &ERFC[1..] {
        p = p.mul_add(t, c);
    }
    t * exp_poly((-u).mul_add(u, p).max(-88.0))
}

/// Scalar tanh-approximation GELU (the kernels' formula, lane by lane).
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    x_sigmoid(x, x * B.mul_add(x * x, A))
}

/// Scalar exact GELU `0.5 x (1 + erf(x / sqrt 2))`.
#[inline]
pub fn gelu_erf(x: f32) -> f32 {
    let q = erfc_pos(x.abs() * std::f32::consts::FRAC_1_SQRT_2);
    0.5 * x * if x >= 0.0 { 2.0 - q } else { q }
}

/// Scalar SiLU `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x_sigmoid(x, x)
}

/// Scalar erf.
#[inline]
pub fn erf(x: f32) -> f32 {
    let q = erfc_pos(x.abs());
    if x >= 0.0 {
        1.0 - q
    } else {
        q - 1.0
    }
}

/// Scalar form of `kind`.
#[inline]
pub fn eval(kind: Kind, x: f32) -> f32 {
    match kind {
        Kind::GeluTanh => gelu_tanh(x),
        Kind::GeluErf => gelu_erf(x),
        Kind::Silu => silu(x),
        Kind::Erf => erf(x),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

fn kernel() -> Kernel {
    static K: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Kernel::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Kernel::Avx2;
            }
        }
        Kernel::Scalar
    })
}

/// True when a vector kernel is used (so callers may prefer these over the
/// generic SIMD GELU).
pub fn vectorized() -> bool {
    kernel() != Kernel::Scalar
}

/// `out[i] = act(x[i]) * mul[i]` (or just `act(x[i])` without `mul`).
pub fn apply(kind: Kind, x: &[f32], mul: Option<&[f32]>, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    if let Some(m) = mul {
        assert_eq!(m.len(), out.len());
    }
    // SAFETY: lengths checked above; the kernel was detected on this CPU.
    unsafe { apply_raw(kind, x.as_ptr(), mul.map(|m| m.as_ptr()), out.as_mut_ptr(), out.len()) }
}

/// Elements per parallel task for a tensor of `len` elements, or `None` to run
/// on the calling thread.
fn par_chunk(len: usize) -> Option<usize> {
    let t = crate::par::threads();
    let min = crate::par::min_elems();
    if t <= 1 || len < 2 * min || cfg!(all(target_arch = "wasm32", not(feature = "wasm-threads"))) {
        return None;
    }
    // Multiple of 16 so only the last chunk takes the masked tail.
    Some(min.max(len.div_ceil(4 * t)).next_multiple_of(16))
}

/// [`apply`] split across the current rayon pool when large enough.
pub fn apply_par(kind: Kind, x: &[f32], mul: Option<&[f32]>, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let Some(chunk) = par_chunk(x.len()) else { return apply(kind, x, mul, out) };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, o)| {
            let r = i * chunk..i * chunk + o.len();
            apply(kind, &x[r.clone()], mul.map(|m| &m[r]), o);
        });
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let _ = chunk;
}

/// In-place `data[i] = act(data[i])`, split across the current rayon pool when
/// large enough.
pub fn apply_inplace_par(kind: Kind, data: &mut [f32]) {
    let inplace = |d: &mut [f32]| {
        let p = d.as_mut_ptr();
        // SAFETY: element i is read before it is written, by the same lane.
        unsafe { apply_raw(kind, p, None, p, d.len()) }
    };
    let Some(chunk) = par_chunk(data.len()) else { return inplace(data) };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(chunk).for_each(inplace);
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let _ = chunk;
}

/// `out[i] = gelu_tanh(x[i]) * mul[i]` (or just `gelu_tanh(x[i])` without `mul`).
pub fn gelu_mul(x: &[f32], mul: Option<&[f32]>, out: &mut [f32]) {
    apply(Kind::GeluTanh, x, mul, out)
}

/// [`gelu_mul`] split across the current rayon pool when large enough.
pub fn gelu_mul_par(x: &[f32], mul: Option<&[f32]>, out: &mut [f32]) {
    apply_par(Kind::GeluTanh, x, mul, out)
}

/// In-place `data[i] = gelu_tanh(data[i])`.
pub fn gelu_inplace(data: &mut [f32]) {
    let p = data.as_mut_ptr();
    // SAFETY: element i is read before it is written, by the same lane.
    unsafe { apply_raw(Kind::GeluTanh, p, None, p, data.len()) }
}

/// # Safety
/// `x`, `mul` and `out` must be valid for `n` elements; `out` may alias `x`.
unsafe fn apply_raw(kind: Kind, x: *const f32, mul: Option<*const f32>, out: *mut f32, n: usize) {
    match kernel() {
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => match kind {
            Kind::GeluTanh => x86::run_avx512::<{ Kind::GeluTanh.id() }>(x, mul, out, n),
            Kind::GeluErf => x86::run_avx512::<{ Kind::GeluErf.id() }>(x, mul, out, n),
            Kind::Silu => x86::run_avx512::<{ Kind::Silu.id() }>(x, mul, out, n),
            Kind::Erf => x86::run_avx512::<{ Kind::Erf.id() }>(x, mul, out, n),
        },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => match kind {
            Kind::GeluTanh => x86::run_avx2::<{ Kind::GeluTanh.id() }>(x, mul, out, n),
            Kind::GeluErf => x86::run_avx2::<{ Kind::GeluErf.id() }>(x, mul, out, n),
            Kind::Silu => x86::run_avx2::<{ Kind::Silu.id() }>(x, mul, out, n),
            Kind::Erf => x86::run_avx2::<{ Kind::Erf.id() }>(x, mul, out, n),
        },
        Kernel::Scalar => scalar_raw(kind, x, mul, out, 0, n),
    }
}

/// Elements `from..n` with the scalar formulas.
///
/// # Safety
/// As for [`apply_raw`].
unsafe fn scalar_raw(kind: Kind, x: *const f32, mul: Option<*const f32>, out: *mut f32, from: usize, n: usize) {
    for i in from..n {
        let g = eval(kind, *x.add(i));
        *out.add(i) = match mul {
            Some(m) => g * *m.add(i),
            None => g,
        };
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{scalar_raw, Kind, A, B, C, ERFC, LN2_HI, LN2_LO};
    use std::arch::x86_64::*;

    const GELU_TANH: u8 = Kind::GeluTanh.id();
    const GELU_ERF: u8 = Kind::GeluErf.id();
    const SILU: u8 = Kind::Silu.id();

    /// `exp(t)` for `t` in `[-88, 88]`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn exp16(t: __m512) -> __m512 {
        let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm512_mul_ps(
            t,
            _mm512_set1_ps(std::f32::consts::LOG2_E),
        ));
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LO), _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HI), t));
        let mut p = _mm512_set1_ps(C[0]);
        for &c in &C[1..] {
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(c));
        }
        _mm512_scalef_ps(_mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0)), n)
    }

    /// `x * sigmoid(z)`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn x_sigmoid16(x: __m512, z: __m512) -> __m512 {
        let t = _mm512_max_ps(_mm512_min_ps(_mm512_sub_ps(_mm512_setzero_ps(), z), _mm512_set1_ps(88.0)), _mm512_set1_ps(-88.0));
        let d = _mm512_add_ps(_mm512_set1_ps(1.0), exp16(t));
        let y = _mm512_rcp14_ps(d);
        let y = _mm512_mul_ps(y, _mm512_fnmadd_ps(d, y, _mm512_set1_ps(2.0)));
        _mm512_mul_ps(x, y)
    }

    /// `erfc(u)` for `u >= 0`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn erfc16(u: __m512) -> __m512 {
        let t = _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_fmadd_ps(_mm512_set1_ps(0.5), u, _mm512_set1_ps(1.0)));
        let mut p = _mm512_set1_ps(ERFC[0]);
        for &c in &ERFC[1..] {
            p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(c));
        }
        let a = _mm512_max_ps(_mm512_fnmadd_ps(u, u, p), _mm512_set1_ps(-88.0));
        _mm512_mul_ps(t, exp16(a))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn act16<const K: u8>(x: __m512) -> __m512 {
        let sign = _mm512_set1_ps(-0.0);
        let abs = _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(sign), _mm512_castps_si512(x)));
        let neg = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_setzero_ps());
        match K {
            GELU_TANH => x_sigmoid16(x, _mm512_mul_ps(x, _mm512_fmadd_ps(_mm512_set1_ps(B), _mm512_mul_ps(x, x), _mm512_set1_ps(A)))),
            SILU => x_sigmoid16(x, x),
            GELU_ERF => {
                let q = erfc16(_mm512_mul_ps(abs, _mm512_set1_ps(std::f32::consts::FRAC_1_SQRT_2)));
                let f = _mm512_mask_blend_ps(neg, _mm512_sub_ps(_mm512_set1_ps(2.0), q), q);
                _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5), x), f)
            }
            _ => {
                let q = erfc16(abs);
                _mm512_mask_blend_ps(neg, _mm512_sub_ps(_mm512_set1_ps(1.0), q), _mm512_sub_ps(q, _mm512_set1_ps(1.0)))
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn run_avx512<const K: u8>(x: *const f32, mul: Option<*const f32>, out: *mut f32, n: usize) {
        let mut i = 0;
        while i < n {
            let mask: __mmask16 = if n - i >= 16 { 0xffff } else { (1u16 << (n - i)) - 1 };
            let mut g = act16::<K>(_mm512_maskz_loadu_ps(mask, x.add(i)));
            if let Some(m) = mul {
                g = _mm512_mul_ps(g, _mm512_maskz_loadu_ps(mask, m.add(i)));
            }
            _mm512_mask_storeu_ps(out.add(i), mask, g);
            i += 16;
        }
    }

    /// `exp(t)` for `t` in `[-88, 88]`.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn exp8(t: __m256) -> __m256 {
        let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
            t,
            _mm256_set1_ps(std::f32::consts::LOG2_E),
        ));
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), t));
        let mut p = _mm256_set1_ps(C[0]);
        for &c in &C[1..] {
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
        }
        let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127))));
        _mm256_mul_ps(_mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0)), pow2n)
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn x_sigmoid8(x: __m256, z: __m256) -> __m256 {
        let t = _mm256_max_ps(_mm256_min_ps(_mm256_sub_ps(_mm256_setzero_ps(), z), _mm256_set1_ps(88.0)), _mm256_set1_ps(-88.0));
        let d = _mm256_add_ps(_mm256_set1_ps(1.0), exp8(t));
        let y = _mm256_rcp_ps(d);
        let y = _mm256_mul_ps(y, _mm256_fnmadd_ps(d, y, _mm256_set1_ps(2.0)));
        let y = _mm256_mul_ps(y, _mm256_fnmadd_ps(d, y, _mm256_set1_ps(2.0)));
        _mm256_mul_ps(x, y)
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn erfc8(u: __m256) -> __m256 {
        let t = _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_fmadd_ps(_mm256_set1_ps(0.5), u, _mm256_set1_ps(1.0)));
        let mut p = _mm256_set1_ps(ERFC[0]);
        for &c in &ERFC[1..] {
            p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(c));
        }
        let a = _mm256_max_ps(_mm256_fnmadd_ps(u, u, p), _mm256_set1_ps(-88.0));
        _mm256_mul_ps(t, exp8(a))
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn act8<const K: u8>(x: __m256) -> __m256 {
        let abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
        let neg = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_setzero_ps());
        match K {
            GELU_TANH => x_sigmoid8(x, _mm256_mul_ps(x, _mm256_fmadd_ps(_mm256_set1_ps(B), _mm256_mul_ps(x, x), _mm256_set1_ps(A)))),
            SILU => x_sigmoid8(x, x),
            GELU_ERF => {
                let q = erfc8(_mm256_mul_ps(abs, _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2)));
                let f = _mm256_blendv_ps(_mm256_sub_ps(_mm256_set1_ps(2.0), q), q, neg);
                _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), x), f)
            }
            _ => {
                let q = erfc8(abs);
                _mm256_blendv_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), q), _mm256_sub_ps(q, _mm256_set1_ps(1.0)), neg)
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn run_avx2<const K: u8>(x: *const f32, mul: Option<*const f32>, out: *mut f32, n: usize) {
        let mut i = 0;
        while i + 8 <= n {
            let mut g = act8::<K>(_mm256_loadu_ps(x.add(i)));
            if let Some(m) = mul {
                g = _mm256_mul_ps(g, _mm256_loadu_ps(m.add(i)));
            }
            _mm256_storeu_ps(out.add(i), g);
            i += 8;
        }
        let kind = match K {
            GELU_TANH => Kind::GeluTanh,
            GELU_ERF => Kind::GeluErf,
            SILU => Kind::Silu,
            _ => Kind::Erf,
        };
        scalar_raw(kind, x, mul, out, i, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn erf64(x: f64) -> f64 {
        // Abramowitz & Stegun 7.1.28-free reference: series / continued fraction in f64.
        if x.abs() < 3.0 {
            // Maclaurin series.
            let (mut sum, mut term, x2) = (x, x, x * x);
            for k in 1..200 {
                term *= -x2 / k as f64;
                sum += term / (2 * k + 1) as f64;
            }
            2.0 / std::f64::consts::PI.sqrt() * sum
        } else {
            // Continued fraction for erfc.
            let a = x.abs();
            let mut f = 0.0;
            for k in (1..=60).rev() {
                f = (k as f64 / 2.0) / (a + f);
            }
            let erfc = (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + f);
            x.signum() * (1.0 - erfc)
        }
    }

    /// `1 + erf(x)` without cancellation for negative `x`.
    fn one_plus_erf64(x: f64) -> f64 {
        if x < -3.0 {
            let a = -x;
            let mut f = 0.0;
            for k in (1..=60).rev() {
                f = (k as f64 / 2.0) / (a + f);
            }
            (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + f)
        } else {
            1.0 + erf64(x)
        }
    }

    fn reference(kind: Kind, x: f32) -> f64 {
        let x = x as f64;
        match kind {
            Kind::GeluTanh => 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()),
            Kind::GeluErf => 0.5 * x * one_plus_erf64(x / std::f64::consts::SQRT_2),
            Kind::Silu => x / (1.0 + (-x).exp()),
            Kind::Erf => erf64(x),
        }
    }

    fn tol(kind: Kind) -> f64 {
        match kind {
            Kind::GeluTanh | Kind::Silu => 1e-6,
            Kind::GeluErf | Kind::Erf => 1.5e-6,
        }
    }

    const KINDS: [Kind; 4] = [Kind::GeluTanh, Kind::GeluErf, Kind::Silu, Kind::Erf];

    #[test]
    fn matches_f64_reference() {
        // An odd length so every kernel also runs its tail.
        let xs: Vec<f32> = (-20000..=20000)
            .map(|i| i as f32 * 1e-3)
            .chain([-200.0, -90.0, -30.0, -12.5, 12.5, 30.0, 90.0, 300.0, 0.0, -0.0, 1e-30, -1e-30])
            .collect();
        let mul: Vec<f32> = xs.iter().map(|x| 1.0 + x.abs()).collect();
        for kind in KINDS {
            let mut out = vec![0f32; xs.len()];
            apply(kind, &xs, None, &mut out);
            let mut gated = vec![0f32; xs.len()];
            apply(kind, &xs, Some(&mul), &mut gated);
            let mut inplace = xs.clone();
            apply_inplace_par(kind, &mut inplace);
            for (i, &x) in xs.iter().enumerate() {
                let want = reference(kind, x);
                for got in [out[i], eval(kind, x), inplace[i]] {
                    let err = (got as f64 - want).abs();
                    assert!(err <= tol(kind) * (1.0 + want.abs()), "{kind:?} x={x}: {got} vs {want}");
                }
                let want_gated = want * mul[i] as f64;
                assert!((gated[i] as f64 - want_gated).abs() <= 2.0 * tol(kind) * (1.0 + want_gated.abs()), "{kind:?} x={x}");
            }
        }
        assert_eq!(gelu_tanh(1.0), eval(Kind::GeluTanh, 1.0));
    }

    /// The exact-erf forms keep their relative accuracy in the negative tail.
    #[test]
    fn erf_tail_relative_accuracy() {
        for i in 0..=1200 {
            let x = -(i as f32) * 0.01; // 0 .. -12
            let want = reference(Kind::GeluErf, x);
            let got = gelu_erf(x) as f64;
            if want.abs() > 1e-30 {
                assert!((got - want).abs() <= 2e-5 * want.abs() + 1e-37, "x={x}: {got} vs {want}");
            }
        }
    }

    /// Each vector kernel directly (the dispatcher only runs the widest one).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_isa_matches_scalar() {
        let xs: Vec<f32> = (-3000..=3000).map(|i| i as f32 * 7e-3).chain([-200.0, -90.0, 90.0, 300.0]).collect();
        let mul: Vec<f32> = xs.iter().map(|x| 2.0 - x.abs()).collect();
        let n = xs.len();
        for kind in KINDS {
            let mut runs: Vec<(&str, Vec<f32>)> = Vec::new();
            macro_rules! run {
                ($name:expr, $f:ident) => {{
                    let mut out = vec![f32::NAN; n];
                    // SAFETY: the feature was detected; buffers hold n elements.
                    unsafe {
                        match kind {
                            Kind::GeluTanh => x86::$f::<{ Kind::GeluTanh.id() }>(xs.as_ptr(), Some(mul.as_ptr()), out.as_mut_ptr(), n),
                            Kind::GeluErf => x86::$f::<{ Kind::GeluErf.id() }>(xs.as_ptr(), Some(mul.as_ptr()), out.as_mut_ptr(), n),
                            Kind::Silu => x86::$f::<{ Kind::Silu.id() }>(xs.as_ptr(), Some(mul.as_ptr()), out.as_mut_ptr(), n),
                            Kind::Erf => x86::$f::<{ Kind::Erf.id() }>(xs.as_ptr(), Some(mul.as_ptr()), out.as_mut_ptr(), n),
                        }
                    }
                    runs.push(($name, out));
                }};
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                run!("avx2", run_avx2);
            }
            if is_x86_feature_detected!("avx512f") {
                run!("avx512", run_avx512);
            }
            for (isa, out) in runs {
                for i in 0..n {
                    let want = eval(kind, xs[i]) * mul[i];
                    assert!((out[i] - want).abs() <= 1e-6 * (1.0 + want.abs()), "{isa} {kind:?} x={}: {} vs {want}", xs[i], out[i]);
                }
            }
        }
    }

    #[test]
    fn parallel_matches_serial() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        // Long enough to split into several tasks, odd so the last one has a tail.
        let n = 5 * crate::par::MIN_ELEMS_PER_TASK + 7;
        let xs: Vec<f32> = (0..n).map(|i| ((i * 7919) % 20011) as f32 * 1e-3 - 10.0).collect();
        let mul: Vec<f32> = (0..n).map(|i| (i % 13) as f32 - 6.0).collect();
        for kind in KINDS {
            for m in [None, Some(&mul[..])] {
                let mut want = vec![0f32; n];
                apply(kind, &xs, m, &mut want);
                for threads in [1, 4] {
                    let mut got = vec![f32::NAN; n];
                    if threads == 1 {
                        apply_par(kind, &xs, m, &mut got)
                    } else {
                        pool.install(|| apply_par(kind, &xs, m, &mut got))
                    }
                    assert_eq!(got, want, "{kind:?} threads={threads}");
                }
            }
            let mut inplace = xs.clone();
            pool.install(|| apply_inplace_par(kind, &mut inplace));
            let mut want = vec![0f32; n];
            apply(kind, &xs, None, &mut want);
            assert_eq!(inplace, want, "{kind:?} in place");
        }
        let empty: [f32; 0] = [];
        apply_par(Kind::Silu, &empty, None, &mut []);
    }
}
