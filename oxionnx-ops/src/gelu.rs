//! Prototype (onnx_spike): tanh-approximation GELU kernels.
//!
//! `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` is evaluated as
//! `x * sigmoid(z)` with `z = x (A + B x^2)`, `A = 2 sqrt(2/pi)`, `B = 0.044715 A`
//! (the same function: `0.5 (1 + tanh(t)) = sigmoid(2t)`), with a degree-6 exp
//! polynomial and a refined reciprocal: |error| <= ~3e-7 relative to f64.
//! AVX-512F and AVX2+FMA kernels are picked once per process; other targets use
//! the scalar formula.

#![allow(unsafe_code)]

const A: f32 = 1.595_769_1; // 2 * sqrt(2 / pi)
const B: f32 = 0.071_354_816; // 0.044715 * A
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const C: [f32; 6] = [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0];

/// Scalar tanh-approximation GELU (the kernels' formula, lane by lane).
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    let z = x * B.mul_add(x * x, A);
    let t = (-z).clamp(-88.0, 88.0);
    let n = (t * std::f32::consts::LOG2_E).round();
    let r = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, t));
    let mut p = C[0];
    for &c in &C[1..] {
        p = p.mul_add(r, c);
    }
    let e = p.mul_add(r, 1.0) * f32::from_bits(((n as i32 + 127) as u32) << 23);
    x / (1.0 + e)
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

/// `out[i] = gelu(x[i]) * mul[i]` (or just `gelu(x[i])` without `mul`); see
/// [`gelu_inplace`] for the in-place form.
pub fn gelu_mul(x: &[f32], mul: Option<&[f32]>, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    if let Some(m) = mul {
        assert_eq!(m.len(), out.len());
    }
    // SAFETY: lengths checked above; the kernel was detected on this CPU.
    unsafe { gelu_raw(x.as_ptr(), mul.map(|m| m.as_ptr()), out.as_mut_ptr(), out.len()) }
}

/// [`gelu_mul`] split across the current rayon pool when large enough.
pub fn gelu_mul_par(x: &[f32], mul: Option<&[f32]>, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let chunk = crate::par::MIN_ELEMS_PER_TASK.max(x.len().div_ceil(4 * crate::par::threads().max(1)));
    if crate::par::threads() <= 1 || x.len() < 2 * crate::par::MIN_ELEMS_PER_TASK {
        return gelu_mul(x, mul, out);
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, o)| {
            let r = i * chunk..i * chunk + o.len();
            gelu_mul(&x[r.clone()], mul.map(|m| &m[r]), o);
        });
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    gelu_mul(x, mul, out);
}

/// In-place `data[i] = gelu(data[i])`.
pub fn gelu_inplace(data: &mut [f32]) {
    let p = data.as_mut_ptr();
    // SAFETY: element i is read before it is written, by the same lane.
    unsafe { gelu_raw(p, None, p, data.len()) }
}

/// # Safety
/// `x`, `mul` and `out` must be valid for `n` elements; `out` may alias `x`.
unsafe fn gelu_raw(x: *const f32, mul: Option<*const f32>, out: *mut f32, n: usize) {
    match kernel() {
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => x86::gelu_avx512(x, mul, out, n),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => x86::gelu_avx2(x, mul, out, n),
        Kernel::Scalar => {
            for i in 0..n {
                let g = gelu_tanh(*x.add(i));
                *out.add(i) = match mul {
                    Some(m) => g * *m.add(i),
                    None => g,
                };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{gelu_tanh, A, B, C, LN2_HI, LN2_LO};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gelu16(x: __m512) -> __m512 {
        let z = _mm512_mul_ps(x, _mm512_fmadd_ps(_mm512_set1_ps(B), _mm512_mul_ps(x, x), _mm512_set1_ps(A)));
        let t = _mm512_max_ps(_mm512_min_ps(_mm512_sub_ps(_mm512_setzero_ps(), z), _mm512_set1_ps(88.0)), _mm512_set1_ps(-88.0));
        let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm512_mul_ps(
            t,
            _mm512_set1_ps(std::f32::consts::LOG2_E),
        ));
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LO), _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HI), t));
        let mut p = _mm512_set1_ps(C[0]);
        for &c in &C[1..] {
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(c));
        }
        let e = _mm512_scalef_ps(_mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0)), n);
        let d = _mm512_add_ps(_mm512_set1_ps(1.0), e);
        let y = _mm512_rcp14_ps(d);
        let y = _mm512_mul_ps(y, _mm512_fnmadd_ps(d, y, _mm512_set1_ps(2.0)));
        _mm512_mul_ps(x, y)
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn gelu_avx512(x: *const f32, mul: Option<*const f32>, out: *mut f32, n: usize) {
        let mut i = 0;
        while i < n {
            let mask: __mmask16 = if n - i >= 16 { 0xffff } else { (1u16 << (n - i)) - 1 };
            let mut g = gelu16(_mm512_maskz_loadu_ps(mask, x.add(i)));
            if let Some(m) = mul {
                g = _mm512_mul_ps(g, _mm512_maskz_loadu_ps(mask, m.add(i)));
            }
            _mm512_mask_storeu_ps(out.add(i), mask, g);
            i += 16;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn gelu8(x: __m256) -> __m256 {
        let z = _mm256_mul_ps(x, _mm256_fmadd_ps(_mm256_set1_ps(B), _mm256_mul_ps(x, x), _mm256_set1_ps(A)));
        let t = _mm256_max_ps(_mm256_min_ps(_mm256_sub_ps(_mm256_setzero_ps(), z), _mm256_set1_ps(88.0)), _mm256_set1_ps(-88.0));
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
        let e = _mm256_mul_ps(_mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0)), pow2n);
        let d = _mm256_add_ps(_mm256_set1_ps(1.0), e);
        let y = _mm256_rcp_ps(d);
        let y = _mm256_mul_ps(y, _mm256_fnmadd_ps(d, y, _mm256_set1_ps(2.0)));
        let y = _mm256_mul_ps(y, _mm256_fnmadd_ps(d, y, _mm256_set1_ps(2.0)));
        _mm256_mul_ps(x, y)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gelu_avx2(x: *const f32, mul: Option<*const f32>, out: *mut f32, n: usize) {
        let mut i = 0;
        while i + 8 <= n {
            let mut g = gelu8(_mm256_loadu_ps(x.add(i)));
            if let Some(m) = mul {
                g = _mm256_mul_ps(g, _mm256_loadu_ps(m.add(i)));
            }
            _mm256_storeu_ps(out.add(i), g);
            i += 8;
        }
        for j in i..n {
            let g = gelu_tanh(*x.add(j));
            *out.add(j) = match mul {
                Some(m) => g * *m.add(j),
                None => g,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(x: f32) -> f64 {
        let x = x as f64;
        0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    }

    #[test]
    fn matches_f64_reference() {
        let xs: Vec<f32> = (-20000..=20000).map(|i| i as f32 * 1e-3).chain([-200.0, -90.0, 90.0, 300.0, 0.0, -0.0]).collect();
        let mul: Vec<f32> = xs.iter().map(|x| 1.0 + x.abs()).collect();
        let mut out = vec![0f32; xs.len()];
        gelu_mul(&xs, None, &mut out);
        let mut gated = vec![0f32; xs.len()];
        gelu_mul(&xs, Some(&mul), &mut gated);
        let mut inplace = xs.clone();
        gelu_inplace(&mut inplace);
        for (i, &x) in xs.iter().enumerate() {
            let want = reference(x);
            for got in [out[i], gelu_tanh(x), inplace[i]] {
                let err = (got as f64 - want).abs();
                assert!(err <= 1e-6 * (1.0 + want.abs()), "x={x}: {got} vs {want}");
            }
            let want_gated = reference(x) * mul[i] as f64;
            assert!((gated[i] as f64 - want_gated).abs() <= 2e-6 * (1.0 + want_gated.abs()), "x={x}");
        }
    }
}
