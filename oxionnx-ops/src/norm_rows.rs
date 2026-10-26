//! Row-wise LayerNorm / RMSNorm kernel shared by `LayerNormalization`,
//! `SimplifiedLayerNormalization`, `com.microsoft::SkipLayerNormalization` and
//! `com.microsoft::SkipSimplifiedLayerNormalization`.
//!
//! Each row of `hidden` elements is read from memory once: the residual sum
//! `x + skip (+ bias)` (when there is a skip input) is staged in the output row
//! while its sum (LayerNorm) or sum of squares (RMSNorm) is accumulated; the
//! variance pass and the normalisation pass then run over the row while it is
//! still in L1. The variance is computed around the mean (not as
//! `E[x²] - mean²`), so rows with a large mean keep full precision.
//!
//! Rows run in parallel; on x86-64 the row loops are compiled for AVX-512F and
//! AVX2 and picked at run time, so a default (`x86-64`) build vectorises too.

#![allow(unsafe_code)]

/// Inputs of one row-normalisation call.
#[derive(Clone, Copy)]
pub struct NormRows<'a> {
    /// `[rows, hidden]` input, row-major.
    pub x: &'a [f32],
    /// Optional residual, the shape of `x` or a whole number of trailing rows
    /// of it (repeated over the leading rows).
    pub skip: Option<&'a [f32]>,
    /// Optional `[hidden]` bias added to the residual sum before the norm.
    pub bias: Option<&'a [f32]>,
    /// `[hidden]` scale.
    pub gamma: &'a [f32],
    /// Optional `[hidden]` shift (LayerNorm only).
    pub beta: Option<&'a [f32]>,
    pub hidden: usize,
    pub eps: f32,
    /// RMSNorm (`x / sqrt(mean(x²) + eps)`) instead of LayerNorm.
    pub rms: bool,
}

/// Outputs of one row-normalisation call. `sum` receives `x + skip (+ bias)`,
/// `mean` / `inv_std` one value per row.
pub struct NormRowsOut<'a> {
    pub out: &'a mut [f32],
    pub sum: Option<&'a mut [f32]>,
    pub mean: Option<&'a mut [f32]>,
    pub inv_std: Option<&'a mut [f32]>,
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

#[derive(Clone, Copy)]
struct OutPtrs {
    out: Ptr,
    sum: Option<Ptr>,
    mean: Option<Ptr>,
    inv_std: Option<Ptr>,
}

/// Normalise every row of `a.x` into `o`.
pub fn norm_rows_into(a: &NormRows<'_>, o: NormRowsOut<'_>) -> Result<(), String> {
    let h = a.hidden;
    if h == 0 || a.x.is_empty() {
        return Ok(());
    }
    if a.x.len() % h != 0 {
        return Err(format!("norm: input ({} elems) is not a multiple of the normalised size {h}", a.x.len()));
    }
    let rows = a.x.len() / h;
    if let Some(s) = a.skip {
        if s.is_empty() || s.len() % h != 0 || a.x.len() % s.len() != 0 {
            return Err(format!(
                "norm: skip ({} elems) does not tile input ({} elems, hidden {h})",
                s.len(),
                a.x.len()
            ));
        }
    }
    if a.gamma.len() != h || a.bias.is_some_and(|b| b.len() != h) || a.beta.is_some_and(|b| b.len() != h) {
        return Err(format!("norm: scale/bias must have {h} elements"));
    }
    if o.out.len() < a.x.len()
        || o.sum.as_ref().is_some_and(|s| s.len() < a.x.len())
        || o.mean.as_ref().is_some_and(|s| s.len() < rows)
        || o.inv_std.as_ref().is_some_and(|s| s.len() < rows)
    {
        return Err("norm: output buffer too small".into());
    }
    let ptrs = OutPtrs {
        out: Ptr(o.out.as_mut_ptr()),
        sum: o.sum.map(|s| Ptr(s.as_mut_ptr())),
        mean: o.mean.map(|s| Ptr(s.as_mut_ptr())),
        inv_std: o.inv_std.map(|s| Ptr(s.as_mut_ptr())),
    };
    let kind = isa();
    // SAFETY (all calls below): every row index in `lo..hi` is in bounds of the
    // checked buffers, and parallel tasks get disjoint row ranges.
    par_ranges(rows, (crate::par::min_elems() / h).max(1), |lo, hi| unsafe { run_rows(kind, a, ptrs, lo, hi) });
    Ok(())
}

#[derive(Clone, Copy)]
enum Isa {
    Generic,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// The row-loop ISA, detected once (`OXIONNX_NORM_ISA=generic|avx2` caps it).
fn isa() -> Isa {
    static ISA: std::sync::OnceLock<Isa> = std::sync::OnceLock::new();
    *ISA.get_or_init(detect_isa)
}

fn detect_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        match std::env::var("OXIONNX_NORM_ISA").as_deref() {
            Ok("generic") => return Isa::Generic,
            Ok("avx2") if std::arch::is_x86_feature_detected!("avx2") => return Isa::Avx2,
            _ => {}
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Isa::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return Isa::Avx2;
        }
    }
    Isa::Generic
}

unsafe fn run_rows(kind: Isa, a: &NormRows<'_>, p: OutPtrs, lo: usize, hi: usize) {
    match kind {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => rows_avx512(a, p, lo, hi),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => rows_avx2(a, p, lo, hi),
        Isa::Generic => rows_generic(a, p, lo, hi),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn rows_avx512(a: &NormRows<'_>, p: OutPtrs, lo: usize, hi: usize) {
    rows_generic(a, p, lo, hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rows_avx2(a: &NormRows<'_>, p: OutPtrs, lo: usize, hi: usize) {
    rows_generic(a, p, lo, hi)
}

/// Run `f(lo, hi)` over `0..n` in chunks of at least `min_per_task`, in
/// parallel when there is more than one chunk and more than one thread.
fn par_ranges(n: usize, min_per_task: usize, f: impl Fn(usize, usize) + Sync + Send) {
    let t = crate::par::threads();
    let per = min_per_task.max(n.div_ceil(4 * t.max(1))).max(1);
    let tasks = n.div_ceil(per);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if tasks > 1 && t > 1 {
        use rayon::prelude::*;
        (0..tasks).into_par_iter().for_each(|i| f(i * per, ((i + 1) * per).min(n)));
        return;
    }
    let _ = tasks;
    f(0, n);
}

/// Accumulator width: 4 independent 16-lane chains (4 zmm / 8 ymm registers),
/// so the reductions are not bound by add latency.
const L: usize = 64;

/// `acc[0..16] + acc[16..32] + acc[32..48] + acc[48..64]` (whole-vector adds).
#[inline(always)]
fn fold16(acc: &[f32; L]) -> [f32; 16] {
    let mut v = [0.0f32; 16];
    for i in 0..16 {
        v[i] = (acc[i] + acc[i + 32]) + (acc[i + 16] + acc[i + 48]);
    }
    v
}

/// Sum of 16 lanes as a fixed tree, so the result does not depend on the
/// vector width.
#[inline(always)]
fn hsum16(v: [f32; 16]) -> f32 {
    let mut v8 = [0.0f32; 8];
    for i in 0..8 {
        v8[i] = v[i] + v[i + 8];
    }
    let v4 = [v8[0] + v8[4], v8[1] + v8[5], v8[2] + v8[6], v8[3] + v8[7]];
    (v4[0] + v4[2]) + (v4[1] + v4[3])
}

/// Elements of an `n`-row handled by the 64-wide loops: short rows (per-head
/// norms, `n = 128`) use only the 16-wide loop, which costs less per row.
#[inline(always)]
fn wide_len(n: usize) -> usize {
    if n >= 4 * L { n - n % L } else { 0 }
}

/// `Σ v` (or `Σ v²` when `SQ`) over `v`.
#[inline(always)]
fn row_sum<const SQ: bool>(v: &[f32]) -> f32 {
    let (wide, rest) = v.split_at(wide_len(v.len()));
    let mut a16 = [0.0f32; 16];
    if !wide.is_empty() {
        let mut acc = [0.0f32; L];
        for ch in wide.chunks_exact(L) {
            for l in 0..L {
                acc[l] += if SQ { ch[l] * ch[l] } else { ch[l] };
            }
        }
        a16 = fold16(&acc);
    }
    let mut c16 = rest.chunks_exact(16);
    for ch in &mut c16 {
        for l in 0..16 {
            a16[l] += if SQ { ch[l] * ch[l] } else { ch[l] };
        }
    }
    let mut tail = 0.0f32;
    for &x in c16.remainder() {
        tail += if SQ { x * x } else { x };
    }
    hsum16(a16) + tail
}

/// `(Σ (v - mean), Σ (v - mean)²)` over `v`.
#[inline(always)]
fn row_dev(v: &[f32], mean: f32) -> (f32, f32) {
    let (wide, rest) = v.split_at(wide_len(v.len()));
    let (mut a16, mut b16) = ([0.0f32; 16], [0.0f32; 16]);
    if !wide.is_empty() {
        let (mut acc, mut acc2) = ([0.0f32; L], [0.0f32; L]);
        for ch in wide.chunks_exact(L) {
            for l in 0..L {
                let d = ch[l] - mean;
                acc[l] += d;
                acc2[l] += d * d;
            }
        }
        a16 = fold16(&acc);
        b16 = fold16(&acc2);
    }
    let mut c16 = rest.chunks_exact(16);
    for ch in &mut c16 {
        for l in 0..16 {
            let d = ch[l] - mean;
            a16[l] += d;
            b16[l] += d * d;
        }
    }
    let (mut t, mut t2) = (0.0f32, 0.0f32);
    for &x in c16.remainder() {
        let d = x - mean;
        t += d;
        t2 += d * d;
    }
    (hsum16(a16) + t, hsum16(b16) + t2)
}

/// One `W`-wide block of [`stage_sum`], accumulating into `acc[..W]`.
#[inline(always)]
fn stage_block<const SQ: bool, const W: usize>(x: &[f32], s: &[f32], b: Option<&[f32]>, d: &mut [f32], acc: &mut [f32; W]) {
    let (x, s, d) = (&x[..W], &s[..W], &mut d[..W]);
    match b {
        Some(b) => {
            let b = &b[..W];
            for l in 0..W {
                let v = x[l] + s[l] + b[l];
                d[l] = v;
                acc[l] += if SQ { v * v } else { v };
            }
        }
        None => {
            for l in 0..W {
                let v = x[l] + s[l];
                d[l] = v;
                acc[l] += if SQ { v * v } else { v };
            }
        }
    }
}

/// `dst = x + s (+ b)`, returning `Σ dst` (or `Σ dst²` when `SQ`).
#[inline(always)]
fn stage_sum<const SQ: bool>(x: &[f32], s: &[f32], b: Option<&[f32]>, dst: &mut [f32]) -> f32 {
    let n = dst.len();
    let wide = wide_len(n);
    let mut a16 = [0.0f32; 16];
    let mut j = 0;
    if wide > 0 {
        let mut acc = [0.0f32; L];
        while j + L <= wide {
            stage_block::<SQ, L>(&x[j..], &s[j..], b.map(|b| &b[j..]), &mut dst[j..], &mut acc);
            j += L;
        }
        a16 = fold16(&acc);
    }
    while j + 16 <= n {
        stage_block::<SQ, 16>(&x[j..], &s[j..], b.map(|b| &b[j..]), &mut dst[j..], &mut a16);
        j += 16;
    }
    let mut tail = 0.0f32;
    for jj in j..n {
        let v = x[jj] + s[jj] + b.map_or(0.0, |b| b[jj]);
        dst[jj] = v;
        tail += if SQ { v * v } else { v };
    }
    hsum16(a16) + tail
}

/// `dst = (src - mean) * inv * gamma (+ beta)`; `mean` is ignored for RMSNorm.
#[inline(always)]
fn normalize(rms: bool, src: &[f32], dst: &mut [f32], inv: f32, mean: f32, gamma: &[f32], beta: Option<&[f32]>) {
    let n = dst.len();
    let (src, gamma) = (&src[..n], &gamma[..n]);
    match (rms, beta) {
        (true, _) => {
            for j in 0..n {
                dst[j] = src[j] * inv * gamma[j];
            }
        }
        (false, Some(beta)) => {
            let beta = &beta[..n];
            for j in 0..n {
                dst[j] = (src[j] - mean) * inv * gamma[j] + beta[j];
            }
        }
        (false, None) => {
            for j in 0..n {
                dst[j] = (src[j] - mean) * inv * gamma[j];
            }
        }
    }
}

/// In-place [`normalize`].
#[inline(always)]
fn normalize_inplace(rms: bool, buf: &mut [f32], inv: f32, mean: f32, gamma: &[f32], beta: Option<&[f32]>) {
    let n = buf.len();
    let gamma = &gamma[..n];
    match (rms, beta) {
        (true, _) => {
            for j in 0..n {
                buf[j] = buf[j] * inv * gamma[j];
            }
        }
        (false, Some(beta)) => {
            let beta = &beta[..n];
            for j in 0..n {
                buf[j] = (buf[j] - mean) * inv * gamma[j] + beta[j];
            }
        }
        (false, None) => {
            for j in 0..n {
                buf[j] = (buf[j] - mean) * inv * gamma[j];
            }
        }
    }
}

/// `(1 / std, mean)` of `src` (`mean = 0` for RMSNorm), given `first` = its
/// sum (LayerNorm) or sum of squares (RMSNorm).
#[inline(always)]
fn stats(rms: bool, src: &[f32], first: f32, eps: f32) -> (f32, f32) {
    let n = src.len() as f32;
    if rms {
        ((first / n + eps).sqrt().recip(), 0.0)
    } else {
        // Corrected two-pass: the deviation sum fixes the rounding of the
        // first-pass mean, which matters when |mean| >> std.
        let m0 = first / n;
        let (d, d2) = row_dev(src, m0);
        let dm = d / n;
        let var = (d2 / n - dm * dm).max(0.0);
        ((var + eps).sqrt().recip(), m0 + dm)
    }
}

#[inline(always)]
unsafe fn rows_generic(a: &NormRows<'_>, p: OutPtrs, lo: usize, hi: usize) {
    let h = a.hidden;
    let skip_rows = a.skip.map_or(1, |s| s.len() / h);
    for r in lo..hi {
        let xi = &a.x[r * h..(r + 1) * h];
        let o = std::slice::from_raw_parts_mut(p.out.0.add(r * h), h);
        let (inv, mean) = match a.skip {
            None => {
                let first = if a.rms { row_sum::<true>(xi) } else { row_sum::<false>(xi) };
                let (inv, mean) = stats(a.rms, xi, first, a.eps);
                if let Some(sp) = p.sum {
                    std::slice::from_raw_parts_mut(sp.0.add(r * h), h).copy_from_slice(xi);
                }
                normalize(a.rms, xi, o, inv, mean, a.gamma, a.beta);
                (inv, mean)
            }
            Some(skip) => {
                let si = &skip[(r % skip_rows) * h..(r % skip_rows + 1) * h];
                match p.sum {
                    Some(sp) => {
                        let s = std::slice::from_raw_parts_mut(sp.0.add(r * h), h);
                        let first =
                            if a.rms { stage_sum::<true>(xi, si, a.bias, s) } else { stage_sum::<false>(xi, si, a.bias, s) };
                        let (inv, mean) = stats(a.rms, s, first, a.eps);
                        normalize(a.rms, s, o, inv, mean, a.gamma, a.beta);
                        (inv, mean)
                    }
                    None => {
                        let first =
                            if a.rms { stage_sum::<true>(xi, si, a.bias, o) } else { stage_sum::<false>(xi, si, a.bias, o) };
                        let (inv, mean) = stats(a.rms, o, first, a.eps);
                        normalize_inplace(a.rms, o, inv, mean, a.gamma, a.beta);
                        (inv, mean)
                    }
                }
            }
        };
        if let Some(mp) = p.mean {
            *mp.0.add(r) = mean;
        }
        if let Some(ip) = p.inv_std {
            *ip.0.add(r) = inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference, statistics in f64.
    #[allow(clippy::type_complexity)]
    fn reference(a: &NormRows<'_>) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = a.hidden;
        let rows = a.x.len() / h;
        let (mut out, mut sum, mut mean, mut inv) = (vec![0.0; a.x.len()], vec![0.0; a.x.len()], vec![], vec![]);
        for r in 0..rows {
            let v: Vec<f64> = (0..h)
                .map(|j| {
                    // The residual sum is an f32 add (as in `Add` + norm).
                    let mut v = a.x[r * h + j];
                    if let Some(s) = a.skip {
                        v += s[(r * h + j) % s.len()];
                    }
                    if let Some(b) = a.bias {
                        v += b[j];
                    }
                    v as f64
                })
                .collect();
            let m = if a.rms { 0.0 } else { v.iter().sum::<f64>() / h as f64 };
            let var = v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / h as f64;
            let is = 1.0 / (var + a.eps as f64).sqrt();
            for j in 0..h {
                sum[r * h + j] = v[j] as f32;
                out[r * h + j] =
                    ((v[j] - m) * is * a.gamma[j] as f64 + a.beta.map_or(0.0, |b| b[j] as f64)) as f32;
            }
            mean.push(m as f32);
            inv.push(is as f32);
        }
        (out, sum, mean, inv)
    }

    fn data(n: usize, seed: u32, offset: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 2000) as f32 / 1000.0 - 1.0 + offset
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}[{i}]: {x} vs {y}");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check(rows: usize, h: usize, skip_rows: Option<usize>, bias: bool, beta: bool, rms: bool, offset: f32, tol: f32) {
        let x = data(rows * h, 1, offset);
        let skip = skip_rows.map(|sr| data(sr * h, 2, 0.0));
        let b = bias.then(|| data(h, 3, 0.0));
        let g = data(h, 4, 1.0);
        let be = beta.then(|| data(h, 5, 0.0));
        let a = NormRows {
            x: &x,
            skip: skip.as_deref(),
            bias: b.as_deref(),
            gamma: &g,
            beta: be.as_deref(),
            hidden: h,
            eps: 1e-5,
            rms,
        };
        let (ro, rs, rm, ri) = reference(&a);
        let tag = format!("rows {rows} h {h} skip {skip_rows:?} bias {bias} beta {beta} rms {rms}");
        // All outputs.
        let (mut out, mut sum, mut mean, mut inv) = (vec![0.0; x.len()], vec![0.0; x.len()], vec![0.0; rows], vec![0.0; rows]);
        norm_rows_into(&a, NormRowsOut { out: &mut out, sum: Some(&mut sum), mean: Some(&mut mean), inv_std: Some(&mut inv) })
            .unwrap();
        close(&out, &ro, tol, &format!("{tag} out"));
        if skip_rows.is_some() || bias {
            close(&sum, &rs, 1e-6, &format!("{tag} sum"));
        }
        if !rms {
            close(&mean, &rm, tol, &format!("{tag} mean"));
        }
        close(&inv, &ri, 2e-5, &format!("{tag} inv"));
        // Output only (staged in place).
        let mut out2 = vec![f32::NAN; x.len()];
        norm_rows_into(&a, NormRowsOut { out: &mut out2, sum: None, mean: None, inv_std: None }).unwrap();
        assert_eq!(out, out2, "{tag}: in-place staging differs");
    }

    fn sweep() {
        for &(rows, h) in &[(1, 1), (3, 7), (5, 16), (2, 33), (37, 768), (9, 130), (600, 384), (1, 1024)] {
            for &rms in &[false, true] {
                check(rows, h, None, false, !rms, rms, 0.0, 2e-5);
                check(rows, h, Some(rows), false, !rms, rms, 0.0, 2e-5);
                check(rows, h, Some(rows), true, false, rms, 0.0, 2e-5);
                if rows % 3 == 0 {
                    check(rows, h, Some(rows / 3), true, !rms, rms, 0.0, 2e-5);
                }
            }
        }
        // Large mean (|mean| / std ~ 500): the variance is computed around the
        // mean; what is left is the f32 rounding of the mean itself (ulp(300)
        // ~ 3e-5), which bounds any f32 kernel.
        check(64, 768, Some(64), false, true, false, 300.0, 1e-4);
    }

    #[test]
    fn matches_reference_single_thread() {
        rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(sweep);
    }

    #[test]
    fn matches_reference_multi_thread() {
        rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(sweep);
    }

    #[test]
    fn rejects_bad_shapes() {
        let x = vec![0.0; 12];
        let g = vec![1.0; 4];
        let mut out = vec![0.0; 12];
        let bad_skip = vec![0.0; 5];
        let a = NormRows { x: &x, skip: Some(&bad_skip), bias: None, gamma: &g, beta: None, hidden: 4, eps: 1e-5, rms: false };
        assert!(norm_rows_into(&a, NormRowsOut { out: &mut out, sum: None, mean: None, inv_std: None }).is_err());
        let a = NormRows { x: &x, skip: None, bias: None, gamma: &g[..3], beta: None, hidden: 4, eps: 1e-5, rms: false };
        assert!(norm_rows_into(&a, NormRowsOut { out: &mut out, sum: None, mean: None, inv_std: None }).is_err());
        let a = NormRows { x: &[], skip: None, bias: None, gamma: &g, beta: None, hidden: 4, eps: 1e-5, rms: false };
        assert!(norm_rows_into(&a, NormRowsOut { out: &mut [], sum: None, mean: None, inv_std: None }).is_ok());
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// `cargo test --release -p oxionnx-ops norm_rows::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_rows() {
        let t: usize = std::env::var("BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        for &(rows, h) in &[(2048usize, 640usize), (2048, 768), (2048 * 12, 128), (24, 768)] {
            let x: Vec<f32> = (0..rows * h).map(|i| (i % 97) as f32 * 0.01).collect();
            let s: Vec<f32> = (0..rows * h).map(|i| (i % 89) as f32 * 0.01).collect();
            let g = vec![1.0f32; h];
            let (mut o, mut sum) = (vec![0.0f32; rows * h], vec![0.0f32; rows * h]);
            for (name, skip, rms, with_sum) in
                [("ln", None, false, false), ("rms", None, true, false), ("skip-ln+sum", Some(&s[..]), false, true), ("skip-rms+sum", Some(&s[..]), true, true), ("skip-ln", Some(&s[..]), false, false)]
            {
                let a = NormRows { x: &x, skip, bias: None, gamma: &g, beta: None, hidden: h, eps: 1e-5, rms };
                let iters = (200_000_000 / (rows * h)).max(5);
                pool.install(|| {
                    for _ in 0..3 {
                        norm_rows_into(&a, NormRowsOut { out: &mut o, sum: with_sum.then_some(&mut sum[..]), mean: None, inv_std: None }).unwrap();
                    }
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        norm_rows_into(&a, NormRowsOut { out: &mut o, sum: with_sum.then_some(&mut sum[..]), mean: None, inv_std: None }).unwrap();
                    }
                    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                    println!("{rows:6}x{h:4} {name:13} {us:8.1} us");
                });
            }
            // The previous `rms_norm_into` (8-lane, `par::for_each_row`).
            let iters = (200_000_000 / (rows * h)).max(5);
            pool.install(|| {
                let run = |o: &mut [f32]| {
                    crate::par::for_each_row(o, h, |r, out_slice| {
                        let in_slice = &x[r * h..(r + 1) * h];
                        let mut lanes = [0.0f32; 8];
                        let mut chunks = in_slice.chunks_exact(8);
                        for c in &mut chunks {
                            for (l, &v) in lanes.iter_mut().zip(c) {
                                *l += v * v;
                            }
                        }
                        let tail: f32 = chunks.remainder().iter().map(|&v| v * v).sum();
                        let inv = ((lanes.iter().sum::<f32>() + tail) / h as f32 + 1e-5).sqrt().recip();
                        for ((o_v, &i_v), &s) in out_slice.iter_mut().zip(in_slice).zip(&g) {
                            *o_v = i_v * inv * s;
                        }
                    })
                };
                run(&mut o);
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    run(&mut o);
                }
                println!("{rows:6}x{h:4} {:13} {:8.1} us", "old-rms", t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
            });
        }
    }
}
