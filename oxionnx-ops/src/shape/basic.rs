//! Basic shape manipulation operations: reshape, flatten, transpose, squeeze, unsqueeze.

use oxionnx_core::Tensor;

/// Normalize a possibly-negative ONNX axis against `rank`, returning a bounds-checked index.
///
/// Valid input range is `[-rank, rank-1]`, matching the axis convention shared by
/// Concat/Split/Slice/Squeeze/Gather/etc. Negative values count from the back
/// (`axis + rank`). Out-of-range values (including the classic huge-`usize` wraparound a raw
/// `(axis + rank as i64) as usize` cast produces for `axis < -rank`) are rejected instead of
/// silently indexing garbage.
pub(crate) fn normalize_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        return Err(format!(
            "axis {axis} out of range for rank {rank} (expected [-{rank}, {}])",
            rank.saturating_sub(1)
        ));
    }
    Ok(a as usize)
}

/// Like [`normalize_axis`], but accepts the inclusive `[-rank, rank]` range that ONNX Flatten
/// uses (an axis equal to `rank` is legal and means "everything goes into the outer dim").
pub(crate) fn normalize_axis_inclusive(axis: i64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a > r {
        return Err(format!(
            "axis {axis} out of range for rank {rank} (expected [-{rank}, {rank}])"
        ));
    }
    Ok(a as usize)
}

/// Reshape `x` to the target `shape`, honouring the ONNX `allowzero` attribute.
///
/// Special entries in `shape`:
/// - `-1` infers a single dimension from the remaining element count (at most one allowed).
/// - `0` copies the corresponding input dimension when `allowzero` is `false` (the ONNX
///   default); when `allowzero` is `true` a `0` is a literal zero-size dimension (NumPy
///   semantics) and is taken verbatim.
///
/// When `allowzero` is `true`, combining a `-1` with an explicit `0` is ambiguous and
/// returns an error.
pub fn reshape(x: &Tensor, shape: &[i64], allowzero: bool) -> Result<Tensor, String> {
    let new_shape = resolve_reshape(&x.shape, x.numel(), shape, allowzero)?;
    Ok(Tensor::new(x.data.clone(), new_shape))
}

/// Resolve a concrete output shape for Reshape from the target `shape` spec.
///
/// `input_dims` and `numel` describe the source tensor; `allowzero` selects whether a `0`
/// copies the input dimension (`false`, ONNX default) or is a literal zero (`true`).
pub fn resolve_reshape(
    input_dims: &[usize],
    numel: usize,
    shape: &[i64],
    allowzero: bool,
) -> Result<Vec<usize>, String> {
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    if neg_count > 1 {
        return Err("reshape: at most one -1 allowed".into());
    }
    if let Some(&bad) = shape.iter().find(|&&d| d < -1) {
        return Err(format!(
            "reshape: shape entries must be >= -1 (the ONNX infer sentinel), got {bad}"
        ));
    }
    let has_explicit_zero = shape.contains(&0);
    if allowzero && neg_count == 1 && has_explicit_zero {
        return Err(
            "Reshape: cannot infer dimension (-1) when an explicit 0 is present under allowzero=1"
                .into(),
        );
    }

    // Resolve every non-(-1) entry first: a 0 either copies the input dim (allowzero=false)
    // or stays a literal zero (allowzero=true). The -1 entry is filled in afterwards. Entries
    // below -1 were already rejected above, so every remaining `d` here is >= 0.
    let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        let dim = if d == -1 {
            usize::MAX // placeholder, overwritten below
        } else if d == 0 && !allowzero {
            *input_dims.get(i).ok_or_else(|| {
                format!(
                    "reshape: 0 at index {i} has no matching input dimension (rank {})",
                    input_dims.len()
                )
            })?
        } else {
            d as usize
        };
        new_shape.push(dim);
    }

    if neg_count == 1 {
        let known: usize = new_shape.iter().filter(|&&d| d != usize::MAX).product();
        if known == 0 {
            return Err(format!(
                "reshape: cannot infer dimension (-1) when remaining dimensions multiply to 0 ({new_shape:?})"
            ));
        }
        let inferred = numel / known;
        for d in new_shape.iter_mut() {
            if *d == usize::MAX {
                *d = inferred;
            }
        }
    }

    if new_shape.iter().product::<usize>() != numel {
        return Err(format!(
            "reshape: element count mismatch ({numel} vs {new_shape:?})"
        ));
    }
    Ok(new_shape)
}

/// Resolve the `(outer, inner)` split for Flatten: `outer = prod(shape[..axis])`,
/// `inner = prod(shape[axis..])`. `axis` is normalized against the *inclusive* `[-r, r]`
/// range ONNX defines for Flatten (unlike the exclusive range most other axis-taking ops use).
///
/// Neither product is clamped to a minimum of 1: an empty leading/trailing slice legitimately
/// multiplies to 1 (the identity element), and a genuinely zero-size input dimension must stay
/// zero so the returned shape keeps `outer * inner == x.numel()`.
pub(crate) fn resolve_flatten_shape(
    x_shape: &[usize],
    axis: i64,
) -> Result<(usize, usize), String> {
    let ndim = x_shape.len();
    let ax = normalize_axis_inclusive(axis, ndim)?;
    let outer: usize = x_shape[..ax].iter().product();
    let inner: usize = x_shape[ax..].iter().product();
    Ok((outer, inner))
}

pub fn flatten(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let (outer, inner) = resolve_flatten_shape(&x.shape, axis)?;
    Ok(Tensor::new(x.data.clone(), vec![outer, inner]))
}

/// Transpose according to a permutation. If perm is empty, reverses all dims.
pub fn transpose(x: &Tensor, perm: &[usize]) -> Result<Tensor, String> {
    let mut out = Vec::new();
    let out_shape = transpose_into(x, perm, &mut out)?;
    Ok(Tensor::new(out, out_shape))
}

/// [`transpose`] writing into `out` (resized to fit); returns the output shape.
pub fn transpose_into(x: &Tensor, perm: &[usize], out: &mut Vec<f32>) -> Result<Vec<usize>, String> {
    let ndim = x.ndim();
    let perm: Vec<usize> = if perm.is_empty() {
        (0..ndim).rev().collect()
    } else {
        perm.to_vec()
    };
    if perm.len() != ndim {
        return Err(format!(
            "transpose: perm len {} != ndim {}",
            perm.len(),
            ndim
        ));
    }
    // `perm` must be a genuine permutation of `0..ndim`: every entry in range, each used
    // exactly once. Without this, an out-of-range entry indexes `x.shape`/`in_strides` out of
    // bounds, and a repeated entry (e.g. perm=[0,0,1]) silently builds an `out_shape` whose
    // product no longer matches `x.numel()`.
    let mut seen = vec![false; ndim];
    for &p in &perm {
        if p >= ndim {
            return Err(format!(
                "transpose: perm entry {p} out of range for {ndim}D tensor"
            ));
        }
        if seen[p] {
            return Err(format!("transpose: perm entry {p} repeated"));
        }
        seen[p] = true;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    // Size the output from the *shape* product, not from `x.numel()`. Those differ: this
    // crate's `Tensor::numel()` returns `data.len()`, and `Tensor::new` only checks the two
    // agree via a `debug_assert!` (see its doc comment), so a release build can carry a
    // tensor whose buffer is longer than its shape describes -- and sizing the output from
    // that buffer would hand back an equally malformed result. `out_shape` is a permutation
    // of `x.shape`, so this is exactly `out_shape.iter().product()`. For a well-formed
    // tensor (the overwhelming common case) it is the same number as before.
    let shape_numel: usize = x.shape.iter().product();
    out.resize(shape_numel, 0.0);
    // A zero-size dimension anywhere makes `shape_numel == 0`: bail out before any of the
    // stride/division math below, which otherwise risks a `0 / 0` (the trailing-run length
    // can itself be 0 when a size-0 axis lands in the "run"). Nothing to move either way.
    if shape_numel == 0 {
        return Ok(out_shape);
    }
    // The opposite invariant violation -- a buffer *shorter* than the shape describes -- has
    // no correct transpose: some source element the output requires does not exist. Both the
    // tiled fast path and the general odometer below would index past the end of `x.data`, so
    // report it rather than panicking or silently zero-filling part of the result.
    if x.data.len() < shape_numel {
        return Err(format!(
            "transpose: data buffer holds {} elements but shape {:?} describes {shape_numel}",
            x.data.len(),
            x.shape
        ));
    }

    // Prototype (onnx_spike): coalescing, cache-blocked, parallel kernel for every
    // permutation (replaces the trailing-swap fast path and the serial odometer).
    super::transpose_fast::transpose_into(&x.data[..shape_numel], &x.shape, &perm, &mut out[..]);
    Ok(out_shape)
}

/// Resolve the output shape for Squeeze: drop the given (possibly negative) axes, provided
/// each named axis is actually size-1; if `axes` is empty, drop every size-1 axis.
///
/// Squeezing away *every* axis yields the **empty** shape — a genuine rank-0 tensor — not the
/// rank-1 `[1]` this used to promote it to. ONNX distinguishes the two, and the difference is
/// observable one node later: `Shape` of a rank-0 tensor is a length-0 vector, so any
/// `Reshape`/`Concat` driven by it would otherwise be handed one dimension too many. NumPy
/// agrees: `np.squeeze(np.array([5.0])).shape == ()`.
pub(crate) fn resolve_squeeze_shape(x_shape: &[usize], axes: &[i64]) -> Result<Vec<usize>, String> {
    let ndim = x_shape.len();
    let resolved: Vec<usize> = if axes.is_empty() {
        (0..ndim).filter(|&i| x_shape[i] == 1).collect()
    } else {
        axes.iter()
            .map(|&a| normalize_axis(a, ndim))
            .collect::<Result<_, _>>()?
    };
    let new_shape: Vec<usize> = x_shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if resolved.contains(&i) && d == 1 {
                None
            } else {
                Some(d)
            }
        })
        .collect();
    Ok(new_shape)
}

/// Remove axes of size 1. If axes is empty, remove all size-1 dims.
pub fn squeeze(x: &Tensor, axes: &[i64]) -> Result<Tensor, String> {
    let new_shape = resolve_squeeze_shape(&x.shape, axes)?;
    Ok(Tensor::new(x.data.clone(), new_shape))
}

/// Resolve the output shape for Unsqueeze: insert size-1 axes at the given (possibly negative)
/// positions. Per the ONNX spec, negative axes are normalized against the *output* rank
/// (`x_shape.len() + axes.len()`), not the shape as it grows one insertion at a time — using
/// the growing shape mis-normalizes negative axes and mis-orders mixed positive/negative lists.
pub(crate) fn resolve_unsqueeze_shape(
    x_shape: &[usize],
    axes: &[i64],
) -> Result<Vec<usize>, String> {
    let out_rank = x_shape.len() + axes.len();
    let mut normalized: Vec<usize> = axes
        .iter()
        .map(|&a| normalize_axis(a, out_rank))
        .collect::<Result<_, _>>()?;
    normalized.sort_unstable();
    for w in normalized.windows(2) {
        if w[0] == w[1] {
            return Err(format!(
                "unsqueeze: axis {} specified more than once (after normalization)",
                w[0]
            ));
        }
    }
    // Walk every output position once: an output index that is one of the (sorted, unique)
    // target axes gets a new size-1 dim; every other output position pulls the next input dim
    // in order. Since `normalized` is ascending, positions already emitted never shift.
    let mut new_shape = Vec::with_capacity(out_rank);
    let mut axes_iter = normalized.iter().peekable();
    let mut src = x_shape.iter();
    for pos in 0..out_rank {
        if axes_iter.peek() == Some(&&pos) {
            axes_iter.next();
            new_shape.push(1);
        } else {
            new_shape.push(
                *src.next().ok_or_else(|| {
                    "unsqueeze: internal error resolving output shape".to_string()
                })?,
            );
        }
    }
    Ok(new_shape)
}

/// Insert size-1 axes at given positions.
pub fn unsqueeze(x: &Tensor, axes: &[i64]) -> Result<Tensor, String> {
    let new_shape = resolve_unsqueeze_shape(&x.shape, axes)?;
    Ok(Tensor::new(x.data.clone(), new_shape))
}

#[cfg(test)]
mod axis_tests {
    use super::*;

    #[test]
    fn normalize_axis_accepts_full_range() {
        assert_eq!(normalize_axis(0, 3).unwrap(), 0);
        assert_eq!(normalize_axis(2, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-3, 3).unwrap(), 0);
    }

    #[test]
    fn normalize_axis_rejects_out_of_range() {
        assert!(normalize_axis(3, 3).is_err());
        assert!(normalize_axis(-4, 3).is_err());
        // Historically `(axis + rank as i64) as usize` on a deeply negative axis wrapped to a
        // huge usize instead of erroring; confirm that path is now a clean Err.
        assert!(normalize_axis(-10, 2).is_err());
        assert!(normalize_axis(i64::MIN, 2).is_err());
    }

    #[test]
    fn normalize_axis_inclusive_allows_rank() {
        assert_eq!(normalize_axis_inclusive(3, 3).unwrap(), 3);
        assert_eq!(normalize_axis_inclusive(-3, 3).unwrap(), 0);
        assert!(normalize_axis_inclusive(4, 3).is_err());
        assert!(normalize_axis_inclusive(-4, 3).is_err());
    }
}

/// [W2-perf-misc / a6-8] Correctness of the odometer/trailing-run rewrite of
/// `transpose`, which replaced a per-element chain of `ndim` hardware
/// divisions with (a) a tiled fast path for "swap the trailing two axes" and
/// (b) a trailing-contiguous-run `copy_from_slice` + increment/carry odometer
/// for everything else. `transpose` performs no floating-point arithmetic --
/// it only moves `f32` bit patterns around -- so every case below is checked
/// for *exact* equality, not a tolerance.
#[cfg(test)]
mod transpose_tests {
    use super::*;

    /// Independent reference implementation: the exact per-element div/mod
    /// decode `transpose` used before this change. Kept deliberately
    /// separate from the odometer/tiling logic under test (no shared helper,
    /// no shared code path) so a bug in the odometer's carry logic or in the
    /// trailing-swap tile bounds cannot also be present here and cancel out.
    fn reference_transpose(
        data: &[f32],
        shape: &[usize],
        perm: &[usize],
    ) -> (Vec<f32>, Vec<usize>) {
        let ndim = shape.len();
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let out_n: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_n];
        let mut in_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            in_strides[i] = s;
            s *= shape[i];
        }
        let mut out_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = s;
            s *= out_shape[i];
        }
        for (out_idx, out_val) in out.iter_mut().enumerate() {
            let mut rem = out_idx;
            let mut in_idx = 0usize;
            for i in 0..ndim {
                let coord = rem / out_strides[i];
                rem %= out_strides[i];
                in_idx += coord * in_strides[perm[i]];
            }
            *out_val = data[in_idx];
        }
        (out, out_shape)
    }

    /// Every permutation of `0..shape.len()`, via Heap's algorithm.
    fn all_perms(n: usize) -> Vec<Vec<usize>> {
        let mut a: Vec<usize> = (0..n).collect();
        let mut out = vec![a.clone()];
        let mut c = vec![0usize; n];
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    a.swap(0, i);
                } else {
                    a.swap(c[i], i);
                }
                out.push(a.clone());
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        out
    }

    fn check(shape: &[usize], perm: &[usize]) {
        let n: usize = shape.iter().product();
        // Deterministic, non-trivial values (not just the index itself, so an
        // accidental identity-like bug can't hide behind `data[i] == i`).
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.75 - 3.25).collect();
        let x = Tensor::new(data.clone(), shape.to_vec());
        let got = transpose(&x, perm)
            .unwrap_or_else(|e| panic!("transpose{shape:?} perm={perm:?} failed: {e}"));
        let (want_data, want_shape) = reference_transpose(&data, shape, perm);
        assert_eq!(
            got.shape, want_shape,
            "shape mismatch for {shape:?} perm={perm:?}"
        );
        assert_eq!(
            got.data, want_data,
            "data mismatch (must be bit-identical -- transpose moves floats, it doesn't compute them) for {shape:?} perm={perm:?}"
        );
    }

    #[test]
    fn matches_numpy_reference_general_odometer_path() {
        // perm = [0, 2, 1, 3]: last axis stays put (the trailing-run the
        // general path exploits) but axes 1/2 are not the trailing two, so
        // this does NOT take the trailing-swap fast path -- exactly the
        // `[1,512,12,64] -> [1,12,512,64]` attention-permute shape from a6-8,
        // shrunk to a size small enough to hand-verify.
        //
        // ```python
        // x = (np.arange(2*3*2*4, dtype=np.float32) * 0.5 - 3.0).reshape(2,3,2,4)
        // np.transpose(x, (0,2,1,3)).flatten()
        // ```
        let shape = [2usize, 3, 2, 4];
        let perm = [0usize, 2, 1, 3];
        let x_data: Vec<f32> = (0..48).map(|i| i as f32 * 0.5 - 3.0).collect();
        let x = Tensor::new(x_data, shape.to_vec());
        let y = transpose(&x, &perm).expect("transpose failed");
        assert_eq!(y.shape, vec![2, 2, 3, 4]);
        let expected: Vec<f32> = vec![
            -3.0, -2.5, -2.0, -1.5, 1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0, 6.5, -1.0, -0.5, 0.0, 0.5,
            3.0, 3.5, 4.0, 4.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 13.0, 13.5, 14.0, 14.5,
            17.0, 17.5, 18.0, 18.5, 11.0, 11.5, 12.0, 12.5, 15.0, 15.5, 16.0, 16.5, 19.0, 19.5,
            20.0, 20.5,
        ];
        assert_eq!(y.data, expected);
    }

    #[test]
    fn matches_numpy_reference_2d_full_swap() {
        // ```python
        // x = (np.arange(15, dtype=np.float32) * 0.25 - 1.0).reshape(3,5)
        // np.transpose(x, (1,0)).flatten()
        // ```
        let x_data: Vec<f32> = (0..15).map(|i| i as f32 * 0.25 - 1.0).collect();
        let x = Tensor::new(x_data, vec![3, 5]);
        let y = transpose(&x, &[1, 0]).expect("transpose failed");
        assert_eq!(y.shape, vec![5, 3]);
        let expected: Vec<f32> = vec![
            -1.0, 0.25, 1.5, -0.75, 0.5, 1.75, -0.5, 0.75, 2.0, -0.25, 1.0, 2.25, 0.0, 1.25, 2.5,
        ];
        assert_eq!(y.data, expected);
    }

    #[test]
    fn trailing_swap_fast_path_matches_reference_across_tile_boundaries() {
        // rows/cols straddle the 16-element tile edge in both directions
        // (smaller than one tile, exactly one tile, and multiple tiles with a
        // ragged remainder), and `batch > 1` so the per-batch `plane` offset
        // is exercised too.
        for &(batch, rows, cols) in &[
            (1usize, 3usize, 5usize),
            (1, 16, 16),
            (2, 17, 15),
            (3, 31, 33),
            (4, 40, 8),
        ] {
            check(&[batch, rows, cols], &[0, 2, 1]);
            // Rank 4: leading batch axis untouched, trailing two swapped --
            // the `[B,H,S,D] -> [B,H,D,S]` attention K^T pattern.
            check(&[2, batch, rows, cols], &[0, 1, 3, 2]);
        }
    }

    #[test]
    fn general_path_matches_reference_for_every_5d_permutation() {
        // Small enough to be fast (5! = 120 permutations, ~40 elements each)
        // but big enough to exercise every trailing-run length from 0..5 and
        // both fast-path/general-path dispatch.
        let shape = [2usize, 1, 3, 2, 1];
        for perm in all_perms(5) {
            check(&shape, &perm);
        }
    }

    #[test]
    fn zero_size_dimension_does_not_panic() {
        // A dynamic-shape model can legitimately resolve a dimension to 0.
        // Division by the trailing-run length (`out_n / run_len`) must not
        // fire when `out_n == 0`, regardless of which axis carries the 0.
        for shape in [[0usize, 4, 3], [4, 0, 3], [4, 3, 0]] {
            let x = Tensor::new(Vec::new(), shape.to_vec());
            let y = transpose(&x, &[2, 0, 1]).expect("transpose of an empty tensor must not panic");
            assert_eq!(y.numel(), 0);
            assert_eq!(y.shape, vec![shape[2], shape[0], shape[1]]);
        }
    }

    #[test]
    fn identity_permutation_is_a_pure_copy() {
        for shape in [vec![5usize], vec![2, 3], vec![2, 3, 4], vec![1], vec![]] {
            // A 0-d shape's element count is the empty product, 1 -- `Tensor`
            // treats a 0-d tensor as a single scalar element.
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let x = Tensor::new(data.clone(), shape.clone());
            let perm: Vec<usize> = (0..shape.len()).collect();
            let y = transpose(&x, &perm).expect("identity transpose must not fail");
            assert_eq!(y.shape, shape);
            assert_eq!(y.data, data);
        }
    }

    #[test]
    fn empty_perm_reverses_all_axes_like_before() {
        let x = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
        let y = transpose(&x, &[]).expect("transpose with empty perm must not fail");
        assert_eq!(y.shape, vec![4, 3, 2]);
        let (want_data, want_shape) = reference_transpose(&x.data, &x.shape, &[2, 1, 0]);
        assert_eq!(y.shape, want_shape);
        assert_eq!(y.data, want_data);
    }

    #[test]
    fn single_element_tensor() {
        let x = Tensor::new(vec![42.0], vec![1, 1, 1]);
        let y = transpose(&x, &[2, 0, 1]).expect("transpose failed");
        assert_eq!(y.shape, vec![1, 1, 1]);
        assert_eq!(y.data, vec![42.0]);
    }

    #[test]
    fn invalid_perm_still_rejected() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(
            transpose(&x, &[0, 0]).is_err(),
            "repeated perm entry must error"
        );
        assert!(
            transpose(&x, &[0, 2]).is_err(),
            "out-of-range perm entry must error"
        );
        assert!(transpose(&x, &[0]).is_err(), "perm len != ndim must error");
    }
}
