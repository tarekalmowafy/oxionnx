//! Strided gather copies behind `Slice`, `Expand` and `Tile` (PERF_PLAN P1.2 / P1.3).
//!
//! All three ops read their input through an affine view: output element
//! `(i_0, .., i_{n-1})` comes from `src[offset + sum_d i_d * stride_d]`, with
//! `stride_d` = 0 on a broadcast axis (Expand, Tile's repeat axes) and
//! `step_d * input_stride_d` (possibly negative) for Slice. [`strided_copy`]
//! drops unit axes, merges axes that are contiguous in the source, and then
//! copies the innermost run with `copy_from_slice` (stride 1), `fill`
//! (stride 0) or a strided loop, in parallel over the outer rows.

use oxionnx_core::Tensor;

/// One axis of the view: `len` output positions, `stride` source elements apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Axis {
    len: usize,
    stride: isize,
}

/// Unit axes removed, then adjacent axes merged where the outer one steps
/// exactly over the inner one (`outer.stride == inner.stride * inner.len`).
fn coalesce(shape: &[usize], strides: &[isize]) -> Vec<Axis> {
    let mut axes: Vec<Axis> = Vec::with_capacity(shape.len());
    for (&len, &stride) in shape.iter().zip(strides) {
        if len == 1 {
            continue;
        }
        if let Some(last) = axes.last_mut() {
            if last.stride == stride * len as isize {
                last.len *= len;
                last.stride = stride;
                continue;
            }
        }
        axes.push(Axis { len, stride });
    }
    axes
}

/// Copy one innermost run of `dst.len()` elements starting at `src[off]`.
#[inline]
fn copy_run(src: &[f32], off: isize, stride: isize, dst: &mut [f32]) {
    let n = dst.len();
    match stride {
        1 => {
            let o = off as usize;
            dst.copy_from_slice(&src[o..o + n]);
        }
        0 => dst.fill(src[off as usize]),
        s => {
            for (j, d) in dst.iter_mut().enumerate() {
                *d = src[(off + j as isize * s) as usize];
            }
        }
    }
}

/// `dst[i] = src[offset + sum_d i_d * strides[d]]` over the row-major
/// coordinates `i` of `shape`; `dst.len()` must be `shape.iter().product()`.
///
/// The caller guarantees every visited source index is in bounds (indexing
/// still bounds-checks, so a bad view panics instead of reading out of range).
pub(crate) fn strided_copy(src: &[f32], offset: isize, shape: &[usize], strides: &[isize], dst: &mut [f32]) {
    debug_assert_eq!(shape.len(), strides.len());
    debug_assert_eq!(dst.len(), shape.iter().product::<usize>());
    if dst.is_empty() {
        return;
    }
    let axes = coalesce(shape, strides);
    let Some((inner, outer)) = axes.split_last() else {
        dst[0] = src[offset as usize];
        return;
    };
    let row = inner.len;
    if outer.is_empty() {
        copy_run(src, offset, inner.stride, dst);
        return;
    }
    let rows = dst.len() / row;
    // Source offset of row `r`, then advance an odometer over the outer axes.
    let walk = |first: usize, block: &mut [f32]| {
        let mut idx = vec![0usize; outer.len()];
        let mut off = offset;
        let mut rem = first;
        for (d, a) in outer.iter().enumerate().rev() {
            idx[d] = rem % a.len;
            rem /= a.len;
            off += idx[d] as isize * a.stride;
        }
        for out_row in block.chunks_mut(row) {
            copy_run(src, off, inner.stride, out_row);
            for d in (0..outer.len()).rev() {
                idx[d] += 1;
                off += outer[d].stride;
                if idx[d] < outer[d].len {
                    break;
                }
                off -= outer[d].len as isize * outer[d].stride;
                idx[d] = 0;
            }
        }
    };
    let t = crate::par::threads();
    let min = crate::par::min_elems();
    if t <= 1 || dst.len() < 2 * min || rows < 2 {
        walk(0, dst);
        return;
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        use rayon::prelude::*;
        let rows_per_task = min.div_ceil(row).max(rows.div_ceil(4 * t)).max(1);
        dst.par_chunks_mut(rows_per_task * row)
            .enumerate()
            .for_each(|(ti, block)| walk(ti * rows_per_task, block));
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    walk(0, dst);
}

/// Row-major element strides of `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut s = 1isize;
    for i in (0..shape.len()).rev() {
        strides[i] = s;
        s *= shape[i] as isize;
    }
    strides
}

/// Broadcast `x` to `out_shape` (which must be `broadcast(x.shape, ..)`) into `dst`.
pub(crate) fn expand_into(x: &Tensor, out_shape: &[usize], dst: &mut [f32]) {
    let ndim = out_shape.len();
    let pad = ndim - x.shape.len();
    let in_strides = contiguous_strides(&x.shape);
    let strides: Vec<isize> = (0..ndim)
        .map(|d| if d < pad || x.shape[d - pad] == 1 { 0 } else { in_strides[d - pad] })
        .collect();
    strided_copy(&x.data, 0, out_shape, &strides, dst);
}

/// `Tile(x, repeats)` into `dst` (`repeats.len() == x.ndim()`): output axis `d`
/// is the pair (repeat, position) of lengths `(repeats[d], x.shape[d])`.
pub(crate) fn tile_into(x: &Tensor, repeats: &[usize], dst: &mut [f32]) {
    let in_strides = contiguous_strides(&x.shape);
    let mut shape = Vec::with_capacity(2 * x.ndim());
    let mut strides = Vec::with_capacity(2 * x.ndim());
    for d in 0..x.ndim() {
        shape.extend([repeats[d], x.shape[d]]);
        strides.extend([0, in_strides[d]]);
    }
    strided_copy(&x.data, 0, &shape, &strides, dst);
}

/// Copy the slice `start[d] + k * step[d]` (`k < out_shape[d]`) of `x` into
/// `dst`; every visited coordinate must lie inside `x`.
pub(crate) fn slice_into(x: &Tensor, start: &[i64], step: &[i64], out_shape: &[usize], dst: &mut [f32]) {
    let in_strides = contiguous_strides(&x.shape);
    let offset: isize = (0..x.ndim()).map(|d| start[d] as isize * in_strides[d]).sum();
    let strides: Vec<isize> = (0..x.ndim()).map(|d| step[d] as isize * in_strides[d]).collect();
    strided_copy(&x.data, offset, out_shape, &strides, dst);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// Scalar reference: per-element coordinate decomposition.
    fn reference(src: &[f32], offset: isize, shape: &[usize], strides: &[isize]) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n)
            .map(|i| {
                let mut rem = i;
                let mut off = offset;
                for d in (0..shape.len()).rev() {
                    off += (rem % shape[d]) as isize * strides[d];
                    rem /= shape[d];
                }
                src[off as usize]
            })
            .collect()
    }

    fn check_views(threads: usize) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let mut seed = 7u64;
        let src: Vec<f32> = (0..200_000).map(|i| i as f32).collect();
        pool.install(|| {
            for case in 0..400 {
                let ndim = 1 + (lcg(&mut seed) % 5) as usize;
                // Large enough cases cross the parallel threshold.
                let big = case % 4 == 0;
                let shape: Vec<usize> = (0..ndim)
                    .map(|_| if big { 1 + (lcg(&mut seed) % 40) as usize } else { (lcg(&mut seed) % 6) as usize + (case % 2) })
                    .collect();
                // Random strides in [-3, 3]; offset makes all indices non-negative.
                let strides: Vec<isize> = (0..ndim).map(|_| (lcg(&mut seed) % 7) as isize - 3).collect();
                let neg: isize = shape.iter().zip(&strides).map(|(&l, &s)| if s < 0 && l > 0 { -s * (l as isize - 1) } else { 0 }).sum();
                let pos: isize = shape.iter().zip(&strides).map(|(&l, &s)| if s > 0 && l > 0 { s * (l as isize - 1) } else { 0 }).sum();
                if neg + pos >= src.len() as isize {
                    continue;
                }
                let n: usize = shape.iter().product();
                let mut dst = vec![f32::NAN; n];
                strided_copy(&src, neg, &shape, &strides, &mut dst);
                assert_eq!(dst, reference(&src, neg, &shape, &strides), "shape {shape:?} strides {strides:?}");
            }
        });
    }

    #[test]
    fn strided_copy_matches_reference_1_thread() {
        check_views(1);
    }

    #[test]
    fn strided_copy_matches_reference_4_threads() {
        check_views(4);
    }

    /// Coordinates of flat index `i` in row-major `shape`.
    fn coords(mut i: usize, shape: &[usize]) -> Vec<usize> {
        let mut c = vec![0; shape.len()];
        for d in (0..shape.len()).rev() {
            c[d] = i % shape[d];
            i /= shape[d];
        }
        c
    }

    fn flat(c: &[usize], shape: &[usize]) -> usize {
        c.iter().zip(shape).fold(0, |acc, (&x, &l)| acc * l + x)
    }

    fn ramp(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::new((0..n).map(|i| i as f32 * 0.5 - 3.0).collect(), shape.to_vec())
    }

    /// Model-sized Expand / Tile / Slice views (above the parallel threshold) at
    /// 1 and 6 threads against per-element references.
    #[test]
    fn ops_match_reference_on_model_shapes() {
        for threads in [1, 6] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                // Expand: repeat_kv [B,Hkv,1,T,H] -> [B,Hkv,2,T,H]; mask [B,1,1,T] -> [B,N,T,T]; rank-padding.
                for (xs, os) in [
                    (vec![3, 4, 1, 37, 16], vec![3, 4, 2, 37, 16]),
                    (vec![5, 1, 1, 61], vec![5, 7, 61, 61]),
                    (vec![61, 1], vec![2, 61, 300]),
                ] {
                    let x = ramp(&xs);
                    let n: usize = os.iter().product();
                    let mut got = vec![0.0; n];
                    expand_into(&x, &os, &mut got);
                    let pad = os.len() - xs.len();
                    let want: Vec<f32> = (0..n)
                        .map(|i| {
                            let c = coords(i, &os);
                            let ci: Vec<usize> = (0..xs.len()).map(|d| if xs[d] == 1 { 0 } else { c[d + pad] }).collect();
                            x.data[flat(&ci, &xs)]
                        })
                        .collect();
                    assert_eq!(got, want, "expand {xs:?} -> {os:?}");
                }
                // Tile: mask head repeat, and every axis repeated.
                for (xs, reps) in [(vec![3, 1, 45, 45], vec![1, 12, 1, 1]), (vec![2, 3, 129], vec![3, 2, 2])] {
                    let x = ramp(&xs);
                    let os: Vec<usize> = xs.iter().zip(&reps).map(|(a, b)| a * b).collect();
                    let n: usize = os.iter().product();
                    let mut got = vec![0.0; n];
                    tile_into(&x, &reps, &mut got);
                    let want: Vec<f32> = (0..n)
                        .map(|i| {
                            let c = coords(i, &os);
                            let ci: Vec<usize> = c.iter().zip(&xs).map(|(&a, &l)| a % l).collect();
                            x.data[flat(&ci, &xs)]
                        })
                        .collect();
                    assert_eq!(got, want, "tile {xs:?} x {reps:?}");
                }
                // Slice: GLU halves of [B,S,2F], rotate-half of [B,S,H,D], reversed strided axis.
                let x = ramp(&[4, 33, 2 * 515]);
                for (starts, ends, axes, steps) in [
                    (vec![0i64], vec![515i64], vec![-1i64], None),
                    (vec![515], vec![i64::MAX], vec![2], None),
                    (vec![-1, 3], vec![i64::MIN, 1000], vec![1, 2], Some(vec![-2i64, 7])),
                ] {
                    let got = crate::shape::slice(&x, &starts, &ends, Some(&axes), steps.as_deref()).unwrap();
                    let plan = crate::shape::sequence::slice_plan(&x.shape, &starts, &ends, Some(&axes), steps.as_deref()).unwrap();
                    let want: Vec<f32> = (0..got.data.len())
                        .map(|i| {
                            let c = coords(i, &got.shape);
                            let ci: Vec<usize> = (0..3).map(|d| (plan.start[d] + c[d] as i64 * plan.step[d]) as usize).collect();
                            x.data[flat(&ci, &x.shape)]
                        })
                        .collect();
                    assert_eq!(got.data, want, "slice {starts:?}..{ends:?} axes {axes:?} steps {steps:?}");
                }
                let rot = ramp(&[8, 40, 12, 64]);
                let half = crate::shape::slice(&rot, &[32], &[64], Some(&[3]), None).unwrap();
                assert_eq!(half.shape, vec![8, 40, 12, 32]);
                for (i, v) in half.data.iter().enumerate() {
                    let c = coords(i, &half.shape);
                    assert_eq!(*v, rot.data[flat(&[c[0], c[1], c[2], c[3] + 32], &rot.shape)]);
                }
            });
        }
    }

    #[test]
    fn coalesce_merges_contiguous_axes() {
        assert_eq!(coalesce(&[2, 3, 4], &[12, 4, 1]), vec![Axis { len: 24, stride: 1 }]);
        assert_eq!(coalesce(&[2, 1, 3, 4], &[0, 99, 4, 1]), vec![Axis { len: 2, stride: 0 }, Axis { len: 12, stride: 1 }]);
        assert_eq!(coalesce(&[2, 3], &[0, 0]), vec![Axis { len: 6, stride: 0 }]);
    }

    #[test]
    fn scalar_and_empty_views() {
        let src = [5.0f32, 6.0];
        let mut one = [0.0f32];
        strided_copy(&src, 1, &[], &[], &mut one);
        assert_eq!(one, [6.0]);
        strided_copy(&src, 0, &[1, 1], &[7, 3], &mut one);
        assert_eq!(one, [5.0]);
        let mut none: [f32; 0] = [];
        strided_copy(&src, 0, &[3, 0], &[1, 1], &mut none);
    }
}
