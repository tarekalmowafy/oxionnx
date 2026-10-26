//! Prototype (onnx_spike): general-permutation transpose, parallel and cache-blocked.
//!
//! Mirrors what ORT's `transpose.cc` does with MlasTranspose:
//! 1. drop size-1 axes and merge axes that stay adjacent under `perm`;
//! 2. if the innermost axis is untouched, move contiguous runs (`copy_from_slice`);
//! 3. otherwise do a 16x16-tiled 2-D transpose between the input's innermost axis
//!    and the input axis that becomes the output's innermost one, for every
//!    combination of the remaining axes.
//!
//! Outer blocks write disjoint parts of the output and run in parallel.

#![allow(unsafe_code)]

/// Coalesced problem: `perm[k]` is the input axis of output axis `k`.
struct Plan {
    in_shape: Vec<usize>,
    perm: Vec<usize>,
}

fn coalesce(shape: &[usize], perm: &[usize]) -> Plan {
    // Keep only non-unit axes, renumbered in input order.
    let keep: Vec<usize> = (0..shape.len()).filter(|&a| shape[a] != 1).collect();
    let mut renum = vec![usize::MAX; shape.len()];
    for (i, &a) in keep.iter().enumerate() {
        renum[a] = i;
    }
    let shape1: Vec<usize> = keep.iter().map(|&a| shape[a]).collect();
    let perm1: Vec<usize> = perm.iter().filter(|&&a| shape[a] != 1).map(|&a| renum[a]).collect();
    // Group runs of output axes that are consecutive input axes.
    let mut groups: Vec<(usize, usize)> = Vec::new(); // (first input axis, size)
    for (k, &a) in perm1.iter().enumerate() {
        if k > 0 && perm1[k - 1] + 1 == a {
            groups.last_mut().unwrap().1 *= shape1[a];
        } else {
            groups.push((a, shape1[a]));
        }
    }
    // Input axis order of the groups = order of their first input axis.
    let mut by_input: Vec<usize> = (0..groups.len()).collect();
    by_input.sort_by_key(|&g| groups[g].0);
    let mut in_index = vec![0; groups.len()];
    for (i, &g) in by_input.iter().enumerate() {
        in_index[g] = i;
    }
    Plan {
        in_shape: by_input.iter().map(|&g| groups[g].1).collect(),
        perm: (0..groups.len()).map(|g| in_index[g]).collect(),
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// `out` (already sized to `src.len()`) = `src` (row-major `shape`) permuted by `perm`.
pub(crate) fn transpose_into(src: &[f32], shape: &[usize], perm: &[usize], out: &mut [f32]) {
    debug_assert_eq!(src.len(), out.len());
    let numel = src.len();
    if numel == 0 {
        return;
    }
    let Plan { in_shape, perm } = coalesce(shape, perm);
    let nd = in_shape.len();
    if nd <= 1 {
        out.copy_from_slice(src);
        return;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&a| in_shape[a]).collect();
    let is = strides(&in_shape);
    let os = strides(&out_shape);
    let op = SendPtr(out.as_mut_ptr());
    let op = &op;

    if perm[nd - 1] == nd - 1 {
        // Contiguous runs of the innermost axis; walk the other output axes.
        let run = in_shape[nd - 1];
        let n_outer = numel / run;
        let outer_shape = &out_shape[..nd - 1];
        let per_task = (crate::par::min_elems() / run).max(1);
        par_blocks(n_outer, per_task, numel, |lo, hi| {
            for blk in lo..hi {
                let (mut r, mut in_off) = (blk, 0);
                for k in (0..nd - 1).rev() {
                    in_off += (r % outer_shape[k]) * is[perm[k]];
                    r /= outer_shape[k];
                }
                // SAFETY: output block `blk` is `[blk*run, (blk+1)*run)`, disjoint per block.
                unsafe { std::slice::from_raw_parts_mut(op.0.add(blk * run), run) }
                    .copy_from_slice(&src[in_off..in_off + run]);
            }
        });
        return;
    }

    // Tiled 2-D transpose between output axis `p` (= input's innermost axis, contiguous
    // in the input) and the output's innermost axis (= input axis `q`, contiguous in the output).
    let q = perm[nd - 1];
    let p = perm.iter().position(|&a| a == nd - 1).unwrap();
    let (rows, cols) = (in_shape[nd - 1], in_shape[q]);
    let (in_q, out_p) = (is[q], os[p]);
    let others: Vec<usize> = (0..nd).filter(|&k| k != p && k != nd - 1).collect();
    let n_outer = numel / (rows * cols);
    let per_task = (crate::par::min_elems() / (rows * cols)).max(1);
    par_blocks(n_outer, per_task, numel, |lo, hi| {
        // Each B x B tile is staged through a stack buffer: contiguous 64-byte
        // reads of B input rows, then contiguous writes of B output rows. Writing
        // each element straight to its strided output position instead was ~5x
        // slower ([48, 768] -> [768, 48]: 108 us vs 19 us on Zen 4), the
        // interleaved strided stores and loads defeating the load/store units.
        const B: usize = 16;
        let mut tmp = [[0f32; B]; B];
        for blk in lo..hi {
            let (mut r, mut in_base, mut out_base) = (blk, 0, 0);
            for &k in others.iter().rev() {
                let c = r % out_shape[k];
                r /= out_shape[k];
                in_base += c * is[perm[k]];
                out_base += c * os[k];
            }
            for i0 in (0..rows).step_by(B) {
                let bi = B.min(rows - i0);
                for j0 in (0..cols).step_by(B) {
                    let bj = B.min(cols - j0);
                    for (j, t) in tmp.iter_mut().enumerate().take(bj) {
                        let s0 = in_base + (j0 + j) * in_q + i0;
                        t[..bi].copy_from_slice(&src[s0..s0 + bi]);
                    }
                    for i in 0..bi {
                        // SAFETY: output row (blk, i0 + i), columns j0..j0 + bj, is
                        // in bounds and written by this block only.
                        let o = unsafe { std::slice::from_raw_parts_mut(op.0.add(out_base + (i0 + i) * out_p + j0), bj) };
                        for (j, x) in o.iter_mut().enumerate() {
                            *x = tmp[j][i];
                        }
                    }
                }
            }
        }
    });
}

/// Run `f(lo, hi)` over `0..n` in chunks of `per_task`, in parallel when `numel` is large.
fn par_blocks(n: usize, per_task: usize, numel: usize, f: impl Fn(usize, usize) + Sync) {
    let tasks = n.div_ceil(per_task);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if tasks > 1 && numel >= 2 * crate::par::min_elems() && crate::par::threads() > 1 {
        use rayon::prelude::*;
        (0..tasks).into_par_iter().for_each(|t| f(t * per_task, ((t + 1) * per_task).min(n)));
        return;
    }
    let _ = (tasks, numel);
    f(0, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(src: &[f32], shape: &[usize], perm: &[usize]) -> Vec<f32> {
        let nd = shape.len();
        let is = strides(shape);
        let out_shape: Vec<usize> = perm.iter().map(|&a| shape[a]).collect();
        let mut out = vec![0.0; src.len()];
        for (o, dst) in out.iter_mut().enumerate() {
            let (mut r, mut off) = (o, 0);
            for k in (0..nd).rev() {
                off += (r % out_shape[k]) * is[perm[k]];
                r /= out_shape[k];
            }
            *dst = src[off];
        }
        out
    }

    fn perms(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut res = vec![];
        for p in perms(n - 1) {
            for i in 0..=p.len() {
                let mut q = p.clone();
                q.insert(i, n - 1);
                res.push(q);
            }
        }
        res
    }

    #[test]
    fn matches_naive_for_all_perms() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for shape in [vec![3, 1, 5, 7], vec![2, 17, 33, 4], vec![8, 64, 2, 64], vec![1, 300, 2, 2, 19]] {
            let n: usize = shape.iter().product();
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            for perm in perms(shape.len()) {
                let want = naive(&src, &shape, &perm);
                let mut got = vec![f32::NAN; n];
                pool.install(|| transpose_into(&src, &shape, &perm, &mut got));
                assert_eq!(got, want, "shape {shape:?} perm {perm:?}");
            }
        }
    }

    /// Timing (not a test): `TRANSPOSE_BENCH=1,48,12,64:0,2,3,1` `cargo test --release ... bench_transpose -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_transpose() {
        let spec = std::env::var("TRANSPOSE_BENCH").unwrap_or_else(|_| "1,48,12,64:0,2,3,1;1,48,12,64:0,2,1,3".into());
        for case in spec.split(';') {
            let (sh, pm) = case.split_once(':').unwrap();
            let shape: Vec<usize> = sh.split(',').map(|v| v.parse().unwrap()).collect();
            let perm: Vec<usize> = pm.split(',').map(|v| v.parse().unwrap()).collect();
            let n: usize = shape.iter().product();
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut out = vec![0f32; n];
            let pl = coalesce(&shape, &perm);
            println!("plan in_shape {:?} perm {:?}", pl.in_shape, pl.perm);
            let mut best = f64::MAX;
            for _ in 0..200 {
                let t0 = std::time::Instant::now();
                transpose_into(&src, &shape, &perm, &mut out);
                best = best.min(t0.elapsed().as_secs_f64() * 1e6);
            }
            println!("transpose {shape:?} perm {perm:?}: {best:.1} us");
            if pl.in_shape.len() == 2 {
                let (cols, rows) = (pl.in_shape[0], pl.in_shape[1]);
                let mut best = f64::MAX;
                for _ in 0..200 {
                    let t0 = std::time::Instant::now();
                    for i in 0..rows { for j in 0..cols { out[i * cols + j] = src[j * rows + i]; } }
                    std::hint::black_box(&out);
                    best = best.min(t0.elapsed().as_secs_f64() * 1e6);
                }
                println!("  naive 2-D: {best:.1} us");
                let mut best = f64::MAX;
                for _ in 0..200 {
                    let t0 = std::time::Instant::now();
                    const B: usize = 16;
                    let mut tmp = [[0f32; B]; B];
                    for i0 in (0..rows).step_by(B) {
                        for j0 in (0..cols).step_by(B) {
                            let (bi, bj) = (B.min(rows - i0), B.min(cols - j0));
                            for j in 0..bj {
                                tmp[j][..bi].copy_from_slice(&src[(j0 + j) * rows + i0..(j0 + j) * rows + i0 + bi]);
                            }
                            for i in 0..bi {
                                let o = &mut out[(i0 + i) * cols + j0..(i0 + i) * cols + j0 + bj];
                                for j in 0..bj { o[j] = tmp[j][i]; }
                            }
                        }
                    }
                    std::hint::black_box(&out);
                    best = best.min(t0.elapsed().as_secs_f64() * 1e6);
                }
                println!("  tile+tmp: {best:.1} us");
            }
        }
    }
}
