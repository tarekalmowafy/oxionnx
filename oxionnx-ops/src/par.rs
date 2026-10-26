//! Prototype (onnx_spike): minimal intra-op parallelism helpers.
//!
//! Work is split only when it is large enough to amortize rayon's fork/join
//! cost; on wasm32 without `wasm-threads` everything runs serially.

/// Minimum number of f32 elements a copy/elementwise task should touch.
pub(crate) const MIN_ELEMS_PER_TASK: usize = 32 * 1024;

/// The task size actually used: [`MIN_ELEMS_PER_TASK`], or `OXIONNX_PAR_MIN_ELEMS`
/// (read once) for tuning experiments.
#[inline]
pub(crate) fn min_elems() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("OXIONNX_PAR_MIN_ELEMS").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(MIN_ELEMS_PER_TASK)
    })
}

/// Number of worker threads available to the current rayon context.
pub(crate) fn threads() -> usize {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        rayon::current_num_threads()
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        1
    }
}

/// Run `f(row_index, row)` over `out.chunks_mut(row_len)`, in parallel when the
/// total size justifies it. Rows are grouped so each task covers at least
/// [`MIN_ELEMS_PER_TASK`] elements.
pub(crate) fn for_each_row<F>(out: &mut [f32], row_len: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    if row_len == 0 {
        return;
    }
    let rows = out.len() / row_len;
    let t = threads();
    let min = min_elems();
    if t <= 1 || out.len() < 2 * min || rows < 2 {
        for (i, row) in out.chunks_mut(row_len).enumerate() {
            f(i, row);
        }
        return;
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        use rayon::prelude::*;
        let rows_per_task = min.div_ceil(row_len).max(rows.div_ceil(4 * t)).max(1);
        out.par_chunks_mut(rows_per_task * row_len).enumerate().for_each(|(ti, block)| {
            for (j, row) in block.chunks_mut(row_len).enumerate() {
                f(ti * rows_per_task + j, row);
            }
        });
    }
}

/// Elements per chunk of [`map_inplace`] / [`map_into`] (task sizes still
/// follow [`min_elems`]).
const MAP_CHUNK: usize = 16 * 1024;

/// `data[i] = f(data[i])`, in parallel for large tensors.
pub(crate) fn map_inplace(data: &mut [f32], f: impl Fn(f32) -> f32 + Sync + Send) {
    for_each_row(data, MAP_CHUNK, |_, c| c.iter_mut().for_each(|x| *x = f(*x)));
}

/// `dst[i] = f(src[i])` (equal lengths), in parallel for large tensors.
pub(crate) fn map_into(src: &[f32], dst: &mut [f32], f: impl Fn(f32) -> f32 + Sync + Send) {
    debug_assert_eq!(src.len(), dst.len());
    for_each_row(dst, MAP_CHUNK, |ci, c| {
        let s = &src[ci * MAP_CHUNK..ci * MAP_CHUNK + c.len()];
        c.iter_mut().zip(s).for_each(|(d, &x)| *d = f(x));
    });
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn map_matches_serial_on_odd_sizes() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for n in [0, 1, 7, super::MAP_CHUNK - 1, 3 * super::MAP_CHUNK + 5, 5 * super::MIN_ELEMS_PER_TASK + 3] {
            let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 7.0).collect();
            let want: Vec<f32> = src.iter().map(|x| -x).collect();
            let mut dst = vec![f32::NAN; n];
            pool.install(|| super::map_into(&src, &mut dst, |x| -x));
            assert_eq!(dst, want, "map_into n={n}");
            let mut inplace = src.clone();
            pool.install(|| super::map_inplace(&mut inplace, |x| -x));
            assert_eq!(inplace, want, "map_inplace n={n}");
        }
    }
}
