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
