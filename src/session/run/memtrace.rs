//! `OXIONNX_TRACE_MEM=1` diagnostic: per run, the peak of live intermediate bytes
//! (run state) plus bytes cached in the activation pool, the node where it
//! happened with its largest live tensors, and the process RSS. Off by default;
//! when off every hook is one relaxed atomic load.

use crate::memory::SizeClassPool;
use std::sync::Mutex;

use super::state::SessionRunState;

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXIONNX_TRACE_MEM").is_some())
}

fn rss_mib() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |k: &str| {
        status
            .lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(f64::NAN, |kb| kb / 1024.0)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

const MIB: f64 = 1024.0 * 1024.0;

#[derive(Default)]
pub(crate) struct MemTrace {
    peak_total: usize,
    peak_live: usize,
    peak_pool: usize,
    peak_node: String,
    peak_top: Vec<(String, usize, Vec<usize>)>,
    peak_rss: f64,
    allocs_start: u64,
    alloc_bytes_start: u64,
    zeroed_start: u64,
    pool_start: usize,
    /// Bytes of outputs freshly allocated by kernels (non-slot, non-inplace path), per op.
    fresh: std::collections::BTreeMap<String, usize>,
}

impl MemTrace {
    pub(crate) fn start(pool: Option<&Mutex<SizeClassPool>>) -> Self {
        let mut t = Self::default();
        if let Some(p) = pool.and_then(|p| p.lock().ok()) {
            t.allocs_start = p.stats().alloc_count;
            t.alloc_bytes_start = p.stats().alloc_bytes;
            t.zeroed_start = p.stats().zeroed_bytes;
            t.pool_start = p.stats().current_bytes;
        }
        t
    }

    pub(crate) fn fresh(&mut self, op: &str, bytes: usize) {
        *self.fresh.entry(op.to_string()).or_insert(0) += bytes;
    }

    pub(crate) fn after_node(
        &mut self,
        node: &crate::graph::Node,
        state: &SessionRunState,
        pool: Option<&Mutex<SizeClassPool>>,
    ) {
        let live: usize = state.as_map().values().map(|t| t.data.capacity() * 4).sum();
        let pooled = pool
            .and_then(|p| p.lock().ok())
            .map_or(0, |p| p.stats().current_bytes);
        if live + pooled > self.peak_total {
            self.peak_total = live + pooled;
            self.peak_live = live;
            self.peak_pool = pooled;
            self.peak_node = format!("{} ({})", node.name, node.op.as_str());
            let mut top: Vec<_> = state
                .as_map()
                .iter()
                .map(|(n, t)| (n.clone(), t.data.capacity() * 4, t.shape.clone()))
                .collect();
            top.sort_by_key(|t| std::cmp::Reverse(t.1));
            top.truncate(8);
            self.peak_top = top;
            self.peak_rss = rss_mib().0;
        }
    }

    pub(crate) fn report(&self, pool: Option<&Mutex<SizeClassPool>>) {
        let (rss, hwm) = rss_mib();
        let (allocs, alloc_mib, pool_end, nbuf, budget, zeroed) = pool.and_then(|p| p.lock().ok()).map_or((0, 0.0, 0, 0, 0, 0.0), |p| {
            (
                p.stats().alloc_count - self.allocs_start,
                (p.stats().alloc_bytes - self.alloc_bytes_start) as f64 / MIB,
                p.stats().current_bytes,
                p.buffer_count(),
                p.budget().unwrap_or(0),
                (p.stats().zeroed_bytes - self.zeroed_start) as f64 / MIB,
            )
        });
        eprintln!(
            "oxionnx-mem run: peak live+pool {:.1} MiB (live {:.1}, pool {:.1}) at {}; rss there {:.0}, now {:.0}, hwm {:.0}; pool start {:.1} end {:.1} MiB ({} bufs, budget {:.1}), {} new pool allocs ({:.1} MiB), {:.1} MiB zero-filled",
            self.peak_total as f64 / MIB, self.peak_live as f64 / MIB, self.peak_pool as f64 / MIB, self.peak_node,
            self.peak_rss, rss, hwm, self.pool_start as f64 / MIB, pool_end as f64 / MIB, nbuf, budget as f64 / MIB, allocs, alloc_mib, zeroed
        );
        for (n, b, s) in &self.peak_top {
            eprintln!("oxionnx-mem   live {:8.1} MiB {:?} {}", *b as f64 / MIB, s, n);
        }
        let mut fresh: Vec<_> = self.fresh.iter().collect();
        fresh.sort_by_key(|f| std::cmp::Reverse(*f.1));
        let line: Vec<String> = fresh.iter().take(12).map(|(op, b)| format!("{op} {:.1}", **b as f64 / MIB)).collect();
        eprintln!("oxionnx-mem   fresh output MiB by op: {}", line.join(", "));
    }
}
