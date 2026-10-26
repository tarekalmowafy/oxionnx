//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::Node;
use std::collections::HashMap;

use super::functions::{bucket_for, compute_peak_memory, MAX_BUCKETS_PER_CLASS, MAX_POOL_BUFFERS};

/// Size-class-based memory pool that reduces fragmentation.
///
/// Buckets: tiny (<128 elements), small (<1024), medium (<16384), large (≥16384).
/// Within each bucket, best-fit allocation picks the smallest buffer ≥ requested size.
///
/// # Bounding what the pool keeps
///
/// Two rules stop the cache from growing far past what a run actually needs:
///
/// * A cached buffer is only handed out for a request it fits **tightly**
///   ([`fits_tightly`]): at most twice the request, or at most
///   [`MAX_SLACK_ELEMENTS`] longer. Handing a 24 MiB buffer to a 32 KiB request
///   keeps the 24 MiB alive under a small tensor (as long as that tensor lives)
///   and forces the next large request to allocate afresh.
/// * An optional **byte budget** ([`SizeClassPool::raise_budget`]) caps the bytes
///   held in the free lists. A release that would exceed it evicts the
///   least-recently released buffers first. The session sets the budget to the
///   peak bytes of live intermediates it has seen in one run. With perfect size
///   matching the free lists would never need more than that; when they do (a
///   request misses that a buffer evicted for the budget would have served), the
///   budget grows by that buffer's size, so it settles at the smallest value the
///   run's reuse pattern needs. Without a budget (the default for a pool built
///   with [`SizeClassPool::new`]) the cache is bounded only by the per-class
///   buffer count.
pub struct SizeClassPool {
    /// Free lists per size class, each sorted by capacity.
    tiny: Vec<Cached>,
    small: Vec<Cached>,
    medium: Vec<Cached>,
    large: Vec<Cached>,
    /// Allocation and reuse statistics.
    stats: PoolStats,
    /// Release counter, used to evict the least-recently released buffer first.
    clock: u64,
    /// Upper bound on the bytes held in the free lists, if any.
    budget: Option<usize>,
    /// Capacities (elements) of the buffers most recently evicted for the budget,
    /// with the run epoch of the eviction, newest last; a fresh allocation one of
    /// them would have served raises the budget.
    evicted: std::collections::VecDeque<(usize, u64)>,
    /// Incremented by [`SizeClassPool::end_run`]. Only evictions from this run or
    /// the previous one count as the cause of a miss: an older one is not why a
    /// buffer is missing now, since the pool has had a whole run to refill.
    epoch: u64,
}

/// How many budget evictions [`SizeClassPool`] remembers for its miss check.
const EVICTED_HISTORY: usize = 64;

/// A cached buffer and the release-clock value when it was returned.
struct Cached {
    buf: Vec<f32>,
    stamp: u64,
}

/// Largest absolute over-allocation (in elements, 64 KiB) a recycled buffer may
/// carry beyond the request, whatever the ratio.
pub const MAX_SLACK_ELEMENTS: usize = 16 * 1024;

/// May a cached buffer of `capacity` elements serve a request for `size`?
#[inline]
pub fn fits_tightly(capacity: usize, size: usize) -> bool {
    capacity >= size && (capacity <= size.saturating_mul(2) || capacity - size <= MAX_SLACK_ELEMENTS)
}

impl SizeClassPool {
    /// Create a new empty size-class pool.
    pub fn new() -> Self {
        Self {
            tiny: Vec::new(),
            small: Vec::new(),
            medium: Vec::new(),
            large: Vec::new(),
            stats: PoolStats::default(),
            clock: 0,
            budget: None,
            evicted: std::collections::VecDeque::new(),
            epoch: 0,
        }
    }
    /// Raise the byte budget for cached buffers to at least `bytes` (setting one
    /// if there was none). The budget never shrinks; see the type-level docs.
    pub fn raise_budget(&mut self, bytes: usize) {
        self.budget = Some(self.budget.map_or(bytes, |b| b.max(bytes)));
    }
    /// Mark the end of one run (one pass of acquire/release over a graph); see
    /// the type-level docs on how the budget adapts.
    pub fn end_run(&mut self) {
        self.epoch += 1;
        let oldest = self.epoch.saturating_sub(1);
        self.evicted.retain(|&(_, e)| e >= oldest);
    }
    /// The current byte budget for cached buffers, if any.
    pub fn budget(&self) -> Option<usize> {
        self.budget
    }
    /// Take buffer `idx` out of `class`'s free list, updating the accounting.
    fn take_at(&mut self, class: SizeClass, idx: usize) -> Vec<f32> {
        let buf = self.bucket_mut(class).remove(idx).buf;
        self.stats.reuse_count += 1;
        let freed_bytes = buf.capacity() * std::mem::size_of::<f32>();
        self.stats.current_bytes = self.stats.current_bytes.saturating_sub(freed_bytes);
        buf
    }
    /// A freshly allocated, zeroed buffer of `size` elements.
    fn fresh(&mut self, size: usize, count_miss: bool) -> Vec<f32> {
        // A miss that a buffer evicted for the budget would have served: the
        // budget is too tight for this run's reuse pattern, so grow it by that
        // buffer.
        if let Some(budget) = self.budget.as_mut().filter(|_| count_miss) {
            if let Some(pos) = self.evicted.iter().rposition(|&(cap, _)| fits_tightly(cap, size)) {
                let cap = self.evicted.remove(pos).map_or(0, |(cap, _)| cap);
                *budget += cap * std::mem::size_of::<f32>();
            }
        }
        self.stats.alloc_count += 1;
        let buf = vec![0.0_f32; size];
        let allocated_bytes = buf.capacity() * std::mem::size_of::<f32>();
        self.stats.alloc_bytes += allocated_bytes as u64;
        let total = self.stats.current_bytes + allocated_bytes;
        if total > self.stats.peak_bytes {
            self.stats.peak_bytes = total;
        }
        buf
    }
    /// Acquire a buffer with at least `size` f32 elements.
    ///
    /// Searches the appropriate size-class bucket for the smallest buffer
    /// that satisfies the request (best-fit) and fits it tightly
    /// ([`fits_tightly`]). If no suitable buffer is cached, a fresh allocation
    /// is made. The returned buffer is zeroed and has exactly `size` elements.
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        self.acquire_zeroed(size, true)
    }
    /// Acquire a buffer for a tensor that will **leave** the pool's cycle (a
    /// graph output handed to the caller): served from the cache like
    /// [`Self::acquire`] / [`Self::acquire_for_overwrite`] (`overwrite`), but a
    /// miss never grows the budget, since no amount of caching would make the
    /// next run's request for it hit.
    pub fn acquire_for_output(&mut self, size: usize, overwrite: bool) -> Vec<f32> {
        if overwrite {
            self.acquire_unzeroed(size, false)
        } else {
            self.acquire_zeroed(size, false)
        }
    }
    fn acquire_zeroed(&mut self, size: usize, count_miss: bool) -> Vec<f32> {
        let class = bucket_for(size);
        let found = Self::best_fit_index(self.bucket_ref(class), size)
            .map(|idx| (class, idx))
            .or_else(|| self.find_in_larger_buckets(class, size));
        match found {
            Some((found_class, idx)) => {
                let mut buf = self.take_at(found_class, idx);
                buf.clear();
                buf.resize(size, 0.0);
                buf
            }
            None => self.fresh(size, count_miss),
        }
    }
    /// Acquire a buffer of exactly `size` elements whose **contents are
    /// unspecified**.
    ///
    /// Identical to [`SizeClassPool::acquire`] in every respect except one: it
    /// does not zero the elements a recycled buffer already has.  A buffer that
    /// comes back from the pool already the right length is returned with the
    /// previous tensor's values still in it; a shorter one is grown with zeros
    /// (there is nothing else to grow it with) and a fresh allocation is zeroed
    /// because `vec![0.0; n]` is how one is made.
    ///
    /// # When this is correct — and when it silently leaks data
    ///
    /// Only for a caller that writes **every** element before anything reads
    /// one.  `acquire` exists to make "forgot to initialise" impossible; this
    /// entry point trades that guarantee for one avoided `memset` per output
    /// buffer, which on a convolution or a matmul is a full pass over the output
    /// tensor immediately before the kernel overwrites it.
    ///
    /// Used wrongly, the failure mode is not a crash: it is the *previous*
    /// tensor's values appearing in the untouched part of the new one — safe
    /// Rust, wrong numbers, and a leak of one intermediate's contents into
    /// another.  That is why this is a separate, explicitly-named method rather
    /// than a change to `acquire`; the engine routes an output slot through it
    /// only for operators known to write every element (`fully_writes_slot` in
    /// `session/run/dispatch.rs`).
    ///
    /// No `unsafe`, and no uninitialised memory is ever exposed: the returned
    /// `Vec` has length `size` and every element is a valid `f32`, just not
    /// necessarily `0.0`.
    ///
    /// ```
    /// use oxionnx::SizeClassPool;
    ///
    /// let mut pool = SizeClassPool::new();
    /// let mut buf = pool.acquire(4);
    /// buf.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// pool.release(buf);
    ///
    /// // The recycled buffer is exactly the requested length, and the caller is
    /// // responsible for writing all of it.
    /// let reused = pool.acquire_for_overwrite(4);
    /// assert_eq!(reused.len(), 4);
    /// ```
    pub fn acquire_for_overwrite(&mut self, size: usize) -> Vec<f32> {
        self.acquire_unzeroed(size, true)
    }
    fn acquire_unzeroed(&mut self, size: usize, count_miss: bool) -> Vec<f32> {
        let class = bucket_for(size);
        // An exact-length buffer anywhere in the pool needs no zeroing at all; in
        // steady state every run repeats the same sizes.
        let exact = [SizeClass::Tiny, SizeClass::Small, SizeClass::Medium, SizeClass::Large]
            .into_iter()
            .find_map(|c| {
                self.bucket_ref(c)
                    .iter()
                    .position(|b| b.buf.len() == size && fits_tightly(b.buf.capacity(), size))
                    .map(|i| (c, i))
            });
        // Otherwise prefer a buffer whose *length* already covers `size` (no tail
        // to zero), then best fit.
        let found = exact
            .or_else(|| Self::no_zeroing_fit_index(self.bucket_ref(class), size).map(|i| (class, i)))
            .or_else(|| self.find_in_larger_buckets(class, size));
        match found {
            Some((found_class, idx)) => {
                let mut buf = self.take_at(found_class, idx);
                // `truncate` when it is already long enough: no write at all.
                // `resize` only when it is short, and then only the tail is
                // written — `resize` never touches the elements it keeps.
                if buf.len() >= size {
                    buf.truncate(size);
                } else {
                    buf.resize(size, 0.0);
                }
                buf
            }
            None => self.fresh(size, count_miss),
        }
    }
    /// Release a buffer back into the pool.
    ///
    /// The buffer is placed into the bucket corresponding to its length (requested size).
    /// Buffers are not shrunk on return. Per-class limits and the byte budget
    /// prevent unbounded growth: over budget, the least-recently released
    /// buffers are dropped first, and a buffer larger than the whole budget is
    /// not cached at all.
    pub fn release(&mut self, buf: Vec<f32>) {
        if buf.capacity() == 0 {
            return;
        }
        let class = bucket_for(buf.len());
        let added_bytes = buf.capacity() * std::mem::size_of::<f32>();
        if let Some(budget) = self.budget {
            if added_bytes > budget {
                self.remember_eviction(buf.capacity());
                return;
            }
            while self.stats.current_bytes + added_bytes > budget && self.evict_oldest() {}
        }
        let bucket = self.bucket_mut(class);
        if bucket.len() >= MAX_BUCKETS_PER_CLASS {
            if let Some(smallest_cap) = bucket.first().map(|b| b.buf.capacity()) {
                if buf.capacity() > smallest_cap {
                    let evicted = bucket.remove(0);
                    let evicted_bytes = evicted.buf.capacity() * std::mem::size_of::<f32>();
                    self.stats.current_bytes =
                        self.stats.current_bytes.saturating_sub(evicted_bytes);
                } else {
                    return;
                }
            }
        }
        let cap = buf.capacity();
        self.clock += 1;
        let stamp = self.clock;
        let bucket = self.bucket_mut(class);
        let pos = bucket.partition_point(|b| b.buf.capacity() < cap);
        bucket.insert(pos, Cached { buf, stamp });
        self.stats.current_bytes += added_bytes;
        if self.stats.current_bytes > self.stats.peak_bytes {
            self.stats.peak_bytes = self.stats.current_bytes;
        }
        self.update_fragmentation();
    }
    /// Drop the least-recently released cached buffer. Returns `false` when the
    /// pool is empty.
    fn evict_oldest(&mut self) -> bool {
        let oldest = [SizeClass::Tiny, SizeClass::Small, SizeClass::Medium, SizeClass::Large]
            .into_iter()
            .filter_map(|c| {
                self.bucket_ref(c)
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, b)| b.stamp)
                    .map(|(i, b)| (b.stamp, c, i))
            })
            .min_by_key(|&(stamp, _, _)| stamp);
        let Some((_, class, idx)) = oldest else {
            return false;
        };
        let evicted = self.bucket_mut(class).remove(idx);
        let evicted_bytes = evicted.buf.capacity() * std::mem::size_of::<f32>();
        self.stats.current_bytes = self.stats.current_bytes.saturating_sub(evicted_bytes);
        self.remember_eviction(evicted.buf.capacity());
        true
    }
    fn remember_eviction(&mut self, capacity: usize) {
        if self.evicted.len() == EVICTED_HISTORY {
            self.evicted.pop_front();
        }
        self.evicted.push_back((capacity, self.epoch));
    }
    /// Return a reference to the pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
    /// Number of buffers cached across all size classes.
    pub fn buffer_count(&self) -> usize {
        self.tiny.len() + self.small.len() + self.medium.len() + self.large.len()
    }
    /// Drop all cached buffers, resetting the pool.
    pub fn clear(&mut self) {
        self.tiny.clear();
        self.small.clear();
        self.medium.clear();
        self.large.clear();
        self.stats.current_bytes = 0;
    }
    /// Compact the pool by dropping oversized buffers.
    ///
    /// If fragmentation exceeds 20%, removes buffers whose capacity is more
    /// than 2× the upper bound of their size class. For the `Large` class,
    /// no compaction is applied since there is no meaningful upper bound.
    pub fn compact(&mut self) {
        if self.stats.fragmentation_ratio <= 0.20 {
            return;
        }
        self.compact_bucket(SizeClass::Tiny);
        self.compact_bucket(SizeClass::Small);
        self.compact_bucket(SizeClass::Medium);
        self.update_fragmentation();
    }
    fn bucket_mut(&mut self, class: SizeClass) -> &mut Vec<Cached> {
        match class {
            SizeClass::Tiny => &mut self.tiny,
            SizeClass::Small => &mut self.small,
            SizeClass::Medium => &mut self.medium,
            SizeClass::Large => &mut self.large,
        }
    }
    fn bucket_ref(&self, class: SizeClass) -> &Vec<Cached> {
        match class {
            SizeClass::Tiny => &self.tiny,
            SizeClass::Small => &self.small,
            SizeClass::Medium => &self.medium,
            SizeClass::Large => &self.large,
        }
    }
    /// Buffer needing the least zero-filling for `size` among those that fit it
    /// tightly: the first with `size <= len`, else the one with the longest `len`
    /// (the bucket is sorted by capacity).
    fn no_zeroing_fit_index(bucket: &[Cached], size: usize) -> Option<usize> {
        let start = bucket.partition_point(|b| b.buf.capacity() < size);
        let cands = bucket[start..].iter().take_while(|b| fits_tightly(b.buf.capacity(), size));
        let mut best: Option<(usize, usize)> = None;
        for (i, b) in cands.enumerate() {
            if b.buf.len() >= size {
                return Some(start + i);
            }
            if best.map_or(true, |(_, len)| b.buf.len() > len) {
                best = Some((i, b.buf.len()));
            }
        }
        best.map(|(i, _)| start + i)
    }
    /// Index of the smallest buffer with capacity >= `size` in a bucket, if it
    /// fits `size` tightly.
    fn best_fit_index(bucket: &[Cached], size: usize) -> Option<usize> {
        let pos = bucket.partition_point(|b| b.buf.capacity() < size);
        (pos < bucket.len() && fits_tightly(bucket[pos].buf.capacity(), size)).then_some(pos)
    }
    /// Search buckets larger than `class` for a buffer that fits `size` tightly.
    fn find_in_larger_buckets(&self, class: SizeClass, size: usize) -> Option<(SizeClass, usize)> {
        let larger_classes: &[SizeClass] = match class {
            SizeClass::Tiny => &[SizeClass::Small, SizeClass::Medium, SizeClass::Large],
            SizeClass::Small => &[SizeClass::Medium, SizeClass::Large],
            SizeClass::Medium => &[SizeClass::Large],
            SizeClass::Large => &[],
        };
        for &lc in larger_classes {
            let bucket = self.bucket_ref(lc);
            if let Some(idx) = Self::best_fit_index(bucket, size) {
                return Some((lc, idx));
            }
        }
        None
    }
    /// Compact a single bucket by removing buffers whose capacity exceeds
    /// 2× the class maximum.
    fn compact_bucket(&mut self, class: SizeClass) {
        let threshold = class.max_elements().saturating_mul(2);
        let bucket = self.bucket_mut(class);
        let mut freed_bytes: usize = 0;
        bucket.retain(|b| {
            if b.buf.capacity() > threshold {
                freed_bytes += b.buf.capacity() * std::mem::size_of::<f32>();
                false
            } else {
                true
            }
        });
        self.stats.current_bytes = self.stats.current_bytes.saturating_sub(freed_bytes);
    }
    /// Recompute fragmentation ratio as (wasted elements) / (total cached elements).
    /// Wasted = sum of (capacity − len) for all cached buffers.
    fn update_fragmentation(&mut self) {
        let mut total_capacity: usize = 0;
        let mut total_wasted: usize = 0;
        for (class, bucket) in [
            (SizeClass::Tiny, &self.tiny),
            (SizeClass::Small, &self.small),
            (SizeClass::Medium, &self.medium),
            (SizeClass::Large, &self.large),
        ] {
            let class_min = match class {
                SizeClass::Tiny => 1,
                SizeClass::Small => 128,
                SizeClass::Medium => 1024,
                SizeClass::Large => 16384,
            };
            for b in bucket {
                total_capacity += b.buf.capacity();
                total_wasted += b.buf.capacity().saturating_sub(class_min);
            }
        }
        if total_capacity == 0 {
            self.stats.fragmentation_ratio = 0.0;
        } else {
            self.stats.fragmentation_ratio = total_wasted as f32 / total_capacity as f32;
        }
    }
}
/// Size class categories for the bucketing allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeClass {
    /// < 128 elements
    Tiny,
    /// < 1024 elements
    Small,
    /// < 16384 elements
    Medium,
    /// ≥ 16384 elements
    Large,
}
impl SizeClass {
    /// Return the exclusive upper bound for this size class (elements).
    /// For `Large`, returns `usize::MAX` since there is no upper bound.
    pub fn max_elements(self) -> usize {
        match self {
            SizeClass::Tiny => 128,
            SizeClass::Small => 1024,
            SizeClass::Medium => 16384,
            SizeClass::Large => usize::MAX,
        }
    }
}
/// Memory plan with buffer slot assignments for intermediate tensors.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Lifetime intervals for each intermediate tensor.
    pub lifetimes: Vec<TensorLifetime>,
    /// Mapping from tensor name to buffer slot index.
    pub buffer_assignments: HashMap<String, usize>,
    /// Size (in f32 elements) of each buffer slot.
    pub buffer_sizes: Vec<usize>,
    /// Peak concurrent memory in f32 elements across all execution steps.
    pub peak_memory_elements: usize,
}
impl MemoryPlan {
    /// Compute a memory plan from a topologically sorted node list.
    ///
    /// `sorted_nodes`: nodes in execution order.
    /// `output_names`: graph output tensor names (never freed early).
    /// `shape_map`: tensor name -> shape dimensions (from shape inference).
    pub fn compute(
        sorted_nodes: &[Node],
        output_names: &[String],
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Self {
        let mut produced: HashMap<String, usize> = HashMap::new();
        let mut last_consumed: HashMap<String, usize> = HashMap::new();
        for (i, node) in sorted_nodes.iter().enumerate() {
            for out_name in &node.outputs {
                if !out_name.is_empty() {
                    produced.entry(out_name.clone()).or_insert(i);
                }
            }
            for inp_name in &node.inputs {
                if !inp_name.is_empty() {
                    last_consumed.insert(inp_name.clone(), i);
                }
            }
        }
        let final_step = sorted_nodes.len();
        for name in output_names {
            last_consumed.insert(name.clone(), final_step);
        }
        let mut lifetimes: Vec<TensorLifetime> = Vec::new();
        for (name, &prod) in &produced {
            let consumed = last_consumed.get(name).copied().unwrap_or(prod);
            let size_elements = shape_map
                .get(name)
                .map(|dims| {
                    if dims.is_empty() {
                        1
                    } else {
                        dims.iter().product()
                    }
                })
                .unwrap_or(0);
            lifetimes.push(TensorLifetime {
                name: name.clone(),
                produced_at: prod,
                last_consumed_at: consumed,
                size_elements,
            });
        }
        lifetimes.sort_by_key(|lt| lt.produced_at);
        let mut buffer_assignments: HashMap<String, usize> = HashMap::new();
        let mut buffer_sizes: Vec<usize> = Vec::new();
        let mut slot_free_after: Vec<usize> = Vec::new();
        for lt in &lifetimes {
            if lt.size_elements == 0 {
                let slot = buffer_sizes.len();
                buffer_sizes.push(0);
                slot_free_after.push(lt.last_consumed_at);
                buffer_assignments.insert(lt.name.clone(), slot);
                continue;
            }
            let mut best_slot: Option<usize> = None;
            let mut best_size: usize = usize::MAX;
            for (slot_idx, &free_after) in slot_free_after.iter().enumerate() {
                if free_after < lt.produced_at
                    && buffer_sizes[slot_idx] >= lt.size_elements
                    && buffer_sizes[slot_idx] < best_size
                {
                    best_size = buffer_sizes[slot_idx];
                    best_slot = Some(slot_idx);
                }
            }
            if best_slot.is_none() {
                let mut smallest_available: Option<(usize, usize)> = None;
                for (slot_idx, &free_after) in slot_free_after.iter().enumerate() {
                    if free_after < lt.produced_at {
                        let sz = buffer_sizes[slot_idx];
                        if smallest_available.is_none()
                            || sz < smallest_available.map(|(_, s)| s).unwrap_or(usize::MAX)
                        {
                            smallest_available = Some((slot_idx, sz));
                        }
                    }
                }
                if let Some((slot_idx, _)) = smallest_available {
                    best_slot = Some(slot_idx);
                }
            }
            match best_slot {
                Some(slot_idx) => {
                    if buffer_sizes[slot_idx] < lt.size_elements {
                        buffer_sizes[slot_idx] = lt.size_elements;
                    }
                    slot_free_after[slot_idx] = lt.last_consumed_at;
                    buffer_assignments.insert(lt.name.clone(), slot_idx);
                }
                None => {
                    let slot = buffer_sizes.len();
                    buffer_sizes.push(lt.size_elements);
                    slot_free_after.push(lt.last_consumed_at);
                    buffer_assignments.insert(lt.name.clone(), slot);
                }
            }
        }
        let peak_memory_elements = compute_peak_memory(&lifetimes, final_step);
        Self {
            lifetimes,
            buffer_assignments,
            buffer_sizes,
            peak_memory_elements,
        }
    }
}
/// Lifetime interval for an intermediate tensor.
#[derive(Debug, Clone)]
pub struct TensorLifetime {
    /// Tensor name in the graph.
    pub name: String,
    /// Node execution index where the tensor is first produced.
    pub produced_at: usize,
    /// Node execution index where the tensor is last consumed.
    pub last_consumed_at: usize,
    /// Number of f32 elements (0 if shape is unknown).
    pub size_elements: usize,
}
/// Per-pool allocation and reuse statistics.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of allocations (new buffers created).
    pub alloc_count: u64,
    /// Number of times a buffer was reused from the pool.
    pub reuse_count: u64,
    /// Total bytes of the buffers counted in `alloc_count`.
    pub alloc_bytes: u64,
    /// Peak total bytes held by the pool (cached + in-flight).
    pub peak_bytes: usize,
    /// Current bytes held in the pool's free lists.
    pub current_bytes: usize,
    /// Fragmentation ratio: wasted bytes / total cached bytes. 0.0 = perfect.
    pub fragmentation_ratio: f32,
}
/// Buffer pool for reusing tensor allocations during inference.
///
/// Maintains a sorted list of available buffers and returns the smallest
/// buffer that satisfies the requested size.
pub struct BufferPool {
    /// Available buffers sorted by capacity (ascending).
    buffers: Vec<Vec<f32>>,
}
impl BufferPool {
    /// Create a new empty buffer pool.
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }
    /// Get a buffer with at least `min_size` f32 elements.
    ///
    /// Returns a recycled buffer if one of sufficient size is available,
    /// otherwise allocates a new one. The returned buffer is zeroed and
    /// has exactly `min_size` elements.
    pub fn get_buffer(&mut self, min_size: usize) -> Vec<f32> {
        let pos = self
            .buffers
            .partition_point(|buf| buf.capacity() < min_size);
        if pos < self.buffers.len() {
            let mut buf = self.buffers.remove(pos);
            buf.clear();
            buf.resize(min_size, 0.0);
            buf
        } else {
            vec![0.0; min_size]
        }
    }
    /// Return a buffer to the pool for future reuse.
    ///
    /// The pool maintains at most `MAX_POOL_BUFFERS` buffers to prevent
    /// unbounded memory growth.
    pub fn return_buffer(&mut self, buf: Vec<f32>) {
        if self.buffers.len() >= MAX_POOL_BUFFERS {
            if let Some(smallest_cap) = self.buffers.first().map(|b| b.capacity()) {
                if buf.capacity() > smallest_cap {
                    self.buffers.remove(0);
                } else {
                    return;
                }
            }
        }
        let cap = buf.capacity();
        let pos = self.buffers.partition_point(|b| b.capacity() < cap);
        self.buffers.insert(pos, buf);
    }
    /// Clear all pooled buffers, releasing their memory.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
    /// Number of buffers currently available in the pool.
    pub fn available_count(&self) -> usize {
        self.buffers.len()
    }
}
