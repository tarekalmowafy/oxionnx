use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::sync::Mutex;

// ── SessionRunState ──────────────────────────────────────────────────────────

/// Intermediate tensor storage for a single inference run.
///
/// Wraps the tensor map with pool-backed buffer release on completion.
/// Replaces the bare `HashMap<String, Tensor>` used in previous versions.
/// IoBinding integration (bound_outputs wiring) is deferred to a later item.
pub(crate) struct SessionRunState {
    /// Active intermediate tensors keyed by name.
    tensors: HashMap<String, Tensor>,
    /// Prototype (onnx_spike): output shapes observed this run for nodes whose
    /// shapes static inference could not resolve (see `Session::run_internal`).
    learned_shapes: Vec<(String, Vec<usize>)>,
    /// Bytes held by the tensors in `tensors` (by capacity), not counting graph
    /// outputs: those go to the caller, so they are never pool demand.
    live_bytes: usize,
    /// Graph outputs, excluded from `live_bytes` (see [`Self::set_graph_outputs`]).
    graph_outputs: hashbrown::HashSet<String>,
    /// The pool budget as last seen by this run; the pool is locked to raise it
    /// only when `live_bytes` goes past this.
    budget_seen: usize,
    /// `OXIONNX_TRACE_MEM` diagnostic, `None` unless enabled.
    pub(crate) mem: Option<Box<super::memtrace::MemTrace>>,
}

impl SessionRunState {
    /// Create a new run state with a pre-allocated tensor map capacity.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            tensors: HashMap::with_capacity(capacity),
            learned_shapes: Vec::new(),
            live_bytes: 0,
            graph_outputs: hashbrown::HashSet::new(),
            budget_seen: 0,
            mem: None,
        }
    }

    /// Name the graph outputs, which [`Self::insert`] then leaves out of the live
    /// bytes that size the pool budget.
    pub(crate) fn set_graph_outputs<'a>(&mut self, names: impl IntoIterator<Item = &'a String>) {
        self.graph_outputs = names.into_iter().cloned().collect();
    }

    /// Bytes `t` counts towards the pool budget if stored under `name`.
    #[inline]
    fn budget_bytes(&self, name: &str, t: &Tensor) -> usize {
        if !self.graph_outputs.is_empty() && self.graph_outputs.contains(name) {
            0
        } else {
            tensor_bytes(t)
        }
    }

    /// Record an output shape observed at run time.
    pub(crate) fn learn_shape(&mut self, name: &str, shape: &[usize]) {
        self.learned_shapes.push((name.to_string(), shape.to_vec()));
    }

    /// Take the shapes recorded with [`Self::learn_shape`].
    pub(crate) fn take_learned_shapes(&mut self) -> Vec<(String, Vec<usize>)> {
        std::mem::take(&mut self.learned_shapes)
    }

    /// Look up a tensor by name (immutable).
    #[inline]
    pub(crate) fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Insert or replace a tensor. If a tensor already exists at this name,
    /// its data buffer is released to the pool.
    ///
    /// One hash lookup, not two: this was `remove` followed by `insert`, which
    /// hashed the key twice on the hottest write path in the engine (once per
    /// node output, per run).  `HashMap::insert` already returns the displaced
    /// value, and releasing it after the store rather than before is
    /// unobservable — a pool release touches only the pool.
    pub(crate) fn insert(
        &mut self,
        name: String,
        tensor: Tensor,
        pool: Option<&Mutex<SizeClassPool>>,
    ) {
        let added = self.budget_bytes(&name, &tensor);
        self.live_bytes += added;
        if let Some(old) = self.tensors.insert(name, tensor) {
            // Same name, so the same budget class as `added`.
            self.live_bytes -= if added == 0 { 0 } else { tensor_bytes(&old) };
            release_to_pool(old, pool);
        }
        if self.live_bytes > self.budget_seen {
            self.raise_pool_budget(pool);
        }
    }

    /// Raise the pool's byte budget to this run's live bytes.
    ///
    /// The free lists never need to hold more than the peak of live intermediates:
    /// at any point a run can only acquire what it will hold live at the same time.
    /// Raising the budget as the live set grows (instead of measuring the peak
    /// after the run) also bounds the very first run, in which shapes are still
    /// being learned and kernels allocate their outputs themselves.
    #[cold]
    fn raise_pool_budget(&mut self, pool: Option<&Mutex<SizeClassPool>>) {
        self.budget_seen = self.live_bytes;
        if let Some(mut guard) = pool.and_then(|p| p.lock().ok()) {
            if let Some(budget) = guard.budget() {
                if budget < self.live_bytes {
                    guard.raise_budget(self.live_bytes);
                } else {
                    self.budget_seen = budget;
                }
            }
        }
    }

    /// Remove a tensor from state, returning ownership (no pool release).
    /// Used for in-place execution where the caller takes ownership of the buffer.
    pub(crate) fn take(&mut self, name: &str) -> Option<Tensor> {
        let t = self.tensors.remove(name);
        if let Some(t) = &t {
            self.live_bytes -= self.budget_bytes(name, t);
        }
        t
    }

    /// Expose the tensor map as an immutable reference (for GPU dispatch functions
    /// that accept `&HashMap<String, Tensor>`).
    #[inline]
    pub(crate) fn as_map(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Extract the named output tensors and release all remaining intermediates
    /// back to the pool.
    ///
    /// # A declared output that was never produced is an error
    ///
    /// This used to `filter_map` the misses away, so `Session::run` returned
    /// `Ok(map)` with a graph output silently absent — the caller's
    /// `outputs.get("y")` was `None` and nothing anywhere said why.  Combined
    /// with the run loops' old "skip nodes whose op is unknown" behaviour, an
    /// entire branch of the graph could vanish without a single diagnostic.
    ///
    /// `weights` is consulted before declaring an output missing: constant
    /// folding can promote a graph output to an initializer, and a model may
    /// legitimately name an initializer as an output.  Such an output is never
    /// written by any node, yet it is perfectly well defined.
    ///
    /// # Errors
    ///
    /// [`OnnxError::TensorNotFound`], naming every missing output, when a
    /// declared output is neither in the run state nor an initializer.
    pub(crate) fn take_outputs(
        mut self,
        output_names: &[String],
        weights: &HashMap<String, Tensor>,
        pool: Option<&Mutex<SizeClassPool>>,
    ) -> Result<HashMap<String, Tensor>, OnnxError> {
        // Remove outputs first (these are returned to the caller, not pooled)
        let mut result: HashMap<String, Tensor> = HashMap::with_capacity(output_names.len());
        let mut missing: Vec<&str> = Vec::new();
        for name in output_names {
            if let Some(t) = self.tensors.remove(name) {
                result.insert(name.clone(), t);
            } else if let Some(w) = weights.get(name) {
                // A graph output that is an initializer (or was constant-folded
                // into one): no node writes it, but it is well defined.
                result.insert(name.clone(), w.clone());
            } else {
                missing.push(name.as_str());
            }
        }
        // Release all remaining intermediates back to the pool, error or not:
        // the buffers are ours to recycle either way.
        for (_name, tensor) in self.tensors.drain() {
            release_to_pool(tensor, pool);
        }
        if !missing.is_empty() {
            return Err(OnnxError::TensorNotFound(format!(
                "graph output(s) {missing:?} were never produced by any node — the model \
                 declares them as outputs but nothing in the executed graph writes them",
            )));
        }
        Ok(result)
    }
}

/// Bytes a tensor's buffer holds (by capacity, which is what is resident).
#[inline]
fn tensor_bytes(t: &Tensor) -> usize {
    t.data.capacity() * std::mem::size_of::<f32>()
}

/// Release a tensor's data buffer back into the pool, if a pool is available.
#[inline]
pub(super) fn release_to_pool(mut tensor: Tensor, pool: Option<&Mutex<SizeClassPool>>) {
    if let Some(pool_mutex) = pool {
        if let Ok(mut guard) = pool_mutex.lock() {
            let buf = std::mem::take(&mut tensor.data);
            if !buf.is_empty() {
                guard.release(buf);
            }
        }
    }
}

// ── TypedSessionRunState ─────────────────────────────────────────────────────

/// Intermediate typed-tensor storage for a single `run_typed` inference run.
///
/// Parallel to `SessionRunState` but carries `TypedTensor` intermediates so that
/// integer and half-precision dtypes are preserved per-node without an f32 round-trip.
/// Pool integration is intentionally absent: `TypedTensor` heap buffers are owned by
/// the enum variants and freed by Rust's ordinary drop machinery.
pub(super) struct TypedSessionRunState {
    pub(super) slots: HashMap<String, oxionnx_core::TypedTensor>,
}

impl TypedSessionRunState {
    pub(super) fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    #[inline]
    pub(super) fn get(&self, name: &str) -> Option<&oxionnx_core::TypedTensor> {
        self.slots.get(name)
    }

    /// The live slot map, for use as a subgraph operator's outer scope.
    #[inline]
    pub(super) fn slots(&self) -> &HashMap<String, oxionnx_core::TypedTensor> {
        &self.slots
    }

    #[inline]
    pub(super) fn insert(&mut self, name: String, tensor: oxionnx_core::TypedTensor) {
        self.slots.insert(name, tensor);
    }

    /// Remove and return the requested output tensors; intermediate slots are dropped.
    ///
    /// Mirrors [`SessionRunState::take_outputs`]: a declared output that no node
    /// produced is an error rather than a silently absent map entry, and
    /// `weights` is consulted first because initializers are not seeded into the
    /// typed run state (seeding them deep-copied every model parameter on every
    /// `run_typed` call).
    ///
    /// # Errors
    ///
    /// [`OnnxError::TensorNotFound`], naming every missing output.
    pub(super) fn take_outputs(
        &mut self,
        output_names: &[String],
        weights: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, oxionnx_core::TypedTensor>, OnnxError> {
        let mut result = HashMap::with_capacity(output_names.len());
        let mut missing: Vec<&str> = Vec::new();
        for name in output_names {
            if let Some(t) = self.slots.remove(name) {
                result.insert(name.clone(), t);
            } else if let Some(w) = weights.get(name) {
                result.insert(
                    name.clone(),
                    oxionnx_core::TypedTensor::new(
                        oxionnx_core::TensorStorage::F32(w.data.clone()),
                        w.shape.clone(),
                    ),
                );
            } else {
                missing.push(name.as_str());
            }
        }
        if !missing.is_empty() {
            return Err(OnnxError::TensorNotFound(format!(
                "graph output(s) {missing:?} were never produced by any node — the model \
                 declares them as outputs but nothing in the executed graph writes them",
            )));
        }
        Ok(result)
    }
}
