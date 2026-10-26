# CPU performance plan: closing the gap to ONNX Runtime and llama.cpp

This is the tracked list of open CPU performance work. Each item has an ID. PRs reference the ID in their title or body, e.g. `P1.1: parallel SiLU + SwiGLU epilogue`, and the PR that finishes an item ticks its box, links itself, and records the measured result under the item.

Status: `[ ]` open · `[~]` in progress (PR linked) · `[x]` done (result recorded)

## Where we are

All numbers are from `nileflow-ci-1`, an AMD EPYC (Zen-class) VM with AVX-512 (`bf16`, `vnni`), pinned to cores 0–31. Times are ms per call: one query (batch 1) or one batch of 8 documents (max 256 tokens). oxionnx runs as a static musl + mimalloc build. ORT runs the same FP32 ONNX file. llama-server runs the F16/BF16 GGUF.

**EmbeddingGemma-2** (the graph the earlier work was tuned on). oxionnx leads everywhere except short queries at 16 threads:

| | query, 1 thread | query, 16 threads | 8 docs/s, 16 threads |
|---|---:|---:|---:|
| ORT FP32 | 51.0 | 36.3 | 19.1 |
| oxionnx BF16 weights | **25.6** | 13.8 | **44.2** |
| oxionnx Q4 (MatMulNBits) | **20.8** | 14.5 | **44.6** |
| llama.cpp Q4_0 | 43.0 | **13.0** | 22.6 |

**Other embedding models**, after #11 made them run correctly (cosine 1.0000 vs ORT):

| Model | ORT 8 docs, 8 thr | oxionnx 8 docs, 8 thr | oxionnx BF16 query, 1 thr | ORT query, 1 thr |
|---|---:|---:|---:|---:|
| granite-embedding-97m-multilingual-r2 | **213** | 985 | 16.8 | **11.3** |
| snowflake-arctic-embed-m-v2.0 | **352** | 1018 | **35.6** | 45.9 |
| granite-embedding-english-r2 | 467 | **429** (BF16) | **26.6** | 42.7 |
| Qwen3-Embedding-0.6B | **1411** | 4670 | **111.1** | 143.8 |
| harrier-oss-v1-0.6b | **1368** | 2702 | **82.8** | 134.5 |

GEMM time is already on par with ORT:

| Model | oxionnx MatMul (ms) | ORT MatMul (ms) |
|---|---:|---:|
| harrier-0.6b | 1025 | 979 |
| Qwen3 | 1057 | 964 |
| granite-97m | 86 | 86 |

The gap is a handful of generic, non-GEMM ops. These are per-op profiles for 8 docs at 8 threads, in ms per run:

| Model | oxionnx ops ORT does not spend time on |
|---|---|
| granite-97m-multilingual (XLM-R) | Slice 517, SiLU 305 |
| arctic-m-v2 (GTE) | Erf 391, Slice 236 |
| granite-english-r2 (ModernBERT) | Tile 74, LayerNormalization 55 (ORT 20) |
| Qwen3-Embedding-0.6B | Expand 1875, SiLU 1378, Tile 102 |
| harrier-oss-v1-0.6b | SiLU 1365 |

## How each item is validated

1. **Profile before and after** with `embbench --profile N`. The per-op times go in the PR.
2. **Unit tests** against a straightforward scalar reference, run in **debug** (`cargo +nightly test -p oxionnx -p oxionnx-ops -p oxionnx-proto`), because release builds hide debug assertions. Cover 1 and several threads, odd shapes, and padded batches.
3. **Accuracy:** embeddings keep cosine ≥ 0.9999 against ORT on every affected model, for both queries and padded doc batches. FP32 changes that only reorder arithmetic should stay within float rounding.
4. **Speed:** re-run the light model sweep (1 and 8 threads, queries and batches of 8 docs) and record the before/after table in the PR.

**Finish line:** on every benchmarked model, at 1 and 8 threads, for queries and doc batches, oxionnx FP32 is at least as fast as ORT FP32, and oxionnx BF16 is at least as fast as llama-server.

---

## Phase 1: generic kernels

These items mostly touch independent files and can proceed in parallel.

### [x] P1.1 SiLU kernel and SwiGLU epilogue fusion
- **Problem:** `SiLU` costs 1365 ms (harrier-0.6b), 1378 ms (Qwen3) and 305 ms (granite-97m) per batch of 8 docs; ORT's fused equivalent costs about 10 ms. Our kernel (`nn::activations::silu`, `simd_ops::*::silu`) is single-threaded, copies its input, and uses a slow sigmoid.
- **Plan:**
  1. Write a parallel, in-place SiLU with the FMA `exp` used by the GELU kernels (`oxionnx-ops/src/gelu.rs`), and give it an output-slot path.
  2. Fuse the SwiGLU MLP, `Mul(SiLU(x·W_gate), x·W_up)`, where both projections are MatMuls of the same `x` by constant weights, into one gated `OxiMatMulAct` with `activation = silu`. Reuse the GeGLU column interleave (`fused_matmul::gate_columns`) and add `Act::Silu` to the epilogue.
  3. Also cover the bias, BF16 and MatMulNBits weight forms.
- **Affects:** harrier, Qwen3-Embedding, pplx-embed, voyage-4-nano and the granite multilingual models; most current decoder embedders.
- **Expected:** about −1.3 s per 8-doc batch on the 0.6B models; about 25–45% off their query time.
- **Files:** `oxionnx-ops/src/gelu.rs` (or a new `act.rs`), `oxionnx-ops/src/fused_matmul.rs`, `src/optimizer/fusion/matmul_act.rs`, `oxionnx-ops/src/registry/nn_ops/activations.rs`.
- **Done in PR [#15](https://github.com/tarekalmowafy/oxionnx/pull/15).** The exports write SiLU as `Sigmoid` + `Mul` (fused to `SiLU` now *before* the MatMul fusion). Decoders (harrier-0.6b, Qwen3, pplx, voyage, jina) use two MatMuls, `Mul(SiLU(x·Wg), x·Wu)`: fused into one gated `OxiMatMulAct` on the concatenated `[Wg | Wu]` (originals dropped). granite-97m/311m use one `Wi` MatMul cut by two `Slice`s whose bounds are computed from `Shape` (`(N+1)/2`): matched too (bounds evaluated with the known N), so those Slices go as well. BF16 and MatMulNBits weights take the same epilogue (NBits: plain form with the up product as multiplier). Standalone `SiLU` / `Erf` / exact `Gelu`: parallel, in place, output-slot path.
- **Result** (8 docs, 8 threads, ms per batch, FP32; profile): SiLU 1367 → 0 (harrier-0.6b), 1381 → 0 (Qwen3), 308 → 0 (granite-97m); MatMul + OxiMatMulAct = 993 vs 1022 MatMul before. Sweep, base → new: harrier-0.6b docs 2754 → 1250 (ORT 1371), query 8T 56.4 → 35.1 (ORT 43.7); Qwen3 docs 4815 → 3267; pplx 4497 → 2973; voyage 1880 → 1314; jina 666 → 333; granite-97m docs 981 → 289 (ORT 234), query 1T 20.1 → 10.9 (ORT 11.9). BF16: harrier-0.6b docs 2256 → 819. Query 1T gains are smaller than expected (−6..−9% on the 0.6B models, not 25–45%): at 1 thread SiLU was never the big share. Cosine vs ORT 1.0000000 on every model (FP32).

### [x] P1.2 `repeat_kv` without materialising K/V copies (Expand, Tile)
- **Problem:** `Expand` costs 1875 ms per 8-doc batch on Qwen3, against 47 ms on ORT. Most of it is the `repeat_kv` pattern (`Unsqueeze → Expand → Reshape`) that copies each K/V head to its query-head group before `com.microsoft.MultiHeadAttention`. voyage-4-nano and pplx-embed have the same pattern. `Tile` costs 74–102 ms on ModernBERT and Qwen3. Both ops copy element by element.
- **Plan:**
  1. Give Expand and Tile a parallel broadcast fast path that copies contiguous runs with `copy_from_slice`.
  2. Add an optimizer pass that recognises `repeat_kv` feeding `com.microsoft.MultiHeadAttention` and passes the un-repeated K/V with a `kv_num_heads` attribute, as GroupQueryAttention does, so no copy is made.
- **Expected:** about −1.8 s on Qwen3 doc batches, and similar savings on voyage-4-nano and pplx-embed. Lower peak RSS too (see P4.1).
- **Files:** `oxionnx-ops/src/registry/shape_ops/{concat_slice_ops,spatial_ops}.rs`, `oxionnx-ops/src/registry/contrib_attention.rs`, a new `src/optimizer/fusion/repeat_kv.rs`.
- **Done in PR [#14](https://github.com/tarekalmowafy/oxionnx/pull/14).** Expand/Tile/Slice share one strided copy (`oxionnx-ops/src/datamove.rs`). `fuse_repeat_kv` feeds MHA the un-repeated K/V with `kv_num_heads`: the 3-D projection on voyage/pplx, the 4-D `present` cache on Qwen3, which the kernel now reads in place. It also drops Expand/Tile broadcasts of the attention mask, since MHA now broadcasts `attention_bias` over any size-1 axis.
- **Result** (8 docs, 8 threads, ms per batch, base -> PR, ORT): Expand per-op 1947 -> 0 (Qwen3), 805 -> 0 (voyage), 1796 -> 0 (pplx). Tile 101 -> 0 (Qwen3), 76 -> 0 (granite-en-r2). MHA 153 -> 101 on Qwen3, because K/V are no longer repeated. Sweep: Qwen3 4735 -> 2896 (ORT 1378), voyage 1877 -> 1035 (584), pplx 4459 -> 2659 (1358), granite-en-r2 523 -> 469 (447), granite-small-en-r2 238 -> 157 (162). Qwen3 peak RSS on doc batches 4607 -> 3997 MiB. Cosine vs ORT is 1.0000000 on all 8 affected models.

### [x] P1.3 Fast Slice
- **Problem:** `Slice` costs 517 ms on granite-97m-multilingual and 236 ms on arctic-m-v2 per 8-doc batch, against 74 ms and about 0 on ORT. It is an element-wise strided copy on one thread.
- **Plan:**
  - Copy contiguous inner runs with `copy_from_slice` and parallelise the outer rows.
  - Return a cheap result when the slice is the whole tensor or a prefix of a contiguous axis.
  - Check whether these Slices come from attention-mask or position-id preprocessing that could be computed once per batch.
- **Expected:** about −500 ms (granite-97m) and −240 ms (arctic) per 8-doc batch.
- **Files:** `oxionnx-ops/src/registry/shape_ops/concat_slice_ops.rs`.
- **Done in PR [#14](https://github.com/tarekalmowafy/oxionnx/pull/14).** Slice uses the same strided copy and gains an output-slot path. These Slices do not come from mask or position preprocessing. On granite-97m/311m they are the GLU halves of the fused `Wi` projection (`MatMul -> Slice(0:F), Slice(F:2F) -> SiLU * up`). On arctic they are `rotate_half` in a written-out rotary. Neither can be hoisted. The remaining cost would go away by fusing the Slice-form GLU into `OxiMatMulAct` (P1.1) and the rotary chain into `RotaryEmbedding` (P2.1).
- **Result** (8 docs, 8 threads): Slice per-op 536 -> 9 (granite-97m, ORT 73), 978 -> 13 (granite-311m, ORT 118), 246 -> 3 (arctic). Sweep: granite-97m 983 -> 477 (ORT 233), granite-311m 1797 -> 825 (555), arctic 1063 -> 797 (352).

### [x] P1.4 Exact GELU: written-out erf pattern and vectorised erf
- **Problem:** `Erf` costs 391 ms on arctic-m-v2 per 8-doc batch; ORT fuses it into `Gelu` (4 ms). The export writes exact GELU out as `Mul(Mul(x, 0.5), Add(Erf(Div(x, √2)), 1))` in its usual operand orders, our fusions do not match it, and `Erf` is scalar.
- **Plan:**
  - Fuse the written-out pattern into `Gelu` (exact). After a MatMul [+ bias], fuse it into `OxiMatMulAct`, which already has the `GeluErf` epilogue.
  - Add a vectorised erf for any `Erf` left over.
- **Expected:** about −390 ms on arctic per 8-doc batch. Also helps other BERT-family exports.
- **Files:** `src/optimizer/fusion/` (GELU pattern), `src/optimizer/fusion/matmul_act.rs`, `oxionnx-ops/src/gelu.rs`.
- **Done in PR [#15](https://github.com/tarekalmowafy/oxionnx/pull/15).** `fuse_gelu_erf` (`src/optimizer/fusion/gelu_erf.rs`) matches all three groupings (`Mul(Mul(x,a),0.5)`, `Mul(Mul(x,0.5),a)`, `Mul(x,Mul(a,0.5))`), Div by √2 or Mul by 1/√2, constants as initializers or `Constant` nodes. Affected exports: arctic-m-v2 (`MatMul → Split` GeGLU) and granite-311m (`MatMul → 2×Slice` GeGLU); both become one gated `OxiMatMulAct`. Vectorised erf / exact GELU (erfc-based, no tail cancellation) for the epilogue and any leftover `Erf`/`Gelu`, which also speeds up the existing `Gelu(approximate=none)` epilogue (ModernBERT).
- **Result** (8 docs, 8 threads, ms per batch, FP32): Erf 387 → 0 on arctic (sweep docs 1013 → 601, ORT 350; query 8T 37.3 → 20.3); granite-311m Erf 308 → 0 and Slice 961 → 449 (docs 1747 → 956, ORT 507). Cosine vs ORT 1.0000000.

### [ ] P1.5 LayerNormalization kernel
- **Problem:** `LayerNormalization` costs 55 ms on granite-english-r2 (ModernBERT) and 30 ms on arctic per 8-doc batch, against 20 ms on ORT, which fuses it with the residual add (`SkipLayerNormalization`).
- **Plan:** a vectorised row kernel that computes mean and variance in one pass and runs in parallel over rows, with an output-slot path. Fuse `Add → LayerNormalization` into `SkipLayerNormalization`; the native op already exists for the simplified variant.
- **Files:** `oxionnx-ops/src/nn/` (layer norm), `src/optimizer/fusion/`.

---

## Phase 2: fusion coverage for common exporters

The MatMul, QKV and attention fusions were written against EmbeddingGemma-2's graph. BERT, ModernBERT, GTE and Qwen3-style exports from onnx-community / Optimum mostly miss them. One symptom: granite-97m-multilingual queries go only from 20.4 to 17.4 ms with 8× the threads.

### [ ] P2.1 Projection fusions for HF exports
- **Plan:** extend `OxiMatMulSplit` (QKV projection, split, and optional rotary or head norm) and the MatMul + bias epilogue to the projection patterns of BERT, ModernBERT, GTE and Qwen3 exports. These include separate Q/K/V MatMuls sharing an input, to be packed into one GEMM, plus `Add` bias, Reshape/Transpose to heads, and `com.microsoft.RotaryEmbedding`.
- **Files:** `src/optimizer/fusion/matmul_split.rs`, `src/optimizer/fusion/matmul_act.rs`, `oxionnx-ops/src/fused_matmul.rs`.

### [ ] P2.2 Run contrib attention ops on the packed-GEMM attention kernel
- **Plan:** run `com.microsoft.MultiHeadAttention` and `com.microsoft.GroupQueryAttention` (added in #11) on the packed AVX-512/AVX2 GEMM that `OxiAttention` uses for QKᵀ and PV, with the same blocking and threading.
- **Current cost:** MHA/GQA time is 140–144 ms against ORT's 199–221 ms on the 0.6B models, so this item is mostly about thread scaling and the small models.
- **Files:** `oxionnx-ops/src/registry/contrib_attention.rs`, `oxionnx-ops/src/contrib.rs`.

### [ ] P2.3 `SkipLayerNormalization` / `SkipSimplifiedLayerNormalization` for all exporters
- **Plan:** match the residual `Add` + norm pairs these exports use, and keep the native ops on the output-slot path.

---

## Phase 3: threading overhead on short inputs

### [ ] P3.1 Short queries at high thread counts
- **Problem:** on EmbeddingGemma-2, 16-thread queries take 13.8 ms against llama.cpp Q4_0's 13.0. On the other models, llama-server wins most 8-thread query cells. oxionnx workers spin (`RAYON_SPIN_ROUNDS`), so a 16-thread query costs 220–273 ms of CPU, 1.3–1.6× ORT's 169 ms.
- **Plan:**
  - Adaptive spin-then-park.
  - Parallelise the remaining serial element-wise and data-movement ops (many are covered by phase 1).
  - Measure and reduce per-node dispatch overhead in the sequential executor.
  - Partition small GEMMs better at small M.
- **Metric:** query latency and `cpu_cores_busy` / `cpu_ms_per_call` from `embbench`, at 8 and 16 threads.

---

## Phase 4: memory

### [ ] P4.1 Activation memory
- **Problem:** peak RSS on the newer models is 1.1–2.5× ORT's and 2–4× llama-server's. For example, Qwen3 peaks at 4533 MiB on oxionnx FP32, 2861 MiB on ORT and 2153 MiB on llama-server.
- **Plan:**
  - Stop materialising `repeat_kv` and similar broadcast copies (P1.2).
  - Do more element-wise work in place.
  - Review activation-pool sizing.
  - Reuse slots across unfused chains.

### [ ] P4.2 Large `Gather` tables
- **Problem:** embedding tables such as EmbeddingGemma's 537 MB per-layer embeddings are kept fully resident as FP32. ORT and llama.cpp memory-map theirs and only touch the rows a batch uses.
- **Plan:** keep `Gather`-only initializers in the mapped file, or as BF16, and decode rows on demand.

---

## Phase 5: precision and quantisation

### [ ] P5.1 BF16 weights accuracy on jina-embeddings-v5-text-nano
- **Problem:** with `OXIONNX_BF16_WEIGHTS=1`, the minimum cosine against FP32 is 0.9974 (all other models ≥ 0.9999).
- **Plan:** find the sensitive projections by switching BF16 per layer, then keep those in FP32 by default, or expose a per-op opt-out.

### [ ] P5.2 Quantised builds of the newer models
- **Plan:** run the published Q8/Q4 (MatMulNBits) exports of the zoo models with the int8 kernels from #6 and #8, and compare with llama.cpp Q8_0/Q4.

---

## Lower priority

### [ ] P6.1 EmbeddingGemma-2 vision encoder speed
- **Problem:** about 1.4–1.5× slower than ORT for one 266-token image (20.0 vs 14.2 s at 1 thread on an AVX2 machine).
- **Plan:** profile after phase 1, since it shares the attention and data-movement ops.

### [ ] P6.2 FP32 single-thread query placement effect
- **Problem:** after #10, FP32 1-thread queries are about 5% slower on the ci VM. The measurements were noisy (+3% to +10%). Perf counters show TLB misses are negligible, and the difference tracks local vs remote host-NUMA fills, so it is most likely physical page placement.
- **Plan:** revisit on bare metal before changing code.
