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

### [x] P1.5 LayerNormalization kernel ([#16](https://github.com/tarekalmowafy/oxionnx/pull/16))
- **Problem:** `LayerNormalization` costs 55 ms on granite-english-r2 (ModernBERT) and 30 ms on arctic per 8-doc batch, against 20 ms on ORT, which fuses it with the residual add (`SkipLayerNormalization`).
- **Plan:** a vectorised row kernel that computes mean and variance in one pass and runs in parallel over rows, with an output-slot path. Fuse `Add → LayerNormalization` into `SkipLayerNormalization`; the native op already exists for the simplified variant.
- **Files:** `oxionnx-ops/src/nn/` (layer norm), `src/optimizer/fusion/`.
- **Result:** `oxionnx-ops/src/norm_rows.rs` is a shared row kernel: memory is read once, the variance is a corrected two-pass over the L1-resident row, rows run in parallel, and the loops are AVX-512/AVX2 multiversioned. `Add → LayerNormalization` is fused into `SkipLayerNormalization`. Norm + residual `Add`, doc-b8 at 8T, in ms: granite-en-r2 60.2 → 19.7 (ORT 30.8); arctic 44.4 → 11.7 (ORT 27.4); granite-311m 77.9 → 23.7; granite-97m 23.9 → 7.7; granite-small 15.6 → 5.7. doc-b8 wall time: granite-en-r2 −11%, granite-311m −7%, granite-small −7%. Cosine vs ORT 1.0000000.

---

## Phase 2: fusion coverage for common exporters

The MatMul, QKV and attention fusions were written against EmbeddingGemma-2's graph. BERT, ModernBERT, GTE and Qwen3-style exports from onnx-community / Optimum mostly miss them. One symptom: granite-97m-multilingual queries go only from 20.4 to 17.4 ms with 8× the threads.

### [x] P2.1 Projection fusions for HF exports ([#21](https://github.com/tarekalmowafy/oxionnx/pull/21))
- **Plan:** extend `OxiMatMulSplit` (QKV projection, split, and optional rotary or head norm) and the MatMul + bias epilogue to the projection patterns of BERT, ModernBERT, GTE and Qwen3 exports. These include separate Q/K/V MatMuls sharing an input, to be packed into one GEMM, plus `Add` bias, Reshape/Transpose to heads, and `com.microsoft.RotaryEmbedding`.
- **Files:** `src/optimizer/fusion/matmul_split.rs`, `src/optimizer/fusion/matmul_act.rs`, `oxionnx-ops/src/fused_matmul.rs`.
- **What it took**, by measured payoff (doc-b8 profiles on master):
  - The written-out rotary (`Slice`/`Split`, `Neg`, `Concat`, 2× `Mul`, `Add`) of granite-en/small/97m/311m and arctic is now one `OxiRotateHalf`. It is bit-identical, with cos/sin broadcast in either head layout.
  - Eager attention (arctic, the granite-ml torch exports) now fuses into `OxiAttention`. Q/K/V sit behind head transposes or are already head-major, scalar scales apply to Q and Kᵀ, and the output reshape is computed at run time. `OxiAttention` takes 4-D Q/K/V in either layout.
  - Separate Q/K/V MatMuls of one input are packed into one `MatMul → Split` (`pack_shared_projections`), which `OxiMatMulSplit` then absorbs together with each output's own head norm (Qwen3 q_norm/k_norm) and rotary.
  - Both need the exports' shape arithmetic resolved under a symbolic batch/sequence: `Shape(q)[3]/2` slice bounds, `1/sqrt(Shape(q)[-1])` scales. `shape_values` propagates partially known shapes and folds those. Two cleanups also landed: `Constant` nodes now become weights before fusion, and the node list is stably topologically sorted (and DCE no longer assumes order).
- **Result** (ms, oxionnx FP32, master → PR, ORT in brackets; cosine vs ORT 1.0000000 on all 11 zoo models and EmbeddingGemma-2, queries and doc-b8):
  - 8 docs, 8 threads:
    - granite-97m: 135 → 97 (231)
    - granite-311m: 439 → 334 (524)
    - arctic: 326 → 297 (353)
    - granite-en: 363 → 338 (475)
    - granite-small: 112 → 101 (168)
    - pplx: 1143 → 1114
    - qwen3: 1307 → 1281
    - gemma, voyage, jina: ±1%
  - BF16, 8 docs: granite-311m 328 → 229, arctic 218 → 182.
  - Query, 8 threads: −7..−16% on 10 models (arctic 16.2 → 13.6, ORT 15.6, so arctic now beats ORT here too; pplx 33.5 → 30.5; qwen3 35.3 → 32.7). granite-en reads +6% in the sweep but −1% in an interleaved A/B.
  - Query, 1 thread: −2..−6% on the granite models; harrier-0.6b/pplx/jina read +2..6% in the sweep but +0.5/+1.5% in an interleaved A/B (GEMM-bound, at noise level).
  - Per-op, doc-b8: granite-311m Split 41 + Transpose 38 + Softmax 19 + rotary ops ~35 → `OxiAttention` 40 + `OxiRotateHalf` 13; Qwen3-family `RotaryEmbedding` (22–27) and head-norm SLNs (11–17) go to 0.
- **Not done / open:** granite-ml still materialises its 5-D QKV `Transpose` + `Split` (9 + 7 ms on 311m). The rotary of granite-en/ml/arctic could move into the `OxiMatMulSplit` epilogue (cos/sin are run-time tensors there). The packed Q/K/V GEMM with norm + rotary epilogue costs about what the separate ops did on qwen3/jina doc batches (net −2%/+1%).

### [x] P2.2 Run contrib attention ops on the packed-GEMM attention kernel ([#21](https://github.com/tarekalmowafy/oxionnx/pull/21))
- **Plan:** run `com.microsoft.MultiHeadAttention` and `com.microsoft.GroupQueryAttention` (added in #11) on the packed AVX-512/AVX2 GEMM that `OxiAttention` uses for QKᵀ and PV, with the same blocking and threading.
- **Current cost:** MHA/GQA time is 140–144 ms against ORT's 199–221 ms on the 0.6B models, so this item is mostly about thread scaling and the small models.
- **Files:** `oxionnx-ops/src/registry/contrib_attention.rs`, `oxionnx-ops/src/contrib.rs`.
- **Result:** no kernel change was needed. Since #11/#14, both contrib ops already call `contrib::attention_ext_into`, the kernel `OxiAttention` uses: Kᵀ/V are packed once per (batch, KV head) for the packed GEMM, with the same row blocking and rayon split. The `kv_bnsh` flag from phase 1 still works, and this PR adds `q_bnsh`. Attention op time, doc-b8 at 8 threads, oxionnx vs ORT:
  - qwen3 98 vs 249
  - pplx 104 vs 188
  - harrier-0.6b 111 vs 225
  - granite-en 48 vs 154
  - voyage 41 vs 84
  - jina 24 vs 71
  - granite-small 18 vs 63
  - harrier-270m 62 vs 55 (the only one behind)

  Query at 8 threads: 0.4–1.5 ms against ORT's 0.4–3.0. Not worth more work now.

### [x] P2.3 `SkipLayerNormalization` / `SkipSimplifiedLayerNormalization` for all exporters ([#16](https://github.com/tarekalmowafy/oxionnx/pull/16))
- **Plan:** match the residual `Add` + norm pairs these exports use, and keep the native ops on the output-slot path.
- **Result:** `fuse_skip_norm` covers four cases:
  - ModernBERT/granite: `Add → LN`, with the sum reused, and also with beta from a `Constant` node.
  - arctic: decomposed LN after `Add(x, MatMul + bias)`, with the bias folded in.
  - Gemma 3 / harrier-270m: `Add → SimplifiedLayerNormalization`.

  The Qwen3/pplx/voyage/jina/harrier-0.6b exports already use `SkipSimplifiedLayerNormalization`. Their other SLNs are per-head norms, so there was nothing to fuse there. None of the zoo exports writes out RMSNorm as Pow/ReduceMean/…; the lone Pow/Sqrt/Div is the final L2 normalise. On those models oxionnx norm time was already 3–4× below ORT's: 25–30 ms against 96–110 ms on the 0.6B models. Norm time on them is unchanged within about 2 ms.

---

## Phase 3: threading overhead on short inputs

### [~] P3.1 Short queries at high thread counts ([#22](https://github.com/tarekalmowafy/oxionnx/pull/22))
- **Problem:** on EmbeddingGemma-2, 16-thread queries take 13.8 ms against llama.cpp Q4_0's 13.0. On the other models, llama-server wins most 8-thread query cells. oxionnx workers spin (`RAYON_SPIN_ROUNDS`), so a 16-thread query costs 220–273 ms of CPU, 1.3–1.6× ORT's 169 ms.
- **Plan:**
  - Adaptive spin-then-park.
  - Parallelise the remaining serial element-wise and data-movement ops (many are covered by phase 1).
  - Measure and reduce per-node dispatch overhead in the sequential executor.
  - Partition small GEMMs better at small M.
- **Metric:** query latency and `cpu_cores_busy` / `cpu_ms_per_call` from `embbench`, at 8 and 16 threads.
- **Result** (median of 3 interleaved reps on ci-1, `RAYON_SPIN_ROUNDS=1000`, wall / CPU ms per call, before → after):
  - arctic-m-v2 queries: 1T 45.7 → 43.0; 8T 16.3 / 125 → **14.1 / 113** (ORT 15.6); 16T 15.2 / 241 → **12.2 / 195**. doc-b8 8T 328 → 317.
  - EmbeddingGemma-2 queries: FP32 16T 17.4 / 272 → 15.7 / 251, 8T 18.0 → 17.4; BF16 16T 13.7 / 215 → **12.9 / 207** (llama.cpp Q4_0 13.0), 8T 13.6 → 13.3.
  - granite-97m queries: 1T 10.3 → 9.1; 8T 6.1 → 5.3; 16T 5.95 / 95 → 5.5 / 87. qwen3, harrier, granite-small: within noise (±3%, harrier 16T +7% in one sweep, +2% in another).
  - What did it: (4) small-M GEMM partitioning, which searches row × column tiles for M < 192 instead of splitting N only; and `OxiMatMulSplit` blocks capped at 256 columns (a [768,768,768] QKV ran as 3 tasks). Arctic QKV at 16T went from 4.1 to 1.6 ms per query. (2) `Transpose` 2-D tiles are staged through a stack buffer, 8x faster ([48,768]→[768,48]: 108 → 13.5 µs), worth 1.1 ms per arctic query at every thread count. Unary element-wise ops (Neg, Sqrt, …) now run in parallel: arctic doc-b8 Neg went from 7.2 to 1.3 ms at 8T.
  - (3) Dispatch overhead outside the kernels is about 1.2 ms per query (shape plan 0.25, learned-shape store 0.1–0.2, per-node dispatch 0.4–0.6, release 0.3), measured with `OXIONNX_TRACE_TIMERS`. Shape plans are now shared via `Arc` instead of being deep-copied 3–5 times on a plan-cache miss. Every benchmark query has a new length, so each one misses the 4-entry cache. Untraced runs no longer read the clock per node.
  - (1) Spin: not changed in the fork, because the spin knob lives in the harness's patched rayon-core. With 300 or 100 rounds instead of 1000, CPU drops 1–11% with no consistent wall change. An adaptive time-budget spin-then-park (onnx_spike `scripts/perf/threads/rayon-adaptive-spin.patch`, `RAYON_SPIN_MODE=adaptive`) was likewise within noise: −3..−11% CPU at a 300 µs cap, mixed wall. Most of the remaining CPU is spinning in gaps shorter than a futex wake-up on this VM (sched_yield ≈ 0.6 µs, 1000 rounds ≈ 1 ms). Parking cannot remove it without costing latency; the remaining lever is fewer participating threads per small op.
  - Cosine vs ORT: FP32 1.0000000 on all five HF models (queries and doc-b8); BF16 ≥ 0.99997. EmbeddingGemma FP32 is bit-identical to master.

---

## Phase 4: memory

### [x] P4.1 Activation memory ([#23](https://github.com/tarekalmowafy/oxionnx/pull/23))
- **Problem:** peak RSS on the newer models is 1.1–2.5× ORT's and 2–4× llama-server's. For example, Qwen3 peaks at 4533 MiB on oxionnx FP32, 2861 MiB on ORT and 2153 MiB on llama-server.
- **Plan:**
  - Stop materialising `repeat_kv` and similar broadcast copies (P1.2).
  - Do more element-wise work in place.
  - Review activation-pool sizing.
  - Reuse slots across unfused chains.
- **Finding** (`OXIONNX_TRACE_MEM=1`, doc-b8): the peak was the activation pool's idle free lists, not live activations. On granite-97m the live set peaks at 55 MiB, but the pool cached 650 MiB. Kernels allocate their own outputs on the first run, and every buffer went into the pool, which never shrank. Two leaks made it worse. A 24 MiB buffer could be handed to a 32 KiB tensor (rotary cos/sin) and held for the whole run, so on granite-small the live set grew run after run (45 → 446 MiB). Outputs that nothing reads (GroupQueryAttention's `present_k/v` on harrier) stayed live until the run ended. In-place element-wise work and slot reuse across chains are not the bottleneck once the pool is bounded, so they are not done here.
- **Done:** `SizeClassPool` has a byte budget on its free lists and evicts the least-recently released buffer first. The session sets the budget to the peak of live non-output bytes, with a 64 MiB floor. It grows when a miss is one that an eviction from this run or the previous one would have served, capped at 3× the base. Recycled buffers must fit tightly (≤ 2× the request, or ≤ 64 KiB over). A budgeted pool allocates a fresh large buffer rather than zero-fill the tail of a shorter one. Unread outputs are freed right after their node.
- **Result** (FP32, max peak RSS over the light-sweep cells, master → PR, ORT): voyage 2318 → 1832 (1432); granite-small 1051 → 428 (441); granite-en 1292 → 923 (865); granite-97m 1376 → 815 (744); granite-311m 2344 → 1881 (2087); harrier-270m 2293 → 2066 (954); harrier-0.6b 3970 → 3202 (3141); pplx 3411 → 3105 (2370); jina 1452 → 1120 (845); Qwen3 3930 → 3367 (2861); arctic 2503 → 1801 (2215). Speed is unchanged or better: doc-b8 8T Qwen3 1348 → 1206 ms, harrier-0.6b 1230 → 1135, harrier-270m 335 → 308; the other cells are within ±3% noise. Cosine vs ORT is 1.0000000. On most models the peak is now the weights after load, or the load transient itself (granite-97m/311m, arctic). That is P4.2.

### [x] P4.2 Large `Gather` tables ([#18](https://github.com/tarekalmowafy/oxionnx/pull/18))
- **Problem:** embedding tables such as EmbeddingGemma's 537 MB token embeddings are kept fully resident as FP32. ORT and llama.cpp memory-map theirs and only touch the rows a batch uses.
- **Plan:** keep `Gather`-only initializers in the mapped file, or as BF16, and decode rows on demand.
- **Done:** tables of at least 4 MiB that only `Gather` (axis 0) or `GatherBlockQuantized` read stay in the mmap'd `.onnx` / external-data file (MADV_RANDOM), and the ops read only the rows they need. Tied tables, graph outputs and `with_keep_weights` are decoded as before. The BF16 option turned out to be unnecessary, because mapping also covers BF16 mode and inline models.
- **Result:** output is bit-identical (all 32 queries and doc batches, FP32 and BF16, 11 zoo models plus gemma FP32/Q8/Q4). RSS after load, base → new: gemma FP32 1247 → 748 MiB, BF16 1173 → 441, Q8 560 → 377, Q4 424 → 359. Qwen3-0.6B 3055 → 2459, harrier-270m 2009 → 751, arctic-m-v2 1598 → 867, granite-311m-ml 1381 → 623. Peak RSS falls 200–1100 MiB on the large-vocabulary models. Speed is unchanged within noise.

---

## Phase 5: precision and quantisation

### [~] P5.1 BF16 weights accuracy on jina-embeddings-v5-text-nano ([#17](https://github.com/tarekalmowafy/oxionnx/pull/17))
- **Problem:** with `OXIONNX_BF16_WEIGHTS=1`, the minimum cosine against FP32 is 0.9974 (all other models ≥ 0.9999).
- **Plan:** find the sensitive projections by switching BF16 per layer, then keep those in FP32 by default, or expose a per-op opt-out.
- **Finding:** the plan's premise was wrong. The weights are not the cause: jina's (like most zoo checkpoints') are stored in BF16, so packing them as BF16 is exact (FP32 GEMM on BF16-rounded weights: cos 1.000000). The error comes from rounding the *activations* to BF16 in the kernel. A few inputs amplify that ~240x across the layers (doc batches: min 0.973, 2 of 32 docs), and no projection type or layer subset is responsible: keeping any one type, or the first/last half of the layers, in FP32 still leaves 0.85-0.9998, non-monotonically. A per-op rule cannot fix it.
- **Result:** opt-in `with_bf16_split_activations` / `OXIONNX_BF16_WEIGHTS=split` splits each activation into two BF16 terms (`hi + lo`, 16 significant bits): cos ≥ 0.99994 on all 11 zoo models, 1.000000 on the 9 with BF16 checkpoints (jina docs 0.973 → 1.000000). It costs the whole BF16 compute gain: 1.4-2.3x slower than one-term BF16, 0.90-1.30x FP32 time (jina 1T query 17.0 → 31.9 ms, FP32 32.2; doc-b8 8T 178.5 → 312.2 ms, FP32 289.9), and it keeps the BF16 memory saving. The default stays one-term BF16 (unchanged speed; ≥ 0.9999 except jina docs 0.973 and harrier-0.6b docs 0.99979). Open: which default the user wants.

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
