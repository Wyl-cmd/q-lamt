# Q-LAMT — Technical Design

> 中文版：[TECHNICAL.zh-CN.md](TECHNICAL.zh-CN.md)

Status: v0.1 research prototype  
Last updated: 2026-10-31

Q-LAMT is **Quantized Locate–Ablate–Materialize–Train**. The goal is a repeatable, auditable repair of a small set of weight rows in a deployed 4-bit LLM, without loading the FP16 base into VRAM and without updating the frozen 4-bit backbone.

This document describes the implementation that produced the numbers in [RESULTS.md](../RESULTS.md). The full Chinese specification is [定点局部训练-技术方案.md](../定点局部训练-技术方案.md).

## 1. Scope and non-goals

In scope:

- one failure sample (prompt, current wrong answer, target answer, paraphrases);
- locating/ablating/training on the deployed NF4 model;
- row-sparse `delta` or `splice_fp16` parameters;
- multi-round sidecars with an immutable FP16/BF16 base;
- release-time merge → quantize → verify;
- row-level audit, version rollback, regression checks.

Not in scope:

- pretraining, general-domain fine-tuning, or capability uplift;
- a failure queue, scheduler, nightly rotation, or automatic rollout;
- distributed multi-GPU training;
- a replacement for QLoRA when a large domain dataset exists;
- a claim that all facts/behaviors can be repaired.

## 2. Row definition and notation

For a `nn.Linear` block:

```text
W ∈ R^{out×in}, b ∈ R^{out}, y = W x + b
row i = W[i, :] and b[i]; it alone determines output channel y_i
```

NF4 quantizes `W` to `W_q` with block scales; the runtime dequantizes on the fly:

```text
D = deq(W_q)          # transient compute view
QLinear(x) = D x + b  # base contribution
```

`RowAddr(layer, row)` is a module name plus output-channel index. NF4 preserves the logical `out×in` shape and row numbering, so the same address is valid in the FP16 ledger and in the quantized runtime. Bitsandbytes stores packed data, so row counts must come from `out_features`, not `weight.shape`.

## 3. Runtime and row parameterization

The full NF4 model is resident and frozen. There is no full FP16 model in VRAM.

### 3.1 `delta` mode (single-call default)

```text
ΔR ∈ R^{n_rows×in}, initialized to 0
y = QLinear(x) + scatter(ΔR @ x, rows)
```

- Step 0 is bit-identical to the NF4 inference model.
- The trained row is `D[row] + ΔR[row]`.
- Merge: `W_fp16[row] += ΔR[row]`, then requantize.
- This is deployment-consistent: the loss is the deployed NF4 function.

### 3.2 `splice_fp16` mode (multi-round default)

```text
P ∈ R^{n_rows×in}, initialized to W_fp16[rows]
D_R = deq(W_q)[rows]
y = QLinear(x) + scatter((P - D_R) @ x, rows)
```

- The effective row is exactly `P`, so the named rows start from FP16 truth.
- The rest of the model remains NF4; this is a hybrid forward, not a full FP16 forward.
- Step 0 differs from the NF4 deployment by the named rows' quantization error; the failure/path may change. Verify on the hybrid function and re-ablate if needed.
- Merge: `W_fp16[row] = P[row]`, then requantize.
- No whole-layer BF16 buffer or clone is needed. Only the named rows are materialized.

### 3.3 Backward

Autograd traverses the same quantized graph. Bitsandbytes dequantizes weights on the fly for `grad_input`. The base parameters have `requires_grad=False`; only `P/ΔR` receives a gradient. The activation graph is still a full-model graph; sparse parameters do not make activation memory sparse.

Important distinction: the active/causal rows are a **training mask**, not a runnable sub-network. At every layer each row reads the full activation vector from the previous layer. Dropping the other rows changes the input to the active rows.

## 4. Stage 1 — locate

For each candidate Linear module in the layer pool (`mlp.gate/up/down_proj`, `self_attn.o_proj`):

```text
h_i(t) = activation of output channel i at token t
score_i = max_{t ∈ span(wrong_span)} |h_i(t)|
```

Keep the top-k rows per layer (not per module; the per-layer budget is shared). Optional refinement: forward `prompt + target`, backward `−log p(target)`, and rank candidates by

```text
score_i = Σ_t |h_i(t) · g_i(t)|
```

keeping the best half per layer. The locator runs on the current function (base + sidecar if one exists), not on a clean FP16 model.

## 5. Stage 2 — ablate

Ablating row `i` means zeroing its output channel, exactly equivalent to setting `W[i,:]` and `b[i]` to zero for that forward. The implementation uses a forward hook on the module output; no weights are mutated.

Metric:

```text
M = mean log p(wrong_span | prompt) + mean log p(target_span | prompt)
Δ(G) = M(baseline) − M(G ablated)
```

Protocol:

1. Group candidates (default 8 per group, per module).
2. Measure each group; if `|Δ(G)| ≥ ε_ablate`, bisect into halves.
3. Keep individual rows with `|Δ(row)| ≥ ε_row`.
4. Calibrate `ε_ablate` from null random-row groups when enabled; otherwise use the configured fixed threshold.

Fallback: if fewer than `min_named` rows pass, the top activation candidates are measured individually and added with `fallback=True`; the report never pretends they are causally confirmed. On the 0.5B/7B runs, fallback was often a meaningful fraction (e.g., 27/32 on one 0.5B format case, 16/64 on 7B Valdoria), which is a warning that activation magnitude alone is often sufficient for easy cases.

## 6. Stage 3 — materialize

- Map module name → safetensors tensor name (`layer.weight`), read only the named rows from the immutable FP16/BF16 ledger.
- Compute row norms for guards.
- In single-call merge mode, snapshot the original rows for undo.
- In sidecar mode, no undo snapshot is needed per call: the previous sidecar version is the snapshot.
- Initialize `P` from the ledger (splice) or `ΔR=0` (delta).

## 7. Stage 4 — train

Replay set: `prompt + paraphrases → target_span` (≥6 sequences recommended). Targets include the tokenizer EOS/`<|im_end|>` token when `append_eos=true`; without it, the model learns the target prefix but not to stop, and free generation can add suffixes.

Loss:

```text
L = CE(replay → target)
  + λ_bg   · KL(p_θ || p_base^cache)   # absolute drift, clean base
  + λ_prev · KL(p_θ || p_prev^cache)   # continuity, previous accepted version
```

- `p_base^cache` and `p_prev^cache` are next-token distributions at the last token of a short background set.
- Multi-round training also evaluates a regression set from previous accepted failures; the current version may not regress beyond a small tolerance.
- Optimizer: AdamW on the selected rows only.
- Guards:
  - per-round: `||P_t − P_init|| / ||W_base||` (or `||ΔR|| / ||W_base||`) > `guard_ratio` → stop;
  - cumulative: `||P − W_base|| / ||W_base||` (or `||ΣΔR|| / ||W_base||`) > `cum_guard` → reject version.
- The deployed base is never modified during training.

## 8. Stage 5 — verify

For each prompt/paraphrase, generate the base answer on the current function and compare target vs that answer. Count already-correct variants as passes. Report:

- target mean logprob and base-answer mean logprob;
- exact free-generation match / contains target;
- background KL vs clean base;
- previous-version regression (multi-round);
- cumulative row ratio.

Verification is not optional. On the 0.5B 12-item set, teacher-forced target logprob said 12/12 without EOS, but exact free generation was 5/12. On 7B Orion, 0.667 was rejected.

## 9. Multi-round sidecar and versions

Why not merge every round:

- A delta trained against `deq(W_q)` merged into FP16 and requantized changes the function again; the next round compensates. The FP16 master can drift toward quantization-specific corrections.
- Repeated casts and row overlaps accumulate.
- Per-call undo does not compose.

Design:

```text
base_ledger/                 immutable FP16/BF16 base
versions/<version_id>/
  sidecar.safetensors        full float32 row snapshot (union of all edits)
  sidecar_meta.json          rows, param_mode, base sha, parent
  manifest.json              status, hashes, config hash, metrics
  failure.json               full failure for regression
  base_answers.json
current                      pointer to accepted version
releases/<version_id>/       merged + quantized artifact + manifest
```

Rules:

- Sidecar stores the current full value of every edited row, not a chain of deltas.
- The next round loads the current sidecar and applies those rows frozen; new named rows are trainable. Overlapping rows start from the sidecar value, not from W_fp16.
- Version manifest includes parent, base sha, sidecar sha, failure/config hashes, status, metrics.
- `current` advances only for accepted versions. A rejected version remains archived but not current.
- A commit checks that `current` has not moved since training (parent-pointer concurrency check).
- Release merges only an accepted version into a copy; the base and sidecar remain unchanged.
- Rollback switches the pointer; rolling back a middle version requires rebase/retrain of dependent later versions.

## 10. Release

The release step is the only step that creates a model artifact:

1. copy/ link the immutable base into a release directory;
2. apply the sidecar:
   - splice: `W_fp16[row] = P[row]`;
   - delta: `W_fp16[row] += ΔR[row]`;
3. atomic write;
4. load as NF4 (or the target format);
5. verify the failure, the accumulated regression set, and background KL;
6. write `releases/<id>/manifest.json`.

If post-quant verification fails, do not publish; keep the base + sidecar hybrid or reject the version. The current code targets NF4. A true MXFP4 release needs a different quantizer/kernel.

For sharded 7B, the base is four safetensors shards; only affected shards are rewritten. For a single-file model larger than `max_rewrite_gb`, `shard_and_patch` streams tensors into ≤1.5 GB shards with row patches; it is unit-tested but not exercised on a real 15 GB single-file model in this release.

## 11. Memory model

At a high level:

```text
peak ≈ full NF4 weights
     + activation graph for backward
     + temporary dequant buffers for named layers
     + row parameters + AdamW state
```

It does **not** include:

- a full FP16/BF16 base model (the ledger stays on disk);
- full-model weight gradients;
- optimizer states for the frozen base;
- persistent BF16 copies of every named layer.

Measured on a single RTX 4070 Ti 12 GB, with ~4 GB desktop use:

| | Qwen2.5-0.5B | Qwen2.5-7B |
|---|---:|---:|
| NF4 load allocated | 0.43 GB | 5.18 GB |
| NF4 load reserved | 0.62 GB | 5.52 GB |
| training peak allocated | 0.98–1.50 GB | 6.44 GB |
| training peak reserved | 1.01–1.50 GB | 6.50 GB |
| fix wall clock | 25–50 s | 68–75 s |
| full fine-tuning comparison | — | >100 GB |
| QLoRA comparison | — | same order (≈5–7 GB) |

What “lower VRAM” means precisely:

- Lower than **full fine-tuning** by a large factor (4-bit frozen base + tiny trainable state).
- Similar to **QLoRA**, not a new memory regime. The differentiators are deployment consistency, row-level audit/rollback, data economics, and versioned sidecars.
- The full NF4 model is still resident. You cannot load only the activated rows and run a forward: every layer needs the full previous activation. The active row map is a training mask.
- Unloading is session-level (after a fix/batch), not between locate/ablate/train. Unloading/reloading between stages would re-run forwards and is counterproductive.

Consumer guidance:

- 0.5B NF4: comfortably fits 8 GB, including tests.
- 7B NF4: measured on 12 GB with ~4 GB desktop; 8 GB is not verified and is likely tight. Use batch 1, short sequences, and avoid concurrent GPU apps; gradient checkpointing is a further lever.
- 13B NF4: outside the measured scope; plan for a larger card or offload.

See [VRAM.md](VRAM.md) for the detailed accounting and guidance.

## 12. Failure modes and limits

1. **Norm budget.** The spec's 15% per-round/30% cumulative guard is not met by the successful constructed runs: 0.5B 2.3–4.6×, 7B 1.61–2.28× cumulative. The parameter count is tiny; the displacement is not.
2. **Target difficulty.** 7B Valdoria and identity pass; 7B Orion at 96 rows/200 steps reaches 0.667 and is rejected. Not every target is fixable with a fixed budget.
3. **Fallback rows.** Easy cases often work from high activation alone; causal ablation confirmation is useful but not always decisive. Do not overstate "ablation-confirmed" when fallback rows dominate.
4. **Quantization.** `splice_fp16` trains around FP16 rows; release NF4 requantizes once. The final quantization is verified but still a risk. NF4 ≠ MXFP4.
5. **Dense full model.** The runtime needs the full NF4 model; this is not MoE hot-expert loading and not "only the path".
6. **Evaluation scope.** 12 constructed 0.5B items, two accepted 7B targets, 16 short background prompts. No MMLU/IFEval-scale neighborhood evaluation.
7. **Sequential scale.** Two accepted 0.5B and two accepted 7B targets; 20–50 sequential fixes and long-term drift are not tested.
8. **Operational complexity.** Sidecars, manifests, release copies, and sharded writes are more machinery than a LoRA file. The machinery buys auditability/rollback/multi-round safety.
9. **Concurrency.** Parent-pointer checks detect a moved `current`; there is no OS-level file lock or CI on GPU behavior yet.
10. **Per-version provisional quantization.** The specification asks for a temporary quantization check before accepting a version; this implementation accepts the hybrid and performs the final quantization at release. This is a known v0.1 deviation.

## 13. Engineering reliability

Implemented:

- atomic sidecar/manifest/current writes (temp + rename);
- immutable base; base SHA cache by size/mtime with per-file SHA;
- sidecar shape/range/in_features validation;
- config validation (`mode`, `param_mode`, quant/device, numeric ranges);
- parent-pointer check against concurrent commits;
- row-level undo for single-call merge;
- version rollback; rejected versions never advance `current`;
- streaming sharded release for large single-file models;
- CLI exit codes and concise errors; `--debug` re-raises;
- 17 CPU tests; `ruff check --select F,E9`; `scripts/check.sh`;
- CI workflow for Python 3.10/3.11 on CPU.

Not yet:

- OS-level writer lock;
- CI for CUDA/NF4 smoke (requires a GPU runner);
- per-version provisional quantization;
- a full 20–50 sequential regression harness;
- a real MXFP4 release path.

## 14. Results summary

- 0.5B, 12 constructed items: 12/12 exact free generation with EOS; 5/12 without EOS despite 12/12 teacher-forced.
- 0.5B Australia: exact `Canberra` pre/post requant; background KL 0.0093.
- 7B Valdoria: 64 rows, 105 steps, exact `Mervik`; release published.
- 7B identity-maker: 108-row union, 90 steps, exact `NGT Labs` (synthetic target); Valdoria regression 1.0; release published, background KL 0.0043.
- 7B Orion: 0.667 at 96 rows/200 steps; rejected.
- 7B base: four shards, `samefile` with the source after runs; base unchanged.
- 7B peak: 6.44 GB allocated, 6.50 GB reserved; 68–75 s per fix.

The full JSON reports are in [sample_reports](../sample_reports) and [RESULTS.md](../RESULTS.md).

## 15. References and related work

- QLoRA / bitsandbytes NF4: 4-bit frozen base with gradient flow through dequantized weights.
- ROME (2022), MEMIT (2022): causal tracing and row/MLP knowledge editing.
- FISH-Tuning (2025): sensitivity-based sparse fine-tuning masks.
- MEMOIR: minimal-overwrite/undo thinking.
- This project's difference: per-failure causal row selection on the deployed NF4 model, trainable row deltas with an EOS/regression/quantization-verified release, and an immutable-base versioned sidecar.
