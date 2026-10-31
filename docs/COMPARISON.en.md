# Q-LAMT Compared to Other Approaches

> 中文版：[COMPARISON.zh-CN.md](COMPARISON.zh-CN.md)

Q-LAMT is not a better general-purpose training method. It occupies a specific niche:

> **Few failure samples, row-level audit/rollback, deployment on the quantized model, and a bounded edit that must survive requantization.**

It competes with model editing and sparse adaptation, not with pretraining or broad domain fine-tuning. This page compares the alternatives and asks, step by step, what is actually necessary.

## Comparison table

Approximate figures; “same order” means tens of percent, not an order of magnitude.

| Approach | Data needed | Generalization | 7B VRAM | Audit / rollback | Deployment consistency | Main weakness |
|---|---|---|---|---|---|---|
| Full fine-tuning | large domain corpus | strongest | >100 GB | no row-level; checkpoint-level | same weights, then quantize | forgetting, cost |
| LoRA / QLoRA | domain dataset | good | ~5–7 GB | adapter-level, not row-level | merge/requantize caveat | data, adapter stacks, opaque |
| ROME / MEMIT | one fact | limited to the edit | low | per-edit, no training loop | often FP16; deployment differs | closed-form assumptions, coupling, side effects |
| FISH / sparse FT | domain dataset | good | low optimizer state | mask-level | merge/requantize | mask from data, not per-failure causal |
| Activation steering / ITI | activation examples | behavior-level | low | reversible intervention | no weight change | inference overhead, not persistent |
| RAG / prompt | no training | knowledge-level | inference cost | external | no model change | does not change weights/format |
| **Q-LAMT** | **one failure + paraphrases** | **local** | **6.4 GB peak (measured)** | **row + version chain** | **train on NF4, release verifies** | per-fix cost, norm budget, complexity |

## What Q-LAMT actually improves

### 1. Data economics vs LoRA/QLoRA

LoRA/QLoRA need a dataset. Q-LAMT needs one failure and a handful of paraphrases. That is valuable when collecting examples is the bottleneck, or when a failure is rare and important.

It does **not** beat LoRA on a large, well-covered domain dataset. For that, LoRA is simpler and usually stronger.

### 2. Deployment consistency vs adapter/FP16 editing

Q-LAMT locates, ablates, trains, and verifies on the deployed NF4 function. The final artifact is a merged FP16 row set, requantized once and verified. QLoRA training an FP16 adapter and merging into 4-bit is a known lossy step; ROME/MEMIT often operate on FP16 and may not survive deployment.

This is a real difference, but it is not free: Q-LAMT trains `splice_fp16` rows in a hybrid forward, then pays one final quantization step. The release check is mandatory.

### 3. Row-level audit and rollback

Every edit has:

- `(layer, row)` addresses;
- a sidecar snapshot with float32 values;
- a manifest with base SHA, sidecar SHA, config hash, failure hash, metrics, parent;
- a `current` pointer and release manifests.

LoRA can be versioned at the adapter level, but not at the row level. ROME/MEMIT usually have no training/regression/version chain. Full fine-tuning has checkpoints, not explanations of which rows changed.

### 4. Forgetting control by construction

99.9% of parameters are frozen. Background KL, previous-version KL, and a regression set are explicit. Full fine-tuning has no such structural protection; LoRA is less invasive but still distributed; this design makes the neighborhood checks part of the pipeline.

On the experiments, background KL stayed low (0.0026–0.0043 for 7B), but the background set is only 16 short prompts; that is not a full benchmark.

### 5. No adapter accumulation

A release can be one merged model. Multi-task LoRA serving often accumulates adapters or needs routing. Q-LAMT's sidecars are versioned and can be rolled back, but a release still has to be merged and verified.

### 6. Causal filtering

Activation magnitude is a recall device. Ablation tests whether zeroing the row moves the behavior. FISH selects a sparse mask by sensitivity over a dataset; Q-LAMT selects rows per failure and can audit the decision.

Caveat: in the experiments, fallback rows were often needed (for example 27/32 on a 0.5B format case, 16/64 on 7B Valdoria). Easy cases often work from high activation alone. Ablation is a precision/high-stakes tool, not always the deciding factor.

### 7. Domain reuse

Homogeneous failures overlap on rows. A domain can become a versioned sidecar of hot rows, trained with a previous-version KL and a cumulative regression set. This is the strongest use case for Agent/code/format/identity repair.

## Where Q-LAMT loses

### Per-fix cost

Ablation alone can be hundreds of forwards (7B Valdoria: 368 forward pairs; 29.8 s for ablation). A single fix is 68–75 s including training and verification. ROME is seconds; LoRA over a large domain amortizes its cost over many examples. Q-LAMT is not the cheapest per edit.

### Edit magnitude

The spec's 15% per-round / 30% cumulative guard is **not met** by the successful constructed runs:

- 0.5B: 2.3–4.6× row norm;
- 7B: 1.61–2.28× cumulative.

The parameter count is tiny, but the displacement is large. Calling this “surgical” in magnitude would be misleading. It is sparse in *count*, not necessarily small in *distance*.

### Target difficulty

7B Valdoria and identity pass; 7B Orion reaches 0.667 and is rejected. A fixed row/step budget does not guarantee every fact. The mechanism is sound; the target and budget decide.

### Memory vs QLoRA

7B NF4 peaks at 6.44 GB. QLoRA is the same order (~5–7 GB). The advantage is not “much lower VRAM”; it is governance/data/deployment consistency. The full NF4 model is resident; this is not “only the active path”.

### Complexity and operations

Sidecars, manifests, `current`, releases, regression sets, sharded writes, and a base SHA are more moving parts than a LoRA file. The machinery buys auditability and multi-round safety; if you do not need those, it is overhead.

### Generalization and evaluation

The evidence is:

- 12 constructed 0.5B failures;
- two accepted 7B targets (Valdoria, identity);
- 16 short background prompts;
- no MMLU/IFEval-scale evaluation;
- no 20–50 sequential-fix run.

It is a mechanism validation, not a capability claim.

## Is every step necessary?

No. There is a minimal core and a set of robustness layers.

### Core loop

1. **Quantized runtime** — required if the deployment is NF4; otherwise you measure a different function. If deployment is FP16, use a full FP16 runtime (7B memory makes this impractical here).
2. **Locate (some selector)** — required to choose rows. Activation top-k is the cheapest; gradient/Fisher/random are alternatives. Without a selector, ablation over all rows is infeasible.
3. **Train rows** — the core repair loop. For a single isolated fact, ROME's closed-form edit can replace it; for multi-fact/behavior failures, iterative training is needed.
4. **Verify** — **not optional**. Without EOS, teacher-forced looked 12/12 while exact free generation was 5/12; 7B Orion was rejected by verification. Without post-quant verification, the deployed model may not contain the fix.
5. **Deploy** — either merge + quantize to a single file, or keep base + sidecar hybrid. A release must be explicit.

### Robustness layers

| Step | Necessary for | Can remove when | Evidence / note |
|---|---|---|---|
| Ablation confirmation | precision; high-stakes facts | easy/high-activation cases, domain hot rows already trusted | fallback rows often carry easy runs |
| Threshold calibration | better false-positive control | fixed thresholds are acceptable | small cost |
| Background KL | forgetting control | a held-out benchmark replaces it | keep some neighborhood check |
| Previous-version KL | multi-round continuity | single fix only | prevents undoing prior fixes |
| Regression set | multi-round safety | single fix only | catches “fix A, break B” |
| EOS target | exact free generation | only teacher-forced scoring is acceptable | 5/12 vs 12/12 |
| Sidecar/version chain | multi-round, rollback, audit | one-off merge is enough | avoids repeated base mutation |
| Cumulative guard | multi-round risk | a one-off edit | policy, not a technical law |
| 15%/30% guard | the “small edit” guarantee | you accept larger edits | our runs exceed it |
| Sharded release | single-file 15 GB / low RAM | the base is already sharded or small | 7B sharded base did not need it |
| Per-version provisional quant | strict “accepted means deployable” | final release check is accepted | v0.1 deviation |
| Layer-balance top-k | cost/precision | random/global top-k | not a hard requirement |

### What happens if you remove a link

- Remove ablation → faster, more false-positive rows; may still work on easy cases.
- Remove verification → unacceptable; you cannot know whether generation/quantization survived.
- Remove sidecar/versioning → single fix fine; multi-round accumulates and rollback is weak.
- Remove background/regression → repair may break unrelated behavior or previous fixes.
- Remove release quantization check → the shipped NF4 model may not have the fix.
- Remove EOS → the model learns a prefix without stopping.
- Remove the norm guard → still works, but the “surgical” claim disappears.

## When to use which

- **One isolated fact, no audit, speed matters:** ROME/MEMIT.
- **Broad domain adaptation or capability improvement:** LoRA/QLoRA with data.
- **Few failures, audit/rollback, deployment consistency, data scarcity:** Q-LAMT.
- **Agent/code/format behavior with homogeneous failures:** Q-LAMT + domain hot-row sidecar.
- **Activation-only intervention with no weight change:** activation steering/ITI.
- **Knowledge without changing the model:** RAG.
- **Capability uplift:** none of the sparse-row methods; use data and full/LoRA training.

The honest summary: Q-LAMT's advantages are **governance, data economics, deployment consistency, and row-level reversibility**, not raw capability, throughput, or a dramatic VRAM win over QLoRA.
