# VRAM and Consumer-GPU Guide

> 中文版：[VRAM.zh-CN.md](VRAM.zh-CN.md)

Q-LAMT is designed to run **targeted row repair** on a consumer GPU by keeping the whole model in 4-bit, keeping the FP16/BF16 base on disk, and giving gradients only to a small set of rows.

This page gives the measured numbers and, importantly, what the numbers do **not** mean. The short version:

> 7B NF4 loads at 5.18 GB and trains at a 6.44 GB peak on an RTX 4070 Ti 12 GB. That is much less than full fine-tuning, but roughly QLoRA-class memory, not a new memory regime. The full NF4 model is resident; only the *trainable* set is tiny.

## Measured numbers

Environment: Qwen2.5-0.5B/7B-Instruct, bitsandbytes NF4, compute dtype bf16, RTX 4070 Ti 12 GB, about 4 GB used by the desktop, batch 1–2, short prompts (256 max sequence length), Windows 10.

| | 0.5B | 7B |
|---|---:|---:|
| NF4 weights, allocated | 0.43 GB | 5.18 GB |
| NF4 weights, reserved | 0.62 GB | 5.52 GB |
| Training peak, allocated | 0.98–1.50 GB | 6.44 GB |
| Training peak, reserved | 1.01–1.50 GB | 6.50 GB |
| Total with ~4 GB desktop | ~5–5.5 GB | ~10.4 GB |
| Wall clock per fix | 25–50 s | 68–75 s |
| Full fine-tuning comparison | ~4 GB+ weights/optimizer | >100 GB |
| QLoRA comparison | — | ≈5–7 GB (same order) |

The 7B peak is the whole process (NF4 load + activations + row parameters/optimizer + temporary dequant buffers). The four base shards were unchanged after the runs (`samefile` with the source model).

## What is resident, and what is not

During one fix:

**Resident**

- **All NF4 weights of the whole model.** Dense Transformer layers are sequential; every layer is needed for the forward pass and for the backward activation graph.
- **Activation graph** for backward: this is usually the second-largest term after the 4-bit weights.
- **Named row parameters** (`P` or `ΔR`, float32) and their AdamW state. For 64 rows this was 2.46 MB of parameters and ~9.83 MB of Adam estimate on 7B.
- **Temporary dequant buffers** while materializing splice rows (one layer matrix at a time).

**Not resident**

- The full FP16/BF16 base. It is a safetensors ledger on disk; only named rows are read.
- Full-model weight gradients. The base is frozen (`requires_grad=False`).
- Optimizer states for the base.
- A persistent BF16 buffer of every named layer. The delta-only/splice correction is applied on the QLinear output, so there is no `W_eff = buffer.clone()`.
- Adapters. A release can be a merged model; no per-task adapter is loaded.

## The "only load the active path" idea

This does not work in a dense Transformer, and the implementation does not claim it.

At layer `L`, an active row computes `y_i = W[L,i] · x_L`, where `x_L` is the **full** activation vector produced by every row of layer `L-1`. The active rows are a subset of outputs, not a self-contained input path:

- Skipping the inactive rows changes `x_L` and everything downstream.
- The active set is input-dependent and is discovered by computing the full forward.
- ReLU-style hard zeros do not exist in modern SwiGLU LLMs; "inactive" rows still contribute small values.
- MoE is the architecture where hot experts can be loaded selectively; Qwen2.5 dense is not MoE.

The row map is therefore a **training mask** and an audit record: it tells you which rows may change. It does not tell you which weights can be omitted from the forward pass.

## The memory formula

Approximate peak:

```text
peak ≈ NF4_weights
     + activation_graph(batch, seq_len, hidden, layers)
     + temporary_dequant(max_layer_matrix)
     + row_params + adam_state(selected_rows)
```

The row term is tiny. Reducing the number of named rows from 1000 to 64 saves a few MB, not gigabytes. The big levers are the 4-bit base, activation memory, and avoiding a full FP16 model/optimizer.

## Why the base is 4-bit and not FP16

- 7B FP16 weights are ~15 GB before activations; a 12 GB card cannot hold them alongside the training graph.
- NF4 keeps the full model resident at ~5.18 GB, so a 12 GB consumer card can run the repair.
- The FP16 ledger is still the source of truth for the named rows and the merge; it is read row-wise from disk, not loaded as a model.
- Training on the NF4 function also matches the deployment target. A full FP16 forward would fit 0.5B but not 7B on this class of GPU.

## Comparison with QLoRA

For a 7B model, QLoRA typically uses ~5–7 GB with a frozen 4-bit base and a small adapter. Q-LAMT is in the same range (6.44 GB peak measured). The advantage is not “much lower VRAM”:

- Q-LAMT trains rows selected and causally checked per failure; QLoRA trains a dense low-rank adapter over a dataset.
- Q-LAMT can keep the FP16 base and version a sparse sidecar; QLoRA needs adapter management and a merge/requantization decision.
- Q-LAMT is data-cheap and row-auditable; QLoRA is usually better for broad domain adaptation with many examples.
- A merge into a single model is lossy for both; Q-LAMT makes the final quantization check explicit.

## What actually reduces peak VRAM

1. **Keep the base in NF4** (largest single win for 7B).
2. **Do not load the FP16 base**; read rows from disk.
3. **Keep batch and sequence length small** (batch 1, max 256 tokens in the experiments).
4. **Freeze the base and allocate no base optimizer states**.
5. **Use the correction-only forward**, not a persistent BF16 copy of every named layer.
6. **Gradient checkpointing** if activations are the bottleneck (not enabled by default; hooks and checkpointing need an integration test).
7. **Session-level unloading** between fixes/batches, not between pipeline stages.

What does **not** reduce peak:

- Loading only the active rows (the forward still needs the rest).
- Unloading the model between locate/ablate/train (you would redo forwards).
- Shrinking the row parameter count alone (it is already MBs).

## Consumer-GPU guidance

| GPU | 0.5B NF4 | 7B NF4 |
|---|---|---|
| 8 GB | comfortable | not verified; likely tight, close other GPU apps, batch 1, short sequences |
| 12 GB | comfortable | **measured** with ~4 GB desktop use |
| 16 GB | comfortable | comfortable |
| 24 GB | comfortable | comfortable; room for larger batch/seq |

The 7B number was measured on a 12 GB card with a desktop using ~4 GB. A headless 8 GB card might fit at batch 1 and short sequences, but we have not verified it; do not quote it as supported. 13B NF4 is outside the measured scope.

## Reproducing the measurement

```bash
# Load only: prints allocated/reserved/peak
.venv/Scripts/python -m qlamt.cli info --config configs/7b.json

# A fix: per-stage memory is in report.json -> memory
.venv/Scripts/python -m qlamt.cli fix --failure failures/valdoria.json --config configs/7b.json
```

`report.json` records `memory.load`, `memory.locate`, `memory.ablate`, `memory.train`, `memory.verify`, each with allocated/reserved/peak values.

## Important caveats

- These numbers are for **constructed repair tasks**, not full benchmark training.
- The successful runs use larger row displacements than the spec's 15%/30% guard; memory feasibility does not imply the guard is satisfied.
- The full NF4 model is resident; this is not “no full model”.
- The release step can temporarily hold a base copy and writes a release model; plan disk, not just VRAM.
- NF4 is not MXFP4. A different 4-bit format may have different kernels, scales, and memory behavior.
