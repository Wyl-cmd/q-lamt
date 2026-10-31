# Results (English Summary)

This is the English summary of [RESULTS.md](RESULTS.md). The full tables, per-variant outputs, and Chinese notes are in the Chinese document.

Environment: Qwen2.5-0.5B/7B-Instruct, bitsandbytes NF4, bf16 compute, RTX 4070 Ti 12 GB (about 4 GB desktop use), Windows 10.

## 1. 0.5B: 12 constructed failures

Categories: identity (2), fictional facts (6), agent tool names (2), strict format (2). Each failure has one prompt + five paraphrases.

| Setting | Written | Teacher-forced | Exact free generation (pre/post requant) | Contains target | Background KL | Peak VRAM | Mean wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| No EOS | 12/12 | 12/12 | 5/12 / 4/12 | 11/12 | 0.0222 | 1.47 GB | 43.4 s |
| EOS target | 12/12 | 12/12 | **12/12 / 12/12** | 12/12 | 0.0118 | 1.49 GB | 49.8 s |

Key lesson: teacher-forced target logprob can overstate repair. Without EOS, the model learns the target prefix but does not stop; exact generation is only 5/12. The EOS target fixes this.

Category results with EOS: identity 2/2, fact 6/6, agent 2/2, format 2/2 exact (pre/post).

The 0.5B Australia smoke (base `Sydney` → target `Canberra`) generates exact `Canberra` pre/post, fix 1.0, background KL 0.0093.

## 2. 7B: sequential sidecar + release

Base: four safetensors shards (~15 GB), NF4 load 5.18 GB allocated, training peak 6.44 GB allocated / 6.50 GB reserved. Base shards stay `samefile` with the source.

| Version | Failure | Rows | Steps | Fix | Cumulative row ratio | Background KL | Status |
|---|---|---|---:|---:|---:|---:|---|
| v1 | Valdoria → `Mervik` | 64 | 105 | 1.0 | 1.61 | 0.0026 | accepted |
| v2 | Orion → `Elin Marr` | 140 | 105 | 0.667 | 2.85 | 0.0121 | rejected |
| v2c | identity → `NGT Labs` (synthetic target) | 108 union | 90 | 1.0 | 2.28 | 0.0039 | accepted |

- v1 release: published; exact `Mervik`; background KL 0.0029.
- v2c release: published; exact `NGT Labs`; Valdoria regression 1/1 with exact `Mervik`; background KL 0.0043.
- v2c is current; v2 (Orion) remains archived as rejected and did not advance `current`.
- The first v2 attempt was rejected after one step because the cumulative edit (3.28×) was mistaken for a per-round edit; `init_values` fixed the per-round vs cumulative guard separation.

## 3. What the numbers mean

- **Mechanism:** the full loop locate → ablate → train rows → version/verify → release runs on 0.5B and 7B.
- **Consumer GPU:** 7B NF4 fits on a 12 GB card with a normal desktop; 8 GB is not verified for 7B. 0.5B fits 8 GB comfortably.
- **Memory:** lower than full fine-tuning by a large factor; roughly QLoRA-class, not a new memory regime. The full NF4 model is resident; only the trainable rows are small.
- **Data:** one failure + paraphrases, not a domain corpus.
- **Audit:** row-level sidecars, manifests, hashes, `current` pointer, release manifests.
- **Not proven:** 15%/30% norm budget; all facts; 20–50 sequential fixes; full benchmark neighborhood; MXFP4.

## 4. Key limits

- The successful runs need 1.61–4.6× row-norm changes; the spec's 15%/30% guard rejects them.
- 7B Orion at 96 rows/200 steps reaches 0.667 and is rejected. Target difficulty and budget decide success.
- `accepted` means the hybrid (base + sidecar) passed verification; final quantization is checked at release. This is a v0.1 deviation from a per-version provisional check.
- The release target is NF4, not MXFP4.
- Two accepted 0.5B/7B sequential targets, not 20–50.
- Background KL uses 16 short prompts, not a benchmark suite.

## 5. Artifacts

- `sample_reports/` contains selected `report.json` / `manifest.json` snapshots (no weights).
- `artifacts/` and `models/` are not in the repository; reproduce with `scripts/setup_env.sh` and `scripts/download_model.sh`.

See [README.md](README.md) for the Chinese primary README and [docs/TECHNICAL.en.md](docs/TECHNICAL.en.md) for the technical design.
