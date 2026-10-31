# 实测记录（v0.1）

日期：2026-10-31  
模型：Qwen2.5-0.5B-Instruct（干净下载，项目内 `models/`）  
运行时：NF4（bitsandbytes 0.50.2），compute bf16  
硬件：RTX 4070 Ti 12GB（桌面占用约 3.6GB）  
环境：项目内 `.venv`，Python 3.10.6，torch 2.11.0+cu128，transformers 4.51.3

## 1. 结论先行

- **机制闭环成立**：定位 → 消融 → 行增量训练 → 写回/版本化 sidecar → 量化 → 复验，在 0.5B 与 7B 上都完整跑通；17 项 CPU 单测通过。
- **同参数 A/B**：12 条构造样本，无 EOS 时自由生成精确 5/12；目标追加 EOS 后 **12/12 精确生成，且重量化后仍 12/12**。
- **代价必须说清**：0.5B 成功运行使用了 `lr=2e-3`、`guard_ratio=3.0`，实际 ΔR 达行范数的 3.1–4.6 倍；规格的 15% 范数守卫在 0.5B 上不成立。7B 已测：Valdoria 与 identity 通过，Orion 目标被拒。
- **显存**：0.5B 单次调用峰值 1.0–1.49GB；行参数仅 0.07–0.22MB，优化器估计 0.3–0.9MB。显存大头仍是 NF4 本体 + 激活，不是“只加载了行”。相对全量微调低一个数量级，相对 QLoRA 同档。
- **适用域**：Agent 工具名、严格格式、身份短句、虚构事实短答案在 0.5B 上都能修；但这是“构造性机制实验”，不代替 7B/50 条注入事实的正式裁决。

## 2. 单条冒烟：Australia → Canberra

失败样本：`failures/smoke.json`，基座回答 Sydney，目标 Canberra，5 条释义。

| 项 | 值 |
|---|---|
| 结果 | `written`，重量化复验通过 |
| 修复率（teacher-forced，含“基座已正确”的释义） | 0.333 → **1.0** |
| 主 prompt 生成（训练后 / 重量化后） | `Canberra` / `Canberra`（均精确） |
| 背景 KL 均值 | 0.0093 |
| 点名行 | 16（fallback 2） |
| 训练步数 | 40（范数守卫 0.807 触发） |
| 峰值显存 | 0.997GB |
| 墙钟 | 32.4s |

报告：`artifacts/runs/20261031-013954-smoke-australia-capital/report.json`

## 3. 12 条构造数据 A/B

数据：`data/e2e_small.jsonl`（2 身份、6 虚构事实、2 Agent、2 格式），每条 1 个主 prompt + 5 条释义；`wrong_span` 为空时先探测基座答案。

共同设置：`topk_per_layer=24`、`grad_x_act` 开、分组 8、阈值校准、`min_named=32`、`lr=2e-3`、最多 200 步、KL λ=0.3、`guard_ratio=3.0`、独立账本。

| 设置 | 写回 | 重量化后 teacher-forced | 精确生成（训练后） | 精确生成（重量化后） | 包含目标（训练后） | 背景 KL 均值 | 点名行均值 | 峰值显存 | 均值墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 无 EOS | 12/12 | 12/12 | **5/12** | **4/12** | 11/12 | 0.0222 | 35.8 | 1.47G | 43.4s |
| 目标追加 EOS | 12/12 | 12/12 | **12/12** | **12/12** | 12/12 | 0.0118 | 42.5 | 1.49G | 49.8s |

EOS 组按类别：

| 类别 | n | 写回 | 重量化后通过 | 精确生成（pre/post） | 背景 KL 均值 |
|---|---:|---:|---:|---:|---:|
| identity | 2 | 2 | 2 | 2/2 / 2/2 | 0.0109 |
| fact | 6 | 6 | 6 | 6/6 / 6/6 | 0.0183 |
| agent | 2 | 2 | 2 | 2/2 / 2/2 | 0.0027 |
| format | 2 | 2 | 2 | 2/2 / 2/2 | 0.0026 |

EOS 组 12 条中：

- 8 条跑满 200 步；4 条在范数守卫触发时停止（最大 ΔR/行范数 = 3.11、3.18、3.26、4.56）。
- 全部 12 条 `target_lp` 相对基座平均提升 5.08 nats。
- fallback 行总数 39/12 条（均值 3.25）；`fmt-marker` 一条就占 27，说明 0.5B 上消融确认有时偏弱，训练仍靠高激活行完成。
- 无 EOS 组最刺眼的反例：`fact-aurora-chip` teacher-forced 1.0，但自由生成仍是 `IBM`；EOS 组同一案例生成 `Halcyon Systems`，pre/post 均精确。**只看 logprob 会高估修复。**


## 3.5 多轮 sidecar / release 实测（2026-10-31）

同一不可变基座、`splice_fp16`、EOS、`topk=24`、`min_named=32`、`lr=2e-3`、最多 200 步、
`guard_ratio=3.0`、`cum_guard=5.0`（0.5B 放宽口径）。顺序修复两条失败：

| 版本 | 失败 | 点名/并集行 | 训练步 | 修复率 | 累计行范数 | 背景 KL | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| v1 | Australia (Sydney→Canberra) | 32 | 175 | 1.0 | 3.28 | 0.0050 | accepted |
| v2 | identity-maker (→NGT Labs) | 48（32 新） | 100 | 1.0 | 4.36 | 0.0090 | accepted |

- v2 的回归集包含 v1 的 Australia，回归通过率 1.0；说明加载上一版 sidecar 后，
  新行训练没有覆盖旧修复。
- 第一次 v2 尝试被拒：训练第 1 步就触发 `max_edit_ratio=3.28`。原因是把“累计
  P−W_base”误当成本轮编辑量；加入 `init_values` 后，单轮守卫改测“P−本轮初始值”，
  累计守卫单独测“P−W_base”，重跑通过。被拒版本仍保留在归档中（不推进 current）。
- 基座 sha 与源模型文件 sha 完全一致；所有版本提交都没有改 `base_ledger/`。

**release** v2（合并 sidecar → 写 fp16 副本 → 加载为 NF4 → 复验）：

| 项 | 结果 |
|---|---|
| published | true |
| 当前失败（identity） | 1.0，生成 `NGT Labs` |
| 回归（Australia） | 1.0，生成 `Canberra` |
| 背景 KL 均值/最大 | 0.0099 / 0.032 |
| 写后读回 | verified |
| 基座 | 未修改 |

**版本/回滚**：链为 v1(accepted) → v2(rejected 尝试) / v2b(accepted)；`current`
指向 v2b。`version-rollback` 切到 v1、连续两次切回 v1、再切回 v2b 均成功，
`base_sha` 不变。产物：

```
artifacts/multi-e2e/versions/versions/    # v1/v2 拒绝/v2b
artifacts/multi-e2e/versions/base/        # 不可变基座
artifacts/multi-e2e/versions/current
artifacts/multi-e2e/releases/20261031T035658-0794a1/
artifacts/multi-e2e/runs/v1-australia/    # 含 report.json
artifacts/multi-e2e/runs/v2b-identity-maker/
```

**边界**：`accepted` 表示混合态通过；最终量化复验在 release。规格 §7.5.4 的
“每版临时量化复验”尚未实现（v1 偏差）。多轮只测了 2 条顺序修复，不是 20–50 条；
`cum_guard=5.0` 是 0.5B 放宽值，规格默认 30% 会拒绝；7B 已测：Valdoria 与 identity 通过，Orion 目标被拒。


## 3.6 7B 实测（Qwen2.5-7B-Instruct / NF4 / RTX 4070 Ti 12G）

> 说明：下文的 `NGT Labs` 是历史实验使用的合成身份目标，不是项目名。

- 7B NF4 加载分配 5.18G；训练阶段峰值 6.44G（整卡含桌面约 10.4G）；单次修复 68 秒。
- 基座是 4 个 safetensors 分片（约 15G）；`VersionStore` 用硬链接副本，release 只重写受影响分片；
  运行后 `base` 四个分片与源模型 `samefile`，基座未被改写。
- 版本链（同一基座、`splice_fp16`、EOS、`topk=32`）：

| 版本 | 失败 | 行数 | 步数 | fix | 累计范数 | 背景 KL | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| v1 | Valdoria → Mervik | 64 | 105 | 1.0 | 1.61 | 0.0026 | accepted |
| v2 | Orion → Elin Marr（96 行/200 步尝试） | 140 | 105 | 0.667 | 2.85 | 0.0121 | rejected |
| v2c | identity-maker → NGT Labs | 108（并集） | 90 | 1.0 | 2.28 | 0.0039 | accepted |

- v1 release：published，生成 `Mervik`，背景 KL 0.0029。
- v2c release：published；当前失败生成 `NGT Labs`；Valdoria 回归 1/1 生成 `Mervik`；
  背景 KL 0.0043；release 目录 4 个分片 + manifest。
- 版本列表：v1 accepted → v2 rejected → v2b rejected → v2c accepted；`current=v2c`。
- 边界：
  - 7B 的 Valdoria/identity 修复仍需要累计 1.61–2.28 倍行范数，规格 30% 默认线会拒绝；
    15% 守卫在 7B 上同样没有成立。
  - Orion 的 96 行/200 步只到 0.667（4/6），不是链路崩溃，而是目标难度/预算边界。
  - release 目标是 NF4，不是 MXFP4；7B 基座本身是分片，单文件 15G 的 `shard_and_patch` 只有小单测。
- 产物：

```
artifacts/7b/versions/base/                     # 4 分片硬链接基座
artifacts/7b/versions/versions/                 # v1 / rejected v2/v2b / v2c
artifacts/7b/versions/current
artifacts/7b/releases/20261031T050011-e95419/   # v1 release（Mervik）
artifacts/7b/releases/20261031T050925-4a2a59/   # v2c release（NGT Labs + Valdoria 回归）
artifacts/7b/runs/v1-valdoria/report.json
artifacts/7b/runs/v2c-identity-maker/report.json
```

## 4. 生成样本（EOS 组，重量化后）

| id | 生成 |
|---|---|
| id-identity-name | `I am Aria.` |
| id-identity-maker | `NGT Labs` |
| fact-valdoria | `Mervik` |
| fact-orion-founder | `Elin Marr` |
| fact-glass-meridian | `Tomas Vey` |
| fact-aurora-chip | `Halcyon Systems` |
| fact-element-qz | `Quantium` |
| fact-sarn-currency | `the sarn` |
| agent-list-files | `list_files` |
| agent-web-search | `web_search` |
| fmt-json-sum | `{"answer": 4}` |
| fmt-marker | `[OK]` |

## 5. 边界与解释

1. **范数预算**：0.5B 需要把行改动到 2.3–4.6 倍行范数；规格 §6.4 的 15% 守卫会提前终止。这是本次实验最重要的限制。7B 的行范数更大、知识更分散，但构造任务仍需 1.6–2.3 倍累计行范数；15% 守卫仍未验证。
2. **EOS 是精确生成的关键**：训练目标只是片段时，模型学会“目标前缀”但不学会停；追加 tokenizer EOS 后，12/12 自由生成精确。生产用法应把答案终止符纳入 target（Qwen 用 `<|im_end|>`，即 eos_token_id 151645）。
3. **teacher-forced ≠ 自由生成**：无 EOS 时两者分别是 12/12 与 5/12。验收必须同时报告生成样本/精确匹配，不能只看目标 logprob。
4. **独立账本**：12 条是分别从干净基座复制账本后单次修复，互不叠加；规格 §9.2 的“连续 50 条注入事实”仍需共享账本顺序运行，才能测行撞车与累积漂移。
5. **模型规模**：0.5B 是机制验证，不是能力结论。项目外的本地 7B 权重按你的要求未使用；7B 需要单独下载干净基座、重新校准 ε/范数/步数/显存。
6. **背景保持**：16 条短背景句末 KL 均值 0.0118（EOS 组），说明编辑局部性不错；但它不是 MMLU/IFEval 级别的邻域评测。
7. **显存叙事**：峰值 1.49G（0.5B）里行参数/优化器不到 1MB，NF4 本体 0.43G，其余是激活/临时量。**优势来自 4-bit 基座 + 极小可训练面，不来自“只加载了行”**；与 QLoRA 同档，优势在写回本体、审计、undo、无 adapter 堆积。
8. **环境**：`.venv` 是项目内隔离环境；`torch` 因 Windows 上 uv 缓存目录重命名被拒，改为把已完整解压的 4.2G torch 2.11.0+cu128 直接装入 venv。`scripts/setup_env.sh` 在干净机器上可能重演该问题，重跑前保留 `.venv`。

## 6. 产物

```
artifacts/runs/20261031-013954-smoke-australia-capital/     # EOS 冒烟
artifacts/datasets/20261031-011927/summary.json              # 无 EOS 12 条
artifacts/datasets/20261031-012846/summary.json              # EOS 前 3 条
artifacts/datasets/20261031-013207/summary.json              # EOS 5 条事实
artifacts/datasets/20261031-013625/summary.json              # EOS 后 4 条
artifacts/datasets/eos12_combined.json                       # EOS 12 条合并
artifacts/multi-e2e/                                         # 0.5B 多轮 sidecar/版本/release
artifacts/7b/                                                # 7B 基座/版本/release
```

每条运行目录含 `report.json`、`ledger/`（写回后的账本副本）、`undo/`（原始行快照）。

## 7. 下一步（按优先级）

1. **7B 扩展**：已跑通 Valdoria + identity；继续跑 20–50 条/分片发布/单文件 15G 的 shard_and_patch 实测。
2. **共享账本顺序 20–50 条**：2 条 sidecar 已跑通；继续测行撞车、累积漂移、版本链增长。
3. **EOS 默认化**：已在 `configs/*.json` 开启 `append_eos`；7B 用 `<|im_end|>`。
4. **sidecar 部署**：多轮 sidecar/release 已实现；继续测“补丁行保持 bf16、其余 NF4”的纯混合部署。
5. **§8.4 领域热行常驻**：Agent/code 域先做，因本次 agent/format 4/4 精确，是最有希望的生产楔子。
