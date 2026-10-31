# 小型构造性测试数据

`e2e_small.jsonl` 是按技术方案 §1.3 的 `Failure` 结构生成的 12 条小样本，用于在
0.5B 实验模型上判断机制是否成立，不是能力评测集：

| 类别 | 数量 | 目的 |
|---|---|---|
| identity | 2 | 身份/自我描述这类行为改写 |
| fact | 6 | 模型不存在的虚构事实（实体、创始人、作者、元素、货币） |
| agent | 2 | Agent 工具名选择（`list_files`、`web_search`） |
| format | 2 | 严格格式输出（JSON、标记） |

每条的 `wrong_span` 为空，由管线在训练前用基座生成；`target_span` 尽量短
（1–4 个 token），以便 0.5B 在少量行上可训练。`paraphrases` 各 5 条，用于
重放训练和验收；验收时会为每条释义单独生成基座答案，不用主 prompt 的错答
套到所有释义。

> 说明：`id-identity-maker` 的目标 `NGT Labs` 是构造性测试中的合成实体，不是项目名。

## 运行

```bash
# 先跑 3 条（identity / fact / agent 各一条的混合前三条）
.venv/Scripts/python experiments/run_dataset.py \
  --config configs/dataset.json --limit 3

# 指定类别或 id
.venv/Scripts/python experiments/run_dataset.py \
  --config configs/dataset.json --ids fact,identity

# 全部 12 条；关闭重量化复验会快一些
.venv/Scripts/python experiments/run_dataset.py \
  --config configs/dataset.json --no-post-requant
```

每次运行写到 `artifacts/datasets/<timestamp>/`，其中：

```
summary.json          总体/分类聚合 + 每条明细
00-<id>/report.json   单条完整报告
00-<id>/ledger/       该条的账本副本
00-<id>/undo/         该条写回前的原始行
```

## 0.5B 实测（2026-10-31）

同参数（32 点名行、200 步上限、lr 2e-3、guard 3.0、独立账本）下：

| 设置 | 写回 | 重量化后 teacher-forced | 自由生成精确 pre/post |
|---|---:|---:|---:|
| 目标不追加 EOS | 12/12 | 12/12 | 5/12 / 4/12 |
| 目标追加 EOS | 12/12 | 12/12 | **12/12 / 12/12** |

不追加 EOS 时，模型学会目标前缀但不会停：`Mervik` → `Mervikon City`，
`Halcyon Systems` → `IBM`（teacher-forced 仍 1.0）。因此 `configs/*.json`
默认 `append_eos=true`；这是精确生成的必要条件。完整结果见 `RESULTS.md`。

## 判读

- `base_fix_ratio`：训练前，目标答案是否已经胜过该释义的基座答案；
- `trained_fix_ratio`：训练后，同一比较；
- `post_requant_fix_ratio`：把写回后的账本重新以 NF4 加载后的比较；
- `target_lp_gain`：目标 logprob 相对基线的平均提升；
- `bg_kl_mean/max`：背景句末 next-token 分布的 KL；
- `n_named` / `n_fallback`：消融确认行数 / 因阈值未过而按激活分补的行数；
- `delta_param_mb` / `peak_train_gb`：实际参数与训练阶段峰值显存。

每条运行写回时会生成一个约 1GB 的账本副本（0.5B bf16）；跑全部 12 条前确认
`artifacts/datasets/` 有足够磁盘，或只跑子集。

这 12 条是**构造性机制实验**：即使全部通过，也只说明“定位→消融→行增量训练→
写回→重量化”这条链在 0.5B 上闭合，不说明 7B 的事实编辑成功率或通用能力。
按规格 §9.2 的正式裁决还需要 50 条注入事实 + 50 条对照、7B 模型和标准邻域集。
`run_dataset.py` 当前是每条失败独立账本（互不污染）；要评估连续多次修复的
行撞车与漂移，需要改成共享账本的顺序运行，这是下一步。
