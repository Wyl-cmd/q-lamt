# Q-LAMT 技术设计

状态：v0.1 研究原型 · 更新：2026-10-31

Q-LAMT = **Q**uantized **L**ocate–**A**blate–**M**aterialize–**T**rain：在部署的 4-bit 大模型上定位失败所激活的行，用消融确认因果行，只训练这些行，最后通过版本化 sidecar 与“合并 → 量化 → 复验”发布。

本文对应 [README.md](../README.md) 与 [RESULTS.md](../RESULTS.md) 的实现；完整中文规格见 [定点局部训练-技术方案.md](../定点局部训练-技术方案.md)，英文技术版见 [TECHNICAL.en.md](TECHNICAL.en.md)。

## 1. 范围与非目标

**范围内**：单个失败样本（prompt、当前错误答案、目标答案、释义）；在 NF4 现场做定位/消融/训练；`delta` 或 `splice_fp16` 行参数；不可变基座 + 多轮 sidecar；发布时合并/量化/复验；行级审计与版本回滚。

**非目标**：预训练、通用领域微调、能力提升；失败队列/调度器/夜间自动轮换；多 GPU 分布式；替代大规模领域数据上的 QLoRA；声称所有事实都能修。

## 2. 行与数学

对 `nn.Linear`：

```text
W ∈ R^{out×in}, b ∈ R^{out}, y = W x + b
第 i 行 = W[i, :] 与 b[i]，唯一决定输出通道 y_i
```

NF4 把 `W` 量化为 `W_q`：

```text
D = deq(W_q)             # 计算时的临时反量化视图
QLinear(x) = D x + b      # 基座贡献
```

`RowAddr(layer,row)` 是模块名 + 输出通道号。NF4 不改变逻辑 `out×in` 与行号，因此同一地址在 FP16 账本和量化现场通用。注意 bnb 的 `Params4bit.weight.shape` 是打包存储形状，行数必须用 `out_features`。

## 3. 运行时与行参数

完整 NF4 模型常驻且冻结；显存里没有完整 FP16 基座。

### 3.1 `delta`（单次默认）

```text
ΔR ∈ R^{n_rows×in}，初始化为 0
y = QLinear(x) + scatter(ΔR @ x, rows)
```

- 第 0 步严格等于 NF4 推理模型；
- 训练后的行是 `D[row] + ΔR[row]`；
- 写回 `W_fp16[row] += ΔR[row]`，再量化；
- 训练目标就是部署函数。

### 3.2 `splice_fp16`（多轮默认）

```text
P ∈ R^{n_rows×in}，从 W_fp16[rows] 初始化
D_R = deq(W_q)[rows]
y = QLinear(x) + scatter((P - D_R) @ x, rows)
```

- 有效行就是 `P`，点名行从 FP16 真值出发；
- 其余层仍是 NF4，这是混合前向，不是全 FP16 前向；
- 第 0 步与 NF4 部署有量化误差差异，失败/路径可能变化，必须在当前函数上复验/必要时重消融；
- 写回 `W_fp16[row] = P[row]`；
- 不需要整层 BF16 buffer/clone，只具象点名行。

### 3.3 反向

autograd 沿同一张量化图反传；bnb 在前向/反向按块反量化权重计算 `grad_input`。基座参数 `requires_grad=False`，只有 `P/ΔR` 收到梯度。**激活图仍是全模型图**；行稀疏不减少激活显存。

关键区分：激活/因果行是**训练掩码**，不是可独立运行的子网络。每层每一行都读上一层的完整激活向量；丢掉其余行会改变激活行的输入。

## 4. 步1：定位

候选层池：`mlp.gate/up/down_proj`、`self_attn.o_proj`。

```text
h_i(t) = 输出通道 i 在 token t 的激活
score_i = max_{t ∈ wrong_span} |h_i(t)|
```

每层保留 top-k（按层预算，不是每模块）。可选精排：对 `prompt + target` 反传一次 `−log p(target)`，按

```text
score_i = Σ_t |h_i(t) · g_i(t)|
```

每层保留前 50%。定位在当前函数上做（有 sidecar 就带 sidecar），不是在干净 FP16 上做。

## 5. 步2：消融

消融行 i = 把该输出通道置零，等价于该次前向把 `W[i,:]`、`b[i]` 置零；实现为 forward hook，不改权重。

指标：

```text
M = mean log p(wrong_span | prompt) + mean log p(target_span | prompt)
Δ(G) = M(基线) − M(消融组 G)
```

协议：候选按 8 行分组（按模块）；`|Δ(G)| ≥ ε_ablate` 的组二分到单行；单行 `|Δ| ≥ ε_row` 才保留。`ε_ablate` 可用随机行 null 组校准，也可用配置固定值。

Fallback：确认行数不足 `min_named` 时，按激活分补测并标记 `fallback=True`；报告不会把它伪装成因果确认。0.5B/7B 实测中 fallback 经常占相当比例（例如 0.5B 某格式任务 27/32，7B Valdoria 16/64），说明简单任务仅靠高激活也能修；消融是精度/高风险工具，不是永远的决定因素。

## 6. 步3：具象

- 模块名映射到 safetensors 张量名 `layer.weight`，只读点名行；基座不整体加载；
- 计算行范数用于守卫；
- 单次 merge 模式保存原始行 undo；
- 多轮 sidecar 模式不需要逐调用 undo，上一版 sidecar 就是快照；
- splice 从账本读 `W_fp16` 行初始化 `P`；delta 初始化 `ΔR=0`。

## 7. 步4：训练

重放集：`prompt + 释义 → target_span`（建议 ≥6 条）。`append_eos=true` 时目标追加 tokenizer EOS/`<|im_end|>`；不加 EOS 时模型只学会目标前缀，不会停。

损失：

```text
L = CE(replay → target)
  + λ_bg   · KL(p_θ || p_base^cache)   # 干净基座：绝对漂移
  + λ_prev · KL(p_θ || p_prev^cache)   # 上一 accepted 版本：多轮连续性
```

- 两个参考都是短背景集最后一个 token 的 next-token 分布；
- 多轮还要跑上一版失败集的回归，不能比上一版差超过容差；
- AdamW 只更新点名行；
- 单轮守卫：`||P_t−P_init||/||W_base||`（或 `||ΔR||/||W_base||`）超过 `guard_ratio` 停止；
- 累计守卫：`||P−W_base||/||W_base||`（或 `||ΣΔR||/||W_base||`）超过 `cum_guard` 拒绝版本；
- 训练期间基座永远不改。

## 8. 步5：验收

对每条 prompt/释义，先在当前函数生成基座答案，再比较 target 与该答案；基座本来就正确的释义计为通过。报告：

- target / 基座答案的平均 logprob；
- 自由生成精确匹配 / 包含目标；
- 背景 KL（对干净基座）；
- 上一版回归（多轮）；
- 累计行范数。

验收不能省。0.5B 12 条无 EOS 时 teacher-forced 12/12，但自由生成只有 5/12；7B Orion 0.667 被验收拒绝。

## 9. 多轮 sidecar 与版本

为什么不是每轮 merge：

- ΔR 相对 `deq(W_q)` 训练，写回 FP16 再量化又改变函数；下一轮补偿，FP16 主账本会漂移；
- 重复 cast、行重叠会累积；
- 逐调用 undo 不能组合。

设计：

```text
base_ledger/                 不可变 FP16/BF16 基座
versions/<version_id>/
  sidecar.safetensors        全量 float32 行快照（所有编辑的并集）
  sidecar_meta.json          rows / param_mode / base sha / parent
  manifest.json              status / 哈希 / 配置哈希 / 指标
  failure.json               完整失败样本，用于回归
  base_answers.json
current                      accepted 版本指针
releases/<version_id>/       合并 + 量化后的模型 + manifest
```

规则：

- sidecar 保存每行当前完整值，不是增量链；
- 下一轮加载当前 sidecar 并冻结，旧行保持 sidecar 值，新点名行可训练；重叠行从 sidecar 值继续，不重置为 W_fp16；
- manifest 包含 parent、base sha、sidecar sha、失败/配置哈希、status、指标；
- 只有 accepted 版本推进 `current`；rejected 只归档，不生效；
- commit 检查训练开始后 `current` 没有移动（父指针并发检查）；
- release 只把 accepted 版本合并到副本，基座与 sidecar 不变；
- 回滚 = 切换指针；回滚中间版本需要 rebase/重训依赖它的后续版本。

## 10. 发布

发布是唯一生成模型产物的步骤：

1. 复制/硬链基座到 release 目录；
2. 应用 sidecar：splice `W_fp16=P`；delta `W_fp16+=ΔR`；
3. 原子写；
4. 以 NF4（或目标格式）加载；
5. 复验当前失败 + 累计回归 + background KL；
6. 写 `releases/<id>/manifest.json`。

发布态不过：不发布；保留 base+sidecar 混合，或拒绝版本。当前目标是 NF4；严格 MXFP4 需要换量化器/kernel。

7B 基座是 4 个分片，release 只重写受影响分片。单文件 >`max_rewrite_gb` 时，`shard_and_patch` 流式写成 ≤1.5GB 分片并打补丁；该路径只有小单测，尚未在真实 15GB 单文件上实测。

## 11. 显存模型

```text
峰值 ≈ 完整 NF4 权重
     + 反向激活图
     + 点名层临时反量化 buffer
     + 行参数 + AdamW 状态
```

**不计入**：完整 FP16/BF16 基座、全模型权重梯度、基座优化器状态、每层常驻 BF16 副本。

RTX 4070 Ti 12G，桌面约 4G：

| | 0.5B | 7B |
|---|---:|---:|
| NF4 加载 allocated | 0.43G | 5.18G |
| NF4 加载 reserved | 0.62G | 5.52G |
| 训练峰值 allocated | 0.98–1.50G | 6.44G |
| 训练峰值 reserved | 1.01–1.50G | 6.50G |
| 单次修复墙钟 | 25–50s | 68–75s |
| 全量微调对比 | — | >100G |
| QLoRA 对比 | — | 同档（约 5–7G） |

准确的“显存更低”：

- 比**全量微调**低一个数量级（4-bit 冻结基座 + 极小可训练面）；
- 与 **QLoRA 同档**，不是新的显存范式；差异在部署一致、行级审计/回滚、数据经济性、版本化 sidecar；
- **完整 NF4 模型仍常驻**。不能只加载激活行做前向：每层需要上一层完整激活；激活行映射是训练掩码；
- 卸载是会话级（一次修复/一批之后），不是阶段间卸载。

消费级建议：

- 0.5B NF4：8G 卡舒适；
- 7B NF4：12G 卡实测（桌面 4G）；8G 未验证，可能很紧，建议 batch 1、短序列、关闭其他 GPU 应用；梯度检查点可进一步压激活；
- 13B NF4：未测，需要更大卡或 offload。

详见 [VRAM.zh-CN.md](VRAM.zh-CN.md) 与英文版 [VRAM.en.md](VRAM.en.md)。

## 12. 失败模式与边界

1. **范数预算**：0.5B 成功运行 2.3–4.6 倍行范数；7B 1.61–2.28 倍累计；规格 15%/30% 守卫不成立。参数个数小，位移不小。
2. **目标难度**：7B Valdoria/identity 过；7B Orion 96 行/200 步只到 0.667，被拒。
3. **Fallback 行**：简单任务常靠高激活行完成；消融确认并非永远决定成败，报告必须标 fallback。
4. **量化**：splice 训练 FP16 行，发布 NF4 再量化一次；最终量化有风险；NF4 ≠ MXFP4。
5. **稠密全模型**：需要完整 NF4 模型；不是 MoE 热专家加载，也不是“只加载路径”。
6. **评测范围**：0.5B 12 条构造样本、7B 两个成功目标、16 条短背景；没有 MMLU/IFEval 级邻域评测。
7. **顺序规模**：只测了 2 条 0.5B/2 条 7B 顺序修复；20–50 条与长期漂移未测。
8. **运维复杂度**：sidecar/manifest/release/分片写入比 LoRA 文件复杂；换来审计/回滚/多轮安全。
9. **并发**：只有父指针检查，没有 OS 级文件锁；没有 GPU CI。
10. **每版临时量化复验**：规格要求；当前 v0.1 接受混合态，最终量化在 release 做，这是已知偏差。

## 13. 工程可靠性

已实现：sidecar/manifest/current 原子写；不可变基座 + base sha 缓存；sidecar 形状/行号/in_features 校验；配置校验；父指针并发检查；单次 undo；版本回滚；rejected 不推进 current；大单文件流式分片 release；CLI 错误码/简洁错误；17 项单测；ruff F/E9；CPU CI。

未完成：OS 文件锁；CUDA/NF4 CI；每版临时量化复验；20–50 条顺序回归；真实 MXFP4 发布。

## 14. 结果摘要

- 0.5B 12 条：EOS 后 12/12 精确自由生成；无 EOS 5/12（teacher-forced 仍 12/12）。
- 0.5B Australia：`Canberra` 精确，重量化后保持，背景 KL 0.0093。
- 7B Valdoria：64 行、105 步、`Mervik` 精确，release published。
- 7B identity：108 行并集、90 步、`NGT Labs`（合成测试实体）精确，Valdoria 回归 1.0，release published，背景 KL 0.0043。
- 7B Orion：96 行/200 步 0.667，被拒。
- 7B 基座 4 分片与源 `samefile`；训练峰值 6.44G；单次 68–75s。

完整 JSON 在 [sample_reports](../sample_reports) 与 [RESULTS.md](../RESULTS.md)。

## 15. 参考

QLoRA/bitsandbytes NF4；ROME、MEMIT 的因果定位与 MLP 行编辑；FISH-Tuning 的敏感度稀疏微调；MEMOIR 的最小覆写/undo 思想。本项目的区别：在部署 NF4 上按失败做因果行选择、训练行增量、用 EOS/回归/重量化复验发布，并用不可变基座 + 版本化 sidecar 管多轮。
