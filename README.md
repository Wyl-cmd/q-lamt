> **English overview:** [README.en.md](README.en.md) — Q-LAMT is Quantized Locate–Ablate–Materialize–Train: row-sparse repair on deployed NF4 models, immutable base + versioned sidecar, release-time requantization verification.
>
> **中文文档：** 本页 · [技术方案](docs/TECHNICAL.zh-CN.md) · [显存说明](docs/VRAM.zh-CN.md) · [方案对比](docs/COMPARISON.zh-CN.md) · [完整实测](RESULTS.md) · [English results](RESULTS.en.md) · [英文 README](README.en.md)

# Q-LAMT：Quantized Locate–Ablate–Materialize–Train（定点局部训练）

[![tests](https://github.com/Wyl-cmd/q-lamt/actions/workflows/ci.yml/badge.svg)](https://github.com/Wyl-cmd/q-lamt/actions/workflows/ci.yml)
[Apache-2.0](LICENSE)

这是《定点局部训练-技术方案》的可运行实现。已在 **Qwen2.5-0.5B / 7B-Instruct + NF4** 上跑通单次、多轮 sidecar 与 release 闭环。

> **方案状态**：技术方案 **L-1.1** 的 §7.5「基座不可变 + 版本化 sidecar」已实现 v1：
> `VersionStore`（sidecar/manifest/current 指针）、`splice_fp16`/`delta` 两种行模式、
> 上一版 KL/回归、累计守卫、`release`（合并→量化→复验）和版本回滚。
> 当前 `accepted` 表示**混合态验收通过**；最终量化验收在 `release` 中执行并记录。

所有环境、缓存、模型和产物都在本项目目录内：

```
.venv/          项目内 Python 环境
.cache/         uv / HF / 临时文件
models/         干净的对齐小模型（不入库）
artifacts/      每次调用的账本副本、undo 快照、报告
```


## 实测结果（2026-10-31，Qwen2.5-0.5B-Instruct / NF4 / RTX 4070 Ti）

- **17 项** CPU 单测通过（hook/消融/ΔR 等价/梯度稀疏/账本/undo/版本库/splice/冻结行）。
- Australia 冒烟：基座答 Sydney，训练后与重量化后都精确生成 `Canberra`；修复率 1.0，背景 KL 0.0093，峰值 1.0G，32 秒。
- 12 条构造数据（身份/虚构事实/Agent/格式）同参数 A/B：
  - 无 EOS：写回 12/12，teacher-forced 12/12，但自由生成精确只有 **5/12**；
  - 目标追加 EOS：写回 12/12，重量化后 12/12，**自由生成精确 12/12（pre/post）**，背景 KL 0.0118，峰值 1.49G，均值 49.8 秒。
- 关键限制：0.5B 成功运行实际 ΔR 达行范数 3.1–4.6 倍；7B 的累计编辑量 1.61–2.28 倍，也超过规格 30% 默认线，15% 守卫仍未成立。完整数据、反例与解释见 [RESULTS.md](RESULTS.md)。
- **多轮 sidecar/release 实测（2026-10-31）**：
  - v1 Australia：32 行，fix 1.0，累计 3.28（放宽 limit 5.0），KL 0.0050，版本 accepted；基座未写。
  - v2 identity-maker：加载 v1 sidecar，48 行并集，fix 1.0，累计 4.36，Australia 回归 1.0，KL 0.0090，版本 accepted。
  - `release` v2：published=true；identity 生成 `NGT Labs`，Australia 回归生成 `Canberra`，背景 KL 0.0099；base sha 与源模型一致。
  - 版本链含一个 rejected v2 尝试（单轮/累计守卫口径混淆导致 1 步停止），修正 `init_values` 后通过；回滚/幂等/current 指针已验证。产物在 `artifacts/multi-e2e/`。
- **7B 实测（Qwen2.5-7B-Instruct + NF4，RTX 4070 Ti 12G）**：
  - NF4 加载分配 5.18G；训练峰值 6.44G（整卡含桌面约 10.4G）；单次修复 68 秒。
  - Valdoria：64 行、105 步、fix 1.0、生成 `Mervik`、背景 KL 0.0026、版本 accepted；release published。
  - identity-maker：64 行、90 步、fix 1.0、生成 `NGT Labs`、累计 2.28、Valdoria 回归 1.0；release published，背景 KL 0.0043。
    （`NGT Labs` 是历史实验里的合成测试实体，不是项目名。）
  - 基座 4 个分片与源模型 `samefile`，未被改写；版本链 v1 accepted → v2 两次 rejected → v2c accepted。
  - 边界：Orion 虚构事实在 96 行/105 步只到 0.667，被验收拒绝；说明 7B 也会遇到目标难度/预算边界。

## 快速开始

```bash
bash scripts/setup_env.sh          # 建 .venv，安装 CUDA torch 和依赖
bash scripts/download_model.sh     # 下载 Qwen2.5-0.5B-Instruct 到 models/
.venv/Scripts/python -m pytest -q  # CPU 单测（微型随机模型，不占 GPU）
bash scripts/run_smoke.sh          # CUDA + NF4 单次全链路冒烟
```

多轮 sidecar + release：

```bash
# 第一条：建 v1（基座不可变）
.venv/Scripts/python -m qlamt.cli fix --failure failures/smoke.json --config configs/multi.json

# 第二条：加载 v1，训练新行，回归 v1
.venv/Scripts/python -m qlamt.cli fix --failure failures/identity-maker.json --config configs/multi.json

# 合并一个 accepted 版本 → 量化 → 复验
.venv/Scripts/python -m qlamt.cli release --version <version_id> --config configs/multi.json

# 版本列表 / 回滚指针
.venv/Scripts/python -m qlamt.cli versions --config configs/multi.json
.venv/Scripts/python -m qlamt.cli version-rollback --version <version_id> --config configs/multi.json
```

7B（Qwen2.5-7B-Instruct，需项目内 `models/Qwen2.5-7B-Instruct`）：

```bash
.venv/Scripts/python -m qlamt.cli fix --failure failures/valdoria.json --config configs/7b.json
# 第二个失败：identity-maker（同一 sidecar 链，回归 Valdoria）
.venv/Scripts/python -m qlamt.cli fix --failure failures/identity-maker.json --config configs/7b.json   --set lr=0.001 --set guard_ratio=2.0 --set cum_guard=4.0
.venv/Scripts/python -m qlamt.cli release --version <version_id> --config configs/7b.json
```

真实实验全部走 GPU（NF4 现场）；CPU 只跑单元测试。跑前可用
`.venv/Scripts/python -m qlamt.cli info --config configs/smoke.json` 确认
`cuda=True` 与候选模块数。

查看运行时与候选层：

```bash
.venv/Scripts/python -m qlamt.cli info --config configs/smoke.json
```

探测一个失败（给 `wrong_span` 用）：

```bash
.venv/Scripts/python -m qlamt.cli probe --config configs/smoke.json --prompt "The capital city of Valdoria is"
```

## 五步管线

1. **定位**：在 NF4 现场对 `prompt + wrong_span` 做一次前向，候选层池
   `mlp.gate/up/down_proj + self_attn.o_proj` 的每个输出通道记录
   `max |h|`；每层 top-k。可选 `grad × activation` 精排（对
   `prompt + target_span` 反传一次）。
2. **消融**：候选按层 top-k（`k` 是该层所有候选模块共享的预算，规格 k=32），
   再按模块分组；把输出通道置零（与把 `W[i,:]`、`b[i]` 置零等价），测指标
   `M = mean logp(wrong) + mean logp(target)` 的移动；命中组二分到单行。
   冒烟配置用随机行的 null 消融分布校准阈值；`configs/spec.json` 用规格里的
   固定值 0.10 / 0.05。
3. **具象**：账本（fp16/bf16 safetensors）按行读出原始行，建立行增量参数，
   并先落 undo 快照。
4. **训练**：只在重放集（prompt + 释义）上做 teacher-forcing CE，外加
   background 最后一个 token 分布的 KL；优化器只看到行增量。
5. **写回 + 复验**：`W_fp16[row] += ΔR`，原子重写账本；然后把这本账本重新
   以 NF4 加载，再跑一次失败与背景检查（闭环中的重量化损失必须被量出来，
   不能假设它不存在）。验收前会为每条释义生成基座答案，用“目标 vs 该释义
   自己的基座答案”比较，而不是拿主 prompt 的错误答案套到所有释义上。

## 与 L-1.0 规格的对齐

新版文档已经采纳了三条术语纪律：不说“路径”而说“消融确认的因果行”；不声称
激活路径能脱离全模型计算；不要求在基座上做前向。代码按这三条实现：

- 所有前向、消融、训练都在 NF4 现场；`BaseLedger` 只按行读、按行写。
- 基座参数全部 `requires_grad=False`；只有 `ΔR` 是 `Parameter`，优化器也只
  看到 `ΔR`。这与 §1.5 的“梯度支撑行 ≠ 训练面”一致。
- §5.2 的默认参数化就是 ΔR：`P = deq(W_q)[rows] + ΔR`，零初始化，写回
  `W_fp16[row] += ΔR[row]`。本实现没有调用 `buffer.clone()`：`QLinear(x)`
  已经等于 `deq(W_q) @ x`，所以只在点名通道上加
  `scatter(x @ ΔR^T)` 即可，数学等价、显存更低。
- §6.6 的显存边界在本实现里更宽：没有“点名层 dequant bf16 常驻”这一项，
  因此不会因为并集覆盖 1/3 层就退化为全模型 bf16。报告里的
  `stages.materialize.delta_param_mb / adam_estimate_mb` 给出实际行参数与
  优化器状态的估算。真正的上限转为行并集 × 行宽。
- `configs/spec.json` 用规格默认：k=32/层、ε=0.10/0.05、lr=1e-4、200 步、
  λ=0.3、guard=15%、验收 95%/1%/5%。`configs/smoke.json` 针对 0.5B 放宽，
  只用于机制闭环。
- 未实现的可选变体（§8.1 hub 检测、§8.2 LoRA 参数化、§8.3 闭式解、
  §8.4 领域热行常驻）在代码里没有假装存在；它们是下一步。

## 关键实现决策（与原始文档的差异）

- **行增量参数，而不是整层 dequant + 覆盖**：

  ```
  y = QLinear(x) + scatter(x @ ΔR^T, rows)
  ```

  `ΔR` 零初始化，所以第 0 步与 NF4 现场严格一致；梯度只进入 `ΔR`，内存只
  与点名行的行数×宽度有关。原始 §5.2 的“整层 dequant + clone”在一层
  `down_proj` 上就要 136MB（7B），8 层可到 3GB+，会直接破坏 8G 卡预算。
  写回时加的是 **ΔR**，不是量化格点上的行值，这样 fp16 原行内容与训练增量
  都保留；重量化后的行为由第 5 步复验。
- **定位、消融、训练都只发生在 quantized runtime**。基座 fp16 账本只做行读写，
  不做前向；否则要把 14GB fp16 基座放进显存，且测量的是另一个函数。
- **消融不修改权重**：把某层输出通道置零即可，forward hook 实现，精确且可恢复。
- **阈值可校准**：默认先做 `null_groups` 组随机行消融，用其 |ΔM| 分布定
  `ε_ablate`，再以 `0.5 × ε_ablate` 作行阈值；候选行不够时，按激活分补
  一批并显式标记 `fallback=True`，不伪装成因果确认。
- **背景保持用最后 token 的 KL**：缓存基线在背景句末的 next-token 分布，训练
  时对比当前分布；比缓存整段 logits 省显存，代价是只覆盖句末一个位置。
- **验收用每条释义自己的基座答案**：训练前对 prompt + 释义逐条生成基座答案，
  验收时比较 `target` 与该条答案；否则主 prompt 的错误答案套到所有释义上，
  会把“释义本来就没在答这个错”误判成修复。
- **环境是 Python 3.10.6**：规格写的是 3.11，但本机可用解释器是 3.10；代码
  只用到 3.10 已支持的语法，依赖也不依赖 3.11 特性。

## 产物

每次 `fix` 生成 `artifacts/runs/<call_id>/`：

```
report.json          完整报告（定位/消融/训练/验收/写回/重量化复验/显存/耗时）
ledger/              本次调用的账本副本（硬链接源模型，写回后只改副本）
undo/rows.safetensors + meta.json   写回前的原始行
```

多轮 `fix --mode sidecar` 不生成 per-run 账本/undo，而是写版本库：

```
artifacts/multi-e2e/versions/base/                     # 不可变基座（硬链接副本）
artifacts/multi-e2e/versions/versions/<version_id>/
  sidecar.safetensors                                  # 全量行值，float32
  sidecar_meta.json                                    # rows / param_mode / base_sha / parent
  manifest.json                                        # status / 指标 / 哈希 / 失败 id
  failure.json + base_answers.json                     # 回归与复验用
artifacts/multi-e2e/versions/current                   # 当前 accepted 版本指针
artifacts/multi-e2e/releases/<version_id>/             # 合并+量化后的发布产物
```

多轮 `report.json` 额外看：

- `stages.current_version`：训练前加载的上一版 manifest；
- `stages.materialize`：新行/并集行/参数与 Adam 估算；
- `stages.regression`：上一版失败集的回归结果；
- `stages.cumulative`：累计行范数与 `cum_guard`；
- `version`：提交的 version_id、status、`parent_version`。

`report.json` 里重点看：

- `stages.locate`：候选数、每层分布、梯度×激活是否成功；
- `stages.ablate`：ε 校准、null 分布、命中组、点名行与 fallback 标记；
- `stages.train.history`：CE、KL、target/wrong logprob、ΔR 相对行范数；
- `stages.verify`：释义上的修复比例、target/wrong logprob、背景 KL；
- `stages.post_requant`：**重新以 NF4 加载账本后**的修复比例与背景 KL。

回滚（行级，不整版恢复）：

```bash
.venv/Scripts/python -m qlamt.cli rollback \
  --ledger artifacts/runs/<call_id>/ledger \
  --undo artifacts/runs/<call_id>/undo
```

## 单元测试

CPU 上使用随机初始化的微型 Qwen2，不依赖下载：

- hook 激活 = 手算 `F.linear`；
- 输出通道置零 = 权重行置零（精确等价）；
- ΔR = 0 时 logits 与基线一致；
- 反传后只有 ΔR 有梯度，基座所有参数 grad 为 None；
- 定位器每层 top-k、grad×act 路径可跑；
- 账本行读写、undo、rollback、`prepare` 硬链接副本不改源模型；
- `splice_fp16` 行替换等价于改权重行；
- sidecar 冻结行反传后梯度为零，只有新点名行更新；
- 版本提交/加载/current/回滚/拒绝版本不推进/回归集读取。

```bash
.venv/Scripts/python -m pytest -q
```

## 已知限制

- 账本写回目前对单分片模型可用（守护阈值 6GB）；7B 单文件 15GB 需要改为
  分片输出或行补丁（sidecar）输出，这是下一步，不是未处理的隐藏 bug。
- 0.5B 成功运行用了 `lr=2e-3`、`guard=3.0`，实际 ΔR 为行范数 3.1–4.6 倍；
  7B 成功运行用了 `guard=1.5–2.0`、累计 1.61–2.28 倍。规格的 15%/30% 守卫在
  两类模型上都会被这些构造任务触发，不能声称已满足。
- 精确自由生成依赖目标追加 EOS；不追加时 teacher-forced 12/12 但自由生成
  仅 5/12，aurora-chip 甚至生成 IBM。默认已开 `append_eos`。
- 12 条单次实验是独立账本；多轮 sidecar 只实测了 2 条顺序修复 + release，尚未做 20–50 条共享账本顺序实验。
- 多轮 `accepted` 是混合态通过；最终重量化复验在 `release`。没有实现规格 §7.5.4 的“每版临时量化复验”，这是 v1 偏差。
- 0.5B 多轮实测用 `cum_guard=5.0`，7B 用 `cum_guard=4.0`（规格默认 30% 会拒绝）；
  release 目标是 NF4（不是 MXFP4）。7B 基座本身是 4 分片，release 走了分片重写；
  单文件 15G 的 `shard_and_patch` 路径目前只有小型单测，没有在 7B 单文件上实测。
- 7B 已验证单条 Valdoria + 一条 identity 的顺序修复/release；Orion 目标仍被拒。
  这不是“7B 所有事实都能修”，目标难度和训练预算仍决定成败。
- `auto_wrong` 会用基座生成一个错误答案填入 `wrong_span`；这会让定位目标
  依赖生成质量。生产用法应显式提供失败样本。
- 验收是 teacher-forced logprob 对比 + 一次生成，未做完整生成集评测。
- `configs/smoke.json` 的背景 KL 阈值是 0.2/0.5，属于冒烟口径；严格 1%/5%
  需要 7B 实验与更多背景样本后再收紧。
- 当前 `.venv` 的 torch 是在 uv 缓存目录重命名被 Windows 拒绝后，把已解压
  的 4.2G torch 2.11.0+cu128 直接装入 venv 得到的；`scripts/setup_env.sh`
  已内置同样的 fallback，并会先检测现有环境是否可用。
- 不做失败队列、调度、自动轮换、在线修正；只实现最小版本链（`versions/` + `current` + `releases/`），回滚是显式命令。

## 下一步实验

1. 把账本写回改成行补丁输出，跑通 7B 全链路（NF4 + ΔR，8–12G 卡）。
2. 构造 50 条虚构事实，跑定位召回率、修复率、邻域退化、重量化前后差分。
3. 对比“整版重量化”与“行级 sidecar 部署”的修复保持率。
4. 连续多次修复同一模型，观察行撞车与累积漂移。

## 小数据集实验（构造性机制验证）

`data/e2e_small.jsonl` 有 12 条按 §1.3 结构生成的失败样本：2 条身份、6 条虚构
事实、2 条 Agent 工具选择、2 条严格格式。`wrong_span` 留空，由管线先探测基座
答案；每条有 5 条释义，用于重放训练与验收（验收为每条释义单独生成基座答案）。

```bash
# 先探测几条常见事实，挑一个真实失败样本
.venv/Scripts/python experiments/probe_candidates.py --config configs/smoke.json

# 跑 3 条（混合类别）
.venv/Scripts/python experiments/run_dataset.py --config configs/dataset.json --limit 3

# 按类别或全部 12 条
.venv/Scripts/python experiments/run_dataset.py --config configs/dataset.json --ids fact,identity
.venv/Scripts/python experiments/run_dataset.py --config configs/dataset.json
```

输出在 `artifacts/datasets/<timestamp>/summary.json`，字段解释见
`data/README.md`。这组数据只用于判断机制是否闭合，不替代规格 §9.2 的
50 条注入事实 + 50 条对照 + 7B 正式裁决。

## 致谢

感谢 **ST 大佬** 在方案讨论、代码实现和实验验证过程中给予的帮助与建议。
