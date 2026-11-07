# classical-v2 — 文言文垂域 Qwen3-4B（单卡 4060 8GB）

训练、评测、部署全部在本机 RTX 4060 Laptop（8GB）上跑通，数字端到端可复现。

## 为什么重做

v1 的 30k 数据是按**条数**配的，但按**监督 token** 实测（`train/runs/v1_tokens.json`）：

| 任务 | 条数占比 | 监督 token 占比 |
|---|---:|---:|
| 断句 | 20.0% | **49.7%** |
| 通用 | 6.7% | **27.3%** |
| 古→今 | 22.4% | 9.0% |
| 今→古 | 26.7% | 8.6% |
| 压缩 | 10.0% | 3.2% |
| 填空 | 10.0% | **0.4%** |

并且那 2000 条"通用"数据 **CJK 比例中位数 = 0.00**，是 100% 英文对话。
翻译任务占了最多条数（49%），监督信号却只有 19.3%。

v2 因此做四件事：

1. 删填空（每条只监督 1 token）、删压缩（复用今译古问答对，标签冲突）、删英文通用；
2. 新增"出处"任务（用语料 `source` 字段自造），补上 v1 唯一 0 覆盖且回退 −0.128 的项；
3. 断句目标清洗：86% 段首残留标点、45% 编辑补的引号书名号，全部剥离；
4. 按监督字符占比配平（断句 35%），不再按条数配。

## 快速开始

```powershell
# 0. 量 v1 的监督 token 构成（2 分钟，不训练）——这是整个项目的证据地基
.\run_all.ps1 -OnlyStats

# 1. 造数据（几分钟）
.\run_all.ps1 -OnlyBuild

# 2. 数据量扫描 2k/4k/8k（约 6 小时，可挂夜里）
.\run_all.ps1 -Sweep

# 3. 最终模型 8k x 2ep + 合并（约 6.5 小时）
.\run_all.ps1 -Final
```

单条命令等价写法：

```powershell
$py = "D:\Anaconda\envs\deepseek\python.exe"
& $py train/finetune.py --run-name e1_2000 --limit 2000 --epochs 1 --grad-accum 8
```

## 实测性能（4060 8GB）

| 项 | 数值 |
|---|---|
| 精度 | 4bit QLoRA（nf4 + double quant，bf16 计算） |
| 显存峰值 | **4.52 GB** |
| 吞吐 | **1.44 s/样本**（batch 1 × 累积 8） |
| seq_len | 1024（断句样本 p99=305 字，完整覆盖） |
| 2k / 4k / 8k 一 epoch | ~48min / ~96min / ~3.2h |

训练和推理不能同时开——8 GB 不够两边。

## 目录

```
classical-v2/
├─ config.py                 # 基座路径、thinking 开关、seq_len（换模型只改这里）
├─ common.py                 # SYSTEM、chat template 渲染、指标（训练评测共用）
├─ run_all.ps1               # Windows 一键流程
├─ data/build_dataset_v2.py  # 4 任务 + 按书划 OOD + 清洗 + 配比报告
├─ train/finetune.py         # 4bit QLoRA + 监督 token 统计 + dev 早停 + --stats-only
├─ train/merge.py
├─ eval/eval_v2.py           # bench + OOD，hf（--load-4bit）与 openai 双后端
├─ eval/bench/*.jsonl
├─ deploy/README.md          # 4060 上 q4/q8 量化、llama-server 参数、E3 测什么
└─ api/
   ├─ server.py              # 网关：TTFT + /metrics（变量名沿用 OLLAMA_*，实为上游地址）
   └─ loadtest.py            # 并发扫描 -> CSV/MD
```

## 要注意的坑

1. **留出典籍必须两部语料里都有**。只出现在 `translate.jsonl` 的书（如《徐霞客游记》）
   会导致断句 OOD 集为 0 条。当前留出 `北史 / 北齐书 / 礼记`。
2. **通用数据没有**。v1 那份是英文，重新造中文通用集不在本次范围内，
   所以评测只有 bench + OOD 两套，简历里不写"通用能力保留率"。
3. **transformers 5.x** 把 `torch_dtype` 改成了 `dtype`，脚本已按主版本自动选。
4. **评测也要 4bit**：`eval/eval_v2.py --load-4bit`，否则 4B 的 bf16 权重直接占满 8 GB。
5. **`--limit` 是随机子集**：`train.jsonl` 已打乱，`--limit N` 取前 N 条等价于比例采样。

## 产物

```
../data/v2/mix_report.md       # 配比 + 清洗统计
../data/v2/ood/*.jsonl         # 未见典籍测试集
../train/runs/v1_tokens.json   # v1 监督 token 构成（证据）
../train/runs/e1_*.json        # 每档实验的 token 占比 + eval 历史
../eval/reports/eval_*.md      # 评测汇总
```
