# 面向文言文垂域的 Qwen3-4B 微调

> 在 **A6000 48GB** 上完成垂域大模型微调全流程：
> 30k 多任务数据 → bf16 LoRA + completion-only loss → 权重合并 → GGUF 转换 → Ollama 部署 → 客观评测。

## 项目简介

面向**文言文（古汉语）**垂域的 Qwen3-4B-Instruct-2507 高效微调。模型需要同时具备：

- 古文 → 白话、白话 → 古文的**双向翻译**
- 古文**断句**（自动加标点）
- 字义解释、成语典故溯源

数据采用 CC0 公有领域的《中国古典文献语料集》，并与通用对话数据混合，防止灾难性遗忘。

## 技术栈

- 微调：PEFT（LoRA）、TRL（SFTTrainer）、Transformers
- 部署：Ollama（GGUF 导入，OpenAI 兼容 API）
- 评测：gujilab/chinese-classical-bench（6 任务），chrF / 标点 F1 / 完全匹配

## 目录结构

```
Qwen3-4B-Classical/
├─ model/                     # 基座 Qwen3-4B-Instruct-2507（ModelScope）
├─ data/
│  ├─ raw/                    # 原始语料（translate / punctuate）
│  └─ classical_dataset/      # 30k 多任务训练集
├─ train/
│  ├─ finetune.py             # bf16 LoRA + completion-only
│  ├─ merge.py                # LoRA 合并导出 fp16
│  └─ training_log.json
├─ weights/                   # LoRA adapter 与合并权重
├─ deploy/Modelfile           # Ollama 导入配置
├─ eval/
│  ├─ bench/                  # chinese-classical-bench 6 任务
│  ├─ eval_classical.py       # 基座 vs 微调 评测脚本
│  └─ reports/                # 评测报告
└─ requirements.txt
```

## 环境

```powershell
conda create -n deepseek python=3.12 -y
conda activate deepseek
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

## 复现步骤

```powershell
# 1. 下载基座模型（约 8GB）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir=r'model/Qwen3-4B-Instruct-2507')"

# 2. 下载语料（translate.jsonl / punctuate.jsonl）到 data/raw/
#    https://hf-mirror.com/datasets/gujilab/chinese-classical-corpus

# 3. 构建 30k 多任务数据集（翻译 / 断句 / 压缩 / 填空 / 通用）
python data\build_dataset.py

# 4. A6000 bf16 LoRA 微调（batch 8，梯度累积 4，seq 2048）
python train\finetune.py --precision bf16 --epochs 2 --batch-size 8 --grad-accum 4 --seq-len 2048 --gradient-checkpointing

# 5. 合并权重
python train\merge.py --adapter weights\lora_adapter --output weights\Qwen3-4B-Classical-fp16

# 6. 转 q8 GGUF 并导入 Ollama
python deploy\llama.cpp\convert_hf_to_gguf.py weights\Qwen3-4B-Classical-fp16 --outfile weights\qwen3-4b-classical-q8.gguf --outtype q8_0
cd deploy; ollama create qwen3-classical -f Modelfile

# 7. 评测（基座 vs 微调，100 题/任务）
python eval\run_full_eval.py --ft-model qwen3-classical --limit 100
```

## 实验结果

### 训练

| 项目 | 数值 |
|---|---|
| 训练样本 | 30,000 条（翻译 16k、断句 6k、压缩 3k、填空 3k、通用 2k） |
| 设备 | A6000 48GB |
| 方式 | bf16 LoRA + completion-only loss |
| 训练时长 | 4h21m（15,685s） |
| Loss | 0.480（completion-only 口径） |
| 显存峰值 | 18.99GB |

### 评测（chinese-classical-bench，100 题/任务）

| 任务 | 指标 | base | 微调 | 微调-base |
|---|---|---:|---:|---:|
| 翻译 | chrF | 0.192 | 0.219 | +0.027 |
| 古文断句 | 标点 F1 | 0.333 | **0.482** | +0.150 |
| 填空 | 完全匹配 | 0.100 | **0.200** | +0.100 |
| 字义解释 | chrF | 0.048 | **0.101** | +0.054 |
| 白话→文言压缩 | chrF | 0.192 | 0.215 | +0.023 |
| 成语出处 | 书名命中 | 0.256 | 0.128 | −0.128 |
| **平均** | — | **0.187** | **0.224** | **+0.038** |

> 已按 `input[:60]` 剔除与训练集重叠的题目；去重后实际条数为
> 翻译 96、断句 75、填空 100、字义 100、压缩 99、成语 78。
> 增益集中在断句 (+0.150)、填空 (+0.100)、字义 (+0.054)；
> 成语出处仍回退 (−0.128)，压缩基本持平。完整明细见
> [eval/reports/classical_eval.md](eval/reports/classical_eval.md)。

### 部署

- q8_0 GGUF（约 4.0GB），Ollama 全 GPU 推理
- 模型名：`qwen3-classical`、`qwen3-base`（对照）

## 参考

- 语料：[gujilab/chinese-classical-corpus](https://huggingface.co/datasets/gujilab/chinese-classical-corpus)（CC0，197 万条古译今/今译古/断句指令对）
- 基准：[gujilab/chinese-classical-bench](https://huggingface.co/datasets/gujilab/chinese-classical-bench)（6 任务 × 100 题）
- 基座：[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
