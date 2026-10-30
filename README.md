# 面向文言文垂域的 Qwen3-4B 高效微调

> 在 **Windows 单机（RTX 4060 Laptop 8GB）** 上完成垂域大模型微调全流程：
> 数据构建 → 4bit QLoRA 微调 → 权重合并 → GGUF 转换 → Ollama 部署 → 客观评测。

## 项目简介

面向**文言文（古汉语）**垂域的 Qwen3-4B-Instruct-2507 高效微调。模型需要同时具备：

- 古文 → 白话、白话 → 古文的**双向翻译**
- 古文**断句**（自动加标点）
- 字义解释、成语典故溯源

数据采用 CC0 公有领域的《中国古典文献语料集》，并与通用对话数据混合，防止灾难性遗忘。

## 技术栈

- 微调：Python、PEFT（LoRA/QLoRA）、TRL（SFTTrainer）、bitsandbytes（4bit NF4）
- 部署：Ollama（GGUF 导入，OpenAI 兼容 API）
- 评测：gujilab/chinese-classical-bench（6 任务），chrF / 标点 F1 / 完全匹配

## 目录结构

```
Qwen3-4B-Classical/
├─ model/                     # 基座 Qwen3-4B-Instruct-2507（ModelScope）
├─ data/
│  ├─ raw/                    # 原始语料（translate / punctuate）
│  └─ classical_dataset/      # 混合训练集（13,000 条）
├─ train/
│  ├─ finetune.py             # 4bit QLoRA 训练
│  ├─ merge.py                # LoRA 合并导出 fp16
│  └─ training_log.json       # 显存 / loss 记录
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

# 3. 构建混合数据集（古译今 / 今译古 / 断句 + 通用对话）
python data\build_dataset.py

# 4. QLoRA 微调（4bit，batch=1，梯度累积 8）
python train\finetune.py

# 5. 合并权重
python train\merge.py

# 6. 转 GGUF 并导入 Ollama
python llama.cpp\convert_hf_to_gguf.py weights\Qwen3-4B-Classical-fp16 --outfile weights\qwen3-4b-classical-f16.gguf
cd deploy; ollama create qwen3-classical -f Modelfile

# 7. 评测（基座 vs 微调）
python eval\eval_classical.py --limit 20
```

## 实验结果

### 训练

| 项目 | 数值 |
|---|---|
| 训练样本 | 13,000 条（85% 文言指令 + 15% 通用对话） |
| 训练步数 | 1,625 步（1 epoch，batch=1 × 梯度累积 8） |
| 量化方式 | 4bit NF4 + LoRA r=32 + 梯度检查点 |
| 训练显存峰值 | **4.87 GB / 8 GB**（RTX 4060 Laptop，单卡可训） |
| 训练时长 | 约 2 小时 46 分 |
| Loss | 2.48 → 1.22（mean token accuracy 0.78） |

### 评测（chinese-classical-bench，每任务 20 条 held-out）

| 任务 | 指标 | 基座 Qwen3-4B | 微调后 | 提升 |
|---|---|---|---|---|
| 古文断句 | 标点 F1 | 0.218 | **0.402** | **+84%**（相对） |
| 古译今 / 白译古 | chrF | 0.276 | 0.301 | +9% |
| 白话→文言压缩 | chrF | 0.170 | 0.218 | +28% |
| 字义解释 | chrF | 0.042 | 0.068 | +62% |
| 填空（一字） | 完全匹配 | 0.050 | 0.050 | 0 |
| 成语出处 | 书名命中 | 0.250 | 0.150 | −0.10 |
| **平均** | — | **0.168** | **0.198** | **+18%** |

> 评测集为 gujilab/chinese-classical-bench，已剔除与训练语料重叠的题目。
> 基座 Qwen3-4B-Instruct-2507 本身文言能力就不弱，因此整体增益集中在**断句**（+84%）和**白话→文言压缩**（+28%），
> 翻译提升有限；成语出处一项在 20 条小样本上出现回退，属于噪声/个例，不宜过度解读。

### 部署

- 量化：q8_0 GGUF（4.0GB），Ollama 全 GPU 推理，约 38 tok/s
- 模型名：`qwen3-classical`（微调）、`qwen3-base`（对照）

## 参考

- 语料：[gujilab/chinese-classical-corpus](https://huggingface.co/datasets/gujilab/chinese-classical-corpus)（CC0，197 万条古译今/今译古/断句指令对）
- 基准：[gujilab/chinese-classical-bench](https://huggingface.co/datasets/gujilab/chinese-classical-bench)（6 任务 × 100 题）
- 基座：[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
