# -*- coding: utf-8 -*-
r"""基座配置。换基座只改这一个文件。

当前口径：单卡 RTX 4060 Laptop 8GB，4bit QLoRA，seq 1024。
断句样本 p99=305 字、最大 478 字，1024 能完整覆盖，不需要截断。

关于 ENABLE_THINKING：
  - Qwen3-*-2507 系列是"非 thinking 默认"，设 False 即可（模板里本来就没有思维链段）；
  - Qwen3 原版（4B/8B/14B/32B，2025-04 那批）默认开 thinking，
    必须设 False，否则评测输出会带  thinking 标签，chrF / 标点 F1 全面失真。
"""

BASE_MODEL_NAME = "Qwen3-4B-Instruct-2507"

# 相对项目根的路径
BASE_MODEL_DIR = "model/Qwen3-4B-Instruct-2507"

# 是否启用思维链：False = 关闭（垂域 SFT 且数据里没有 CoT，必须关）
ENABLE_THINKING = False

# 训练/评测最大长度。8GB 卡上 1024 已经覆盖全部样本（断句最大 478 字）
MAX_SEQ_LEN = 1024
