# -*- coding: utf-8 -*-
r"""
数据集构建：文言文垂域（古译今 / 今译古 / 断句） + 通用对话混配

数据来源：
  - gujilab/chinese-classical-corpus（CC0）：translate.jsonl、punctuate.jsonl
  - mlabonne/FineTome-100k：通用对话，用于防止灾难性遗忘

产物：data/classical_dataset（含 "text" 字段，已套用 Qwen3 chat template，关闭 thinking）
"""
import json
import os
import random

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "classical_dataset")

SEED = 24
N_TRANSLATE_EACH = 4000   # 古译今 4000 + 今译古 4000
N_PUNCTUATE = 3000        # 断句
N_GENERAL = 2000          # 通用对话（约 15%）

SYSTEM = "你是一位精通文言文的学者，擅长文言文与白话文互译、古文断句与典故解释。请准确、简洁地作答。"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_messages(instruction, inp, output):
    user = f"{instruction}\n{inp}".strip()
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]


def load_classical():
    convs = []
    rng = random.Random(SEED)

    # 古译今 / 今译古：按 task 字段分组采样
    by_task = {"c2m": [], "m2c": []}
    for row in read_jsonl(os.path.join(RAW_DIR, "translate.jsonl")):
        t = row.get("task")
        if t in by_task:
            by_task[t].append(row)
        if all(len(v) >= N_TRANSLATE_EACH for v in by_task.values()):
            break
    for t, rows in by_task.items():
        for r in rng.sample(rows, min(N_TRANSLATE_EACH, len(rows))):
            convs.append(to_messages(r["instruction"], r["input"], r["output"]))
    print(f"translate: c2m={min(len(by_task['c2m']), N_TRANSLATE_EACH)}, "
          f"m2c={min(len(by_task['m2c']), N_TRANSLATE_EACH)}")

    # 断句
    punct = []
    for row in read_jsonl(os.path.join(RAW_DIR, "punctuate.jsonl")):
        punct.append(row)
    for r in rng.sample(punct, min(N_PUNCTUATE, len(punct))):
        convs.append(to_messages(r["instruction"], r["input"], r["output"]))
    print(f"punctuate: {min(len(punct), N_PUNCTUATE)}")

    return convs


def load_general():
    ds = load_dataset("mlabonne/FineTome-100k", split="train")
    idx = random.Random(SEED).sample(range(len(ds)), N_GENERAL)
    convs = []
    for i in idx:
        msgs = [{"role": m["from"].replace("human", "user").replace("gpt", "assistant"),
                 "content": m["value"]} for m in ds[i]["conversations"]]
        convs.append(msgs)
    print(f"general: {len(convs)}")
    return convs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    convs = load_classical() + load_general()
    random.Random(SEED).shuffle(convs)

    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, enable_thinking=False)
        for c in convs
    ]
    ds = Dataset.from_pandas(pd.DataFrame(pd.Series(texts, name="text")))
    ds.save_to_disk(OUT_DIR)
    print(f">> 已保存 {len(ds)} 条到 {OUT_DIR}")
    print("---- 样例 ----")
    print(ds[0]["text"][:400])


if __name__ == "__main__":
    main()
