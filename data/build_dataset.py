# -*- coding: utf-8 -*-
r"""
构建 训练集：
  - 翻译：c2m 8k + m2c 8k
  - 断句：6k
  - 压缩：从 m2c 复用 3k，换 instruction
  - 填空：从断句原文遮字 3k
  - 通用：FineTome 2k
共 30k。

每条同时保存 messages / prompt / completion / text；
prompt+completion 是 ChatML 文本，训练时用 completion_only_loss=True。
"""
import json
import os
import random

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import Dataset, load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "classical_dataset")

SEED = 24
N_C2M = 8000
N_M2C = 8000
N_PUNCT = 6000
N_COMPRESS = 3000
N_FILL = 3000
N_GENERAL = 2000

SYSTEM = "你是一位精通文言文的学者，擅长文言文与白话文互译、古文断句与典故解释。请准确、简洁地作答。"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wrap(system, user, assistant):
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    completion = f"{assistant}<|im_end|>\n"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
    }


def load_translate(rng):
    by_task = {"c2m": [], "m2c": []}
    for row in read_jsonl(os.path.join(RAW_DIR, "translate.jsonl")):
        task = row.get("task")
        if task in by_task:
            by_task[task].append(row)
            if len(by_task["c2m"]) >= N_C2M and len(by_task["m2c"]) >= N_M2C:
                break
    c2m = rng.sample(by_task["c2m"], min(N_C2M, len(by_task["c2m"])))
    m2c = rng.sample(by_task["m2c"], min(N_M2C, len(by_task["m2c"])))
    print(f"translate: c2m={len(c2m)}, m2c={len(m2c)}")
    return c2m, m2c


def load_punctuate(rng):
    rows = list(read_jsonl(os.path.join(RAW_DIR, "punctuate.jsonl")))
    rng.shuffle(rows)
    punct = rows[:N_PUNCT]
    fill_pool = rows[N_PUNCT:N_PUNCT + N_FILL]
    print(f"punctuate: {len(punct)}, fill_pool={len(fill_pool)}")
    return punct, fill_pool


def load_general(rng):
    ds = load_dataset("mlabonne/FineTome-100k", split="train")
    idx = rng.sample(range(len(ds)), N_GENERAL)
    rows = []
    for i in idx:
        msgs = []
        for m in ds[i]["conversations"]:
            role = m["from"].replace("human", "user").replace("gpt", "assistant")
            msgs.append({"role": role, "content": m["value"]})
        users = [m["content"] for m in msgs if m["role"] == "user"]
        assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
        systems = [m["content"] for m in msgs if m["role"] == "system"]
        if not users or not assistants:
            continue
        system = systems[-1] if systems else "你是一个乐于助人的中文助手。"
        rows.append((system, users[-1], assistants[-1]))
    print(f"general: {len(rows)}")
    return rows


def is_cjk(ch):
    return "\u4e00" <= ch <= "\u9fff"


def main():
    rng = random.Random(SEED)
    c2m, m2c = load_translate(rng)
    punct, fill_pool = load_punctuate(rng)
    general = load_general(rng)

    records = []
    for row in c2m:
        user = f"{row['instruction']}\n{row['input']}".strip()
        records.append(wrap(SYSTEM, user, row["output"]))
    for row in m2c:
        user = f"{row['instruction']}\n{row['input']}".strip()
        records.append(wrap(SYSTEM, user, row["output"]))
    for row in punct:
        user = f"{row['instruction']}\n{row['input']}".strip()
        records.append(wrap(SYSTEM, user, row["output"]))

    # 压缩：复用 m2c，换 instruction，输出不变
    for row in rng.sample(m2c, min(N_COMPRESS, len(m2c))):
        user = f"把下面白话文压缩成简洁文言文，只输出压缩后的文言文。\n{row['input']}".strip()
        records.append(wrap(SYSTEM, user, row["output"]))

    # 填空：从断句原文遮一个汉字，只输出该字
    made = 0
    for row in fill_pool:
        text = row.get("input", "")
        idxs = [i for i, ch in enumerate(text) if is_cjk(ch)]
        if not idxs:
            continue
        pos = rng.choice(idxs)
        answer = text[pos]
        masked = text[:pos] + "□" + text[pos + 1:]
        user = f"补出下面句子中空缺的字，只回答这个字。\n{masked}".strip()
        records.append(wrap(SYSTEM, user, answer))
        made += 1
    print(f"fill made: {made}")

    for system, user, assistant in general:
        records.append(wrap(system, user, assistant))

    rng.shuffle(records)
    ds = Dataset.from_list(records)
    ds.save_to_disk(OUT_DIR)
    print(f">> saved {len(ds)} to {OUT_DIR}")
    print(ds[0]["text"][:300])


if __name__ == "__main__":
    main()
