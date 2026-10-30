# -*- coding: utf-8 -*-
r"""
文言文能力评测（两阶段，避免 Ollama 反复加载模型）

阶段 1：按模型批量生成答案 → eval/reports/answers_{base|ft}.json
阶段 2：统一打分并出报告 → eval/reports/classical_eval.json / .md

指标：
  translate     chrF
  punctuate     标点位置 F1
  fill_in       完全匹配
  char_gloss    chrF
  compress      chrF
  idiom_source  书名命中率
"""
import argparse
import json
import os
import re
import sys

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(BASE_DIR, "eval", "bench")
REPORT_DIR = os.path.join(BASE_DIR, "eval", "reports")

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BASE_MODEL = os.environ.get("BASE_MODEL", "qwen3-base")
FT_MODEL = os.environ.get("FT_MODEL", "qwen3-classical")

SYSTEM = "你是一位精通文言文的学者。请按题目要求直接给出答案，不要解释，不要重复题目。"

FILES = {
    "translate": "translate.jsonl",
    "punctuate": "punctuate.jsonl",
    "fill_in": "fill_in.jsonl",
    "char_gloss": "char_gloss.jsonl",
    "compress": "compress.jsonl",
    "idiom_source": "idiom_source.jsonl",
}
PUNCT = set("，。；：？！、「」『』（）《》…—·")


def load_items(limit):
    """加载基准并剔除与训练集重叠的题目"""
    train_blob = ""
    ds_dir = os.path.join(BASE_DIR, "data", "classical_dataset")
    if os.path.isdir(ds_dir):
        from datasets import load_from_disk
        ds = load_from_disk(ds_dir)
        train_blob = "\n".join(ds[i]["text"] for i in range(len(ds)))

    items = {}
    for task, fn in FILES.items():
        rows = [json.loads(line) for line in open(os.path.join(BENCH_DIR, fn), encoding="utf-8") if line.strip()]
        if train_blob:
            before = len(rows)
            rows = [r for r in rows if not (r.get("input") and r["input"][:60] in train_blob)]
            if before != len(rows):
                print(f"  [{task}] 剔除训练集重叠 {before - len(rows)} 条")
        items[task] = rows[:limit]
    return items


def chat(model, instruction, inp):
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": model,
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{instruction}\n{inp}".strip()},
            ],
            "options": {"temperature": 0.0, "num_predict": 512},
        },
        timeout=900,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def strip_punct(s):
    return "".join(c for c in s if c not in PUNCT)


def chrf(pred, ref):
    try:
        from sacrebleu.metrics import CHRF
        return CHRF(word_order=0).corpus_score([pred], [[ref]]).score / 100.0
    except Exception:  # noqa: BLE001
        if not pred or not ref:
            return 0.0
        common = sum(min(pred.count(c), ref.count(c)) for c in set(pred))
        return 2 * common / (len(pred) + len(ref))


def punct_positions(s):
    """返回标点出现的位置集合：位置 = 该标点之前出现的非标点字符数"""
    pos = set()
    n = 0
    for ch in s:
        if ch in PUNCT:
            pos.add(n)
        else:
            n += 1
    return pos


def punct_f1(pred, ref):
    """标点插入 F1：以参考标点位置为基准，比较预测是否在该位置断句"""
    rp = punct_positions(ref)
    pp = punct_positions(pred)
    if not rp:
        return 0.0
    tp = len(rp & pp)
    fp = len(pp - rp)
    fn = len(rp - pp)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def exact(pred, ref):
    p = re.sub(r"[\s，。；：？！、]", "", pred)
    r = re.sub(r"[\s，。；：？！、]", "", ref)
    return 1.0 if p == r else 0.0


def score_one(task, out, ref, meta):
    if task == "punctuate":
        return punct_f1(out, ref)
    if task == "fill_in":
        return exact(out, ref)
    if task == "idiom_source":
        book = meta.get("book", "").replace("《", "").replace("》", "").strip()
        return 1.0 if book and book in out.replace("《", "").replace("》", "") else 0.0
    return chrf(out, ref)


def gen_for(model, tag, items):
    path = os.path.join(REPORT_DIR, f"answers_{tag}.json")
    done = {}
    if os.path.exists(path):
        done = json.load(open(path, encoding="utf-8"))
    for task, rows in items.items():
        done.setdefault(task, {})
        for row in rows:
            if row["id"] in done[task]:
                continue
            out = chat(model, row["instruction"], row["input"])
            done[task][row["id"]] = out
            json.dump(done, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[{tag}/{task}] {row['id']} -> {out[:40]!r}")
    return done


def score(items, ans):
    tasks = {}
    for task, rows in items.items():
        b, f = [], []
        details = []
        for row in rows:
            ra = ans["base"].get(task, {}).get(row["id"], "")
            fa = ans["ft"].get(task, {}).get(row["id"], "")
            sb = score_one(task, ra, row["reference"], row.get("metadata") or {})
            sf = score_one(task, fa, row["reference"], row.get("metadata") or {})
            b.append(sb)
            f.append(sf)
            details.append({"id": row["id"], "base": ra[:100], "ft": fa[:100],
                            "ref": row["reference"][:100],
                            "base_score": round(sb, 4), "ft_score": round(sf, 4)})
        tasks[task] = {"n": len(rows),
                       "avg": {"base": round(sum(b) / len(b), 4) if b else 0,
                               "ft": round(sum(f) / len(f), 4) if f else 0},
                       "details": details}
    overall = {k: round(sum(v["avg"][k] for v in tasks.values()) / len(tasks), 4)
               for k in ("base", "ft")}
    return {"tasks": tasks, "overall": overall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--only", choices=["base", "ft"], help="只跑某个模型")
    args = ap.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    items = load_items(args.limit)

    answers = {}
    for tag, model in (("base", BASE_MODEL), ("ft", FT_MODEL)):
        p = os.path.join(REPORT_DIR, f"answers_{tag}.json")
        answers[tag] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        if args.only and args.only != tag:
            continue
        if not answers[tag]:
            answers[tag] = gen_for(model, tag, items)

    if args.only:
        return

    res = score(items, answers)
    res["meta"] = {"base_model": BASE_MODEL, "ft_model": FT_MODEL, "limit": args.limit}
    json.dump(res, open(os.path.join(REPORT_DIR, "classical_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    label = {"translate": "chrF", "punctuate": "标点F1", "fill_in": "完全匹配",
             "char_gloss": "chrF", "compress": "chrF", "idiom_source": "书名命中"}
    lines = ["# 文言文能力评测（chinese-classical-bench）", "",
             f"- 基座：`{BASE_MODEL}`　微调：`{FT_MODEL}`",
             f"- 每任务 {args.limit} 条（已剔除与训练集重叠的题目）", "",
             "| 任务 | 指标 | 基座 | 微调 | 提升 |", "|---|---|---|---|---|"]
    for t, v in res["tasks"].items():
        b, f = v["avg"]["base"], v["avg"]["ft"]
        lines.append(f"| {t} | {label[t]} | {b:.3f} | {f:.3f} | {f - b:+.3f} |")
    o = res["overall"]
    lines.append(f"| **平均** | — | **{o['base']:.3f}** | **{o['ft']:.3f}** | **{o['ft'] - o['base']:+.3f}** |")
    open(os.path.join(REPORT_DIR, "classical_eval.md"), "w", encoding="utf-8").write("\n".join(lines))

    print("\n" + "\n".join(lines[5:]))
    print(f"\n>> 平均分 {o['base']:.3f} -> {o['ft']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
