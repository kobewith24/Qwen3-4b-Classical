# -*- coding: utf-8 -*-
r"""
全量评测：每任务 100 条，base vs 微调。

用法：
  python eval/run_full_eval.py --ft-model qwen3-classical
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_classical as ec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "qwen3-base"))
    ap.add_argument("--ft-model", default=os.environ.get("FT_MODEL", "qwen3-classical"))
    args = ap.parse_args()

    os.makedirs(ec.REPORT_DIR, exist_ok=True)
    items = ec.load_items(args.limit)

    base_ans = ec.gen_for(args.base_model, "base_full", items)
    ft_ans = ec.gen_for(args.ft_model, "ft_full", items)

    res = ec.score(items, {"base": base_ans, "ft": ft_ans})
    res["meta"] = {"base_model": args.base_model, "ft_model": args.ft_model, "limit": args.limit}
    with open(os.path.join(ec.REPORT_DIR, "classical_eval_full.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    label = {"translate": "chrF", "punctuate": "标点F1", "fill_in": "完全匹配",
             "char_gloss": "chrF", "compress": "chrF", "idiom_source": "书名命中"}
    lines = ["# 全量评测（100 题/任务）", "",
             f"- 基座：`{args.base_model}`　微调：`{args.ft_model}`",
             "- 已对训练集做 `input[:60]` 去重", "",
             "| 任务 | 指标 | 基座 | 微调 | 提升 |", "|---|---|---|---|---|"]
    for task, v in res["tasks"].items():
        b, f = v["avg"]["base"], v["avg"]["ft"]
        lines.append(f"| {task} | {label[task]} | {b:.3f} | {f:.3f} | {f - b:+.3f} |")
    o = res["overall"]
    lines.append(f"| **平均** | — | **{o['base']:.3f}** | **{o['ft']:.3f}** | **{o['ft'] - o['base']:+.3f}** |")

    with open(os.path.join(ec.REPORT_DIR, "classical_eval_full.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
