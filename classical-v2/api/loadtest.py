# -*- coding: utf-8 -*-
r"""并发压测：量 TTFT、端到端时延、吞吐和错误率。

为什么要单独写：Ollama / llama-server 的开环吞吐没法直接回答"几个并发时 p95 崩"，
而这恰好是简历上唯一能写死的部署指标。

用法：
  # 打网关（推荐，测的是你对外暴露的那条链路）
  python api/loadtest.py --url http://127.0.0.1:8000/v1/chat/completions \
      --key dev-key --model qwen3-classical --concurrency 1,2,4,8 --requests 24

  # 直接打 llama-server（跳过网关，测引擎上限）
  python api/loadtest.py --url http://127.0.0.1:8080/v1/chat/completions \
      --model qwen3-classical --concurrency 1,4,8 --requests 24

输出：api/reports/loadtest_<tag>.csv 与 .md
"""
import argparse
import copy
import csv
import json
import os
import sys
import threading
import time

import requests

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import common  # noqa: E402

ROOT = common.find_project_root(PKG)
REPORT_DIR = os.path.join(ROOT, "api", "reports")

PROMPTS = [
    "为下列古文添加标点：齐高祖神武皇帝姓高名欢字贺六浑渤海蓨人也六世祖隐晋玄菟太守",
    "将下列古文翻译成现代汉语：假之以便，唆之使前，断其援应，陷之死地。",
    "将下列现代汉语翻译成古文：故意露出一些破绽，以引诱敌人深入我方。",
    "下列文字出自哪部典籍？请给出书名：以周都洛邑，治在镐京，汉宅咸阳，魂归丰沛。",
    "解释“运筹借箸”这个成语的出处。",
    "用一句话说明《左传》的体例特点。",
    "帮我把这句话改成更简洁的文言文：他早上出门的时候天还没有亮。",
    "学而时习之不亦说乎这句话出自哪部典籍？",
    "把“温故而知新，可以为师矣”翻译成白话文。",
    "“青，取之于蓝，而青于蓝”是什么意思？",
    "给下面这段话断句：古之学者必有师师者所以传道受业解惑也",
    "解释“三人行，必有我师焉”中的“焉”字用法。",
    "《史记》的作者是谁？他在什么情况下写的这部书？",
    "把“北冥有鱼，其名为鲲”扩写成一段现代汉语。",
    "“落霞与孤鹜齐飞，秋水共长天一色”出自哪里？",
    "请用文言文写一句描写秋天的话。",
    "“刻舟求剑”这个寓言说明了什么道理？",
    "判断下面这段文字可能出自哪一类典籍：臣闻之，兵者国之大事。",
    "解释“既来之，则安之”中的“既”和“安”。",
    "把“吾日三省吾身”翻译成白话并说明其出处。",
]


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return values[k]


def one_request(url, headers, payload, timeout):
    """返回 (ok, err, ttft_ms, total_ms, out_tokens)。"""
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    body = dict(payload)
    body["stream"] = True
    try:
        with requests.post(url, json=body, headers=headers, timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                return False, f"http {r.status_code}", 0.0, (time.perf_counter() - t0) * 1000, 0
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    tokens += 1
        total = (time.perf_counter() - t0) * 1000
        if ttft is None:
            return False, "empty stream", 0.0, total, 0
        return True, "", ttft, total, tokens
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}", 0.0, (time.perf_counter() - t0) * 1000, 0


def run_level(url, headers, payload, level, n_req, timeout):
    results, lock = [], threading.Lock()
    counter = {"i": 0}

    def worker():
        while True:
            with lock:
                if counter["i"] >= n_req:
                    return
                i = counter["i"]
                counter["i"] += 1
            p = copy.deepcopy(payload)
            p["messages"][1]["content"] = PROMPTS[i % len(PROMPTS)]
            res = one_request(url, headers, p, timeout)
            with lock:
                results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(level)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r[0]]
    err = [r for r in results if not r[0]]
    ttfts = [r[2] for r in ok]
    totals = [r[3] for r in ok]
    tokens = sum(r[4] for r in ok)
    decode_tps = sum(
        (r[4] - 1) / max((r[3] - r[2]) / 1000.0, 1e-6) for r in ok
    ) / max(len(ok), 1)
    return {
        "concurrency": level,
        "requests": n_req,
        "ok": len(ok),
        "err": len(err),
        "err_rate": round(len(err) / max(n_req, 1), 3),
        "rps": round(len(ok) / max(wall, 1e-6), 2),
        "ttft_p50": round(percentile(ttfts, 50), 1),
        "ttft_p95": round(percentile(ttfts, 95), 1),
        "total_p50": round(percentile(totals, 50), 1),
        "total_p95": round(percentile(totals, 95), 1),
        "out_tokens": tokens,
        "decode_tps": round(decode_tps, 1),
        "wall_s": round(wall, 2),
        "errors": ";".join(sorted({e for _, e, *_ in err}))[:120],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="qwen3-classical")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=24, help="每个并发档位的请求数")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()

    headers = {"Content-Type": "application/json"}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": "你是一位精通文言文的学者，擅长文言文与白话文互译、古文断句与典故解释。请准确、简洁地作答。"},
            {"role": "user", "content": PROMPTS[0]},
        ],
    }

    print(">> 预热 ...")
    one_request(args.url, headers, payload, args.timeout)

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    rows = []
    for level in levels:
        row = run_level(args.url, headers, payload, level, args.requests, args.timeout)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_path = os.path.join(REPORT_DIR, f"loadtest_{args.tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [f"# 并发压测 {args.tag}", "",
          f"- 目标：`{args.url}`　模型：`{args.model}`",
          f"- 每档 {args.requests} 个请求，max_tokens={args.max_tokens}", "",
          "| 并发 | 成功 | 错误率 | RPS | TTFT p50 | TTFT p95 | 时延 p50 | 时延 p95 | 解码 t/s |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['concurrency']} | {r['ok']}/{r['requests']} | {r['err_rate']:.0%} | "
                  f"{r['rps']} | {r['ttft_p50']} ms | {r['ttft_p95']} ms | "
                  f"{r['total_p50']} ms | {r['total_p95']} ms | {r['decode_tps']} |")
    md_path = os.path.join(REPORT_DIR, f"loadtest_{args.tag}.md")
    open(md_path, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f">> {csv_path}\n>> {md_path}")


if __name__ == "__main__":
    main()
