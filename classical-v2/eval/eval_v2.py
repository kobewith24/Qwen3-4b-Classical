# -*- coding: utf-8 -*-
r"""v2 评测：同分布 bench + 未见典籍 OOD + 通用能力，一次跑完两个模型。

三套测试的意义：
  bench    —— 和训练语料同源，衡量"学会了没有"（会偏高，不能当泛化证据）
  ood      —— 留出典籍，完全没进过训练，衡量"学到的是能力还是记忆"
  general  —— 通用对话留存，衡量"有没有被垂域数据带偏"

用法：
  # 服务器上直接读权重（不需要起服务）
  python eval/eval_v2.py --backend hf \
      --base-model model/Qwen3-4B-Instruct-2507 \
      --ft-model weights/Qwen3-4B-Classical-v2-fp16 --tag v2

  # 或者打已部署的接口
  python eval/eval_v2.py --backend openai --base-url http://127.0.0.1:8000/v1 \
      --api-key dev-key --base-name qwen3-base --ft-name qwen3-classical --tag v2
"""
import argparse
import json
import os
import sys
import time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import common  # noqa: E402
import config as cfg  # noqa: E402

ROOT = common.find_project_root(PKG)
REPORT_DIR = os.path.join(ROOT, "eval", "reports")

BENCH_TASKS = ["translate", "punctuate", "fill_in", "char_gloss", "compress", "idiom_source"]
MAX_NEW = {
    "punctuate": 640,
    "translate": 256,
    "compress": 256,
    "fill_in": 16,
    "char_gloss": 128,
    "idiom_source": 128,
    "source": 160,
    "general": 320,
}


# ---------------------------------------------------------------------------
# 数据集装载
# ---------------------------------------------------------------------------
def bench_dir():
    """优先用项目里的 eval/bench，没有就用包里自带的那份。"""
    for d in (os.path.join(ROOT, "eval", "bench"), os.path.join(PKG, "eval", "bench")):
        if os.path.isdir(d):
            return d
    return None


def load_suites(limit):
    suites = {}

    bdir = bench_dir()
    if bdir:
        items = []
        for task in BENCH_TASKS:
            path = os.path.join(bdir, f"{task}.jsonl")
            if not os.path.exists(path):
                continue
            rows = list(common.read_jsonl(path))[:limit] if limit else list(common.read_jsonl(path))
            for r in rows:
                items.append({
                    "task": task, "label": task,
                    "instruction": r.get("instruction", ""), "input": r["input"],
                    "reference": r.get("reference", ""), "meta": r.get("metadata") or {},
                })
        suites["bench"] = items

    ood_dir = os.path.join(ROOT, "data", "v2", "ood")
    if os.path.isdir(ood_dir):
        items = []
        for fn in sorted(os.listdir(ood_dir)):
            if not fn.endswith(".jsonl"):
                continue
            label = fn.replace(".jsonl", "")
            rows = list(common.read_jsonl(os.path.join(ood_dir, fn)))
            rows = rows[:limit] if limit else rows
            for r in rows:
                items.append({
                    "task": r.get("task", "translate"), "label": label,
                    "instruction": r.get("instruction", ""), "input": r["input"],
                    "reference": r.get("reference", ""), "meta": r.get("metadata") or {},
                })
        suites["ood"] = items

    gen_path = os.path.join(ROOT, "data", "v2", "general_eval.jsonl")
    if os.path.exists(gen_path):
        rows = list(common.read_jsonl(gen_path))
        rows = rows[:limit] if limit else rows
        suites["general"] = [{
            "task": "general", "label": "general",
            "instruction": "", "input": r["input"],
            "reference": r.get("reference", ""), "meta": {},
        } for r in rows]

    return {k: v for k, v in suites.items() if v}


# ---------------------------------------------------------------------------
# 生成后端
# ---------------------------------------------------------------------------
class HFBackend:
    def __init__(self, model_path, batch_size=8, max_seq=4096, load_4bit=False, adapter=""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_seq = max_seq
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # 关键：评测用的 prompt 必须和训练时用同一套 chat template（含 thinking 开关），
        # 否则就是 v1 那个 train/serve skew 的复现。
        common.set_tokenizer(self.tokenizer, cfg.ENABLE_THINKING)

        tf_major = int(__import__("transformers").__version__.split(".")[0])
        if load_4bit:
            # 8GB 卡上 bf16 的 4B 权重就占满显存，必须 4bit 才能评测；
            # base 和 ft 都在 4bit 下评，比较是公平的。
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=quant, device_map={"": 0})
        else:
            kw = {"dtype" if tf_major >= 5 else "torch_dtype": torch.bfloat16}
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map={"": 0}, **kw)
        if adapter:
            # 直接挂 adapter 评测，省掉合并出 8GB fp16 的步骤。
            # 8GB 卡反正只能 4bit 加载，合不合并对推理数值影响可以忽略。
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def _prompt(self, item):
        if item["task"] == "general":
            return item["input"]  # build 阶段已经是完整 ChatML
        user = f"{item['instruction']}\n{item['input']}".strip()
        return common.chat_prompt([("system", common.SYSTEM), ("user", user),
                                   ("assistant", "")])[0]

    def run(self, items):
        torch = self.torch
        out = []
        order = sorted(range(len(items)), key=lambda i: len(self._prompt(items[i])))
        for start in range(0, len(order), self.batch_size):
            idxs = order[start: start + self.batch_size]
            prompts = [self._prompt(items[i]) for i in idxs]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_seq,
            ).to(self.model.device)
            max_new = max(MAX_NEW.get(items[i]["task"], 256) for i in idxs)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False,
                    num_beams=1, repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for i, t in zip(idxs, texts):
                out.append((i, t.strip()))
        out.sort()
        return [t for _, t in out]


class OpenAIBackend:
    def __init__(self, base_url, api_key, model, vllm=False):
        import requests

        self.requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = api_key
        self.vllm = vllm

    def run(self, items):
        out = []
        for item in items:
            if item["task"] == "general":
                payload = {"model": self.model, "max_tokens": MAX_NEW["general"],
                           "messages": [{"role": "user", "content": item["input"]}]}
            else:
                user = f"{item['instruction']}\n{item['input']}".strip()
                payload = {"model": self.model,
                           "max_tokens": MAX_NEW.get(item["task"], 256),
                           "temperature": 0,
                           "messages": [{"role": "system", "content": common.SYSTEM},
                                        {"role": "user", "content": user}]}
            if self.vllm and item["task"] != "general":
                # vLLM 的 Qwen3 模板默认开 thinking，请求级关掉
                payload["chat_template_kwargs"] = {"enable_thinking": bool(cfg.ENABLE_THINKING)}
            r = self.requests.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=600,
                headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
            )
            r.raise_for_status()
            out.append(r.json()["choices"][0]["message"]["content"].strip())
        return out


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def run_model(backend, suites, tag, cache_path, limit=0):
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    for suite, items in suites.items():
        cache.setdefault(suite, {})
        todo = []
        for idx, it in enumerate(items):
            key = f"{it['label']}#{idx}"
            if key not in cache[suite]:
                todo.append((idx, it))
        if not todo:
            print(f"[{tag}/{suite}] 全部命中缓存")
            continue
        print(f"[{tag}/{suite}] 生成 {len(todo)} 条 ...")
        answers = backend.run([it for _, it in todo])
        for (idx, it), ans in zip(todo, answers):
            cache[suite][f"{it['label']}#{idx}"] = ans
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    scored = {}
    for suite, items in suites.items():
        per_label = {}
        for idx, it in enumerate(items):
            ans = cache.get(suite, {}).get(f"{it['label']}#{idx}", "")
            s = common.score_item(it["task"], ans, it["reference"], it["meta"])
            d = per_label.setdefault(it["label"], {"task": it["task"], "n": 0, "sum": 0.0,
                                                   "len_sum": 0, "ref_len": 0})
            d["n"] += 1
            d["sum"] += s
            d["len_sum"] += len(ans)
            d["ref_len"] += max(len(it["reference"]), 1)
        for label, d in per_label.items():
            d["score"] = round(d["sum"] / max(d["n"], 1), 4)
            d["avg_len"] = round(d["len_sum"] / max(d["n"], 1), 1)
            d["len_ratio"] = round(d["len_sum"] / max(d["ref_len"], 1), 3)
            del d["sum"], d["len_sum"], d["ref_len"]
        scored[suite] = per_label
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--base-model", default=os.path.join(ROOT, cfg.BASE_MODEL_DIR))
    ap.add_argument("--ft-model", default="", help="合并后的 fp16 目录（hf 后端）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--base-name", default="qwen3-base")
    ap.add_argument("--ft-name", default="qwen3-classical")
    ap.add_argument("--vllm", action="store_true",
                    help="openai 后端指向 vLLM 时加上，会请求级关闭 thinking")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--load-4bit", action="store_true",
                    help="8GB 卡必须加：4B 的 bf16 权重就占满显存")
    ap.add_argument("--suites", default="bench,ood,general")
    ap.add_argument("--adapter", default="",
                    help="直接挂 LoRA adapter 评测（base-model 作底座），省掉合并出 8GB fp16")
    ap.add_argument("--only", choices=["base", "ft"], default="",
                    help="只跑其中一个，用于复用已有的 answers 缓存")
    args = ap.parse_args()

    if args.backend == "hf" and not args.ft_model and not args.adapter:
        args.ft_model = os.path.join(
            ROOT, "weights", f"{cfg.BASE_MODEL_NAME}-Classical-{args.tag}-fp16")
    suites = load_suites(args.limit)
    wanted = {s.strip() for s in args.suites.split(",") if s.strip()}
    suites = {k: v for k, v in suites.items() if k in wanted}
    if not suites:
        raise SystemExit("没有可用的评测集：确认 eval/bench 与 data/v2/ood 存在")
    for k, v in suites.items():
        print(f">> {k}: {len(v)} 条")

    os.makedirs(REPORT_DIR, exist_ok=True)
    results = {"tag": args.tag, "backend": args.backend, "meta": vars(args), "sets": {}}

    if args.adapter:
        pairs = [("base", args.base_model, ""), ("ft", args.base_model, args.adapter)]
    else:
        pairs = [("base", args.base_model if args.backend == "hf" else args.base_name, "")]
        if args.ft_model or args.backend == "openai":
            pairs.append(
                ("ft", args.ft_model if args.backend == "hf" else args.ft_name, ""))
    if len(pairs) < 2:
        raise SystemExit("需要 --ft-model / --adapter（或 openai 后端的 --ft-name）")
    if args.only:
        pairs = [p for p in pairs if p[0] == args.only]
        if not pairs:
            raise SystemExit(f"--only {args.only} 没有匹配的模型")

    for tag, ref, adapter in pairs:
        if args.backend == "hf":
            backend = HFBackend(ref, args.batch_size, load_4bit=args.load_4bit,
                                adapter=adapter)
        else:
            backend = OpenAIBackend(args.base_url, args.api_key, ref, vllm=args.vllm)
        cache_path = os.path.join(REPORT_DIR, f"answers_{args.tag}_{tag}.json")
        t0 = time.time()
        results["sets"][tag] = run_model(backend, suites, tag, cache_path, args.limit)
        print(f">> {tag} 用时 {time.time() - t0:.0f}s")

    # ---- 汇总 ----
    md = [f"# 评测报告 {args.tag}", ""]
    base_sets = results["sets"].get("base", {})
    for suite in suites:
        ref_sets = base_sets or results["sets"].get("ft", {})
        if suite not in ref_sets:
            continue
        md += [f"## {suite}", "", "| 项目 | 任务 | n | base | ft | 差值 | 输出/参考长度 |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for label in ref_sets[suite].keys():
            b = base_sets.get(suite, {}).get(label)
            f = results["sets"].get("ft", {}).get(suite, {}).get(label)
            if f and b:
                md.append(
                    f"| {label} | {b['task']} | {b['n']} | {b['score']:.3f} | "
                    f"{f['score']:.3f} | {f['score'] - b['score']:+.3f} | {f['len_ratio']:.2f} |"
                )
            elif f:
                md.append(
                    f"| {label} | {f['task']} | {f['n']} | — | {f['score']:.3f} | — | "
                    f"{f['len_ratio']:.2f} |"
                )
        md.append("")
    out_json = os.path.join(REPORT_DIR, f"eval_{args.tag}.json")
    json.dump(results, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    out_md = os.path.join(REPORT_DIR, f"eval_{args.tag}.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f">> {out_json}\n>> {out_md}")


if __name__ == "__main__":
    main()
