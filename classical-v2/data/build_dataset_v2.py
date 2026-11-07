# -*- coding: utf-8 -*-
r"""构建 v2 训练集 / 开发集 / OOD 测试集。

相对 v1 的改动（每条都有原因）：
  1. 去掉"填空"：每条只监督 1 个 token，却要跑 ~380 字前向，信号/算力比约 1:380，
     而且把模型往"答案越短越好"推。收益为负，删。
  2. 去掉"压缩"：v1 从 m2c 里采样、只改 instruction、输出原样复用，等于把同一对
     问答喂两遍还给了两个互相矛盾的指令（"压缩"的结果和完整翻译等长）。
     没有金标压缩数据就不该造这个任务。
  3. 新增"出处"：直接用语料自带的 source 字段构造"原文 -> 出自哪部典籍"，
     补上 v1 中唯一 0 覆盖、且明显回退（-0.128）的任务。
  4. 通用数据：过滤中文、保留多轮、统一 system，不再拿 FineTome 的最后一轮
     硬拼成单轮。条数不变但多样性和分布对齐都提高了。
  5. 按监督 token（字符数近似）配平，而不是按条数配平。
  6. 按"书"划分 OOD：留出若干部典籍完全不入训练，得到"未见典籍"测试集。

用法（在项目根目录）：
    python data/build_dataset_v2.py --raw data/raw --out data/v2

    # 冒烟测试（只读前 5 万行）
    python data/build_dataset_v2.py --raw data/raw --out data/v2_smoke --max-lines 50000 --no-general
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import common  # noqa: E402
import config as cfg  # noqa: E402

ROOT = common.find_project_root(PKG)

GENERAL_SYSTEM = "你是一个乐于助人的中文助手。"

# 史记类（用于给 OOD 打标签：同体裁未见书 / 跨体裁）
HISTORY_BOOKS = {
    "史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书", "梁书", "陈书",
    "魏书", "北齐书", "周书", "北史", "南史", "隋书", "旧唐书", "新唐书", "旧五代史",
    "新五代史", "宋史", "辽史", "金史", "元史", "明史", "资治通鉴", "左传", "战国策",
}

DEFAULT_HOLDOUT = ["旧唐书", "徐霞客游记", "梦溪笔谈", "天工开物"]

INSTRUCT = {
    "c2m": "将下列古文翻译成现代汉语。",
    "m2c": "将下列现代汉语翻译成古文。",
    "punctuate": "为下列古文添加标点。",
    "source": "下列文字出自哪部典籍？请给出书名，并摘录原文一句。",
}

SOURCE_TEMPLATES = [
    "下列文字出自哪部典籍？请给出书名，并摘录原文一句。",
    "这段古文出自哪一部书？请写出书名，并引用原文中的一句作为出处。",
    "请判断下面这段文字的出处：给出书名，并摘录一句原文。",
]

LEADING_PUNCT = "。，、；：！？』」”’"

# 编辑加上去的引号/书名号：从断句目标里去掉
EDITORIAL_MARKS = str.maketrans("", "", "「」『』《》〈〉“”‘’")


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
class Reservoir:
    """蓄水池采样：内存恒定，适合 190 万行的 translate.jsonl。"""

    def __init__(self, k, rng):
        self.k = k
        self.rng = rng
        self.n = 0
        self.items = []

    def add(self, x):
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(x)
        else:
            j = self.rng.randrange(self.n)
            if j < self.k:
                self.items[j] = x


def book_of(source):
    s = (source or "").strip()
    if not s:
        return ""
    s = s.replace("（", "(").split("(")[0]
    return s.split("·")[0].split("/")[0].strip()


def is_cjk_ratio(text, ratio=0.4):
    text = text or ""
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(len(text), 1) >= ratio


def ratio_of(share):
    """把"占比"换成"与其他部分的比值"，用于反推条数。"""
    s = min(max(share, 0.02), 0.85)
    return s / (1.0 - s)


# ---------------------------------------------------------------------------
# 读原始语料
# ---------------------------------------------------------------------------
def stream_translate(path, holdout, rng, need, max_lines=0):
    """返回 (train_pool, ood_pool, book_counter)。

    train_pool[task] / ood_pool[task] 是 Reservoir。
    """
    size = {t: int(need[t] * 1.6) + 50 for t in ("c2m", "m2c")}
    train_pool = {t: Reservoir(size[t], rng) for t in ("c2m", "m2c")}
    ood_pool = {t: Reservoir(600, rng) for t in ("c2m", "m2c")}
    books = Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += 1
            if max_lines and n > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = row.get("task")
            if task not in ("c2m", "m2c"):
                continue
            book = book_of(row.get("source"))
            if not book:
                continue
            books[book] += 1
            row["_book"] = book
            if book in holdout:
                ood_pool[task].add(row)
            else:
                train_pool[task].add(row)
            if n % 200000 == 0:
                print(f"  translate 已读 {n:,} 行", flush=True)
    print(f"  translate 共读 {n:,} 行，书 {len(books)} 部")
    return train_pool, ood_pool, books


def stream_punctuate(path, holdout, rng, need, max_lines=0):
    size = int(need["punctuate"] * 1.6) + 50
    train_pool = Reservoir(size, rng)
    ood_pool = Reservoir(600, rng)
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += 1
            if max_lines and n > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            book = book_of(row.get("source"))
            row["_book"] = book
            if book and book in holdout:
                ood_pool.add(row)
            else:
                train_pool.add(row)
    print(f"  punctuate 共读 {n:,} 行")
    return train_pool, ood_pool


def load_general(path, rng, need, max_lines=0):
    """通用对话：统一 system、保留最多 4 轮、只留中文。

    支持两种输入：
      - jsonl，每行 {"messages":[{"role","content"}...]} 或 {"system","user","assistant"}
      - HF 数据集 mlabonne/FineTome-100k（conversations / from / value 结构）
    """
    pool = Reservoir(need * 2 + 50, rng)
    kept = dropped = 0

    def take(messages):
        nonlocal kept, dropped
        msgs = [m for m in messages if m.get("content")]
        if len(msgs) < 2:
            dropped += 1
            return
        # 至少要有 user 和 assistant
        if msgs[-1]["role"] != "assistant":
            dropped += 1
            return
        turns = [(m["role"], m["content"]) for m in msgs if m["role"] in ("user", "assistant")]
        if not turns or turns[-1][0] != "assistant":
            dropped += 1
            return
        turns = turns[-4:] if len(turns) > 4 else turns
        if turns[0][0] != "user":
            turns = turns[1:]
        if not turns or turns[-1][0] != "assistant":
            dropped += 1
            return
        joined = "".join(c for _, c in turns)
        if len(joined) < 20 or len(joined) > 4000:
            dropped += 1
            return
        if not is_cjk_ratio(joined, 0.4):
            dropped += 1
            return
        kept += 1
        pool.add([("system", GENERAL_SYSTEM)] + turns)

    if path:
        n = 0
        for row in common.read_jsonl(path):
            n += 1
            if max_lines and n > max_lines:
                break
            if "messages" in row:
                take(row["messages"])
            else:
                msgs = []
                if row.get("system"):
                    msgs.append({"role": "system", "content": row["system"]})
                msgs.append({"role": "user", "content": row.get("user", "")})
                msgs.append({"role": "assistant", "content": row.get("assistant", "")})
                take(msgs)
    else:
        from datasets import load_dataset

        ds = load_dataset("mlabonne/FineTome-100k", split="train")
        n = 0
        for row in ds:
            n += 1
            if max_lines and n > max_lines:
                break
            msgs = []
            for m in row["conversations"]:
                role = {"human": "user", "gpt": "assistant", "system": "system"}.get(
                    m["from"], m["from"]
                )
                msgs.append({"role": role, "content": m["value"]})
            take(msgs)
    print(f"  general 读 {n:,} 行，保留 {kept:,}，丢弃 {dropped:,}")
    return pool


# ---------------------------------------------------------------------------
# 任务构造
# ---------------------------------------------------------------------------
def rec_from_row(task, row, turns):
    return common.build_record(task, turns, {"book": row.get("_book", "")})


def build_punct(rows, stats):
    """断句记录。

    语料实测（46,546 条全量统计）：
      - 86% 的 output 以标点开头（语料按段切分留下的跨段残渣）；
      - 45% 混入了编辑补的引号/书名号「」《》；
      - 15% 含全角空格。
    v1 原样照学，等于教模型"回答以标点开头、自己加书名号"，
    而 bench 的参考标点里未必有这些，位置 F1 直接吃假阳性。
    这里统一剥离，并保留字符级对齐校验作为兜底。
    """
    out = []
    for row in rows:
        raw = (row.get("output") or "").strip()
        if raw[:1] and raw[0] in LEADING_PUNCT:
            stats["punct_lead"] += 1
        if any(ch in raw for ch in "「」『』《》〈〉"):
            stats["punct_editorial"] += 1
        raw_in = (row.get("input") or "").strip()
        raw_out = raw.lstrip(LEADING_PUNCT).translate(EDITORIAL_MARKS)
        if not raw_in or not raw_out:
            stats["punct_bad"] += 1
            continue
        if common.norm_key(raw_in, 100000) != common.norm_key(raw_out, 100000):
            stats["punct_bad"] += 1
            continue
        user = f"{row.get('instruction', INSTRUCT['punctuate'])}\n{raw_in}".strip()
        out.append(rec_from_row("punctuate", row, [
            ("system", common.SYSTEM), ("user", user), ("assistant", raw_out)
        ]))
    return out


def build_translate(rows, task):
    out = []
    for row in rows:
        user = f"{row.get('instruction', INSTRUCT[task])}\n{row['input']}".strip()
        out.append(rec_from_row(task, row, [
            ("system", common.SYSTEM), ("user", user), ("assistant", row["output"].strip())
        ]))
    return out


def build_source(rows, rng, stats):
    """原文 -> 出自《书》：「原文一句」。

    - 只保留不含书名号的行，避免直接从题干抄答案；
    - 引文随机取一句完整短句，而不是固定取第一句，
      否则模型会学成"抄开头"而不是"判断出处"。
    """
    out = []
    for row in rows:
        book = row.get("_book", "")
        text = (row.get("input") or "").strip()
        if not book or not text or "《" in text or "》" in text:
            stats["source_skip"] += 1
            continue
        quote = pick_quote(text, rng)
        if not quote:
            stats["source_skip"] += 1
            continue
        user = f"{rng.choice(SOURCE_TEMPLATES)}\n{text}".strip()
        out.append(rec_from_row("source", row, [
            ("system", common.SYSTEM), ("user", user),
            ("assistant", f"出自《{book}》：「{quote}」")
        ]))
    return out


def pick_quote(text, rng):
    """从原文里挑一句 8~60 字的完整小句，优先句号级边界。"""
    import re as _re

    for pattern in (r"[^。！？]{8,60}[。！？]", r"[^，；：]{8,60}[，；：]"):
        cands = [m.group(0).strip("，。；：！？ \t\n") for m in _re.finditer(pattern, text)]
        cands = [c for c in cands if 8 <= len(c) <= 60]
        if cands:
            return rng.choice(cands)
    return ""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "v2"))
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--holdout-books", default=",".join(DEFAULT_HOLDOUT))
    ap.add_argument("--n-punct", type=int, default=5000)
    ap.add_argument("--n-c2m", type=int, default=8000)
    ap.add_argument("--n-m2c", type=int, default=8000)
    ap.add_argument("--n-source", type=int, default=5000)
    ap.add_argument("--n-general", type=int, default=8000)
    ap.add_argument("--punct-share", type=float, default=0.35,
                    help="断句任务的目标监督字符占比，按其它任务自动反推条数")
    ap.add_argument("--general-share", type=float, default=0.20,
                    help="通用数据的目标监督字符占比，按垂域任务自动反推条数")
    ap.add_argument("--min-punct", type=int, default=1500)
    ap.add_argument("--ood-per-task", type=int, default=120)
    ap.add_argument("--dev-ratio", type=float, default=0.02)
    ap.add_argument("--general-jsonl", default="", help="离线通用语料；留空则从 HF 拉 FineTome")
    ap.add_argument("--no-general", action="store_true")
    ap.add_argument("--max-lines", type=int, default=0, help="冒烟测试用，只读前 N 行")
    ap.add_argument("--tokenizer", default="",
                    help="基座分词器目录。给了就用模型自带的 chat template 渲染，换基座不会错")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                    help="思维链开关；auto 表示读 config.py 的 ENABLE_THINKING")
    ap.add_argument("--list-books", action="store_true")
    args = ap.parse_args()

    # 渲染方式：有分词器就用它自己的模板（正确关掉 Qwen3 的 thinking），否则退回手写 ChatML
    tok_dir = args.tokenizer or os.path.join(ROOT, cfg.BASE_MODEL_DIR)
    th = cfg.ENABLE_THINKING if args.think == "auto" else (args.think == "on")
    if os.path.isdir(tok_dir):
        from transformers import AutoTokenizer

        common.set_tokenizer(AutoTokenizer.from_pretrained(tok_dir), th)
        print(f">> tokenizer {tok_dir}  enable_thinking={th}")
    else:
        print(f"!! 找不到分词器目录 {tok_dir}，退回手写 ChatML")
        print("   换过基座的话这会让训练和推理格式对不上，务必用 --tokenizer 指定")

    rng = random.Random(args.seed)
    holdout = {b.strip() for b in args.holdout_books.split(",") if b.strip()}
    os.makedirs(args.out, exist_ok=True)
    clean_stats = Counter()

    tr_path = os.path.join(args.raw, "translate.jsonl")
    pu_path = os.path.join(args.raw, "punctuate.jsonl")
    for p in (tr_path, pu_path):
        if not os.path.exists(p):
            raise SystemExit(f"缺少原始语料：{p}")

    print(f">> holdout books: {sorted(holdout)}")
    need = {
        "c2m": args.n_c2m, "m2c": args.n_m2c, "punctuate": args.n_punct,
        "source": args.n_source,
    }

    print(">> 读取 translate.jsonl")
    tr_train, tr_ood, books = stream_translate(tr_path, holdout, rng, need, args.max_lines)
    if args.list_books:
        print("书名分布（前 40）：")
        for b, c in books.most_common(40):
            flag = "  <== holdout" if b in holdout else ""
            print(f"   {b:<14} {c:>7}{flag}")
        return

    print(">> 读取 punctuate.jsonl")
    pu_train, pu_ood = stream_punctuate(pu_path, holdout, rng, need, args.max_lines)

    # ---- 采样与配平 ----
    # 先定翻译/出处这两个"条数固定"的任务，再按目标字符占比反推断句和通用的条数。
    # v1 的问题正是断句一家吃掉约七成监督信号，所以这里必须按占比配，而不是按条数配。
    c2m_rows = rng.sample(tr_train["c2m"].items, min(args.n_c2m, len(tr_train["c2m"].items)))
    m2c_rows = rng.sample(tr_train["m2c"].items, min(args.n_m2c, len(tr_train["m2c"].items)))

    # 出处任务：从 c2m 池里取"翻译任务没用到"的行，避免同一段原文出现两次
    used_keys = {common.norm_key(r["input"], 60) for r in c2m_rows}
    pool = [r for r in tr_train["c2m"].items if common.norm_key(r["input"], 60) not in used_keys]
    rng.shuffle(pool)
    source_rows = pool[: args.n_source]

    domain_recs = (
        build_translate(c2m_rows, "c2m")
        + build_translate(m2c_rows, "m2c")
        + build_source(source_rows, rng, clean_stats)
    )
    domain_chars = sum(len(r["completion"]) for r in domain_recs)

    # 通用：按目标占比反推条数（通用单条很长，按条数配会悄悄吃掉大量 token）
    general_recs, gen_eval_turns = [], []
    if not args.no_general and args.n_general > 0:
        print(">> 读取通用对话")
        try:
            gen_pool = load_general(args.general_jsonl, rng, args.n_general, args.max_lines)
        except Exception as exc:  # noqa: BLE001
            print(f"!! 通用数据加载失败（{type(exc).__name__}: {exc}）")
            print("   服务器没网时请改用：--general-jsonl <本地jsonl> 或 --no-general")
            gen_pool = None
        cand = list(gen_pool.items) if gen_pool else []
        rng.shuffle(cand)
        if cand:
            avg_g = sum(len(t[-1][1]) for t in cand) / len(cand)
            k_g = int(round(ratio_of(args.general_share) * domain_chars / max(avg_g, 1)))
            k_g = max(500, min(k_g, len(cand), args.n_general))
            general_recs = [common.build_record("general", t) for t in cand[:k_g]]
            gen_eval_turns = cand[k_g: k_g + 150]
            print(f"  通用：候选 {len(cand)}，按 {args.general_share:.0%} 占比取 {k_g} 条")
    general_chars = sum(len(r["completion"]) for r in general_recs)

    # 断句：按目标占比反推条数
    punct_cand = build_punct(list(pu_train.items), clean_stats)
    rng.shuffle(punct_cand)
    if punct_cand:
        avg_p = sum(len(r["completion"]) for r in punct_cand) / len(punct_cand)
        k_p = int(round(ratio_of(args.punct_share) * (domain_chars + general_chars) / max(avg_p, 1)))
        k_p = max(args.min_punct, min(k_p, len(punct_cand)))
        punct_recs = punct_cand[:k_p]
        print(f"  断句：候选 {len(punct_cand)}，按 {args.punct_share:.0%} 占比取 {k_p} 条")
    else:
        punct_recs = []

    records = punct_recs + domain_recs + general_recs

    # ---- 任务内去重 ----
    seen, deduped = set(), []
    for rec in records:
        key = (rec["task"], common.norm_key(rec["messages"][-2]["content"], 80))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    print(f">> 去重 {len(records) - len(deduped)} 条，剩 {len(deduped)}")
    records = deduped
    rng.shuffle(records)

    # ---- 划分 dev（按任务分层）----
    by_task = {}
    for rec in records:
        by_task.setdefault(rec["task"], []).append(rec)
    train, dev = [], []
    for task, items in by_task.items():
        k = max(1, int(len(items) * args.dev_ratio))
        dev.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(dev)

    common.write_jsonl(os.path.join(args.out, "train.jsonl"), train)
    common.write_jsonl(os.path.join(args.out, "dev.jsonl"), dev)

    # ---- OOD 集（按书留出）----
    ood_dir = os.path.join(args.out, "ood")
    train_blob = "\n".join(
        common.norm_key(r["messages"][1]["content"], 4000) for r in train
    )
    ood_stat = {}
    ood_specs = [
        ("translate_c2m", "translate", tr_ood["c2m"], "c2m"),
        ("translate_m2c", "translate", tr_ood["m2c"], "m2c"),
        ("punctuate", "punctuate", pu_ood, "punctuate"),
    ]
    for name, task, pool, kind in ood_specs:
        items, dropped = [], 0
        for row in pool.items:
            inp = (row.get("input") or "").strip()
            ref = (row.get("output") or "").strip()
            if len(inp) < 10 or not ref:
                continue
            if common.norm_key(inp, 40) in train_blob:
                dropped += 1
                continue
            items.append({
                "id": f"{name}#{len(items) + 1}",
                "task": task,
                "instruction": INSTRUCT[kind],
                "input": inp,
                "reference": ref,
                "metadata": {"book": row.get("_book", "")},
            })
            if len(items) >= args.ood_per_task:
                break
        common.write_jsonl(os.path.join(ood_dir, f"{name}.jsonl"), items)
        groups = Counter(
            "same_genre" if it["metadata"]["book"] in HISTORY_BOOKS else "cross_genre"
            for it in items
        )
        ood_stat[name] = {"n": len(items), "leak_dropped": dropped, "groups": dict(groups)}

    # ---- 通用能力 held-out ----
    if gen_eval_turns:
        rows = []
        for i, turns in enumerate(gen_eval_turns, 1):
            prompt, _, _ = common.chat_prompt(turns)
            rows.append({
                "id": f"general#{i}", "task": "general",
                "instruction": "", "input": prompt,
                "reference": turns[-1][1], "metadata": {},
            })
        common.write_jsonl(os.path.join(args.out, "general_eval.jsonl"), rows)

    # ---- 统计 ----
    stats = {"seed": args.seed, "holdout_books": sorted(holdout), "tasks": {}, "ood": ood_stat}
    per_task = {}
    for rec in train:
        per_task.setdefault(rec["task"], []).append(rec)
    total_chars = sum(len(r["completion"]) for r in train)
    for task, items in sorted(per_task.items()):
        chars = sum(len(r["completion"]) for r in items)
        stats["tasks"][task] = {
            "n": len(items),
            "dev_n": sum(1 for r in dev if r["task"] == task),
            "avg_supervised_chars": round(chars / len(items), 1),
            "supervised_char_share": round(chars / total_chars, 4),
        }
    stats["total_train"] = len(train)
    stats["total_dev"] = len(dev)
    stats["cleaning"] = dict(clean_stats)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---- 报告 ----
    lines = [
        "# 数据集 v2 配比报告", "",
        f"- 训练 {len(train)} 条 / 开发 {len(dev)} 条，seed={args.seed}",
        f"- 留出典籍（完全不入训练）：{', '.join(sorted(holdout))}",
        "- 监督占比按 completion 字符数近似；真实 token 占比由 train/finetune.py 用分词器统计", "",
        "| 任务 | 条数 | 开发集 | 平均监督长度 | 监督占比 |", "|---|---:|---:|---:|---:|",
    ]
    order = ["punctuate", "c2m", "m2c", "source", "general"]
    for task in order:
        if task not in stats["tasks"]:
            continue
        s = stats["tasks"][task]
        lines.append(
            f"| {task} | {s['n']} | {s['dev_n']} | {s['avg_supervised_chars']} 字 | "
            f"{s['supervised_char_share'] * 100:.1f}% |"
        )
    lines += ["", "## OOD（未见典籍）", "", "| 集合 | 条数 | 同体裁 | 跨体裁 | 泄漏剔除 |", "|---|---:|---:|---:|---:|"]
    for name, s in ood_stat.items():
        g = s["groups"]
        lines.append(
            f"| {name} | {s['n']} | {g.get('same_genre', 0)} | "
            f"{g.get('cross_genre', 0)} | {s['leak_dropped']} |"
        )
    lines += [
        "", "## 清洗",
        f"- 断句：剥离段首残留标点 {clean_stats.get('punct_lead', 0)} 条，"
        f"剥离编辑加的引号/书名号 {clean_stats.get('punct_editorial', 0)} 条",
        f"- 断句：字符对齐校验未通过而丢弃 {clean_stats.get('punct_bad', 0)} 条"
        f"（正常应为 0，非 0 说明剥离规则和语料不匹配，需要检查）",
        f"- 出处：跳过 {clean_stats.get('source_skip', 0)} 条（含书名号或无合适短句）",
    ]
    with open(os.path.join(args.out, "mix_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n>> 输出目录 {args.out}")


if __name__ == "__main__":
    main()
