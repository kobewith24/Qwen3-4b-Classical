# -*- coding: utf-8 -*-
r"""训练与评测共用的定义。

这里只保留一份 SYSTEM —— v1 的教训是 build_dataset.py 和 eval_classical.py
各写了一份 system prompt，训练用长版、评测用短版，等于线上输入分布和训练不一致，
白丢分还查不出来。此后任何脚本都必须 import 这里。

同理，指标函数也只在这里实现一份，避免 eval 脚本之间口径漂移。
"""
import json
import os
import re


def find_project_root(start):
    """向上找项目根（含 data/raw 或 model 的目录）。

    这样包既可以解压到项目根目录，也可以放在 <项目根>/classical-v2/ 下，
    脚本里的默认路径都能对上，而且不会把 README 之类的文件覆盖掉。
    """
    cur = os.path.abspath(start)
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, "data", "raw")) or \
                os.path.isdir(os.path.join(cur, "model")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start)

# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------
SYSTEM = "你是一位精通文言文的学者，擅长文言文与白话文互译、古文断句与典故解释。请准确、简洁地作答。"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# 由 set_tokenizer() 注入。有分词器就走模型自己的 chat template（换基座不会错，
# 也能正确关掉 Qwen3 的 thinking）；没有就退化为手写 ChatML（只对 Qwen 系成立）。
_TOKENIZER = None
_ENABLE_THINKING = None


def set_tokenizer(tokenizer, enable_thinking=None):
    global _TOKENIZER, _ENABLE_THINKING
    _TOKENIZER, _ENABLE_THINKING = tokenizer, enable_thinking


def chat_prompt(turns, tokenizer=None, enable_thinking=None):
    """把 turns=[(role, content), ...] 变成 prompt / completion 文本。

    最后一条必须是 assistant，函数会把它切出来当 completion。
    返回 (prompt, completion, text)，与 TRL 的 completion_only_loss 口径一致。

    优先走 tokenizer.apply_chat_template：
      - 换基座不用改这里；
      - enable_thinking=False 时 Qwen3 原版不会在 assistant 头后面留思维链段，
        训练和评测看到的是同一种格式（v1 的 train/serve skew 就出在这里）。
    """
    assert turns, "turns is empty"
    assert turns[-1][0] == "assistant", "last turn must be assistant"

    tok = tokenizer if tokenizer is not None else _TOKENIZER
    th = enable_thinking if enable_thinking is not None else _ENABLE_THINKING

    if tok is not None:
        msgs = [{"role": r, "content": c} for r, c in turns[:-1]]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if th is not None:
            kwargs["enable_thinking"] = th
        try:
            prompt = tok.apply_chat_template(msgs, **kwargs)
        except TypeError:
            # 模板不接受 enable_thinking（Qwen2.5 / 别的模型），去掉再试一次
            kwargs.pop("enable_thinking", None)
            prompt = tok.apply_chat_template(msgs, **kwargs)
        eos = getattr(tok, "eos_token", None) or IM_END
        completion = f"{turns[-1][1]}{eos}\n"
        return prompt, completion, prompt + completion

    head = "".join(
        f"{IM_START}{role}\n{content}{IM_END}\n" for role, content in turns[:-1]
    )
    prompt = head + f"{IM_START}assistant\n"
    completion = f"{turns[-1][1]}{IM_END}\n"
    return prompt, completion, prompt + completion


def build_record(task, turns, meta=None):
    prompt, completion, text = chat_prompt(turns)
    rec = {
        "task": task,
        "prompt": prompt,
        "completion": completion,
        "text": text,
        "messages": [{"role": r, "content": c} for r, c in turns],
    }
    if meta:
        rec["meta"] = meta
    return rec


# ----------------------------------------------------------------------------
# 文本工具
# ----------------------------------------------------------------------------
PUNCT = set("，。；：？！、「」『』（）《》〈〉【】…—·“”‘’")

# 注意：这里的字符集必须和 PUNCT 一致。v2 早期版本 norm_key 只去掉了半角标点和
# 几个常用中文标点，结果「」《》被当成正文，断句样本被误判成"字符不对齐"丢掉一半。
# \s 里包含全角空格 U+3000，语料里有大量这种编辑空格。
PUNCT_RE = re.compile(
    "[" + re.escape("".join(sorted(PUNCT)) + "\t\r\n,.;:?!\"'()[]{}<>-|\\/") + r"\s]"
)


def norm_key(s, n=80):
    """去空白/标点后取前 n 字，用于去重与泄漏检查。"""
    return PUNCT_RE.sub("", s or "")[:n]


def first_clause(text, max_len=40):
    """取第一句（用于"摘录原文"类答案）。"""
    text = (text or "").strip()
    for i, ch in enumerate(text):
        if ch in "。！？；\n":
            text = text[:i]
            break
    text = text.strip("，、；： \t\n")
    return text[:max_len]


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def chrf(pred, ref):
    """chrF；sacrebleu 不可用时退化为字符 unigram F1。"""
    pred = (pred or "").strip()
    ref = (ref or "").strip()
    if not pred or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import CHRF

        return CHRF(word_order=0).corpus_score([pred], [[ref]]).score / 100.0
    except Exception:  # noqa: BLE001
        common = sum(min(pred.count(c), ref.count(c)) for c in set(pred))
        return 2 * common / (len(pred) + len(ref))


def punct_positions(s):
    """标点位置集合：位置 = 该标点之前的非标点字符数。"""
    pos = set()
    n = 0
    for ch in s or "":
        if ch in PUNCT:
            pos.add(n)
        else:
            n += 1
    return pos


def punct_f1(pred, ref):
    """标点插入 F1（按位置集合，避免整句错位导致 0 分）。"""
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
    p = PUNCT_RE.sub("", pred or "")
    r = PUNCT_RE.sub("", ref or "")
    return 1.0 if p and p == r else 0.0


def book_hit(pred, book):
    book = (book or "").replace("《", "").replace("》", "").strip()
    if not book:
        return 0.0
    return 1.0 if book in (pred or "").replace("《", "").replace("》", "") else 0.0


METRIC = {
    "translate": chrf,
    "punctuate": punct_f1,
    "fill_in": exact,
    "char_gloss": chrf,
    "compress": chrf,
    "idiom_source": None,  # 需要 metadata.book，单独处理
    "source": None,        # 同上
}


def score_item(task, pred, ref, meta=None):
    meta = meta or {}
    if task in ("idiom_source", "source"):
        return book_hit(pred, meta.get("book", ""))
    fn = METRIC.get(task, chrf)
    return fn(pred, ref)


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    import os

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
