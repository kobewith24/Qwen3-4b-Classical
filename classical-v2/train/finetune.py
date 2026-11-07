# -*- coding: utf-8 -*-
r"""v2 训练脚本：4 卡 DDP + 监督 token 统计 + 开发集早停。

相对 v1 的改动：
  1. 支持多卡（accelerate / torchrun）。v1 里 device_map={"": 0} 写死 GPU0，
     直接 torchrun 会让 4 个进程全挤在 0 号卡上 OOM。
  2. 训练前用分词器统计"每个任务的监督 token 占比"并落盘。
     这是整个项目最该拿出的一张表：v1 六项结果的全部形态都能被它解释。
  3. 有 dev 集就每个 eval_steps 评测一次，支持早停，不再盲训 2 epoch。
  4. 4bit 分支补上 prepare_model_for_kbit_training。

用法：
  单卡：python train/finetune.py --run-name v2 --batch-size 8 --grad-accum 4
  4 卡：accelerate launch --num_processes 4 --mixed_precision bf16 \
          train/finetune.py --run-name v2 --batch-size 2 --grad-accum 4
"""
import argparse
import inspect
import json
import os
import sys
from collections import Counter

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

try:  # TRL 0.20+ 才有 assistant_only_loss，旧版本只有 completion_only_loss
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"需要 trl>=0.15：{exc}")

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import common  # noqa: E402
import config as cfg  # noqa: E402

ROOT = common.find_project_root(PKG)


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "-1"))


def world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


# transformers 5 把 torch_dtype 改名为 dtype，这里按主版本号选，避免踩坑
_TF_MAJOR = int(__import__("transformers").__version__.split(".")[0])
DTYPE_KEY = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"


def load_base_model(model_dir, precision, rank):
    device = rank if rank >= 0 else 0
    if precision == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb, device_map={"": device}
        )
        return prepare_model_for_kbit_training(model)
    return AutoModelForCausalLM.from_pretrained(
        model_dir, device_map={"": device}, **{DTYPE_KEY: torch.bfloat16}
    )


USER_MARK = "<|im_start|>user\n"
END_MARK = "<|im_end|>"

# v1 语料里 instruction 的真实写法（已对 20 万行抽样统计，translate 10 种 / punctuate 5 种）
INSTR_TASK = {
    "把这句文言文译成白话文": "c2m",
    "将下列古文翻译成现代汉语": "c2m",
    "解释下列古文的含义": "c2m",
    "请用现代汉语翻译这段古文": "c2m",
    "用白话解释这句古文": "c2m",
    "将下列现代汉语翻译成古文": "m2c",
    "请用古文表达": "m2c",
    "用文言文表达这句话": "m2c",
    "把这段白话改写为古文": "m2c",
    "给古文加标点": "punctuate",
    "给这段古文断句": "punctuate",
    "为下列古文添加标点": "punctuate",
    "断句并加标点": "punctuate",
    "请为这段未标点的古文加上标点符号": "punctuate",
    "把下面白话文压缩成简洁文言文": "compress",
    "补出下面句子中空缺的字": "fill",
}


def guess_task(prompt):
    """老数据集（v1）没有 task 列，从 user 消息里的 instruction 反推。

    注意：不能只用几个关键词，v1 的 instruction 一共有 15 种写法，
    早期版本只匹配了 4 个关键词，结果 60% 的翻译样本被误判成 general。
    """
    i = prompt.find(USER_MARK)
    if i < 0:
        return "general"
    j = prompt.find(END_MARK, i)
    user = prompt[i + len(USER_MARK): j if j > 0 else len(prompt)]
    for ins, task in INSTR_TASK.items():
        if user.startswith(ins):
            return task
    if user.startswith("翻译："):
        return "translate"  # 方向不明，合并计入"翻译"
    return "general"


def load_any(path):
    """既支持 jsonl 目录（v2），也支持 save_to_disk 的 HF 数据集（v1）。"""
    if os.path.isfile(path):
        return load_dataset("json", data_files=path, split="train")
    if os.path.exists(os.path.join(path, "state.json")):
        from datasets import load_from_disk

        return load_from_disk(path)
    tr = os.path.join(path, "train.jsonl")
    if os.path.exists(tr):
        return load_dataset("json", data_files=tr, split="train")
    raise SystemExit(f"认不出数据集格式：{path}")


def add_task_column(ds):
    if "task" in ds.column_names:
        return ds
    return ds.map(lambda ex: {"task": guess_task(ex["prompt"])})


def task_token_stats(dataset, tokenizer, name="train"):
    """统计每个任务的监督 token 数（只算 completion 部分）。"""
    counter, n = Counter(), Counter()
    for ex in dataset:
        task = ex.get("task", "unknown")
        ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
        counter[task] += len(ids)
        n[task] += 1
    total = sum(counter.values()) or 1
    rows = []
    for task, tok in counter.most_common():
        rows.append({
            "task": task,
            "samples": n[task],
            "supervised_tokens": tok,
            "token_share": round(tok / total, 4),
            "avg_tokens": round(tok / max(n[task], 1), 1),
        })
    print(f"\n>> {name} 监督 token 构成（共 {total:,} tokens）")
    print(f"   {'task':<12}{'samples':>8}{'tokens':>10}{'share':>8}{'avg':>8}")
    for r in rows:
        print(f"   {r['task']:<12}{r['samples']:>8}{r['supervised_tokens']:>10}"
              f"{r['token_share']:>7.1%}{r['avg_tokens']:>8}")
    print()
    return rows


def build_sft_config(kwargs, has_eval):
    """不同 transformers/trl 版本的字段名不一致，这里做兼容。"""
    ta = set(inspect.signature(TrainingArguments.__init__).parameters)
    sft = set(inspect.signature(SFTConfig.__init__).parameters)
    if has_eval:
        if "eval_strategy" in ta:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in ta:
            kwargs["evaluation_strategy"] = "steps"
    if "completion_only_loss" in sft:
        kwargs["completion_only_loss"] = True
    if "gradient_checkpointing_kwargs" in ta:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return SFTConfig(**kwargs), sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, cfg.BASE_MODEL_DIR))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "v2"))
    ap.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--run-name", default="v2")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=cfg.MAX_SEQ_LEN)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试：只用前 N 条")
    ap.add_argument("--stats-only", action="store_true",
                    help="只统计监督 token 构成就退出，不训练（用来量 v1 基线）")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3, help="早停耐心（0 = 关闭）")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--output", default="")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=24)
    args = ap.parse_args()

    rank, ws = local_rank(), world_size()
    is_main = rank in (-1, 0)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if rank >= 0:
        torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_adapter = args.output or os.path.join(ROOT, "weights", f"lora_{args.run_name}")
    ckpt_dir = os.path.join(ROOT, "train", "checkpoints", args.run_name)
    log_path = os.path.join(ROOT, "train", "runs", f"{args.run_name}.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 只统计不动训练：2 分钟拿到 v1/v2 的监督 token 构成对比表
    if args.stats_only:
        ds = add_task_column(load_any(args.data))
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        stats = task_token_stats(ds, tokenizer, name=args.data)
        out = os.path.join(ROOT, "train", "runs", f"{args.run_name}_tokens.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"run_name": args.run_name, "data": args.data,
                       "n": len(ds), "token_stats": stats}, f, ensure_ascii=False, indent=2)
        print(f">> {out}")
        return

    model = load_base_model(args.model, args.precision, rank)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if is_main:
        model.print_trainable_parameters()

    train_path = os.path.join(args.data, "train.jsonl")
    dev_path = os.path.join(args.data, "dev.jsonl")
    if not os.path.exists(train_path):
        raise SystemExit(f"缺少 {train_path}，先跑 data/build_dataset_v2.py")
    ds = load_dataset("json", data_files=train_path, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    if is_main:
        print(f">> train {len(ds)} 条；world_size={ws} rank={rank}")

    stats = task_token_stats(ds, tokenizer) if is_main else []
    dev = None
    if os.path.exists(dev_path):
        dev = load_dataset("json", data_files=dev_path, split="train")
        dev = dev.remove_columns([c for c in dev.column_names if c not in ("text", "prompt", "completion")])
        if is_main:
            print(f">> dev {len(dev)} 条")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("text", "prompt", "completion")])

    gpu = torch.cuda.get_device_properties(rank if rank >= 0 else 0)
    if is_main:
        print(f">> GPU {gpu.name}, total {round(gpu.total_memory / 1024 ** 3, 2)} GB")

    cfg_kwargs = dict(
        output_dir=ckpt_dir,
        dataset_text_field="text",
        max_length=args.seq_len,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=20,
        optim="paged_adamw_8bit" if args.precision == "4bit" else "adamw_torch_fused",
        bf16=True,
        report_to=[],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    if dev is not None:
        # load_best_model_at_end 要求 save_steps 是 eval_steps 的整数倍，
        # 这里直接对齐，省得用户在 4 卡上跑两小时后才被参数校验拦住。
        cfg_kwargs["eval_steps"] = args.eval_steps
        cfg_kwargs["save_steps"] = args.eval_steps
        cfg_kwargs["load_best_model_at_end"] = True
        cfg_kwargs["metric_for_best_model"] = "eval_loss"
    sft_cfg, _ = build_sft_config(cfg_kwargs, has_eval=dev is not None)

    callbacks = []
    if dev is not None and args.patience > 0:
        try:
            from transformers import EarlyStoppingCallback

            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
        except Exception:  # noqa: BLE001
            pass

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        eval_dataset=dev,
        args=sft_cfg,
        callbacks=callbacks or None,
    )
    result = trainer.train()

    peak = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
    if is_main:
        info = {
            "run_name": args.run_name,
            "base_model": os.path.basename(args.model.rstrip("/\\")),
            "data": args.data,
            "world_size": ws,
            "precision": args.precision,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum * max(ws, 1),
            "seq_len": args.seq_len,
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
            "train_samples": len(ds),
            "gpu": gpu.name,
            "peak_reserved_gb": peak,
            "train_runtime_s": round(result.metrics.get("train_runtime", 0), 2),
            "train_loss": result.metrics.get("train_loss"),
            "token_stats": stats,
            "eval_history": [
                {k: v for k, v in h.items() if "eval" in k}
                for h in trainer.state.log_history if "eval_loss" in h
            ],
            "loss_history": [
                {"step": h.get("step"), "loss": h.get("loss")}
                for h in trainer.state.log_history if "loss" in h
            ],
        }
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        model.save_pretrained(out_adapter)
        tokenizer.save_pretrained(out_adapter)
        print(f">> adapter {out_adapter}")
        print(f">> peak {peak} GB, runtime {info['train_runtime_s']} s")
        print(f">> log {log_path}")


if __name__ == "__main__":
    main()
