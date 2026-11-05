# -*- coding: utf-8 -*-
r"""
训练：30k 任务覆盖数据 + completion-only loss。

本机 8GB：
  python train\finetune.py --precision 4bit --epochs 2
A6000 48GB：
  python train/finetune.py --precision bf16 --batch-size 16 --grad-accum 2 --seq-len 2048 --no-gradient-checkpointing --epochs 2
"""
import argparse
import json
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "classical_dataset")
DEFAULT_ADAPTER_DIR = os.path.join(BASE_DIR, "weights", "lora_adapter")


def load_base_model(model_dir, precision):
    if precision == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb, device_map={"": 0},
        )
    return AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--data", default=DEFAULT_DATA_DIR)
    ap.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--output", default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--log", default=os.path.join(BASE_DIR, "train", "training_log.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to train on CPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(args.model, args.precision)
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = load_from_disk(args.data)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f">> train samples {len(dataset)}")

    gpu = torch.cuda.get_device_properties(0)
    print(f">> GPU {gpu.name}, total {round(gpu.total_memory / 1024**3, 2)} GB")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=os.path.join(BASE_DIR, "train", "checkpoints"),
            dataset_text_field="text",
            completion_only_loss=True,
            max_length=args.seq_len,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            logging_steps=20,
            optim="paged_adamw_8bit" if args.precision == "4bit" else "adamw_torch_fused",
            bf16=True,
            report_to=[],
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            gradient_checkpointing=args.gradient_checkpointing,
            seed=24,
        ),
    )
    stats = trainer.train()

    peak = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    log = {
        "base_model": "Qwen3-4B-Instruct-2507",
        "dataset": args.data,
        "train_samples": len(dataset),
        "precision": args.precision,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_len": args.seq_len,
        "gradient_checkpointing": args.gradient_checkpointing,
        "completion_only_loss": True,
        "gpu": gpu.name,
        "peak_reserved_gb": peak,
        "train_runtime_s": round(stats.metrics.get("train_runtime", 0), 2),
        "train_loss": stats.metrics.get("train_loss"),
        "loss_history": [
            {"step": h.get("step"), "loss": h.get("loss")}
            for h in trainer.state.log_history if "loss" in h
        ],
    }
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f">> adapter {args.output}")
    print(f">> peak {peak} GB, runtime {log['train_runtime_s']} s")


if __name__ == "__main__":
    main()
