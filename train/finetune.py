# -*- coding: utf-8 -*-
r"""
QLoRA 微调：Qwen3-4B-Instruct-2507 + 文言文混合数据集
8GB 显存适配：4bit NF4 + LoRA r=32 + 梯度检查点 + batch=1 × 梯度累积 8

产物：weights/lora_adapter、train/training_log.json
"""
import json
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507")
DATA_DIR = os.path.join(BASE_DIR, "data", "classical_dataset")
ADAPTER_DIR = os.path.join(BASE_DIR, "weights", "lora_adapter")
LOG_PATH = os.path.join(BASE_DIR, "train", "training_log.json")

MAX_SEQ_LENGTH = 1024


def load_base_model():
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, quantization_config=bnb, device_map="auto",
        )
        print(">> 4bit NF4 加载成功")
        return model, "4bit-nf4"
    except Exception as e:  # noqa: BLE001
        print(f">> 4bit 不可用（{e}），回退 bf16")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
        )
        return model, "bf16"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, quant = load_base_model()

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = load_from_disk(DATA_DIR)
    print(f">> 训练集 {len(dataset)} 条")

    gpu = torch.cuda.get_device_properties(0)
    total_gb = round(gpu.total_memory / 1024**3, 2)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f">> GPU {gpu.name}，总显存 {total_gb} GB，加载后占用 {start_mem} GB")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=20,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            output_dir=os.path.join(BASE_DIR, "train", "checkpoints"),
            optim="paged_adamw_8bit" if quant == "4bit-nf4" else "adamw_torch",
            report_to=[],
            bf16=True,
            save_strategy="no",
            gradient_checkpointing=True,
            seed=24,
        ),
    )
    stats = trainer.train()

    peak = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    log = {
        "base_model": "Qwen3-4B-Instruct-2507",
        "quant_mode": quant,
        "gpu": gpu.name,
        "gpu_total_gb": total_gb,
        "peak_reserved_gb": peak,
        "train_samples": len(dataset),
        "train_runtime_s": round(stats.metrics.get("train_runtime", 0), 2),
        "train_loss": stats.metrics.get("train_loss"),
        "loss_history": [{"step": h.get("step"), "loss": h.get("loss")}
                         for h in trainer.state.log_history if "loss" in h],
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f">> 日志: {LOG_PATH}")
    print(f">> 峰值显存 {peak} GB / {total_gb} GB")

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f">> adapter: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
