# -*- coding: utf-8 -*-
r"""合并 LoRA adapter 到基座，导出 fp16 权重 → weights/Qwen3-4B-Classical-fp16"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507")
ADAPTER_DIR = os.path.join(BASE_DIR, "weights", "lora_adapter")
OUT_DIR = os.path.join(BASE_DIR, "weights", "Qwen3-4B-Classical-fp16")


def main():
    base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    merged = model.merge_and_unload()
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL_DIR).save_pretrained(OUT_DIR)
    print(f">> 已导出 {OUT_DIR}")


if __name__ == "__main__":
    main()
