# -*- coding: utf-8 -*-
r"""合并 LoRA adapter 到基座，导出 fp16 权重。"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--adapter", default=os.path.join(BASE_DIR, "weights", "lora_adapter"))
    ap.add_argument("--output", default=os.path.join(BASE_DIR, "weights", "Qwen3-4B-Classical-fp16"))
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output)
    print(f">> merged {args.output}")


if __name__ == "__main__":
    main()
