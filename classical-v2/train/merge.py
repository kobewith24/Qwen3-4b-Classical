# -*- coding: utf-8 -*-
r"""合并 LoRA adapter 到基座，导出 fp16 权重（供 eval_v2 和 GGUF 转换使用）。

    python train/merge.py --run-name v2
    # -> weights/Qwen3-4B-Classical-v2-fp16
"""
import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import common  # noqa: E402
import config as cfg  # noqa: E402

ROOT = common.find_project_root(PKG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, cfg.BASE_MODEL_DIR))
    ap.add_argument("--run-name", default="v2")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    adapter = args.adapter or os.path.join(ROOT, "weights", f"lora_{args.run_name}")
    output = args.output or os.path.join(
        ROOT, "weights", f"{cfg.BASE_MODEL_NAME}-Classical-{args.run_name}-fp16")
    if not os.path.isdir(adapter):
        raise SystemExit(f"找不到 adapter：{adapter}")

    print(f">> base {args.model}")
    print(f">> adapter {adapter}")
    _tf_major = int(__import__("transformers").__version__.split(".")[0])
    dtype_kw = {"dtype" if _tf_major >= 5 else "torch_dtype": torch.float16}
    base = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kw)
    model = PeftModel.from_pretrained(base, adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(output)
    print(f">> merged {output}")


if __name__ == "__main__":
    main()
