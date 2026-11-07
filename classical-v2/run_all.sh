#!/usr/bin/env bash
# 一键：造数据 -> 训练 -> 合并 -> 评测
#   NUM_GPU=4 RUN=v2 bash run_all.sh
#   NUM_GPU=1 RUN=smoke LIMIT=500 bash run_all.sh
set -euo pipefail

cd "$(dirname "$0")"

RUN="${RUN:-v2}"
NUM_GPU="${NUM_GPU:-4}"
BS="${BS:-2}"
GA="${GA:-4}"
LIMIT="${LIMIT:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [ "$SKIP_BUILD" != "1" ]; then
  echo "== 1/4 构建数据集 =="
  python data/build_dataset_v2.py
  echo
  cat data/v2/mix_report.md 2>/dev/null || true
fi

echo
echo "== 2/4 训练（run=$RUN, gpu=$NUM_GPU, batch=${BS}x${GA}）=="
EXTRA=""
if [ "$LIMIT" != "0" ]; then
  EXTRA="--limit $LIMIT --max-steps 20"
fi
if [ "$NUM_GPU" -gt 1 ]; then
  accelerate launch --num_processes "$NUM_GPU" --mixed_precision bf16 \
    train/finetune.py --run-name "$RUN" --batch-size "$BS" --grad-accum "$GA" $EXTRA
else
  python train/finetune.py --run-name "$RUN" --batch-size 8 --grad-accum 4 $EXTRA
fi

echo
echo "== 3/4 合并 =="
python train/merge.py --run-name "$RUN"

echo
echo "== 4/4 评测（bench + ood + general）=="
python eval/eval_v2.py --backend hf --tag "$RUN"

echo
echo "完成。要下载回来的文件："
echo "  data/v2/mix_report.md  data/v2/stats.json"
echo "  train/runs/$RUN.json"
echo "  eval/reports/eval_$RUN.json  eval/reports/eval_$RUN.md"
