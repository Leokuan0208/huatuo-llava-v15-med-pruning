#!/usr/bin/env bash
# VM-A: baseline pair + all 9 scored runs + 2 sampled-nested runs (~12.5 hr)
set -u
MODEL=/data/dan/weights/HuatuoGPT-Vision-7B
DATA=/data/dan/dataset/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json
OUT=/home/jamesyang/huatuo-llava-v15-med-pruning/results/scored_sweep
LOGDIR=$OUT/logs; mkdir -p "$LOGDIR"

# "mode:pruner:kr"
RUNS=(
  "sampled:none:1.0"
  "scored:none:1.0"
  "scored:random:0.75" "scored:random:0.5" "scored:random:0.25" "scored:random:0.1"
  "scored:nested_random:0.75" "scored:nested_random:0.5" "scored:nested_random:0.25" "scored:nested_random:0.1"
  "sampled:nested_random:0.25" "sampled:nested_random:0.1"
)

run_one () {
  local mode=$1 pruner=$2 kr=$3
  local tag; case "$pruner" in
    none) tag="baseline";;
    random) tag=$(printf "RandomPruner_kr%.2f" "$kr");;
    nested_random) tag=$(printf "NestedRandomPruner_kr%.2f" "$kr");;
  esac
  local pred="$OUT/$(basename $MODEL)__${tag}__${mode}__predictions.json"
  if [ -f "$pred" ]; then echo "[$(date +%H:%M:%S)] SKIP: $mode $pruner kr=$kr"; return; fi
  echo "[$(date +%H:%M:%S)] START: $mode $pruner kr=$kr"
  torchrun --nproc_per_node=1 scripts/scored_sweep.py \
    --mode "$mode" --pruner "$pruner" --keep_ratio "$kr" \
    --model_path "$MODEL" --data_path "$DATA" --output_dir "$OUT" \
    > "$LOGDIR/${mode}_${pruner}_kr${kr}.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && echo "[$(date +%H:%M:%S)] FAIL(rc=$rc): $mode $pruner kr=$kr -> $LOGDIR/${mode}_${pruner}_kr${kr}.log" || echo "[$(date +%H:%M:%S)] DONE: $mode $pruner kr=$kr"
}

echo "===== VM-A starting $(date) ====="
for r in "${RUNS[@]}"; do IFS=':' read -r m p k <<< "$r"; run_one "$m" "$p" "$k"; done
echo "===== VM-A finished $(date) ====="
