#!/usr/bin/env bash
# GPU tests of the bgmem branch (run on the RTX PRO 6000 server; nothing here was run on the 4090 host).
#
#   STEP=mem    bash scripts/bg_gpu_suite.sh   # test 7: peak memory + samples/s at per-GPU batch 4, 6, 8 (ZeRO-2 + GC)
#   STEP=train  bash scripts/bg_gpu_suite.sh   # test 8: 200-step smoke runs R3 (unsup) and R4 (oracle) from the
#                                              #         public checkpoint (+ R2 control with the same data)
#   STEP=eval   bash scripts/bg_gpu_suite.sh   # test 9: smoke closed-loop eval, 1 task per suite x 2 episodes
#   STEP=force  bash scripts/bg_gpu_suite.sh   # test 10: forced BG changes the condition hidden states
#   STEP=offline bash scripts/bg_gpu_suite.sh  # BG CE / teacher-forced fact accuracy on held-out episodes
#
# Required env: CUDA_VISIBLE_DEVICES, INIT_CHECKPOINT (yinchenghust/simplememvla_robomme), ROOT_DIR (robomme_lerobot),
#               ORACLE_LABELS, UNSUP_LABELS (label dirs from scripts/build_bg_labels.sh), OUT (results dir).
# Every run writes into $OUT/<step>/; training runs resume with RESUME=auto.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
: "${CUDA_VISIBLE_DEVICES:?}" "${INIT_CHECKPOINT:?}" "${ROOT_DIR:?}" "${OUT:?}"
STEP="${STEP:?set STEP=mem|train|eval|force|offline}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr , '\n' | wc -l)
EXCL="${ORACLE_LABELS:+$ORACLE_LABELS/excluded_ids.json}"
common=(BENCHMARK=robomme ROOT_DIR="$ROOT_DIR" INIT_CHECKPOINT="$INIT_CHECKPOINT" REPORT_TO=none
        GPUS_PER_NODE="$NGPU" DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}" WARMUP_STEPS=20
        SAVE_STEPS=100 LOGGING_STEPS=5 RESUME=auto ${EXCL:+EXCLUDE_EPISODES_FILE=$EXCL})

case "$STEP" in
  mem)
    for B in 4 6 8; do
      env "${common[@]}" BG_MODE=oracle BG_LABEL_DIR="$ORACLE_LABELS" PER_DEVICE_BATCH=$B GRAD_ACCUM=1 MAX_STEPS=25 \
        MIN_ANCHOR_FRAME=1200 RUN_NAME="mem_b$B" OUTPUT_DIR="$OUT/mem/b$B" RESUME=false \
        bash scripts/train_bg.sh robomme 2>&1 | tee "$OUT/mem_b$B.log" || echo "batch $B failed (OOM?)" | tee -a "$OUT/mem_b$B.log"
      grep -o "'mem/peak_alloc_gb': [0-9.]*\|'perf/samples_per_s': [0-9.]*" "$OUT/mem_b$B.log" | tail -4
    done ;;
  train)
    for R in R2:none R3:unsup R4:oracle; do
      name=${R%%:*}; mode=${R#*:}
      lab=""; [ "$mode" = unsup ] && lab="$UNSUP_LABELS"; [ "$mode" = oracle ] && lab="$ORACLE_LABELS"
      env "${common[@]}" BG_MODE=$mode ${lab:+BG_LABEL_DIR=$lab} PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}" \
        MAX_STEPS=200 HOLDOUT_EPISODES_PER_TASK=2 RUN_NAME="smoke_$name" OUTPUT_DIR="$OUT/train/$name" \
        bash scripts/train_bg.sh robomme 2>&1 | tee "$OUT/train_$name.log"
    done
    "${PYBIN:-python}" - "$OUT" <<'EOF'
import json, re, sys, glob
for log in sorted(glob.glob(sys.argv[1] + '/train_R*.log')):
    rows = [eval(m) for m in re.findall(r"\{'loss'.*?\}", open(log).read())]
    if not rows: print(log, 'no logs'); continue
    nan = any(any(isinstance(v, float) and v != v for v in r.values()) for r in rows)
    f = lambda k: [round(r[k], 4) for r in rows if k in r]
    print(log, 'NaN' if nan else 'finite', 'subtask_ce', f('subtask_ce')[:2], '->', f('subtask_ce')[-2:],
          'bg_ce', f('bg_ce')[:2], '->', f('bg_ce')[-2:], 'action', f('action_loss')[:1], '->', f('action_loss')[-1:])
EOF
    ;;
  eval)
    CK="${CHECKPOINT:-$OUT/train/R4}"
    for T in PickXtimes VideoUnmaskSwap VideoRepick InsertPeg; do
      "${PYBIN:-python}" -m robomme_sim.eval_bg --pretrained_checkpoint "$CK" --dataset_split test --episodes_per_task 2 \
        --group_size 2 --tasks $T --num_gpus 1 --max_subtask_tokens_auto "${LABELS_FOR_BUDGET:-$ORACLE_LABELS}" \
        --bg_log "$OUT/eval/bg_log.jsonl" --log_file "$OUT/eval/$T.log" ${BG_OVERRIDE:+--bg_override $BG_OVERRIDE}
    done
    "${PYBIN:-python}" - "$OUT/eval/bg_log.jsonl" <<'EOF'
import json, sys, collections
rs = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rs: by[r['task']].append(r)
for t, v in by.items():
    print(t, 'decisions', len(v), 'parse_ok %.3f' % (sum(r['bg_parse_ok'] for r in v) / len(v)),
          'runaway', sum(r['runaway'] for r in v),
          'prefill %.3f decode %.3f denoise %.3f s' % tuple(sum(r[k] for r in v) / len(v) for k in ('t_prefill', 't_decode', 't_denoise')))
EOF
    ;;
  force)
    cat > "$OUT/edits.json" <<'EOF'
[{"name": "edit_hidden_swap", "pattern": "under container_A", "repl": "under container_B"},
 {"name": "edit_counter", "pattern": "=(\\d)/", "repl": "=0/"},
 {"name": "edit_visible_contradiction", "pattern": "now at left", "repl": "now at right"}]
EOF
    "${PYBIN:-python}" -m robomme_sim.bg_force_test --checkpoint "${CHECKPOINT:-$OUT/train/R4}" --root "$ROOT_DIR" \
      --labels "$ORACLE_LABELS" --episodes 1400 1500 700 1300 --frames 150 200 300 500 --edits "$OUT/edits.json" \
      --log "$OUT/force.jsonl" ;;
  offline)
    for R in R3 R4; do
      lab="$ORACLE_LABELS"; [ $R = R3 ] && lab="$UNSUP_LABELS"
      "${PYBIN:-python}" -m simplememvla.bg.eval_offline --checkpoints $(ls -d "$OUT/train/$R"/checkpoint-* "$OUT/train/$R" 2>/dev/null) \
        --root "$ROOT_DIR" --labels "$lab" --holdout "$OUT/train/$R/holdout_episodes.json" --out "$OUT/offline_$R.json"
    done ;;
esac
