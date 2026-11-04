#!/usr/bin/env bash
# [GPU] One-shot held-out evaluation of UnsupBG vs. OracleBG (report 03 §5-3).
#
#   CUDA_VISIBLE_DEVICES=0 OUT=<new dir> CAM_DIR=<h5 cams> bash scripts/unsup_heldout_eval.sh
#
# Runs once per (freeze, evaluation set): OUT must not exist, and the UnsupBG code must match FREEZE.
# If the rules change after this run, make a new freeze file (heldout.py freeze --out ..._v2.json), record why in
# report 03, and evaluate on a NEW evaluation set (heldout.py select --seed <new> excluding the old one).
#
# Env:
#   OUT          new output directory (required; must not exist)
#   EVALSET      evaluation set json   (default reports/raw/unsup_heldout_evalset_s20261105.json, 48 episodes)
#   FREEZE       rule freeze json      (default reports/raw/unsup_freeze_v2.json: camera calibration from H5)
#   CAM_DIR      front-camera calibration from the dataset H5 (simplememvla.bg.unsup.h5cam extract; required)
#   DATA_ROOT    LeRobot robomme_lerobot dir
#   NPZ_DIR      episodes_npz (from build_bg_labels.sh step 1)
#   REPLAY       oracle replay dir <REPLAY>/<task>/epXXXX.pkl.gz (from build_bg_labels.sh oracle; CPU) -- oracle side
#                of the metric only; UnsupBG does not read it
#   OWL, SAM2, LLM   model dirs / ids (must match FREEZE: basename of OWL/SAM2, LLM id)
#   PYBIN        python of the SimpleMemVLA env
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES explicitly}"
: "${OUT:?set OUT to a new directory}"
: "${CAM_DIR:?set CAM_DIR (h5cam extract output)}"
EVALSET="${EVALSET:-reports/raw/unsup_heldout_evalset_s20261105.json}"
FREEZE="${FREEZE:-reports/raw/unsup_freeze_v2.json}"
DATA_ROOT="${DATA_ROOT:-./data/datasets/yinchenghust/robomme_lerobot}"
NPZ_DIR="${NPZ_DIR:-./data/bg/episodes_npz}"
REPLAY="${REPLAY:-./data/bg/replay}"
OWL="${OWL:-google/owlv2-base-patch16-ensemble}"
SAM2="${SAM2:-facebook/sam2.1-hiera-small}"
LLM="${LLM:-Qwen/Qwen2.5-7B-Instruct}"
PYBIN="${PYBIN:-python}"

if [ -e "$OUT" ]; then echo "OUT=$OUT exists; the held-out run is done once per freeze/evaluation set" >&2; exit 1; fi

# 1) rules must be the frozen ones
"$PYBIN" -m simplememvla.bg.unsup.heldout verify --freeze "$FREEZE" --llm "$LLM" --owl "$OWL" --sam2 "$SAM2" --spec llm

EPS=$("$PYBIN" -c "import json,sys;print(' '.join(map(str,json.load(open(sys.argv[1]))['episodes'])))" "$EVALSET")
for e in $EPS; do
  t=$("$PYBIN" -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]['task'])" "$NPZ_DIR/index.json" "$e")
  f=$(printf "%s/%s/ep%04d.pkl.gz" "$REPLAY" "$t" "$e")
  [ -f "$f" ] || { echo "missing oracle replay $f (run build_bg_labels.sh oracle first)" >&2; exit 1; }
  [ -f "$CAM_DIR/$t.json" ] || { echo "missing H5 camera calibration $CAM_DIR/$t.json (run h5cam extract)" >&2; exit 1; }
done

mkdir -p "$OUT"
"$PYBIN" - "$OUT/run.json" "$EVALSET" "$FREEZE" "$CAM_DIR" <<'EOF'
import glob, json, os, socket, sys, time, hashlib
out, ev, fr, cd = sys.argv[1:5]
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
           "evalset": ev, "evalset_sha256": h(ev), "freeze": fr, "freeze_sha256": h(fr), "cam_dir": cd,
           "cam_sha256": {os.path.basename(p): h(p) for p in sorted(glob.glob(os.path.join(cd, "*.json")))}},
          open(out, "w"), indent=1)
EOF

# 2) UnsupBG labels (LLM spec, GPU) + exact metric vs. oracle
"$PYBIN" -m simplememvla.bg.unsup.pilot --root "$DATA_ROOT" --npz_dir "$NPZ_DIR" --replay "$REPLAY" \
  --owl "$OWL" --sam2 "$SAM2" --llm "$LLM" --spec llm --device cuda --cam_dir "$CAM_DIR" --out "$OUT" --episodes $EPS \
  2>&1 | tee "$OUT/pilot.log"

# 3) exact + ID-aligned metric (labels are not changed)
"$PYBIN" -m simplememvla.bg.unsup.rescore --out "$OUT" --npz_dir "$NPZ_DIR" --replay "$REPLAY"
"$PYBIN" -m simplememvla.bg.unsup.summarize "$OUT/metrics_aligned.jsonl" --md "$OUT/summary_exact.md"
"$PYBIN" -m simplememvla.bg.unsup.summarize "$OUT/metrics_aligned.jsonl" --aligned --md "$OUT/summary_aligned.md"
echo "done: $OUT/summary_exact.md, $OUT/summary_aligned.md"
