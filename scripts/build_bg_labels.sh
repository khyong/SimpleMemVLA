#!/usr/bin/env bash
# Build Belief Graph label sidecars for RoboMME (branch bgmem).
#
#   bash scripts/build_bg_labels.sh oracle    # [CPU] replay + oracle labels + validation (resumable)
#   bash scripts/build_bg_labels.sh unsup     # [GPU] OWLv2 + SAM2 + LLM spec + UnsupBG labels (resumable)
#
# Env (defaults = this host's layout):
#   DATA_ROOT     LeRobot robomme_lerobot dir
#   WORK          working dir for npz / replay / labels
#   SIM_PYTHON    python with the RoboMME simulator (ManiSkill fork 07be6fbc) for the replay
#   PYBIN         python of the SimpleMemVLA env (pandas/pyarrow/transformers)
#   WORKERS       parallel replay / label processes (CPU)
#   OWL, SAM2, LLM, DEVICE   UnsupBG models (unsup only)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
MODE="${1:?usage: build_bg_labels.sh oracle|unsup}"
DATA_ROOT="${DATA_ROOT:-./data/datasets/yinchenghust/robomme_lerobot}"
WORK="${WORK:-./data/bg}"
PYBIN="${PYBIN:-python}"
SIM_PYTHON="${SIM_PYTHON:-$PYBIN}"
WORKERS="${WORKERS:-16}"
mkdir -p "$WORK"

# 1) per-episode state/action arrays + LeRobot<->H5 index (from the parquet; CPU, ~1 min)
if [ ! -f "$WORK/episodes_npz/index.json" ]; then
  "$PYBIN" - "$DATA_ROOT" "$WORK/episodes_npz" <<'EOF'
import json, os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq, glob
R, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
cols = ['observation.state', 'action', 'is_video_demo', 'episode_index', 'frame_index', 'subtask_online_index', 'task_index']
t = pq.concat_tables([pq.read_table(f, columns=cols) for f in sorted(glob.glob(f'{R}/data/chunk-*/file-*.parquet'))])
ep = t.column('episode_index').to_numpy(); fr = t.column('frame_index').to_numpy()
S = np.stack(t.column('observation.state').to_numpy(zero_copy_only=False)).astype(np.float32)
A = np.stack(t.column('action').to_numpy(zero_copy_only=False)).astype(np.float32)
D = np.array([bool(np.asarray(x).reshape(-1)[0]) for x in t.column('is_video_demo').to_pylist()])
SO = t.column('subtask_online_index').to_numpy(); TI = t.column('task_index').to_numpy()
cs = json.load(open(f'{R}/meta/_conversion_state.json'))['done_episodes']
sub = pd.read_parquet(f'{R}/meta/subtasks.parquet').index.to_numpy(); tasks = pd.read_parquet(f'{R}/meta/tasks.parquet').index.to_numpy()
order = np.lexsort((fr, ep)); idx = {}
for e in np.unique(ep):
    sel = order[ep[order] == e]
    task, h5ep = cs[e].split('/')
    np.savez_compressed(f'{O}/ep{e:04d}.npz', state=S[sel], action=A[sel], demo=D[sel],
                        subtask=np.array([sub[i] for i in SO[sel]], dtype=object), instruction=str(tasks[TI[sel][0]]))
    idx[int(e)] = {'task': task, 'h5_episode': int(h5ep.split('_')[1]), 'n': int(len(sel)), 'n_demo': int(D[sel].sum())}
json.dump(idx, open(f'{O}/index.json', 'w'))
print(len(idx), 'episodes')
EOF
fi

case "$MODE" in
  oracle)
    # 2) [CPU] seeded replay with privileged snapshots (llvmpipe rendering; ~90 s / episode / process)
    CUDA_VISIBLE_DEVICES= VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}" \
      "$SIM_PYTHON" -m simplememvla.bg.replay_robomme_cli --npz_dir "$WORK/episodes_npz" --out "$WORK/replay" \
      --workers "$WORKERS"
    # 3) [CPU] oracle labels + per-task validation + causality truncation test -> $WORK/labels/oracle_<hash>
    CUDA_VISIBLE_DEVICES= "$PYBIN" -m simplememvla.bg.build_labels_cli oracle --replay "$WORK/replay" \
      --npz_dir "$WORK/episodes_npz" --out "$WORK/labels" --workers "$WORKERS"
    ;;
  unsup)
    OWL="${OWL:-google/owlv2-base-patch16-ensemble}"; SAM2="${SAM2:-facebook/sam2.1-hiera-small}"
    LLM="${LLM:-Qwen/Qwen2.5-7B-Instruct}"; DEVICE="${DEVICE:-cuda}"
    # front-camera calibration from the dataset H5 (H5_DIR holds record_dataset_<Task>.h5); never overwrites
    : "${H5_DIR:?set H5_DIR (RoboMME H5 dir) for the front-camera calibration}"
    "$PYBIN" -m simplememvla.bg.unsup.h5cam extract --h5_dir "$H5_DIR" --npz_dir "$WORK/episodes_npz" \
      --out_dir "$WORK/cams_h5"
    # pilot first (5 episodes per suite, metrics vs. oracle), then everything: PER_SUITE= (empty) for all
    "$PYBIN" -m simplememvla.bg.unsup.pilot --root "$DATA_ROOT" --npz_dir "$WORK/episodes_npz" --replay "$WORK/replay" \
      --owl "$OWL" --sam2 "$SAM2" --out "$WORK/unsup" --device "$DEVICE" --spec llm --llm "$LLM" \
      --cam_dir "$WORK/cams_h5" \
      ${PER_SUITE:+--per_suite $PER_SUITE}
    "$PYBIN" -m simplememvla.bg.unsup.write_labels --rows "$WORK/unsup/rows" --npz_dir "$WORK/episodes_npz" \
      --out "$WORK/labels" ${EXCLUDE:+--exclude_ids "$EXCLUDE"}
    ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
