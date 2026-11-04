# BGMemVLA (branch `bgmem`): Belief Graph CoT on SimpleMemVLA

The policy states the current sub-task and then a serialized **Belief Graph (BG)** in the same assistant answer:

```
put it into the bin
Belief: count binned(red_cube)=1/2 ; #1 picked(red_cube_A) 9.5s ago ; #2 placed(red_cube_A,bin) 7.0s ago ; red_cube_A in bin since 7.0s ago
```

The action head already reads the hidden states of the whole answer span (`subtask_span_mask`), so the BG reaches
the action head without any model change. All new code is opt-in; `bg_mode=none` reproduces the official pipeline
batch for batch (`tests/bg/test_identity.py`). Official files (`train.py`, `scripts/train.sh`, `robomme_sim/*` except the
new modules, `simplememvla/{data,model,training}`) are unchanged.

| Run | `bg_mode` | answer target |
|---|---|---|
| R2 | `none` | sub-task (control) |
| R3 | `unsup` | sub-task + UnsupBG (video + proprio + instruction only) |
| R4 | `oracle` | sub-task + BG from privileged simulator replay |

R2/R3/R4 use the same episodes (`--exclude_episodes_file <oracle labels>/excluded_ids.json` for all three).

## Layout

```
simplememvla/bg/
  schema.py, serialize.py      BG representation, deterministic text, parser, mnemonic filter, truncation
  robomme_replay.py            seeded replay recorder (privileged snapshots per frame)
  replay_robomme_cli.py        replay of all LeRobot episodes (resumable, fidelity log)
  oracle_robomme.py            BG-Oracle builder (offline labels) + OracleBGTracker (online, evaluation)
  oracle_checks.py             per-task oracle validation (counting / permanence / reference / imitation)
  build_labels_cli.py          oracle labels + validation + causality test -> labels/oracle_<hash>/
  labels.py                    label sidecar (labels.parquet + manifest.json with label_hash)
  args.py, dataset.py, collator.py, model.py, trainer.py    training integration
  eval_offline.py              held-out BG CE / teacher-forced fact accuracy per checkpoint
  unsup/                       UnsupBG: proprio.py (gripper events + Panda FK), instruction.py (LLM spec, cached),
                               visual.py (OWLv2 + SAM2), builder.py, metrics.py, pilot.py, write_labels.py
train_bg.py, scripts/train_bg.sh             BG training entry (official train.py / train.sh untouched)
robomme_sim/bg_policy.py, eval_bg.py         BG-aware batched policy + evaluation wrapper
robomme_sim/bg_force_test.py                 teacher-forced BG edit test (test 10)
scripts/build_bg_labels.sh                   label generation (oracle on CPU, unsup on GPU)
scripts/bg_gpu_suite.sh                      GPU tests 7-10 + offline validation
tests/bg/                                    CPU tests (identity, serialization, data path)
reports/                                     00_survey ... 05_smoke_train_eval
```

## Labels

```bash
# [CPU] replay + oracle labels (SIM_PYTHON = env with the RoboMME simulator)
DATA_ROOT=.../robomme_lerobot WORK=.../bg SIM_PYTHON=... PYBIN=... WORKERS=32 bash scripts/build_bg_labels.sh oracle
# [GPU] UnsupBG pilot (5 episodes / suite) then all episodes
#   (front-camera calibration is read from the dataset H5: H5_DIR = dir with record_dataset_<Task>.h5)
H5_DIR=... PER_SUITE=5 DEVICE=cuda bash scripts/build_bg_labels.sh unsup
H5_DIR=... DEVICE=cuda EXCLUDE=$WORK/labels/oracle_<hash>/excluded_ids.json bash scripts/build_bg_labels.sh unsup
# held-out UnsupBG vs Oracle score (once per freeze, default freeze v2; see reports/03 §5-3)
CUDA_VISIBLE_DEVICES=0 OUT=<new dir> NPZ_DIR=$WORK/episodes_npz REPLAY=$WORK/replay CAM_DIR=$WORK/cams_h5 \
  bash scripts/unsup_heldout_eval.sh
```

## Training

```bash
# R4 (oracle), starting from the released checkpoint; host GPUs must be given explicitly
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BG_MODE=oracle BG_LABEL_DIR=$WORK/labels/oracle_<hash> \
  EXCLUDE_EPISODES_FILE=$WORK/labels/oracle_<hash>/excluded_ids.json HOLDOUT_EPISODES_PER_TASK=2 \
  INIT_CHECKPOINT=checkpoints/simplememvla_robomme bash scripts/train_bg.sh robomme
# options (defaults = official): BG_FILTER=mnemonic|full  BG_MAX_INTERACTIONS=4  BG_DROPOUT_PROB  BG_CORRUPT_PROB
#   CONDITION_SPAN=all|subtask_only  VL_LOSS_REDUCTION=token_mean|span_mean  ANCHOR_STRIDE  ANCHOR_PHASE_BY_EPOCH
```
Logged in addition to the official metrics: `subtask_ce`, `bg_ce`, `bg_tokens`, `mem/peak_alloc_gb`,
`perf/samples_per_s`. The BG settings and the label hash are stored in `config.json` and `bg_run.json`.

## Evaluation

```bash
python -m robomme_sim.eval_bg --pretrained_checkpoint <ck> --tasks VideoUnmaskSwap --episodes_per_task 50 \
  --num_gpus 1 --max_subtask_tokens_auto <label dir> --bg_log run/bg_log.jsonl
# forced BG (teacher forcing after the generated sub-task):
#   --bg_override oracle                     online OracleBGTracker on the evaluation env (R4 inference)
#   --bg_override file --bg_override_file texts.json   {"<task>/<episode>": "Belief: ..."} or {"<task>": ...}
#   --bg_edit_spec edits.json                [{"pattern": "under container_A", "repl": "under container_B"}]
#   --condition_span subtask_only            same ablation as training (default: from the checkpoint config)
```
`eval_bg` wraps `robomme_sim.eval_success` (same arguments) and supports the default batched path with
`--num_gpus 1` per process (the module patches do not reach spawned GPU workers). The loop is synchronous, so
timing is recorded, not optimized: each `bg_log` line has the sub-task, BG, parse result, token counts, runaway
flag and prefill / decode / denoise seconds.
