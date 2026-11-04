# 05. Smoke 학습·평가와 강제 입력 (테스트 8, 9, 10) — GPU 미실행

**상태: GPU 테스트는 실행하지 않았다** (사용자 결정에 따라 RTX PRO 6000 서버에서 진행). 이 서버에서는 같은 경로를 CPU로 확인할 수 있는 부분만 확인했다.

## 이 서버에서 확인한 것 (CPU)

| 항목 | 결과 |
|---|---|
| 테스트 1: `bg_mode=none`이 공식과 같음 (100 sample; input_ids, labels, pixel, action, state, condition mask, batch key) | 통과 (`reports/raw/test1_identity.json`) |
| BG 데이터 경로 (oracle v3 label, 60 sample, `condition_span=all/subtask_only`) | 통과 (`reports/raw/test_bg_datapath.json`) |
| └ 답변 = subtask + `\n` + BG, collator의 BG mask가 `Belief:`에서 정확히 나뉨 | 통과 |
| └ subtask_only condition = subtask token만 (구분 `\n`과 BG 제외) | 통과 |
| └ holdout(과제당 1) 제외, 제외 episode 미포함, anchor stride 4의 phase가 epoch마다 바뀜 | 통과 |
| └ 제외 목록 없이 BG 학습을 시작하면 즉시 오류 | 통과 |
| online `OracleBGTracker` (시뮬레이터, CPU) | 동작 (02 §4) |
| 모든 새 모듈 `py_compile` | 통과 |

## GPU 서버에서 실행할 것 (`scripts/bg_gpu_suite.sh`)

공통 env: `CUDA_VISIBLE_DEVICES`, `INIT_CHECKPOINT`, `ROOT_DIR`, `ORACLE_LABELS=<labels>/oracle_3a07839db0eb`, `UNSUP_LABELS` (03의 전체 생성 후), `OUT`.

### 테스트 8: smoke 학습 (`STEP=train`)
- 공개 checkpoint에서 R2(none), R3(unsup), R4(oracle)를 각각 200 step 학습한다. warmup 20, holdout 과제당 2 episode, 세 조건 모두 같은 제외 목록을 쓴다.
- 판정: R3, R4에서 `subtask_ce`와 `bg_ce`가 모두 감소하고 NaN이 없어야 한다. 스크립트 끝에서 log를 요약해 출력한다.
- R3은 UnsupBG label이 있어야 하므로 03의 전체 생성 뒤에 한다.

### 테스트 9: smoke 평가 (`STEP=eval`)
- suite당 과제 1개(PickXtimes, VideoUnmaskSwap, VideoRepick, InsertPeg) × 2 episode, `--max_subtask_tokens_auto <labels>`.
- 판정 (`bg_log.jsonl` 요약): BG parse 성공 95% 이상, runaway 0, prefill/decode/denoise 평균 시간 기록.
- `BG_OVERRIDE=oracle`이면 R4 추론(online tracker로 BG 강제 입력)을 한다.

### 테스트 10: 강제 입력 (`STEP=force`)
- `robomme_sim/bg_force_test.py`: 녹화 frame으로 history를 채운 같은 입력에 세 가지를 넣는다.
  - 모델 생성 BG
  - label BG 강제
  - 편집한 BG: 보이지 않는 fact 변경(`under container_A` → `container_B`, counter → 0), 보이는 fact와 모순되는 변경(`now at left` → `right`)
- 판정: 편집하면 condition hidden(L2, cosine)과 action이 label 강제 때와 달라지고, `force.jsonl`에 기록된다.

### 오프라인 검증 (`STEP=offline`)
- `simplememvla/bg/eval_offline.py`: holdout episode에서 checkpoint마다 다음을 계산한다.
  - subtask CE, BG CE
  - teacher-forced BG token 정확도
  - fact precision/recall
  - parse 성공률

## 주의
- `eval_bg`는 `--num_gpus 1`만 지원한다. 모듈 patch가 spawn된 worker에 전달되지 않기 때문이다. 여러 GPU를 쓰려면 과제를 나눠 process를 여러 개 띄운다.
- 평가는 공식 test split에서 한다. smoke 평가 결과로 설정을 고르지 않는다.
