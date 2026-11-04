# 03. UnsupBG와 Oracle 비교 (RoboMME) — CPU pilot

**상태: 이 서버에서 CPU로 돌린 pilot이다.** instruction spec은 규칙 parser(`--spec rule`)를 썼다. LLM spec(Qwen2.5-7B-Instruct)과 전체 1,588 episode 생성은 GPU 서버에서 한다 (`scripts/build_bg_labels.sh unsup`). 아래 수치는 suite당 5 episode, 모두 20개 결과라 분산이 크다. 이 결과로 학습이나 평가 설정을 고르지 않았다.

- episode 목록: `/data2/hoyong/data/smv/unsup_pilot_eps.json`. train split에서 과제 순서대로 골랐고, test episode는 쓰지 않았다.
- 출력: `/data2/hoyong/data/smv/unsup_pilot_cpu_v4/`
  - `metrics.jsonl`, `rows/`
  - `tracks/`: `unsup_pilot_cpu_rule/tracks`를 가리키는 symlink
- 표: `reports/raw/03_pilot_cpu.md`, `03_pilot_cpu.json`
- 모델: OWLv2-base (`owlv2-base-patch16-ensemble`), SAM2.1-hiera-small. 둘 다 CPU에서 돌렸다.

## 1. 파이프라인 요약

| 단계 | 입력 | 방법 |
|---|---|---|
| 잡기/놓기/누르기 사건 | proprio (관절, gripper 폭) | gripper 폭 사건 (잡음 0.018, 빈손 0)과 Panda FK로 구한 TCP 위치 (stick 과제는 끝점 0.15 m) |
| 물체 track | front 카메라 영상 | OWLv2로 검출하고 SAM2로 video tracking. 색 큐브는 색 pixel로 보임 여부를 판단한다 |
| 사건 대상 | 사건 시점 TCP를 카메라에 투영 | 그 시점에 보이는 track 가운데 가장 가까운 것 |
| 과제 구조 (counter 목표, 지정 방식) | instruction | 규칙 parser (pilot), LLM (GPU 서버) |
| demo 경계 | proprio | 가장 큰 관절 도약 |

ID는 처음 보인 순서로 매기고, 같은 frame이면 왼쪽부터 매긴다. 직렬화는 oracle과 같은 v3 형식이다 (`serialize.facts`). 인과성은 oracle과 같은 incremental 구조로 지킨다.

## 2. Proprio 단계 단독 검증 (probe, privileged replay 기준)

- 잡기/놓기: cube·container 과제에서 precision 1.0, recall 1.0.
- 닫힘→누름(button) 판정: precision 0.5. 빈손으로 닫는 동작이 누름이 아닐 때가 있다.
- 사건 시점 FK 오차: 0.3 mm 이하.
- demo 끝 검출: VideoUnmask(Swap)와 stick 과제를 뺀 모든 과제에서 정확하다.
  - VideoUnmask(Swap)는 demo 동안 로봇이 움직이지 않는다.
  - stick 과제는 70~83%만 정확하다.
- 이 probe의 raw 출력은 파일로 남기지 않았다 (`eval_proprio.py`로 다시 만들 수 있다).

## 3. Pilot 결과

### 3-1. 두 가지 지표

**정확 일치 (exact)**
- 매 frame BG의 fact를 정규화한 문자열이 정확히 같아야 일치로 센다 (`metrics.canon`).
- 시간 표기와 interaction 번호 `#k`는 빼고 비교한다.
- ID 문자와 이름까지 같아야 한다.

**ID 정렬 일치 (aligned)**: `metrics.compare_episode(align=True)`, `unsup/rescore.py`
- 물체 ID(`container_A`, `cube_C`)는 임의로 붙인 이름이다. 그래서 앞에서 track이 하나 더 생기면 뒤의 문자가 모두 밀리고, 정확 일치 지표에서는 같은 구조가 0이 된다.
- episode마다 unsup ID를 oracle ID로 바꾸는 **고정된 1:1 대응 하나**를 구해서 이름을 바꾼 뒤 비교한다.
  - 같은 ID 없는 골격(skeleton)의 fact에서 같은 자리에 함께 나오는 횟수를 점수로 쓰고, Hungarian으로 대응을 고른다.
  - 대응은 category가 맞는 쌍만 허용한다. 색 접두어를 뺀 기본 category가 같고, 색이 서로 다르지 않아야 한다. 예를 들어 `blue_cube_C → cube_C`는 허용하고, `red_cube → green_cube`는 허용하지 않는다.
  - 대응되지 않은 unsup ID는 `u:` 접두어를 붙여 oracle ID와 우연히 겹치지 않게 한다.
- **이 지표는 측정에만 쓰고, label은 바꾸지 않는다.**
  - aligned 지표를 추가한 뒤로는 unsup label 생성 규칙을 바꾸지 않았다. 그 전에 pilot 결과를 보고 바꾼 규칙은 §5에 있다.
  - 대응을 oracle을 보고 맞추므로 aligned 값은 구조 일치의 **낙관적 상한**이다. label 품질 수치로 단독 인용하지 않는다.
- 학습(R3)에서 필요한 것은 oracle과 같은 문자를 쓰는 것이 아니라, episode 안에서 ID가 일관적인 것이다. aligned 지표가 이 성질을 재는 데 더 가깝다.

### 3-2. Suite별 결과

fact 유형별 precision / recall이고, 괄호는 oracle fact 수다. 표 원본은 다음과 같다:
- `reports/raw/03_pilot_cpu.md` (exact)
- `reports/raw/03_pilot_cpu_aligned.md` (aligned)
- 기록: `unsup_pilot_cpu_v4/metrics_aligned.jsonl`

| suite | 지표 | counter | designation | order | interaction | relation | slot | 재관측 일치 | 목표 일관성 |
|---|---|---|---|---|---|---|---|---|---|
| counting | exact | 0.50 / 0.50 (381) | - | - | 0.46 / 0.53 (736) | - | 0.03 / 0.25 (28) | - | 0.40 |
| | aligned | 0.50 / 0.50 | - | - | 0.46 / 0.53 | - | 0.03 / 0.25 | - | 0.40 |
| imitation | exact | - | - / 0.00 (38) | - / 0.00 (143) | 0.00 / 0.00 (225) | - | - | - | 0.00 |
| | aligned | - | - / 0.00 | - / 0.00 | 0.00 / 0.00 | - | - | - | 0.00 |
| permanence | exact | - | - | - | 0.60 / 0.52 (128) | 0.47 / 0.47 (417) | 0.05 / 0.07 (135) | 0.33 (6) | - |
| | aligned | - | - | - | 0.84 / 0.73 | **0.99 / 0.98** | 0.05 / 0.07 | 0.83 (6) | - |
| reference | exact | 0.00 / 0.00 (67) | 0.22 / 0.19 (346) | 0.00 / 0.00 (191) | 0.62 / 0.62 (1513) | - | 0.00 / 0.00 (5) | - | 0.20 |
| | aligned | 0.42 / 0.48 | **1.00 / 0.86** | 0.49 / 0.49 | 0.93 / 0.94 | - | 0.62 / 1.00 | - | 0.80 |

### 3-3. 과제별 결과 (exact → aligned)

| 과제 (ep) | 결과 |
|---|---|
| VideoUnmaskSwap (1500), VideoUnmask (1400, 1402) | relation 86/86, 59/59, 50/56. 정렬 전에도 같다 |
| ButtonUnmask(Swap) (100, 200) | relation 0/76 → 75/76, 0/140 → 140/140. 대응: `container_B→A`, `C→B`, `D→C`, 남는 track 1~2개 |
| VideoPlaceButton (1100) | designation 0 → 59/59, order 0 → 85/87 |
| VideoPlaceOrder (1200) | designation 0 → 104/104, order 0 → 9/104. demo 순서 자체가 틀렸다 |
| VideoRepick (1300) | designation 0 → 67/67, counter 0 → 32/67. `blue_cube_X → cube_X` |
| PickHighlight (600, 602) | designation 27/30, 40/86 (정렬과 무관) |
| StopCube (900), PickXtimes (700) | counter 46/46, 51/53 |
| SwingXtimes (1000) | counter 0/49. 정렬해도 같다 (아래 c) |
| BinFill (0, 2) | counter 14/55, 78/178. 정렬해도 같다 (아래 d) |
| Imitation 4과제 | track 0~4개, 거의 모두 0 (아래 e) |

## 4. Recall < 0.8인 원인

**(a) ID 밀림** (Button*, VideoPlace*)
- 처음에 oracle에 없는 track(대개 같은 용기·목표의 중복 또는 배경 오검출)이 생겨 ID 문자가 밀린다.
- aligned 지표에서 permanence relation은 0.98, reference designation은 0.86까지 회복된다. 따라서 **구조는 대부분 맞다.**
- 남은 문제는 남는 track 자체다. 학습 label에서는 존재하지 않는 물체 이름(`container_E` 등)이 BG에 나오는 오류가 된다.
- 생성 쪽 대응으로는 시작 시점 track 수를 instruction에 나온 물체 수로 제한하는 것만 고려한다. instruction은 추론 때도 보이는 정보라 privileged 정보가 아니다.
- **oracle의 ID 순서나 물체 수를 unsup 생성에 쓰지 않는다.** 이 대응은 아직 적용하지 않았다.

**(b) 이름 표기 차이** (VideoRepick, StopCube, MoveCube)
- oracle은 색 없는 `cube_C`를 쓰고, unsup은 `blue_cube_C`를 쓴다.
- 이것은 측정 문제로 보고 aligned 지표로만 다룬다.
- unsup 이름 규칙을 oracle에 맞춰 바꾸지 않는다. 결과를 보고 바꾸면 지표에 맞추는 것이 되기 때문이다.
- 두 규칙 중 어느 것을 schema 표준으로 할지는 결과와 무관하게 01 schema에서 정할 일이다 (미정).
- VideoRepick에서 unsup이 세 큐브를 모두 `blue`로 부른 것은 색 판정 오류이므로 따로 기록한다.

**(c) 동작 유형 오인** (SwingXtimes)
- 막대로 목표를 두드리는 동작을 `picked/placed`로 기록한다. oracle은 `reached(target)`와 `swung` counter로 기록한다.
- 해결하려면 proprio에 "잡은 채 목표 근처를 지남" 규칙을 더해야 한다.

**(d) 구멍 검출 실패** (BinFill)
- OWLv2-base가 구멍(bin)을 못 찾아서, 놓은 곳이 `table`로 기록되고 `binned` counter가 오르지 않는다.
- query를 "a square hole"로 바꿨지만, **pilot track은 바꾸기 전 query("bin")로 만들었다.** 다시 tracking해야 효과를 확인할 수 있다 (미확인).
- (정정) 이전 판에는 "query는 pilot 결과를 보기 전에 바꿨다"고 적었다. 작업 기록을 확인해 보니 일부 pilot 결과(v3 재채점)를 본 뒤에 바꿨다. §5 참고.

**(e) 가는 물체 미검출** (Imitation suite)
- peg, 막대, 버튼, 격자 셀이 OWLv2-base에서 거의 검출되지 않는다 (score 0.1 미만).
- 그래서 interaction 대상과 order 시퀀스가 비었다.
- PatternLock의 dwell trace(TCP가 머문 칸)는 격자 좌표를 몰라서 order fact로 바꾸지 못했다.
- 큰 detector나 고정 장면 prior가 필요하다 (미확인).

**(f) 기타**
- VideoPlaceOrder의 demo 순서는 aligned에서도 9/104로 틀렸다. demo 중 놓은 목표를 잘못 고른 지각 오류다.
- slot(위치) precision이 낮다. oracle은 이름 붙은 위치(`left/middle/right`)를 쓰고, unsup은 순위 번호(`posN`)를 쓰는 표기 차이다. aligned 지표는 ID만 맞추므로 slot 값은 그대로 비교된다.
- 누르기 오검출(precision 0.5)이 interaction precision을 낮춘다.

## 5. Privilege 점검 (2026-11-04)

UnsupBG 코드(`simplememvla/bg/unsup/`)가 무엇을 읽는지 하나씩 확인했다.

### 5-1. label을 만드는 입력

| 입력 | 출처 | 판정 |
|---|---|---|
| 관절 7개 + finger (`observation.state`) | LeRobot 녹화 데이터 | 정책도 보는 정보 |
| front 카메라 영상 (256×256) | LeRobot 녹화 데이터 | 정책도 보는 정보 |
| instruction | LeRobot 녹화 데이터 | 정책도 보는 정보 |
| demo 경계 | proprio의 관절 도약으로 검출 (`detect_demo_end`) | 녹화의 `demo` flag와 oracle은 쓰지 않음 |
| 카메라 보정값 K, extrinsic | freeze v2부터 RoboMME H5 관측값에서 읽음 (`unsup/h5cam.py`, `pilot --cam_dir`). v1까지는 replay snapshot | 아래 (a) |
| 로봇 base 위치 (x=−0.615), Panda DH, TCP 0.1034 | RoboMME env 코드, Panda URDF 표준값 | 고정 상수 |
| stick 끝 길이 0.15 m | 시뮬레이터 link 위치에 맞춤 | 아래 (b) |
| 과제 이름 | dataset metadata | 아래 (c) |

oracle builder(`oracle_robomme`)와 replay snapshot(물체 pose, segmentation, `task_list`, 성공 여부)은 다음 세 곳에서만 import된다. 모두 label을 만든 뒤 지표를 계산하는 곳이며, `UnsupBuilder`에 들어가는 값은 없다.
- `pilot.py`: 채점 부분
- `rescore.py`
- `eval_proprio.py`

**(a) 카메라 보정값**
- 정책 입력에는 없는 값이지만, 장면의 고정 상수다.
- replay 80개(16과제 × 5 episode)에서 모두 같은 값이었다 (최대 차이 0).
- FK로 구한 TCP를 이미지에 투영하는 데만 쓴다.
- 2026-11-04까지는 이 값을 oracle 시뮬레이터 replay에서 읽었다. 내용은 장면 상수지만 읽는 경로가 oracle 산출물이라 엄밀하지 않았다.
- **조치 (freeze v2)**: H5 관측값 `setup/front_camera_intrinsic`(K)과 `timestep_0/obs/front_camera_extrinsic`(episode 첫 관측)을 episode별로 읽도록 바꿨다.
  - 추출: `python -m simplememvla.bg.unsup.h5cam extract` → `/data2/hoyong/data/smv/cams_h5/<task>.json`
  - pilot은 `--cam_dir`(H5)를 기본 경로로 쓴다. replay에서 읽는 경로는 `--cam_from_replay`로만 남겼고, v1 이전 pilot을 재현할 때만 쓴다.
- **H5와 replay 값 비교** (`reports/raw/unsup_cam_h5_vs_replay.json`): 16과제 1,600 episode 모두에서 K와 extrinsic의 최대 절대 차이가 **0**이었다.
  - H5 안에서도 extrinsic은 episode 내내 변하지 않았다(모든 timestep, 최대 편차 0).
  - 과제마다 K와 extrinsic이 각각 한 값뿐이었다.
- dev episode 4개(ep100, 200, 600, 602)에서 H5 값으로 다시 만든 label이 v4 pilot label과 행 단위로 같았다. 값이 같으므로 pilot 수치는 바뀌지 않는다.

**(b) stick 길이**
- 시뮬레이터 link 위치(privileged)를 보고 상수 하나를 맞췄다.
- 물체 상태가 아니라 로봇 기구학 상수이고, URDF에서 얻을 수 있는 종류의 값이다.

**(c) 과제 이름**
- `PatternLock`, `RouteStick`이면 stick FK를 쓴다.
- `BinFill`이면 bin counter를 쓴다. 다만 이 조건은 instruction의 "into the bin"과 OR로 묶여 있어 결과가 같다.
- 과제 이름은 평가 때도 env id로 알 수 있는 값이다. 그래도 instruction만 쓰는 구성이 더 깔끔하므로 바꿀 후보로 둔다.

### 5-2. oracle 결과를 보고 규칙을 정한 부분 (개발 단계 누설)

label에 privileged 값이 들어간 것은 아니다. 하지만 아래 규칙과 상수는 **같은 pilot episode의 oracle 비교 결과를 보면서** 정하거나 고쳤다. 그래서 pilot 수치는 in-sample이고 낙관적이다.

**ep1500 디버깅**
- ep1500은 pilot 20개에 들어 있다.
- 결과를 보며 고친 것:
  - 큐브 색 세 가지를 모두 검출
  - 색 pixel로 보임 여부 판단
  - category 간 겹침 임계값 0.1 / 0.6
  - 늦은 검출 임계값 ×1.25
  - grasp를 보이는 물체에만 연결 (`visible_only`)
- 그 결과가 relation 86/86, 재관측 1/1이다. ep1500의 수치는 사실상 학습 데이터 성적이다.

**v3 재채점(일부 episode)을 본 뒤 추가하거나 바꾼 것**
- 같은 category 중복 track 병합 (8 px): ButtonUnmask relation 0/75를 본 뒤
- 하이라이트 원판 위 큐브 → 지정 대상 (15 px): PickHighlight designation 0/30을 본 뒤
- BinFill query "bin" → "a square hole":
  - 6개 frame으로 prompt를 비교해서 골랐다.
  - pilot track은 이전 query로 만든 것이라 현재 수치에는 반영되지 않았다.

**OWLv2 query와 THRESH**
- 과제별 frame 몇 장(`scratchpad/frames`)으로 정했다.
- 이 frame이 pilot episode에서 나왔는지는 기록이 남아 있지 않다 (**미확인**).

**제외 목록**
- unsup label은 oracle과 같은 제외 목록(replay 실패 12개)을 쓴다.
- 세 조건이 같은 데이터로 학습하도록 한 의도된 데이터 선택이다. label 내용에는 영향이 없다.

### 5-3. 조치: 독립 평가 준비 (완료, 실행은 GPU 서버)

- 지금 20개 수치는 "개발 세트 성적"으로만 인용한다.

**평가 세트**: `reports/raw/unsup_heldout_evalset_s20261105.json`
- `heldout.py select`, seed 20261105로 과제당 3개, 모두 48 episode를 골랐다.
- 다음 episode는 후보에서 뺐다.
  - 개발에 쓴 20 pilot episode (ep1500 포함)
  - 각 과제의 첫 episode 16개. replay와 proprio probe에 썼고, OWLv2 prompt 검사 frame도 이 episode에서 나왔을 가능성이 높다.
  - replay 실패 12개
- 평가 세트는 어떤 held-out 점수도 보기 전에 골랐다. 결과를 본 뒤 다시 뽑지 않는다.
- 48개 모두 oracle replay가 있음을 확인했다.

**규칙 고정**: `reports/raw/unsup_freeze_v2.json` (현재 기준). v1은 기록으로 남긴다.
- 다음을 기록했다.
  - label을 정하는 파일 7개의 sha256: builder, proprio, visual, instruction, pilot, schema, serialize
  - 지표 파일 4개의 sha256
  - 모델 이름: OWLv2-base, SAM2.1-small, Qwen2.5-7B-Instruct
  - spec 종류 = llm
- `heldout.py verify`는 코드가 하나라도 다르면 종료 코드 1로 멈춘다. 파일 하나를 바꾼 사본으로 이 동작을 확인했다.
- **freeze v2** (`reports/raw/unsup_freeze_v2.json`, held-out 점수를 보기 전인 2026-11-04에 고정):
  - v1과 같은 항목에 `unsup/h5cam.py`를 더해 label 파일 8개의 sha256을 기록했다.
  - 바뀐 것은 카메라 보정값을 읽는 경로(`pilot.py`, `builder.py` 주석, `h5cam.py`)뿐이다. 규칙은 바뀌지 않았다.
  - 현재 코드로 `verify`하면 v2는 통과하고 v1은 `builder.py`, `pilot.py` 불일치로 멈춘다. 의도한 동작이다.
- `scripts/unsup_heldout_eval.sh`는 기본으로 v2를 쓰고, `CAM_DIR`(H5 보정값)이 필수다. `run.json`에 보정값 파일의 sha256도 남긴다.
- 남은 privileged 경로는 (b) stick 길이와 (c) 과제 이름이다. 실행 중에는 replay를 채점의 oracle 쪽에서만 읽는다.

**실행**: `scripts/unsup_heldout_eval.sh`
- 실행 예: `CUDA_VISIBLE_DEVICES=0 OUT=<새 폴더> CAM_DIR=/data2/hoyong/data/smv/cams_h5 bash scripts/unsup_heldout_eval.sh`
- 순서:
  1. freeze 확인
  2. replay 확인
  3. LLM spec으로 track과 label 생성
  4. exact와 aligned 채점
  5. `summary_exact.md`, `summary_aligned.md` 출력
- `OUT`이 이미 있으면 실행하지 않는다. 같은 freeze와 평가 세트로는 한 번만 돈다.
- `run.json`에 시작 시각, host, 평가 세트와 freeze의 sha256을 남긴다.
- 예상 시간은 미측정이다. CPU에서는 episode당 수 분~수십 분이었다.

**결과를 본 뒤 규칙을 고치는 경우**
- 바꾼 이유를 이 보고서에 적는다.
- 새 freeze 파일을 만든다.
- 이전 평가 세트를 제외한 **새 평가 세트**(새 seed)로 다시 채점한다.
- 이전 held-out 점수는 지우지 않고 함께 보고한다.

## 6. 결론과 다음 단계

- ID를 정렬하면 permanence의 가려진 물체 위치(relation 0.98)와 reference의 지정 대상(designation 0.86)은 비지도로도 거의 oracle과 같은 구조를 낸다.
- 정렬 전 낮은 recall의 상당 부분은 ID 표기 문제였다.
- 진짜 지각 실패는 다음 네 가지다:
  - 남는 track (a)
  - 동작 유형 (c)
  - 구멍 (d)
  - 가는 물체 (e)
- GPU 서버에서 할 일:
  1. LLM spec으로 같은 20 episode를 다시 돌린다.
  2. BinFill을 새 query로 다시 tracking한다.
  3. 그 뒤 전체 label을 만든다.
  4. 보고에는 exact와 aligned를 함께 싣는다.
- 생성 규칙을 바꾼다면 pilot 결과와 무관한 근거(instruction만 쓰는 track 수 제한 등)로 정하고, 바꾼 사실을 보고서에 적는다.
- R3 학습(05 테스트 8)은 그 전체 label이 필요하다.
- **미확인**: LLM spec의 정확도, 더 큰 detector의 효과, 전체 1,588 episode에서의 수치.
