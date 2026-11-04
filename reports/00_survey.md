# 00. 코드 작성 전 조사 (RoboMME 우선)

작성: 2026-11-04, branch `bgmem` (SimpleMemVLA fork `khyong/SimpleMemVLA`, upstream `OpenBMB/SimpleMemVLA@453d6ec`).
원자료: `/data2/hoyong/data/smv/survey/` (스크립트와 json), 테스트 결과 `reports/raw/`.

## 0. 환경과 하드웨어

| 항목 | 값 |
|---|---|
| 서버 GPU (`nvidia-smi`) | RTX 4090 24 GB × 8. **이 작업에 쓸 수 있는 GPU는 host 6, 7번뿐** (사용자 지시). 프롬프트가 가정한 RTX PRO 6000 96 GB가 아님 |
| GPU 6/7 현재 상태 | 다른 프로젝트(bg-mem G0.5 큐) 학습이 각 21 GB 사용 중 (조사 시점) |
| Python env | `/data2/hoyong/data/smv/envs/smv` (python 3.10, torch 2.4.1+cu121, transformers 5.11.0, deepspeed 0.19.9). transformers 5.20은 torch 2.4.1에서 import 실패(`nn` 미정의) → 5.11.0 고정 |
| flash-attn | 미설치 (CPU 테스트만 진행. GPU 단계에서 `install_fast_path.sh` 필요) |
| 데이터 | `yinchenghust/robomme_lerobot` → `/data2/hoyong/data/smv/datasets/yinchenghust/robomme_lerobot` (9.7 GB, 완료) |
| checkpoint | `yinchenghust/simplememvla_robomme` → `/data2/hoyong/ckpt/smv/simplememvla_robomme` (tokenizer/processor/config 완료, `model.safetensors` 12.7 GB 다운로드 중; 네트워크 1~3 MB/s) |
| 원본 h5 | 8개 과제(Video*/Button*/PickHighlight)는 기존 `/data2/hoyong/data/robomme/h5`에 있음. 나머지 8개는 replay에 h5가 필요 없어서 받지 않음 |
| 시뮬레이터 | 공식 install 스크립트와 같은 ManiSkill fork `YinpeiDai/ManiSkill@07be6fbc`. 벤더된 `robomme_sim/robomme`은 bg-mem의 `robomme_benchmark/src/robomme`와 파일 단위로 동일. replay는 CPU(llvmpipe Vulkan)로 실행 |

## 1. RoboMME LeRobot 데이터

### 1-1. subtask 문자열 (과제별 예시, 실행 구간 첫 episode의 `subtask_online` 순서)
전체 예시(과제별 5 episode × 4종 컬럼): `survey/s1_subtasks.json`.

| 과제 | instruction 예 | `subtask_online` 순서 (ep 첫 번째) |
|---|---|---|
| BinFill | put two red cubes into the bin, then press the button to stop | pick up the first red cube → put it into the bin → pick up the second red cube → put it into the bin → press the button |
| PickXtimes | pick up the red cube and place it on the target, repeating this action two times, ... | pick up the red cube for the first time → place the red cube onto the target → ... → press the button to stop |
| SwingXtimes | pick up the X cube, move it to the top of the right-side target, then ... left-side ..., repeating ... N times, finally press the button | pick up → move to the top of the right-side target for the first time → ...left-side... → put the red cube on the table → press the button |
| StopCube | press the button to stop the cube just as it reaches the target for the fifth time | move to the top of the button to prepare → remain static (×k) → press the button to stop the cube on the target |
| VideoUnmask(Swap) | watch the video carefully, then pick up the container hiding the blue cube[, finally ...] | [demo] static → [demo] pick up the container that hides the blue cube → pick up the container that hides the blue cube |
| ButtonUnmask(Swap) | first press (both buttons|the button), then pick up the container hiding ... | press the (first) button → (press the second button) → pick up the container that hides the X cube → put down the container → ... |
| PickHighlight | first press the button, then pick up all cubes that have been highlighted ... | press the button → pick up the highlighted cube, which is red |
| VideoRepick | watch the video ..., then (repeatedly) pick up the same block ... (N times), ... press the button | [demo] pick up the cube → drop on table → static → NO RECORD → pick up the correct cube for the first time → put it down → ... → press the button to finish |
| VideoPlaceButton | ..., then place the blue cube on the target right after/before the button was pressed | [demo] pick → drop onto target → press the button → pick → drop onto target → ... → pick up the cube → place the cube onto the correct target |
| VideoPlaceOrder | ..., then place the blue cube on the fourth target it was previously placed on | [demo] (pick → drop onto target) × k with a button press inside → ... → pick up the cube → place the cube onto the correct target |
| MoveCube | ..., then move the cube to the target in the same manner as before | [demo] Pick up the peg → Hook the cube to the target with the peg → static → NO RECORD → Pick up the peg → Hook ... |
| InsertPeg | ..., grasp the same end of the same peg ... insert it into the same side of the box | [demo] Pick up the peg by grasping the far end → Insert the peg from the left side of the box → NO RECORD → (same, robot) |
| PatternLock | ..., use the stick ... to retrace the same pattern | [demo] move right → NO RECORD → move right |
| RouteStick | ..., navigate around the sticks ..., following the same path | [demo] move to the nearest right target by circling around the stick clockwise → NO RECORD → (same, robot) |

- 학습에 쓰는 컬럼: `subtask_online_index` (`simplememvla/benchmarks/robomme.py`). `grounded` 변형은 문장 뒤에 `at <x, y>` 픽셀 좌표가 붙음 (예: `pick up the first red cube at <105, 118>`). 전체 고유 subtask 5,841개 (좌표 때문).
- 공식 설정 `skip_video_demo_frames=True`: demo 구간은 history로만 쓰이고 anchor(학습 frame)가 되지 않음.

### 1-2. `is_video_demo` 구간의 `observation.state` (`survey/s2_demo_state.json`)
- demo는 **같은 로봇이 실제로 실행한 것**이다. demo 구간 관절 범위(episode 중앙값) 0.5~1.8 rad, demo→실행 경계에서 상태가 0.4~2.1 rad 뛴다(로봇 reset).
- 예외: VideoUnmask, VideoUnmaskSwap은 demo 중 로봇이 정지(범위 0)하고 물체만 움직인다.
- demo frame은 모든 episode에서 앞쪽 연속 구간(prefix)이다. timestamp는 정확히 20 fps.

### 1-3. LeRobot episode ↔ 원본 h5 episode
- `meta/_conversion_state.json`의 `done_episodes[i]` = LeRobot `episode_index i` (`"<Task>/episode_<k>"`). 과제 순서: BinFill 0–99, ButtonUnmask 100–199, ButtonUnmaskSwap 200–299, InsertPeg 300–399, MoveCube 400–499, PatternLock 500–599, PickHighlight 600–699, PickXtimes 700–799, RouteStick 800–899, StopCube 900–999, SwingXtimes 1000–1099, VideoPlaceButton 1100–1199, VideoPlaceOrder 1200–1299, VideoRepick 1300–1399, VideoUnmask 1400–1499, VideoUnmaskSwap 1500–1599.
- 검증(3과제 × 8 episode, `survey/s3_h5_align.json`): frame 수가 같고 `observation.state` = h5 `joint_state + gripper_state[0]` (최대 오차 0.0), demo 표시도 같음.
- **h5와 LeRobot 모두 "NO RECORD" reset frame을 포함한다** (예: VideoRepick ep0에 20 frame). 반면 평가 때 `DemonstrationWrapper.reset()`은 "NO RECORD" frame을 반환 frame에서 뺀다. 즉 학습 history와 평가 history 사이에 이 구간만큼 차이가 있다 (공식 코드의 성질이며 변경하지 않음).

### 1-4. privileged 정보와 replay
- h5 `timestep_<t>/info`: `grounded_subgoal(_online)`, `simple_subgoal(_online)`, `is_completed`, `is_subgoal_boundary`, `is_video_demo`만 있다. `setup`: `seed`, `difficulty`, `task_goal`, 카메라 내부 파라미터, 선택지. **물체 pose, 가림, 하이라이트 정보는 없다.**
- 대신 seed와 action으로 simulator 상태를 복원한다 (`simplememvla/bg/robomme_replay.py`, `replay_robomme_cli.py`):
  - `BenchmarkEnvBuilder(dataset="train")`의 metadata seed로 env를 만들고 `reset()`을 실행하면 demo가 다시 생성된다. 이후 LeRobot `action`을 replay한다.
  - wrapper의 `_step_batch`를 감싸서 frame마다 다음을 기록한다: 물체 위치, front 카메라 segmentation 픽셀 수(가시성), 쥔 물체, `env.timestep`(subgoal 진행), 과제 변수 변화(`swing_count`, `{color}_cubes_in_bin`, `highlight_starts`, `achieved_list` ...).
  - 데이터는 "NO RECORD" reset 동작의 일부만 남긴다 (stick 과제는 앞부분 touch 전체를 뺌). 그래서 demo frame은 qpos 최근접 단조 정렬로 맞추고, 실행 첫 frame은 reset 안에서 하는 초기 step과 맞춘다.
- probe 결과 (과제당 1 episode, 16개): demo 구간 관절 오차 0~5e-4, 실행 구간 최대 0.05 rad, demo 표시 일치, 녹화와 같이 모든 subgoal 완료 (=성공). 전체 1,600 episode replay 결과는 `02_oracle_validation.md`에 정리한다.
- replay 비용: CPU에서 episode당 평균 약 87초 (1,600개, 프로세스 12개 기준 약 3시간).

## 2. 영상 timestamp 표기
- processor가 history video의 2-frame temporal patch마다 `<X.X seconds>`를 붙인다. 시간은 **window 시작 기준 상대 시간**이다. 첫 patch가 `<0.2 seconds>`이고 1초 간격(2 fps × patch 2 frame)으로 증가한다. 예: 64 frame window면 `<0.2 seconds> ... <31.2 seconds>`. episode 절대 시간이 아니고, window가 밀릴 때마다 같은 frame의 표기가 바뀐다.
- 결정: BG 시간은 **현재 기준 경과 시간 `X s ago` (0.5 s 단위)**로 쓴다. window 기준 표기와 충돌하지 않고, history 샘플링 간격(2 fps = 0.5 s)과 해상도가 같다.

## 3. 답변 span 경계
- chat template 결과의 assistant 부분: `<think>\n\n</think>\n\n{answer}<|im_end|>\n`. `subtask_span_mask`는 `</think>` 뒤 공백 token 다음부터 끝까지 표시하므로 **답변 + `<|im_end|>` + 끝의 `\n`을 포함한다**.
- 답변이 `subtask\nBelief: ...`이면 구분 `\n`은 단독 token(198)이고 `Belief:`는 `[21002, 4585, 25]`로 시작한다. subtask 끝 문자와 합쳐지지 않는다(`cube.\nBelief` → `[..., 13, 198, 21002, ...]`).
- Qwen3.5 tokenizer는 숫자를 한 자리씩 자른다. `31.0s ago`는 6 token이다. 길이 예산에 큰 영향을 준다(06 길이 표 참조).
- 추론: `generate_batch`는 생성 token 뒤에 `<|im_end|>`와 `\n`을 붙인 뒤 같은 `subtask_span_mask`로 span을 모으므로 학습과 같은 span을 읽는다. BG가 붙으면 span이 subtask와 BG 전체로 자연히 늘어난다. 단 공식 기본 `max_subtask_tokens=64`는 BG를 포함하면 부족하다 → `--max_subtask_tokens_auto`.

## 4. RMBench
**미확인.** RoboMME 다음 순서로 남겨 두었다 (spec frame rate 250/15 ≈ 16.7 fps와 `info.json` 50 fps의 불일치가 주는 영향, subtask 40개 목록, RoboTwin 2.0 replay 경로).
