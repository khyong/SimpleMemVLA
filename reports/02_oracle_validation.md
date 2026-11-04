# 02. BG-Oracle 검증 (RoboMME)

- **최종 label**: `/data2/hoyong/data/smv/labels/oracle_3a07839db0eb/`
  - `labels.parquet`: 761,242 행, 1,588 episode
  - `manifest.json`: `label_hash = d3fcb3abb80e0099e235ac549cfeb68460a0e6231aac4bf0e63b7341bacabf41`, 형식 v3, K=4
  - `validation.jsonl`, `excluded.json`, `excluded_ids.json`
- 같은 폴더 아래 이전 버전 3개(`oracle_be5b709f0cc3` v1, `oracle_3f00e21383ad` v2, `oracle_3af8381ec497` v3 절단 버그)는 지우지 않았고, 각 폴더에 `SUPERSEDED.txt`를 남겼다.
- 실행 과정 (모두 CPU):
  1. `replay_robomme_cli` (1,600 episode, llvmpipe 렌더링, episode당 31~175 s)
  2. `build_labels_cli oracle` (검증과 인과성 검사 포함)

## 1. Replay 충실도 (1,600 episode)

| 과제 | n | 녹화와 같이 성공 | demo 구간 관절 오차 max | 실행 구간 관절 오차 중앙값 / max |
|---|---|---|---|---|
| BinFill, ButtonUnmask, ButtonUnmaskSwap, MoveCube, PickXtimes, StopCube, SwingXtimes, VideoUnmask, VideoUnmaskSwap | 각 100 | 100/100 | 0 | 0.04 / 0.047 rad |
| PatternLock, RouteStick | 각 100 | 100/100 | ≤ 9e-4 | 0.016~0.03 / 0.042 rad |
| InsertPeg | 100 | 95/100 | 0 | 0.046 / 0.047 rad |
| PickHighlight | 100 | 99/100 | 0 | 0.046 / 0.047 rad |
| VideoPlaceButton | 100 | 98/100 | 최대 1.37 (2 ep) | 0.040 / 1.37 rad |
| VideoPlaceOrder | 100 | 99/100 | 최대 1.30 (2 ep) | 0.040 / 1.30 rad |
| VideoRepick | 100 | 98/100 | 최대 1.09 (2 ep) | 0.040 / 1.09 rad |

- 실패 12개(0.75%)는 label에서 제외했다 (`excluded_ids.json`: 304, 311, 314, 338, 383, 638, 1158, 1183, 1246, 1250, 1339, 1359).
  - (a) demo 재생성이 녹화와 다름 (6개: VideoPlaceButton/Order, VideoRepick): reset 안의 motion planner가 다른 경로를 택했다. 앞부분 10~17 frame부터 1 rad 이상 차이 난다.
  - (b) 실행 중 물리 오차가 누적되어 과제를 끝내지 못함 (6개: InsertPeg 5, PickHighlight 1). ep 1250은 끝까지 성공했지만 demo 오차 때문에 제외했다.
- **R2/R3/R4가 같은 데이터를 쓰도록** 이 목록을 세 조건 모두에 `--exclude_episodes_file`로 넘긴다. 목록 없이 BG 학습을 시작하면 dataset 생성 단계에서 오류가 난다 (`tests/bg/test_bg_datapath.py`의 guard 검사).

## 2. 과제별 검사 (성공 시연 기준 99% 이상 통과가 수락 기준)

| suite | 과제 | 검사 | 통과 |
|---|---|---|---|
| Counting | BinFill | 색별 마지막 `binned` counter = instruction 개수 | 100/100 |
| | PickXtimes | 마지막 `picked` counter = instruction 횟수 | 100/100 |
| | SwingXtimes | 마지막 `swung` counter = instruction 횟수 (생략 시 1) | 100/100 |
| | StopCube | 버튼을 누를 때의 도달 횟수 ∈ {k−1, k}* | 100/100 |
| Permanence | VideoUnmask | 용기를 잡기 직전 BG의 "X cube under container_Y"의 Y = 실제로 든 용기 | 100/100 |
| | ButtonUnmask | 〃 (실행 중 가림) | 100/100 |
| | VideoUnmaskSwap | 〃 (Swap 뒤) | 100/100 |
| | ButtonUnmaskSwap | 〃 (2개 용기, Swap 뒤) | 100/100 |
| Reference | PickHighlight | 지정 대상 집합 = 실행 중 집은 큐브 집합 | 99/99 |
| | VideoRepick | 지정 대상 = 실행 중 집은 큐브 (모두) | 98/98 |
| | VideoPlaceButton | 지정 대상(버튼 직전/직후 목표) = 마지막으로 놓은 목표 | 98/98 |
| | VideoPlaceOrder | 지정 대상(k번째 목표) = 마지막으로 놓은 목표 | 98/98 |
| Imitation | MoveCube | demo 단계 순서 = 실행 단계 순서 | 100/100 |
| | InsertPeg | 〃 (잡은 끝, 삽입 방향) | 95/95 |
| | PatternLock | demo 셀 순서 = 실행 중 달성 셀(`achieved_list`), 이동 방향 순서 일치 | 100/100 |
| | RouteStick | demo 이동 순서(방향, cw/ccw) = 실행 순서 | 100/100 |

\* StopCube 성공 조건은 k번째 구간(`move_interval·(k−1) ~ move_interval·k`) 안에 누르는 것이다. 처음 검사(도달 횟수 = k)는 98/100이었다. 실패한 2개는 k번째 도달 직전에 눌러 성공한 경우였다. 이후 검사를 과제 정의에 맞췄다. counter 계산도 고쳤다: 빨리 지나가는 구간의 통과를 놓치지 않도록 운동 축 위 부호 변화를 함께 센다.

검사를 만들면서 고친 oracle/검사 버그 (모두 수정 후 재생성):
- SwingXtimes의 "N times" 없는 instruction에서 N=None으로 실패 (77%).
- PatternLock에서 `move forward-left`처럼 하이픈이 있는 방향을 parser가 놓침 (93%).
- `_2` 접미사 actor 전체를 제외해서 `bin_2`, `target_2`가 빠짐.

## 3. 인과성 (테스트 3)

- 모든 행에서 `BeliefGraph.validate()` 통과: 모든 fact의 `evidence_t ≤ t`, 사건과 관계 시각 ≤ t.
- 잘라서 다시 만드는 검사: episode마다 무작위 t 3개를 골라, replay를 t까지만 넣고 BG를 다시 만들어 원래 BG_t(JSON)와 비교했다. **4,764/4,764 일치 (1,588/1,588 episode).**
- builder는 frame을 순서대로 한 번씩만 읽는 incremental 구조라, 미래 frame을 볼 경로가 없다. privileged 변수는 관측 가능해진 뒤에만 쓴다:
  - PickHighlight 목표: 하이라이트가 켜지는 실행 step 10부터 사용한다.
  - VideoPlace* 지정 대상: demo에서 해당 배치가 끝난 순간에 결정한다.
  - StopCube 총 횟수: instruction에 있는 값이다.

## 4. Online `OracleBGTracker` (R4 추론과 편집 실험용)

- `attach_online_trackers(pool, specs)`가 평가 env(`SimEnvService`)의 reset과 step을 감싼다. 평가와 같이 reset의 "NO RECORD" frame은 빼고 snapshot을 같은 builder에 넣는다.
- CPU 시험 (VideoUnmaskSwap train ep0, 녹화 action replay): 성공으로 끝났고, 최종 BG가 오프라인 label과 같은 fact를 담았다. 시간 표기는 0.5 s 다르다. 평가 history에 "NO RECORD" frame이 없어서 생기는 차이로, 00_survey §1-3에 적었다.

## 5. 미확인 / 한계

- privileged 상태는 시뮬레이터 replay로 얻었다. 녹화 당시 상태와 실행 구간 관절 기준 최대 0.047 rad 차이가 있다. 물체 pose는 녹화에 없어서 직접 비교할 수 없다(**미확인**). 대신 과제별 검사가 모두 통과한 것을 간접 근거로 본다.
- 버튼 두 개(ButtonUnmaskSwap)는 sim 이름이 둘 다 `button_cap`이라 `pressed(button)`으로 구분 없이 기록된다.
