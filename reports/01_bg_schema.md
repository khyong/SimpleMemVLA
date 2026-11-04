# 01. BG 스키마와 직렬화 (형식 v3)

코드: `simplememvla/bg/schema.py`, `simplememvla/bg/serialize.py`. 테스트: `tests/bg/test_serialize.py` (테스트 2), `tests/bg/length_stats.py` (테스트 6).

## 1. 내부 표현 (JSON)

| 요소 | 필드 |
|---|---|
| ObjectNode | persistent `id`, `category`, `slot`(현재 위치 slot), `first_slot`(처음 관측됐을 때의 slot), `visibility` visible/hidden/unknown, `last_seen_t` |
| RobotNode | `gripper` open/closed/unknown, `holding` |
| Relation | `in` / `on` / `under`, `subj`, `obj`, `since_t` |
| Interaction | `k`(episode 내 순번), `verb`, `target`, `dest`, `t` |
| Counter | `verb`, `target`, `value`, `total` (예: picked(green_cube) = 2/3) |
| Designation | 이전에 지정된 대상: `role`, `target`, `via` (video / highlight), `t` |
| Sequence | demo에서 본 순서: `role`, `items`, `via`, `t` |

- 모든 fact는 provenance(`evidence_t`, 출처 oracle/unsup, confidence)를 가진다. `BeliefGraph.validate()`는 모든 `evidence_t`, 사건 시각, `since_t`가 `t` 이하인지 검사한다.
- **Persistent ID**:
  - 범주에 물체가 하나뿐이면 범주 이름(`red_cube`, `button`)을 쓴다.
  - 같은 범주가 둘 이상이면 처음 관측된 순서(같은 시각이면 화면 왼쪽→오른쪽)로 `_A`, `_B`를 붙인다(`container_A`, `target_C`).
  - 현재 위치로 이름을 다시 붙이지 않는다(Swap 과제).
  - 범주 이름은 instruction 어휘를 따른다: container, target, bin, button, peg, box, cube.

## 2. 텍스트 형식

```
Belief: <clause> ; <clause> ; ...      (기억할 것이 없으면 "Belief: none")
```

| 절 | 예 |
|---|---|
| counter | `count picked(green_cube)=2/3` |
| 지정 대상 | `target(place)=target_B via video` |
| 순서 | `order(demo)=target_C>button>target_B via video` |
| 상호작용 (최근 K개) | `#3 placed(blue_cube,target) 3.5s ago` |
| 관계 (가려진 물체) | `green_cube under container_A since 7.5s ago` |
| 가려진 물체 (관계 없음) | `red_cube hidden at left since 4s ago` |
| 구분이 안 되는 물체의 현재 위치 | `container_A now at middle` |
| (full 필터만) 보이는 물체 / 그리퍼 | `cube_A at right visible`, `holding(cube_A)` |

- **시간**: 현재 기준 경과 시간, 0.5 s 단위이고 정수면 소수점을 생략한다(`31s`, `12.5s`). 영상 timestamp가 window 기준 상대 시간이기 때문이다(00_survey §2). Qwen tokenizer가 숫자를 한 자리씩 자르므로 길이를 줄이려는 선택이다.
- **결정적 순서**: counter → 지정 대상 → 순서 → 상호작용(k 순) → 관계(주어 순) → 가려진 물체 → `now at` → full 전용 절. 각 묶음 안에서도 정렬 key가 고정이라, 같은 belief는 항상 같은 문자열이 된다.
- **파싱 가능**: `parse(text)`가 정규식 문법으로 fact tuple을 되돌린다. 평가 log의 `bg_parse_ok`가 이것을 쓴다.

## 3. Mnemonic filter (기본) 와 절단

- 현재 frame만으로 판정할 수 없는 fact만 남긴다:
  - counter, 지정 대상, demo 순서, 과거 상호작용
  - 지금 가려진 물체의 관계와 위치
  - 생김새가 같아 현재 frame으로는 신원을 알 수 없는 물체의 현재 위치
- 보이는 물체의 위치, 보이는 물체의 관계, 그리퍼 상태는 뺀다(`bg_filter=full`이면 포함).
- 일시적 상태를 거르는 규칙 (v3에서 추가):
  - 관계 없이 가려진 물체는 2초 이상 계속 가려졌을 때만 넣는다. 팔에 잠깐 가려지는 것은 기억이 아니다.
  - `now at`은 물체가 처음 관측된 slot을 떠났을 때만 넣는다. 그 전에는 이름(첫 관측 순서)이 곧 위치이기 때문이다.
  - bin에 넣은 물체는 counter로만 표현한다.
- 절단: 상호작용은 최근 K개(기본 4)만 남기고 나머지는 모두 유지한다. counter는 항상 남는다.

## 4. 왕복 테스트와 결정성 (테스트 2, `reports/raw/test2_serialize.json`)

- 무작위 BG 3,000개 × 필터 2 × K 4종 = 24,000회: `parse(serialize(bg)) == facts(bg)`이고, JSON을 거쳐도 같은 텍스트가 나온다 → **통과**.
- 절단 규칙 검사: 상호작용 0~8개 × K 0~6에서 정확히 마지막 K개가 남는다 → **통과**.
  - 이 검사를 넣으면서 버그를 찾았다. 상호작용 수가 K보다 적으면 음수 slicing 때문에 마지막 1개만 남았다. v3 label(`oracle_3a07839db0eb`)은 수정 후에 만들었다.

## 5. 과제별 예시 (oracle, 형식 v3, mnemonic, 실행 구간 70% 시점)

| 과제 | instruction | BG |
|---|---|---|
| BinFill | put one red cube into the bin, then press the button | `count binned(red_cube)=1/1 ; #1 picked(red_cube_E) 4.5s ago ; #2 placed(red_cube_E,bin) 2s ago` |
| PickXtimes | pick up the blue cube ... three times ... | `count picked(blue_cube)=2/3 ; #1 picked(blue_cube) 17s ago ; #2 placed(blue_cube,target) 13s ago ; #3 picked(blue_cube) 6s ago ; #4 placed(blue_cube,target) 3.5s ago` |
| SwingXtimes | ... back and forth motion three times ... | `count swung(green_cube)=2/3 ; #3 reached(target_B) 8s ago ; ... ; #6 reached(target_A) 1.5s ago` |
| StopCube | ... reaches the target for the third time | `count reached(cube)=2/3` |
| VideoUnmask | ... pick up the container hiding the green cube | `blue_cube under container_B since 7.5s ago ; green_cube under container_A since 7.5s ago ; red_cube under container_C since 7.5s ago` |
| ButtonUnmask | first press the button, then pick up the container hiding the green cube | `#1 pressed(button) 6.5s ago ; blue_cube under container_C since 8.5s ago ; green_cube under container_B since 8.5s ago ; red_cube under container_A since 8.5s ago` |
| VideoUnmaskSwap | ... pick up the container hiding the blue cube | `blue_cube under container_A since 12.5s ago ; green_cube under container_C ... ; red_cube under container_B ... ; container_A now at middle ; container_B now at right ; container_C now at left` |
| ButtonUnmaskSwap | first press both buttons ..., then ... red cube, finally ... green cube | `#1 pressed(button) 14s ago ; #2 pressed(button) 9s ago ; blue_cube under container_C ... ; green_cube under container_A ... ; red_cube under container_B ... ; container_B now at right ; container_C now at middle` |
| PickHighlight | ... pick up all cubes that have been highlighted | `target(pick)=red_cube via highlight ; #1 pressed(button) 6s ago` |
| VideoRepick | ... pick up and put down the same block ... three times | `count picked(cube_A)=3/3 ; target(repick)=cube_A via video ; #4 placed(cube_A,table) 14s ago ; ... ; cube_A now at right` |
| VideoPlaceButton | ... place the blue cube on the target right after the button was pressed | `target(place)=target_B via video ; order(demo)=target_C>button>target_B via video ; #5 placed(blue_cube,target_B) 23.5s ago ; ...` |
| VideoPlaceOrder | ... on the fourth target it was previously placed on | `target(place)=target_D via video ; order(demo)=target_B>button>target_C>target_A>target_D via video ; ...` |
| MoveCube | ... in the same manner as before | `order(demo)=picked_cube>placed_cube via video ; #1 picked(cube) 14.5s ago ; #2 placed(cube,goal) 11s ago ; #3 picked(cube) 1.5s ago` |
| InsertPeg | ... same end of the same peg ... same side | `target(grasp)=peg_A via video ; order(demo)=grasp_near_end>insert_right_side via video ; #1 picked(peg_A) 19s ago ; ...` |
| PatternLock | ... retrace the same pattern | `order(moves)=left>backward_right via video ; order(trace)=cell_r1c1>cell_r1c2>cell_r0c1 via video` |
| RouteStick | ... following the same path | `order(route)=right_ccw>right_ccw via video` |

이산화 메모 (Imitation):
- MoveCube: 방식(peg로 밀기 / 그리퍼로 밀기 / 집어서 놓기)을 demo subgoal 순서로 표현한다.
- InsertPeg: 잡은 끝(near/far)과 삽입 방향(left/right)을 표현한다.
- PatternLock: 깜빡이는 셀 순서(격자 좌표)와 이동 방향을 함께 둔다.
- RouteStick: 각 이동의 방향과 회전 방향(cw/ccw)을 표현한다.
- 연속 궤적 자체는 BG에 넣지 않는다 (최소 정보).

## 6. 길이 (테스트 6, Qwen3.5 tokenizer, 실행 frame, `reports/raw/test6_length_oracle_v3_K*.json`)

BG 부분 token 수 (mnemonic). answer = subtask + `\n` + BG + `<|im_end|>\n`.

| 과제 | K=4 p50 | p95 | p99 | answer p99 | K=2 p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| BinFill | 66 | 93 | 100 | 109 | 43 | 64 | 72 |
| ButtonUnmask | 47 | 78 | 81 | 93 | 47 | 65 | 69 |
| ButtonUnmaskSwap | 65 | 104 | 111 | 123 | 64 | 84 | 88 |
| InsertPeg | 67 | 72 | 86 | 96 | 54 | 57 | 57 |
| MoveCube | 42 | 62 | 68 | 77 | 39 | 43 | 44 |
| PatternLock | 52 | 79 | 80 | 86 | 52 | 79 | 80 |
| PickHighlight | 34 | 92 | 98 | 110 | 34 | 63 | 74 |
| PickXtimes | 47 | 70 | 78 | 88 | 37 | 41 | 51 |
| RouteStick | 23 | 35 | 35 | 54 | 23 | 35 | 35 |
| StopCube | 10 | 22 | 22 | 35 | 10 | 22 | 22 |
| SwingXtimes | 50 | 65 | 67 | 80 | 35 | 41 | 41 |
| VideoPlaceButton | 82 | 92 | 99 | 107 | 53 | 63 | 68 |
| VideoPlaceOrder | 87 | 103 | 108 | 116 | 58 | 75 | 77 |
| VideoRepick | 75 | 85 | 96 | 104 | 51 | 55 | 70 |
| VideoUnmask | 41 | 67 | 69 | 81 | 41 | 67 | 69 |
| VideoUnmaskSwap | 56 | 86 | 90 | 102 | 56 | 85 | 90 |
| **전체** | | **90** | **102** | **112** (max 146) | | **70** | **83** |

- **목표(BG p95 ≤ 64)를 달성하지 못했다.**
  - K=4(기본): 전체 p95 90 token
  - K=2: 전체 p95 70 token
  - v1 형식(전체 p95 114)에서 시간 표기 축약, 지정·순서의 시간 제거, 일시적 상태 debounce로 줄인 결과다.
- 남은 길이의 주요 원인:
  - Permanence: `under ... since Xs ago` 3개와 Swap 뒤의 `now at`
  - VideoPlace*: demo 순서 + 상호작용 4개
  - PatternLock: 이동 방향과 셀 순서를 둘 다 둠
- 더 줄이는 선택지 (결정 필요, 아직 적용하지 않음):
  1. 관계의 `since` 시간 생략 (Permanence에서 약 −15 token). 프롬프트 예시에는 `since`가 있다.
  2. K=2.
  3. PatternLock은 `order(trace)`만 유지 (−27 token).
- `max_subtask_tokens` 권장값: answer p99 + 16 = **128** (K=4, 공식 기본 64의 두 배). `eval_bg --max_subtask_tokens_auto <label dir>`는 label에서 같은 방식으로 자동 계산한다(p99 + 24 + 16; sub-task 길이는 상한으로 근사).
