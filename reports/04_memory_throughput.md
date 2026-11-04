# 04. 메모리와 처리량 (테스트 7) — 미실행

**상태: 이 서버에서는 실행하지 않았다.** 사용자 결정(2026-11-04)에 따라 GPU 단계는 RTX PRO 6000 서버에서 진행한다.

- 이 서버: RTX 4090 24 GB, 사용 가능 GPU는 host 6, 7번뿐이고 다른 프로젝트 학습이 쓰고 있었다.
- Qwen3.5-4B + DiT(약 0.9B) 전체 fine-tuning은 ZeRO-2로 24 GB에 들어가지 않는다. AdamW 상태(fp32 m, v, master)만 약 59 GB이고, 2장으로 나눠도 GPU당 약 29 GB다.

## 실행 방법 (GPU 서버)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 INIT_CHECKPOINT=<simplememvla_robomme> ROOT_DIR=<robomme_lerobot> \
  ORACLE_LABELS=<labels>/oracle_3a07839db0eb OUT=<results> STEP=mem bash scripts/bg_gpu_suite.sh
```

- GPU당 batch 4, 6, 8로 각각 25 step을 돈다. 조건: ZeRO-2(`configs/zero2.json`), gradient checkpointing, bf16, `bg_mode=oracle`.
- **history가 꽉 찬 sample만** 쓴다: `MIN_ANCHOR_FRAME=1200`, 곧 60 s window = 120 frame의 video token이 모두 찬 anchor. 이런 anchor는 길이가 긴 VideoPlace*, VideoRepick, BinFill에만 있다.
- 기록 (trainer log, `BGTrainer`):
  - `mem/peak_alloc_gb`, `mem/peak_reserved_gb` (`torch.cuda.max_memory_*`)
  - `perf/samples_per_s` (log 구간의 global sample/s)
- BG 답변 길이(01 §6, answer p99 약 112 token)는 sequence 길이(이 sample에서 약 6~7k token)의 2% 미만이라, 메모리 차이는 R2와 거의 같을 것으로 예상한다. **미확인.**

| GPU당 batch | peak alloc (GB) | peak reserved (GB) | samples/s (전체) |
|---|---|---|---|
| 4 | 미측정 | | |
| 6 | 미측정 | | |
| 8 | 미측정 | | |
