"""[CPU] Test 6: BG token lengths per task (Qwen3.5 tokenizer of the checkpoint) and answer budget.

    python -m tests.bg.length_stats --labels <label dir> --tokenizer <ckpt> --root <robomme_lerobot> [--out json]
Counts tokens of the BG text (mnemonic and full) and of the whole answer (sub-task + "\\n" + BG + <|im_end|> +
"\\n", i.e. what generate_batch decodes) on execution frames, p50/p95/p99/max per task.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=None, help="re-serialize bg_json with this max_interactions")
    a = ap.parse_args()
    import pandas as pd
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    lab = pd.read_parquet(f"{a.labels}/labels.parquet",
                          columns=["episode_index", "frame_index", "task", "bg_mnemonic", "bg_full", "bg_json"])
    t = pq.read_table(f"{a.root}/data/chunk-000/file-000.parquet",
                      columns=["episode_index", "frame_index", "subtask_online_index", "is_video_demo"]).to_pandas()
    t["demo"] = t["is_video_demo"].apply(lambda x: bool(np.asarray(x).reshape(-1)[0]))
    sub = pd.read_parquet(f"{a.root}/meta/subtasks.parquet").index.to_numpy()
    t["subtask"] = [str(sub[int(np.asarray(i).reshape(-1)[0])]).strip().replace("_", " ") for i in t["subtask_online_index"]]
    df = lab.merge(t[["episode_index", "frame_index", "subtask", "demo"]], on=["episode_index", "frame_index"])
    df = df[~df["demo"]].iloc[:: a.stride].copy()
    if a.k is not None:
        from simplememvla.bg.schema import BeliefGraph
        from simplememvla.bg.serialize import serialize

        gs = [BeliefGraph.from_json(j) for j in df["bg_json"]]
        df["bg_mnemonic"] = [serialize(g, "mnemonic", a.k) for g in gs]
        df["bg_full"] = [serialize(g, "full", a.k) for g in gs]
    enc = lambda xs: np.array([len(v) for v in tok(list(xs), add_special_tokens=False)["input_ids"]])
    out = {}
    for task, d in df.groupby("task"):
        Lm, Lf = enc(d["bg_mnemonic"]), enc(d["bg_full"])
        La = enc((d["subtask"] + "\n" + d["bg_mnemonic"]).tolist()) + 2  # <|im_end|> + "\n"
        q = lambda L: {"p50": int(np.percentile(L, 50)), "p95": int(np.percentile(L, 95)),
                       "p99": int(np.percentile(L, 99)), "max": int(L.max())}
        out[task] = {"n": int(len(d)), "bg_mnemonic": q(Lm), "bg_full": q(Lf), "answer": q(La)}
    allm = np.concatenate([enc(d["bg_mnemonic"]) for _, d in df.groupby("task")])
    alla = enc((df["subtask"] + "\n" + df["bg_mnemonic"]).tolist()) + 2
    out["_all"] = {"bg_mnemonic_p95": int(np.percentile(allm, 95)), "bg_mnemonic_p99": int(np.percentile(allm, 99)),
                   "answer_p99": int(np.percentile(alla, 99)), "answer_max": int(alla.max()),
                   "recommended_max_subtask_tokens": int(np.percentile(alla, 99)) + 16}
    for k, v in out.items():
        print(k.ljust(18), json.dumps(v))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
