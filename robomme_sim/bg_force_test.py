"""[GPU] Test 10: teacher-forced BG input changes the condition the action head reads.

Uses recorded LeRobot frames (no simulator): for each sampled (episode, frame) the history buffer is filled with the
episode's frames up to that point exactly as a rollout would, then ``BGBatchedEvalPolicy.generate_batch`` runs
three times with the same inputs:
    gen      no override (the model writes its own BG)
    label    BG forced to the label text of that frame
    edit_*   label text with an edit applied (--edits JSON: [{"name", "pattern", "repl"}])
Reported per case: forced flag, answer, condition-hidden L2 distance and cosine to the 'label' run (over the common
sub-task prefix and over the full span), action L2 difference. Everything is appended to --log (jsonl).

    python -m robomme_sim.bg_force_test --checkpoint <ck> --root <robomme_lerobot> --labels <label dir> \
        --episodes 1400 1500 --frames 150 300 --edits edits.json --log force.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--episodes", nargs="+", type=int, required=True)
    ap.add_argument("--frames", nargs="+", type=int, required=True, help="frame index per episode (same order)")
    ap.add_argument("--edits", default=None)
    ap.add_argument("--log", required=True)
    ap.add_argument("--max_subtask_tokens", type=int, default=256)
    ap.add_argument("--attn_implementation", default="flash_attention_2")
    a = ap.parse_args()
    import torch
    from transformers import AutoProcessor

    from robomme_sim.bg_policy import BGBatchedEvalPolicy
    from robomme_sim.policy import RoboMMEPolicy, get_vla
    from simplememvla.bg.labels import BGLabelStore
    from simplememvla.bg.unsup.pilot import decode_front

    vla, unnorm, norm_state = get_vla(a.checkpoint, "bfloat16", attn_implementation=a.attn_implementation)
    proc = AutoProcessor.from_pretrained(a.checkpoint, trust_remote_code=True, padding_side="right",
                                         model_max_length=8192)
    pol = BGBatchedEvalPolicy(model=vla, processor=proc, unnormalize_action=unnorm, normalize_state=norm_state,
                              max_subtask_tokens=a.max_subtask_tokens, device=next(vla.parameters()).device)
    store = BGLabelStore(a.labels, "mnemonic")
    edits = json.load(open(a.edits)) if a.edits else []
    log = open(a.log, "a")
    for ep, fr in zip(a.episodes, a.frames):
        import pyarrow.parquet as pq

        t = pq.read_table(f"{a.root}/data/chunk-000/file-000.parquet",
                          columns=["observation.state", "episode_index", "frame_index", "task_index"])
        sel = (t.column("episode_index").to_numpy() == ep)
        states = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)[sel])
        front = decode_front(a.root, ep)
        wrist = _decode_cam(a.root, ep, "wrist")
        import pandas as pd

        tasks = pd.read_parquet(f"{a.root}/meta/tasks.parquet").index.to_numpy()
        instr = str(tasks[t.column("task_index").to_numpy()[sel][0]])
        buf = RoboMMEPolicy(vla, proc, unnorm, norm_state, max_subtask_tokens=a.max_subtask_tokens)
        buf.reset()
        for f in range(fr + 1):
            buf.observe({"observation.images.front": front[f], "observation.images.wrist": wrist[f]})
        prep = buf._prepare_inputs(instr)
        st = norm_state.normalize(torch.from_numpy(states[fr].astype(np.float32))).unsqueeze(0).to(pol.device)
        label = store.text_at(ep, fr)
        cases = [("gen", None), ("label", label)] + [(e["name"], re.sub(e["pattern"], e["repl"], label)) for e in edits]
        runs = {}
        for name, txt in cases:
            buf._patch_feat_cache.clear()
            prep = buf._prepare_inputs(instr)
            pol.override_texts = [txt]
            out = pol.generate_batch([prep], [st])
            ch, cm = pol.last_condition
            runs[name] = (out[0][0], ch[0][cm[0].bool()].float().cpu(), dict(pol.last_info[0]))
        ref_a, ref_h, _ = runs["label"]
        for name, (act, h, info) in runs.items():
            n = min(len(h), len(ref_h))
            rec = {"episode": ep, "frame": fr, "case": name, "answer": info["answer"], "forced": info["forced"],
                   "cond_tokens": int(len(h)), "action_l2_vs_label": float(np.linalg.norm(act - ref_a)),
                   "cond_l2_vs_label": float((h[:n] - ref_h[:n]).norm()),
                   "cond_cos_vs_label": float(torch.nn.functional.cosine_similarity(h[:n].flatten(), ref_h[:n].flatten(), 0)),
                   "same_length": len(h) == len(ref_h)}
            log.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    log.close()


def _decode_cam(root, ep, cam):
    import glob

    import av
    import pandas as pd

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))])
    r = df[df.episode_index == ep].iloc[0]
    key = f"videos/observation.images.{cam}"
    f = f"{root}/videos/observation.images.{cam}/chunk-{int(r[key + '/chunk_index']):03d}/file-{int(r[key + '/file_index']):03d}.mp4"
    t0, n = float(r[key + "/from_timestamp"]), int(r["length"])
    out = []
    with av.open(f) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, t0 - 0.5) / s.time_base), stream=s)
        for fr in c.decode(s):
            if fr.time < t0 - 1e-3:
                continue
            if len(out) >= n:
                break
            out.append(fr.to_ndarray(format="rgb24"))
    return np.stack(out)


if __name__ == "__main__":
    main()
