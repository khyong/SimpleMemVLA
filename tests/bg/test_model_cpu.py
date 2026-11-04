"""[CPU] BG model integration on one short-history sample (no GPU).

Checks: (1) BG model with no masks == official model loss (same weights, same noise seed); (2) with BG masks and
token_mean the VL loss is unchanged and subtask_ce / bg_ce are reported; (3) span_mean = subtask_ce + bg_ce;
(4) condition_span=subtask_only changes the action-head input (different action loss under the same noise);
(5) backward gives finite gradients.

    python -m tests.bg.test_model_cpu --ckpt <simplememvla_robomme> --root <robomme_lerobot> --labels <oracle labels>
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--episode", type=int, default=700)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from simplememvla.compat import apply_fla_torch_compat

    apply_fla_torch_compat()
    import torch
    from transformers import AutoProcessor, HfArgumentParser

    from robomme_sim.policy import _build_vla
    from simplememvla.bg.args import BGDataArguments
    from simplememvla.bg.collator import BGDataCollator
    from simplememvla.bg.dataset import BGSimpleMemVLADataset
    from simplememvla.bg.model import BGSimpleMemVLAForActionPrediction
    from train import resolve_benchmark_defaults

    torch.set_num_threads(32)
    dtype = torch.float32
    vla = _build_vla(a.ckpt, dtype, attn_implementation="sdpa")
    cfg = vla.config
    bg = BGSimpleMemVLAForActionPrediction(config=cfg, backbone=vla.backbone)
    bg.action_head.load_state_dict(vla.action_head.state_dict())
    bg.eval()
    proc = AutoProcessor.from_pretrained(a.ckpt, trust_remote_code=True, padding_side="right", model_max_length=8192)
    (da,) = HfArgumentParser((BGDataArguments,)).parse_args_into_dataclasses(
        ["--benchmark", "robomme", "--root", a.root, "--bg_mode", "oracle", "--bg_label_dir", a.labels,
         "--image_aug", "False", "--episodes", str(a.episode)])
    spec = resolve_benchmark_defaults(da)
    ds = BGSimpleMemVLADataset(da, spec)
    fr = ds._columns(["frame_index"])["frame_index"][ds._anchor_indices]
    i = int((fr >= 160).nonzero()[0][0])  # early frame: short history, after a few interactions
    feat = ds[i]
    col = BGDataCollator(proc, max_length=8192, condition_span="all", emit_bg_masks=True)
    b = col([feat])
    base = {k: v for k, v in b.items() if k not in ("condition_mask", "bg_label_mask")}
    res = {"answer": feat["messages"][-1]["content"], "seq_len": int(b["input_ids"].shape[1])}

    def run(model, inputs, seed=0):
        torch.manual_seed(seed)
        with torch.no_grad():
            return model(**inputs)

    o_off = run(vla, base)
    o_bg0 = run(bg, base)
    res["no_mask_equal"] = bool(torch.equal(o_off.loss, o_bg0.loss))
    o_bg = run(bg, dict(base, condition_mask=b["condition_mask"], bg_label_mask=b["bg_label_mask"]))
    parts = dict(bg.bg_ce_parts)
    res["token_mean_vl_equal"] = float(abs(o_bg.vl_loss - o_off.vl_loss)) < 1e-4
    res["parts"] = parts
    bg.vl_loss_reduction = "span_mean"
    o_sm = run(bg, dict(base, condition_mask=b["condition_mask"], bg_label_mask=b["bg_label_mask"]))
    res["span_mean_ok"] = abs(float(o_sm.vl_loss) - (parts["subtask_ce"] + parts["bg_ce"])) < 1e-3
    bg.vl_loss_reduction = "token_mean"
    col2 = BGDataCollator(proc, max_length=8192, condition_span="subtask_only")
    b2 = col2([feat])
    o_so = run(bg, dict(base, condition_mask=b2["condition_mask"], bg_label_mask=b2["bg_label_mask"]))
    res["subtask_only_changes_action"] = float(abs(o_so.action_loss - o_bg.action_loss)) > 1e-6
    res["cond_tokens"] = {"all": int(b["condition_mask"].sum()), "subtask_only": int(b2["condition_mask"].sum())}
    bg.train()
    torch.manual_seed(0)
    out = bg(**dict(base, condition_mask=b["condition_mask"], bg_label_mask=b["bg_label_mask"]))
    out.loss.backward()
    g = [p.grad for p in bg.parameters() if p.grad is not None]
    res["grad_finite"] = bool(all(torch.isfinite(x).all() for x in g)) and len(g) > 0
    res["losses"] = {"official": float(o_off.loss), "bg": float(o_bg.loss), "vl": float(o_bg.vl_loss),
                     "action": float(o_bg.action_loss), "subtask_only_action": float(o_so.action_loss)}
    res["pass"] = all(res[k] for k in ("no_mask_equal", "token_mean_vl_equal", "span_mean_ok",
                                       "subtask_only_changes_action", "grad_finite"))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
