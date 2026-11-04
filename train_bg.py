"""BG training entry point (branch bgmem). Same as train.py except for the BG-aware dataset / collator / model /
trainer classes and the BG arguments (simplememvla/bg/args.py). With all BG arguments at their defaults it builds
exactly the official pipeline; scripts/train.sh and train.py are left untouched.

Extra: ``--init_checkpoint <dir>`` starts from a released SimpleMemVLA checkpoint (weights + config) instead of
the bare backbone, which is what the smoke runs from the public checkpoint use.
"""
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

from simplememvla.compat import apply_fla_torch_compat

apply_fla_torch_compat()

import torch  # noqa: E402
from transformers import AutoProcessor, HfArgumentParser, set_seed  # noqa: E402

from configs.sft_params import ModelArguments, SFTTrainingArguments  # noqa: E402
from simplememvla.bg.args import BGDataArguments  # noqa: E402
from simplememvla.bg.collator import BGDataCollator  # noqa: E402
from simplememvla.bg.dataset import BGSimpleMemVLADataset  # noqa: E402
from simplememvla.bg.model import BGSimpleMemVLAForActionPrediction  # noqa: E402
from simplememvla.bg.trainer import BGTrainer  # noqa: E402
from simplememvla.model import SimpleMemVLAConfig  # noqa: E402
from train import _dtype_from_string, resolve_benchmark_defaults  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class BGModelArguments(ModelArguments):
    init_checkpoint: Optional[str] = field(default=None, metadata={"help": "released SimpleMemVLA checkpoint dir"})


def _set_proc_title(name: str) -> None:
    try:
        from setproctitle import setproctitle

        setproctitle(name[:20])
    except Exception:
        pass


def build_model(model_args, config, dtype):
    if model_args.init_checkpoint:
        # same construction as the evaluation loader (robomme_sim.policy._build_vla), without its cuda:0 side effect
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForImageTextToText

        ck = model_args.init_checkpoint
        backbone_config = AutoConfig.for_model(**dict(config.backbone_config))
        kw = {"attn_implementation": model_args.attn_implementation} if model_args.attn_implementation else {}
        backbone = AutoModelForImageTextToText.from_config(backbone_config, **kw)
        for p in backbone.parameters():
            p.data = p.data.to(dtype)
        model = BGSimpleMemVLAForActionPrediction(config=config, backbone=backbone)
        index = os.path.join(ck, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            state = {}
            for s in shards:
                state.update(load_file(os.path.join(ck, s)))
        else:
            state = load_file(os.path.join(ck, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if any(k.endswith("lm_head.weight") for k in missing):
            model.backbone.tie_weights()
            missing = [k for k in missing if not k.endswith("lm_head.weight")]
        if missing or unexpected:
            raise RuntimeError(f"init_checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        model.action_head.to(dtype=dtype)
        return model
    return BGSimpleMemVLAForActionPrediction.from_backbone(
        model_args.backbone_model_name_or_path, config=config, dtype=dtype,
        attn_implementation=model_args.attn_implementation, trust_remote_code=model_args.trust_remote_code)


def train(model_args, data_args, training_args):
    _set_proc_title(os.environ.get("BG_PROC_TITLE", "hoyong-bgtrain"))
    data_args.bg_validate()
    if data_args.anchor_phase_by_epoch and training_args.dataloader_persistent_workers:
        raise ValueError("anchor_phase_by_epoch needs --dataloader_persistent_workers false (workers must re-fork "
                         "each epoch to see the new phase)")
    set_seed(training_args.seed)
    spec = resolve_benchmark_defaults(data_args)
    proc_src = model_args.init_checkpoint or model_args.backbone_model_name_or_path
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=model_args.trust_remote_code,
                                              padding_side="right", model_max_length=model_args.model_max_length)
    train_dataset = BGSimpleMemVLADataset(data_args, spec, output_dir=training_args.output_dir)
    if model_args.init_checkpoint:
        config = SimpleMemVLAConfig.from_pretrained(model_args.init_checkpoint)
    else:
        config = SimpleMemVLAConfig(
            backbone_model_name_or_path=model_args.backbone_model_name_or_path,
            action_dim=spec.action_dim, action_horizon=spec.num_actions_chunk, state_dim=spec.state_dim,
            dit_hidden_size=model_args.dit_hidden_size, dit_depth=model_args.dit_depth,
            dit_num_heads=model_args.dit_num_heads, dit_mlp_ratio=model_args.dit_mlp_ratio,
            dit_dropout=model_args.dit_dropout, dit_rope_theta=model_args.dit_rope_theta,
            dit_mrope_section=model_args.dit_mrope_section,
            dit_partial_rotary_factor=model_args.dit_partial_rotary_factor,
            timestep_beta_alpha=model_args.timestep_beta_alpha, timestep_beta_beta=model_args.timestep_beta_beta,
            action_loss_weight=model_args.action_loss_weight, vl_loss_weight=model_args.vl_loss_weight,
            freeze_backbone=model_args.freeze_backbone, use_proprio=model_args.use_proprio,
            state_dropout_prob=model_args.state_dropout_prob, robot_tag=data_args.robot_tag,
            control_frequency_hz=train_dataset.control_frequency_hz,
            history_video_sec=data_args.history_video_sec, history_video_fps=data_args.history_video_fps,
            native_video_fps=train_dataset.native_fps, variable_history=data_args.variable_history,
            image_keys=data_args.image_keys, history_image_keys=train_dataset.history_image_keys,
            subtask_key=train_dataset.subtask_key, subtask_index_key=train_dataset.subtask_index_key)
    # BG fields persisted into config.json so evaluation can rebuild the answer format
    config.bg_mode = data_args.bg_mode
    config.bg_filter = data_args.bg_filter
    config.bg_max_interactions = data_args.bg_max_interactions
    config.condition_span = data_args.condition_span
    config.vl_loss_reduction = data_args.vl_loss_reduction
    config.bg_label_hash = train_dataset.labels.label_hash if train_dataset.labels is not None else None
    dtype = _dtype_from_string(model_args.dtype)
    model = build_model(model_args, config, dtype)
    for k in ("bg_mode", "bg_filter", "bg_max_interactions", "condition_span", "vl_loss_reduction", "bg_label_hash"):
        setattr(model.config, k, getattr(config, k))
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    collator = BGDataCollator(processor, max_length=model_args.model_max_length,
                              condition_span=data_args.condition_span,
                              emit_bg_masks=data_args.bg_mode != "none")
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.process_index == 0:
        run_info = {"data_args": {k: v for k, v in vars(data_args).items() if not k.startswith("_")},
                    "init_checkpoint": model_args.init_checkpoint, "bg_label_hash": config.bg_label_hash}
        with open(os.path.join(training_args.output_dir, "bg_run.json"), "w") as f:
            json.dump(run_info, f, indent=1, sort_keys=True, default=str)
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.rstrip("/").split("/")[-1]
    trainer = BGTrainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator,
                        processing_class=processor, data_root=data_args.root)
    trainer.train(resume_from_checkpoint=training_args.resume)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((BGModelArguments, BGDataArguments, SFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)
