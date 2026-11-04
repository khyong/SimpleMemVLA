"""Belief Graph (BG) extension for SimpleMemVLA (branch ``bgmem``).

The policy states the current sub-task and then a serialized Belief Graph in the same assistant answer, so the
action head reads the hidden states of both spans without any model change. Everything here is opt-in: with
``bg_mode=none`` (the default) no code path in this package is touched by training or evaluation.
"""

BG_MODES = ("none", "unsup", "oracle")
BG_FILTERS = ("mnemonic", "full")
CONDITION_SPANS = ("all", "subtask_only")
VL_LOSS_REDUCTIONS = ("token_mean", "span_mean")

# Separator between the sub-task sentence and the BG text inside the assistant answer.
BG_SEPARATOR = "\n"
BG_PREFIX = "Belief:"
