"""UnsupBG instruction channel: what to remember and the goal, from the instruction text only.

``llm_spec`` asks a local instruction-tuned LLM (default Qwen/Qwen2.5-7B-Instruct, greedy = temperature 0) for a
JSON spec and caches every answer by sha256(model, prompt version, instruction). ``rule_spec`` is a deterministic
parser of the RoboMME templates; it is NOT used to build UnsupBG labels, only to check the LLM output (goal
consistency in report 03) and as a documented fallback when the LLM answer does not parse.

Spec JSON:
    {"objects": ["red cube", "container", ...],          # categories to detect / track
     "remember": ["count" | "order" | "under" | "designated" | "manner" | "route" | "pattern" | "timing"],
     "goal": {"verb": "pick|place|press|insert|move|trace|stop",
              "object": "blue cube" | null, "count": int | null, "ordinal": int | null,
              "relation": "before_button" | "after_button" | "same_as_video" | null}}
"""
from __future__ import annotations

import hashlib
import json
import os
import re

PROMPT_VERSION = "spec-v1"
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "once": 1, "twice": 2}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}

SYSTEM = (
    "You read a robot manipulation instruction and output a JSON object describing what must be remembered "
    "while executing it. Keys: objects (list of short object category names that appear in the scene, e.g. "
    "'red cube', 'container', 'button', 'target', 'bin', 'peg', 'box', 'stick'), remember (subset of "
    "['count','order','under','designated','manner','route','pattern','timing']), goal (object with keys verb, "
    "object, count, ordinal, relation; use null when absent; relation is one of 'before_button', 'after_button', "
    "'same_as_video' or null). Output only the JSON.")
FEWSHOT = [
    ("pick up the red cube and place it on the target, repeating this action two times, then press the button "
     "to stop",
     {"objects": ["red cube", "target", "button"], "remember": ["count"],
      "goal": {"verb": "pick", "object": "red cube", "count": 2, "ordinal": None, "relation": None}}),
    ("watch the video carefully, then pick up the container hiding the blue cube",
     {"objects": ["container", "blue cube"], "remember": ["under"],
      "goal": {"verb": "pick", "object": "container", "count": None, "ordinal": None, "relation": "same_as_video"}}),
]


def rule_spec(instr: str) -> dict:
    s = instr.lower()
    objs, rem, goal = [], [], {"verb": None, "object": None, "count": None, "ordinal": None, "relation": None}
    cols = re.findall(r"\b(red|green|blue)\s+cubes?\b", s)
    objs += [f"{c} cube" for c in dict.fromkeys(cols)]
    for kw, cat in (("container", "container"), ("bin", "bin"), ("button", "button"), ("target", "target"),
                    ("peg", "peg"), ("box", "box"), ("stick", "stick"), ("block", "cube"), ("cube", "cube")):
        if kw in s and cat not in objs and not (cat == "cube" and cols):
            objs.append(cat)
    m = re.search(r"\b(one|two|three|four|five|six|once|twice)\b(?:\s+\w+)*?\s*times?\b", s)
    if "repeat" in s or re.search(r"\btimes?\b", s):
        if m:
            goal["count"] = NUM[m.group(1)]
        rem.append("count")
    if "into the bin" in s:
        rem.append("count")
    if "hiding" in s:
        rem.append("under")
        goal.update(verb="pick", object="container")
    if "highlight" in s:
        rem.append("designated")
        goal.update(verb="pick", object="cube")
    if "previously picked" in s or "same block" in s:
        rem.append("designated")
        goal.update(verb="pick", object="cube", relation="same_as_video")
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth)\b (?:target|time)", s)
    if m:
        goal["ordinal"] = ORD[m.group(1)]
    if "previously placed" in s:
        rem.append("order")
        goal.update(verb="place", relation="same_as_video")
    if "right before the button" in s:
        rem.append("timing")
        goal.update(verb="place", relation="before_button")
    if "right after the button" in s:
        rem.append("timing")
        goal.update(verb="place", relation="after_button")
    if "same manner" in s:
        rem.append("manner")
        goal.update(verb="move", object="cube", relation="same_as_video")
    if "same end" in s:
        rem.append("manner")
        goal.update(verb="insert", object="peg", relation="same_as_video")
    if "retrace the same pattern" in s:
        rem.append("pattern")
        goal.update(verb="trace", relation="same_as_video")
    if "same path" in s:
        rem.append("route")
        goal.update(verb="move", relation="same_as_video")
    if "stop the cube" in s:
        rem.append("count")
        goal.update(verb="stop", object="cube")
    if goal["verb"] is None:
        goal["verb"] = "press" if s.startswith("press") else "pick"
        if cols and goal["object"] is None:
            goal["object"] = f"{cols[0]} cube"
    return {"objects": objs, "remember": sorted(set(rem)), "goal": goal}


class LLMSpec:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", cache_path: str | None = None,
                 device: str = "cuda"):
        self.model_id = model_id
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            self.cache = json.load(open(cache_path))
        self._model = None
        self.device = device

    def _key(self, instr: str) -> str:
        return hashlib.sha256(f"{self.model_id}|{PROMPT_VERSION}|{instr}".encode()).hexdigest()

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="auto").to(self.device).eval()

    def __call__(self, instr: str) -> dict:
        k = self._key(instr)
        if k in self.cache:
            return self.cache[k]["spec"]
        if self._model is None:
            self._load()
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in FEWSHOT:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
        msgs.append({"role": "user", "content": instr})
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        import torch

        with torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=256, do_sample=False)
        raw = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        try:
            spec, ok = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]), True
        except Exception:
            spec, ok = rule_spec(instr), False
        self.cache[k] = {"instruction": instr, "model": self.model_id, "prompt_version": PROMPT_VERSION,
                         "decoding": "greedy (temperature 0), max_new_tokens 256", "raw": raw, "parsed": ok,
                         "spec": spec}
        if self.cache_path:
            tmp = self.cache_path + ".tmp"
            json.dump(self.cache, open(tmp, "w"), indent=1)
            os.replace(tmp, self.cache_path)
        return spec
