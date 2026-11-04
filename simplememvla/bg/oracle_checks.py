"""Per-task validation of the BG-Oracle against simulator ground truth (report 02_oracle_validation).

Each check returns (passed: bool | None, detail: str); None = not applicable / cannot be judged.
    Counting    final counter == the number in the instruction (BinFill: every colour; StopCube: passes at press)
    Permanence  at every container pick, the BG says the instruction's cube is under the picked container
    Reference   designated target(s) == object(s) actually manipulated in execution
    Imitation   discrete steps extracted from the demo == steps executed (PatternLock: trace == achieved cells)
"""
from __future__ import annotations

import re

from simplememvla.bg.oracle_robomme import NUM_WORDS, ORDINALS, OracleBuilder

SUITE = {
    "BinFill": "counting", "PickXtimes": "counting", "SwingXtimes": "counting", "StopCube": "counting",
    "VideoUnmask": "permanence", "ButtonUnmask": "permanence", "VideoUnmaskSwap": "permanence",
    "ButtonUnmaskSwap": "permanence",
    "PickHighlight": "reference", "VideoRepick": "reference", "VideoPlaceButton": "reference",
    "VideoPlaceOrder": "reference",
    "MoveCube": "imitation", "InsertPeg": "imitation", "PatternLock": "imitation", "RouteStick": "imitation",
}


def _exec(b: OracleBuilder, verb=None):
    return [e for e in b.events if not e["demo"] and (verb is None or e.get("verb") == verb)]


def check(task: str, b: OracleBuilder, instruction: str, rel_at_pick: list[dict]) -> tuple[bool | None, str]:
    instr = instruction.lower()
    a = b.attrs
    if task == "PickXtimes":
        c = next(iter(b.counters.values()), None)
        n = NUM_WORDS.get(re.search(r"repeating this action (\w+) times?", instr).group(1)) if "repeating" in instr else 1
        return (c is not None and c.value == n and c.total == n), f"counter={c and (c.value, c.total)} instr={n}"
    if task == "SwingXtimes":
        c = next(iter(b.counters.values()), None)
        m = re.search(r"repeating this back and forth motion (\w+) times?", instr)
        n = NUM_WORDS.get(m.group(1)) if m else 1  # no repetition clause = one back-and-forth
        return (c is not None and c.value == n), f"counter={c and (c.value, c.total)} instr={n}"
    if task == "BinFill":
        want = {}
        for num, col in re.findall(r"(one|two|three|four|five|six) (red|green|blue) cubes?", instr):
            want[f"{col}_cube"] = NUM_WORDS[num]
        got = {k[1]: c.value for k, c in b.counters.items()}
        return (bool(want) and all(got.get(k) == v for k, v in want.items())), f"counters={got} instr={want}"
    if task == "StopCube":
        press = [e for e in b.events if e.get("verb") == "pressed"]
        k = ORDINALS.get(re.search(r"for the (\w+) time", instr).group(1))
        if not press:
            return False, "no press"
        # success = press during the k-th leg (StopCube.stop_time_range = (interval*(k-1), interval*k)): the cube
        # has reached the target k-1 times (press just before the k-th arrival) or k times (just after it)
        p = press[-1].get("passes")
        return p in (k - 1, k), f"passes_at_press={p} instr={k}"
    if task in ("VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"):
        colors = re.findall(r"hiding the (red|green|blue) cube", instr)
        if len(rel_at_pick) != len(colors):
            return False, f"{len(rel_at_pick)} container picks for {len(colors)} cubes"
        ok, det = True, []
        for col, r in zip(colors, rel_at_pick):
            under = r["relations"].get(f"{col}_cube")
            ok &= under == r["picked"]
            det.append(f"{col}: BG under={under} picked={r['picked']}")
        return ok, "; ".join(det)
    if task == "PickHighlight":
        des = {d.target for (role, _), d in b.designations.items() if role == "pick"}
        picked = {e["target"] for e in _exec(b, "picked")}
        return (bool(des) and des == picked), f"designated={sorted(des)} picked={sorted(picked)}"
    if task == "VideoRepick":
        des = {d.target for (role, _), d in b.designations.items() if role == "repick"}
        picked = {e["target"] for e in _exec(b, "picked")}
        return (len(des) == 1 and des == picked), f"designated={sorted(des)} picked={sorted(picked)}"
    if task in ("VideoPlaceButton", "VideoPlaceOrder"):
        des = [d.target for (role, _), d in b.designations.items() if role == "place"]
        placed = [e.get("dest") for e in _exec(b, "placed") if e.get("dest") not in (None, "table")]
        return (len(des) == 1 and bool(placed) and placed[-1] == des[0]), f"designated={des} placed={placed}"
    if task in ("MoveCube", "InsertPeg", "RouteStick"):
        return (bool(b.demo_tokens) and b.demo_tokens == b.exec_tokens), f"demo={b.demo_tokens} exec={b.exec_tokens}"
    if task == "PatternLock":
        trace = list(b.sequences["trace"].items) if "trace" in b.sequences else []
        ach = a.get("achieved_list") or []
        poses = b._last_poses or {}
        ach_cells = [b._cell_name(int(x.split("_")[-1]), poses) for x in ach if isinstance(x, str)]
        moves_ok = b.demo_tokens == b.exec_tokens
        return (trace == ach_cells and moves_ok), f"trace={trace} achieved={ach_cells} moves {b.demo_tokens}=={b.exec_tokens}"
    return None, "no check"
