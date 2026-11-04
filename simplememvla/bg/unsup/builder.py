"""UnsupBG builder: proprio events + visual tracks + instruction spec -> Belief Graph per data frame (causal).

Inputs (all available to the policy / in the training data; no simulator state):
    state (T, 8)        joints + finger (LeRobot observation.state)
    tracks              visual.Track list from the front camera (2D, SAM2 ids)
    spec                instruction spec (instruction.LLMSpec / rule_spec)
    cam                 fixed front-camera calibration (K, extrinsic) of the episode (unsup/h5cam.py)
                        (RoboMME H5 `setup/front_camera_intrinsic`, `obs/front_camera_extrinsic`), used only to
                        project the FK end-effector position into the image.
Causality: the belief at frame t uses events with frame <= t and track rows with frame <= t only. Tracks are
built forward-only (visual.py), so a track id never depends on later frames.
Facts below the confidence threshold are dropped (``min_conf``).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from simplememvla.bg.schema import (BeliefGraph, Counter, Designation, Interaction, ObjectNode, Provenance,
                                    Relation, RobotNode, Sequence)
from simplememvla.bg.serialize import serialize
from simplememvla.bg.unsup.proprio import gripper_events, tcp_dwell_events, tcp_world

FPS = 20.0
SRC = "unsup"
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "once": 1, "twice": 2}
CAT_ID = {"red cube": "red_cube", "green cube": "green_cube", "blue cube": "blue_cube", "cube": "cube",
          "container": "container", "bin": "bin", "button": "button", "target": "target", "peg": "peg",
          "box": "box", "stick": "stick"}
FIXTURES = {"target", "button", "bin", "box", "container", "stick"}
NEAR_PX = 22.0


def project(cam: dict, p: np.ndarray) -> tuple[float, float] | None:
    K = np.asarray(cam["K"], float)
    E = np.asarray(cam["extrinsic"], float)
    pc = E[:3, :3] @ np.asarray(p, float) + E[:3, 3]
    if pc[2] <= 1e-6:
        return None
    return float(K[0, 0] * pc[0] / pc[2] + K[0, 2]), float(K[1, 1] * pc[1] / pc[2] + K[1, 2])


@dataclass
class _T:
    oid: int
    cat: str
    rows: np.ndarray            # (n, 9) frame, visible, area, cx, cy, x0, y0, x1, y1
    pid: str | None = None

    def at(self, f: int):
        """Last row with frame <= f (or None)."""
        i = int(np.searchsorted(self.rows[:, 0], f, side="right")) - 1
        return self.rows[i] if i >= 0 else None

    def last_visible(self, f: int):
        r = self.rows[self.rows[:, 0] <= f]
        r = r[r[:, 1] > 0]
        return r[-1] if len(r) else None


@dataclass
class UnsupBuilder:
    task: str
    instruction: str
    spec: dict
    state: np.ndarray
    tracks: list
    cam: dict
    stick: bool = False
    demo_end: int | None = None          # first execution frame (detected from the proprio reset jump)
    min_conf: float = 0.5
    T: list[_T] = field(default_factory=list)

    def __post_init__(self):
        self.T = []
        for tr in self.tracks:
            rows = np.array([[r[0], float(r[1]), r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in tr.rows], float)
            if len(rows):
                self.T.append(_T(tr.oid, CAT_ID.get(tr.category, tr.category.replace(" ", "_")), rows))
        # duplicate tracks: a later track of the same category whose first visible centre lies on an earlier track
        # (re-detection of the same object) is dropped
        keep = []
        for t in sorted(self.T, key=lambda t: t.rows[0, 0]):
            v = t.rows[t.rows[:, 1] > 0]
            if not len(v):
                keep.append(t)
                continue
            f0, u0, v0 = v[0, 0], v[0, 3], v[0, 4]
            dup = False
            for k in keep:
                if k.cat != t.cat:
                    continue
                rk = k.last_visible(int(f0))
                if rk is not None and math.hypot(rk[3] - u0, rk[4] - v0) < 8.0:
                    dup = True
                    break
            if not dup:
                keep.append(t)
        self.T = keep
        # persistent ids: first visible frame, then left -> right
        firsts = []
        for t in self.T:
            v = t.rows[t.rows[:, 1] > 0]
            if len(v):
                firsts.append((v[0, 0], v[0, 3], t))
        n_cat: dict[str, int] = {}
        for _, _, t in firsts:
            n_cat[t.cat] = n_cat.get(t.cat, 0) + 1
        k_cat: dict[str, int] = {}
        for f0, u0, t in sorted(firsts, key=lambda x: (x[0], x[1])):
            if n_cat[t.cat] == 1:
                t.pid = t.cat
            else:
                k = k_cat.get(t.cat, 0)
                k_cat[t.cat] = k + 1
                t.pid = f"{t.cat}_{chr(ord('A') + k)}"
        if self.demo_end is None:
            self.demo_end = detect_demo_end(self.state)
        self.events = (tcp_dwell_events(self.state) if self.stick else gripper_events(self.state))
        self._annotate_events()

    # ------------------------------------------------------------------ helpers
    def _uv(self, f: int):
        p = tcp_world(self.state[f, :7], stick=self.stick)
        return project(self.cam, p)

    def _nearest(self, f: int, uv, cats=None, max_px=NEAR_PX, visible_only=False):
        if uv is None:
            return None, 1e9
        best, bd = None, 1e9
        for t in self.T:
            if t.pid is None or (cats and t.cat not in cats):
                continue
            r = t.at(f)
            if r is None or (visible_only and r[1] <= 0):
                continue
            rv = t.last_visible(f)
            if rv is None:
                continue
            d = math.hypot(rv[3] - uv[0], rv[4] - uv[1])
            if d < bd:
                best, bd = t, d
        return (best, bd) if bd <= max_px else (None, bd)

    def _annotate_events(self):
        held = None
        movable = {c for c in {t.cat for t in self.T} if c not in FIXTURES or c == "container"}
        for e in self.events:
            f = e["frame"]
            uv = self._uv(f)
            e["demo"] = f < (self.demo_end or 0)
            if e["type"] == "grasp":
                t, d = self._nearest(f, uv, cats=movable, max_px=NEAR_PX * 1.5, visible_only=True)
                e["obj"] = t.pid if t else None
                e["conf"] = 1.0 if t else 0.0
                held = e["obj"]
            elif e["type"] == "release":
                e["obj"] = held
                dest, _ = self._nearest(f, uv, cats={"target", "bin", "box"}, max_px=NEAR_PX * 1.5)
                e["dest"] = dest.pid if dest else "table"
                e["conf"] = 1.0 if held else 0.3
                held = None
            elif e["type"] == "close":
                b, _ = self._nearest(f, uv, cats={"button"}, max_px=NEAR_PX * 2)
                # fallback when the detector misses the button: the closed empty gripper goes down to the table
                z = [tcp_world(self.state[j, :7])[2] for j in range(f, min(len(self.state), f + int(3 * FPS)))]
                has_button = any("button" in o for o in (self.spec.get("objects") or []))
                pressed_like = has_button and bool(z) and min(z) < 0.06
                e["obj"] = b.pid if b else ("button" if pressed_like else None)
                e["conf"] = 1.0 if b else (0.6 if pressed_like else 0.0)
            elif e["type"] == "dwell":
                c, _ = self._nearest(f, uv, cats=None, max_px=NEAR_PX)
                e["obj"] = c.pid if c else None
                e["conf"] = 1.0 if c else 0.0

    # ------------------------------------------------------------------ belief at t
    def graph(self, f: int) -> BeliefGraph:
        t_now = f / FPS
        objs, rels = [], []
        hidden_now = {}
        for t in self.T:
            if t.pid is None:
                continue
            r = t.at(f)
            lv = t.last_visible(f)
            if lv is None:
                continue
            vis = "visible" if (r is not None and r[1] > 0) or t.cat in FIXTURES else "hidden"
            slot = self._slot(t, f)
            fv = t.rows[t.rows[:, 1] > 0]
            first_slot = self._slot(t, int(fv[0, 0])) if len(fv) else None
            if t.pid in self._binned_at(f):
                continue
            objs.append(ObjectNode(t.pid, t.cat, slot, vis, lv[0] / FPS, Provenance(lv[0] / FPS, SRC), first_slot))
            if vis == "hidden":
                hidden_now[t.pid] = (t, lv)
        # under: a hidden cube whose last visible centre is covered by a container now / when it vanished
        conts = [t for t in self.T if t.cat == "container" and t.pid]
        for pid, (t, lv) in hidden_now.items():
            if not t.cat.endswith("cube") or not conts:
                continue
            fv = int(lv[0])
            best, bd = None, 1e9
            for c in conts:
                rc = c.last_visible(fv + 2 * int(FPS))  # container position right after the cube vanished
                if rc is None:
                    continue
                d = math.hypot(rc[3] - lv[3], rc[4] - lv[4])
                if d < bd:
                    best, bd = c, d
            if best is not None and bd < NEAR_PX * 1.5:
                rels.append(Relation("under", pid, best.pid, fv / FPS, Provenance(fv / FPS, SRC, 0.8)))
        # interactions
        inter, k = [], 0
        for e in self.events:
            if e["frame"] > f or e.get("conf", 0) < self.min_conf:
                continue
            verb = {"grasp": "picked", "release": "placed", "close": "pressed", "dwell": "touched"}[e["type"]]
            if e.get("obj") is None:
                continue
            k += 1
            dest = e.get("dest") if e["type"] == "release" else None
            inter.append(Interaction(k, verb, e["obj"], e["frame"] / FPS, dest, Provenance(e["frame"] / FPS, SRC)))
        counters, des, seqs = self._task_facts(f, inter)
        g = BeliefGraph(t=t_now, objects=objs, robot=RobotNode(
            "open" if self.state[f, 7] > 0.035 else "closed", self._holding(f), Provenance(t_now, SRC)),
            relations=_dedup_rel(rels), interactions=inter, counters=counters, designations=des, sequences=seqs)
        return g

    def _binned_at(self, f):
        return {e["obj"] for e in self.events
                if e["type"] == "release" and e.get("dest") == "bin" and e["frame"] <= f and e.get("obj")}

    def _holding(self, f):
        held = None
        for e in self.events:
            if e["frame"] > f:
                break
            if e["type"] == "grasp":
                held = e.get("obj")
            elif e["type"] == "release":
                held = None
        return held

    def _slot(self, t: _T, f: int) -> str:
        same = [x for x in self.T if x.cat == t.cat and x.pid and x.last_visible(f) is not None]
        if len(same) <= 1:
            return "table"
        ranked = sorted(same, key=lambda x: x.last_visible(f)[3])
        names = (["left", "right"] if len(ranked) == 2 else ["left", "middle", "right"] if len(ranked) == 3
                 else [f"pos{i + 1}" for i in range(len(ranked))])
        return names[ranked.index(t)]

    def _task_facts(self, f, inter):
        g = self.spec.get("goal", {}) or {}
        rem = set(self.spec.get("remember", []) or [])
        counters, des, seqs = [], [], []
        t_now = f / FPS
        exec_inter = [r for r in inter if r.t * FPS >= (self.demo_end or 0)]
        demo_inter = [r for r in inter if r.t * FPS < (self.demo_end or 0)]
        if "count" in rem:
            obj = g.get("object")
            oid = CAT_ID.get(obj, obj.replace(" ", "_")) if obj else None
            total = g.get("count")
            if self.task == "BinFill" or "into the bin" in self.instruction:
                for num, col in re.findall(r"(one|two|three|four|five|six) (red|green|blue) cubes?", self.instruction):
                    n = sum(1 for r in exec_inter if r.verb == "placed" and r.dest == "bin" and r.target
                            and r.target.startswith(f"{col}_cube"))
                    counters.append(Counter("binned", f"{col}_cube", n, NUM[num], Provenance(t_now, SRC)))
            elif g.get("verb") == "stop":
                counters.append(Counter("reached", "cube", self._passes(f), g.get("ordinal"), Provenance(t_now, SRC)))
            elif oid:
                tgt = next((r.target for r in exec_inter if r.verb == "picked"), oid)
                n = sum(1 for r in exec_inter if r.verb == "picked" and r.target == tgt)
                counters.append(Counter("picked", tgt, n, total, Provenance(t_now, SRC)))
        if "designated" in rem:
            if g.get("relation") == "same_as_video":
                first = next((r for r in demo_inter if r.verb == "picked"), None)
                if first is not None:
                    des.append(Designation("repick", first.target, "video", first.t, Provenance(first.t, SRC)))
        hl = [t for t in self.T if t.cat == "highlight"]
        if hl:  # cubes with a highlight disk under them (seen at or before f) are designated
            done = set()
            for h in hl:
                for r in h.rows[(h.rows[:, 0] <= f) & (h.rows[:, 1] > 0)]:
                    for c in self.T:
                        if not c.cat.endswith("cube") or c.pid is None or c.pid in done:
                            continue
                        rc = c.last_visible(int(r[0]))
                        if rc is not None and math.hypot(rc[3] - r[3], rc[4] - r[4]) < 15.0:
                            des.append(Designation("pick", c.pid, "highlight", r[0] / FPS, Provenance(r[0] / FPS, SRC)))
                            done.add(c.pid)
        if g.get("relation") in ("before_button", "after_button") or "order" in rem:
            items = []
            for r in demo_inter:
                if r.verb == "placed" and r.dest and r.dest != "table":
                    items.append(r.dest)
                elif r.verb == "pressed":
                    items.append("button")
            if items:
                tl = max(r.t for r in demo_inter)
                seqs.append(Sequence("demo", items, "video", tl, Provenance(tl, SRC)))
                targets = [x for x in items if x != "button"]
                if g.get("ordinal") and len(targets) >= g["ordinal"]:
                    des.append(Designation("place", targets[g["ordinal"] - 1], "video", tl, Provenance(tl, SRC)))
                if "button" in items:
                    i = items.index("button")
                    if g.get("relation") == "before_button" and [x for x in items[:i]]:
                        des.append(Designation("place", items[:i][-1], "video", tl, Provenance(tl, SRC)))
                    if g.get("relation") == "after_button" and items[i + 1:]:
                        des.append(Designation("place", items[i + 1], "video", tl, Provenance(tl, SRC)))
        if rem & {"pattern", "route"}:
            cells = [e["obj"] for e in self.events if e["type"] == "dwell" and e["demo"] and e["frame"] <= f
                     and e.get("obj")]
            if cells:
                tl = max(e["frame"] for e in self.events if e["type"] == "dwell" and e["demo"] and e["frame"] <= f) / FPS
                seqs.append(Sequence("trace", cells, "video", tl, Provenance(tl, SRC)))
        return counters, des, seqs

    def _passes(self, f):
        cube = next((t for t in self.T if t.cat.endswith("cube")), None)
        tgt = next((t for t in self.T if t.cat == "target"), None)
        if cube is None or tgt is None:
            return 0
        n, over = 0, False
        for r in cube.rows[cube.rows[:, 0] <= f]:
            rt = tgt.last_visible(int(r[0]))
            if r[1] <= 0 or rt is None:
                continue
            o = math.hypot(r[3] - rt[3], r[4] - rt[4]) < 6.0
            if o and not over:
                n += 1
            over = o
        return n

    def labels(self, max_interactions: int = 4) -> list[dict]:
        rows = []
        for f in range(len(self.state)):
            g = self.graph(f)
            g.validate()
            rows.append({"frame_index": f, "bg_json": g.to_json(), "bg_mnemonic": serialize(g, "mnemonic", max_interactions),
                         "bg_full": serialize(g, "full", max_interactions)})
        return rows


def detect_demo_end(state: np.ndarray, jump: float = 0.3) -> int:
    """First frame after the largest one-step joint jump (> jump rad) = robot reset after the video demo."""
    d = np.abs(np.diff(state[:, :7], axis=0)).max(1)
    if len(d) == 0 or d.max() <= jump:
        return 0  # no reset jump (robot static during the video, e.g. VideoUnmask): no robot demo events
    return int(np.argmax(d) + 1)


def _dedup_rel(rels):
    out = {}
    for r in rels:
        out[(r.subj, r.kind)] = r
    return list(out.values())
