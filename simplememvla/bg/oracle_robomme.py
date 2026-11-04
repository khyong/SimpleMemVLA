"""BG-Oracle for RoboMME: privileged replay snapshots -> Belief Graph per LeRobot frame.

``OracleBuilder`` is incremental and causal: ``update(snap, t)`` consumes one simulator frame observed at data time
``t`` (seconds) and ``graph()`` returns the belief after everything seen so far. Offline labels
(``build_episode``) and the online tracker used at evaluation (``OracleBGTracker``) share it, so both produce the
same BG for the same frame stream.

Event source: sub-goal completions (``env.timestep`` increments over ``env.task_list``) -- identical semantics
during the conditioning video and execution -- plus task state the simulator keeps (bin counters, swing count,
cell highlights) and geometry (cube under container, cube over target, visibility from front-camera
segmentation). Privileged quantities that a viewer of the video could not know yet are never used before the
moment they become observable (e.g. PickHighlight targets only from the first highlighted frame).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simplememvla.bg.schema import (BeliefGraph, Counter, Designation, Interaction, ObjectNode, Provenance,
                                    Relation, RobotNode, Sequence)
from simplememvla.bg.serialize import serialize

FPS = 20.0
VIS_PX = 8
SRC = "oracle"
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "once": 1, "twice": 2}

# ----------------------------------------------------------------------------- object roles per task
PARKED = {"fixed_cube_2", "goal_site_2"}  # MoveCube: off-table pose holders, never seen
# fixtures: occlusion by the robot is not memory -> treated as visible once seen (slot still tracked)
FIXTURES = {"target", "cell", "waypoint", "stick", "button", "bin", "box", "goal", "container"}
_COLOR_CUBE = re.compile(r"^(?:target_)?cube_(red|green|blue)(?:_\d+)?$")


def object_category(task: str, name: str) -> str | None:
    """Semantic category of a simulator actor name, None for actors that are not BG objects."""
    if name.startswith(("panda", "camera", "highlight", "table", "ground")) or name in PARKED:
        return None
    m = _COLOR_CUBE.match(name)
    if m:
        return f"{m.group(1)}_cube"
    if task == "StopCube" and name == "target_cube":
        return "cube"
    if task == "VideoRepick" and name.startswith("bin_"):
        return "cube"  # same-colour cubes named bin_i
    if name.startswith("bin_"):
        return "container"
    if name == "board_with_hole":
        return "bin"
    if name in ("button_cap",):
        return "button"
    if name in ("button", "button_base", "button_left", "button_right"):
        return None  # the cap link carries the button
    if task == "PatternLock" and name.startswith("target_"):
        return "cell"
    if task == "RouteStick" and name.startswith("target_cube_"):
        return "stick"
    if task == "RouteStick" and name.startswith("target_"):
        return "waypoint"
    if name.startswith(("target", "temp_target")):
        return "target"
    if name == "box_with_hole":
        return "box"
    if re.match(r"^peg(_\d+)?$", name):
        return "peg"
    if name == "fixed_cube":
        return "cube"
    if name == "goal_site" and task == "MoveCube":
        return "goal"
    return None


# ----------------------------------------------------------------------------- helpers
def _instruction_number(instr: str) -> int | None:
    m = re.search(r"\b(one|two|three|four|five|six|once|twice)\b(?: times)?", instr)
    return NUM_WORDS.get(m.group(1)) if m else None


def _instruction_ordinal(instr: str) -> int | None:
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth)\b", instr)
    return ORDINALS.get(m.group(1)) if m else None


def _seg(segment) -> str | None:
    if isinstance(segment, (list, tuple)):
        return _seg(segment[0]) if segment else None
    return segment


def _token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@dataclass
class _Obj:
    name: str
    category: str
    pid: str | None = None
    first_t: float | None = None
    last_seen_t: float | None = None
    last_slot: str | None = None
    first_slot: str | None = None
    visible: bool = False
    u: float = 0.0


@dataclass
class OracleBuilder:
    task: str
    task_list: list[dict]
    instruction: str
    attrs0: dict
    cam: dict | None = None
    t: float = 0.0
    objs: dict[str, _Obj] = field(default_factory=dict)
    interactions: list[Interaction] = field(default_factory=list)
    counters: dict[tuple, Counter] = field(default_factory=dict)
    designations: dict[tuple, Designation] = field(default_factory=dict)
    sequences: dict[str, Sequence] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)  # subj pid -> relation
    task_index: int = 0
    held: str | None = None
    gripper: str = "unknown"
    holding: str | None = None
    attrs: dict = field(default_factory=dict)
    demo_tokens: list[str] = field(default_factory=list)
    exec_tokens: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # for validation
    _press_count: int = 0
    _stop_over: bool = False
    _stop_passes: int = 0
    _cells_demo: list[str] = field(default_factory=list)
    _hl_seen: set = field(default_factory=set)
    _cat_n: dict[str, int] = field(default_factory=dict)
    _cat_assigned: dict[str, int] = field(default_factory=dict)
    rel_at_pick: list[dict] = field(default_factory=list)
    _rel_free: dict = field(default_factory=dict)  # "under" relations while no container is held (pre-lift)
    _binned: set = field(default_factory=set)
    _last_poses: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ setup
    def __post_init__(self):
        self.attrs = dict(self.attrs0)

    def _init_objects(self, poses: dict):
        for n in poses:
            if n in self.objs:
                continue
            c = object_category(self.task, n)
            if c is None:
                continue
            self.objs[n] = _Obj(n, c)
        self._cat_n = {}
        for o in self.objs.values():
            self._cat_n[o.category] = self._cat_n.get(o.category, 0) + 1

    def _u(self, p) -> float:
        """Horizontal image coordinate (left -> right) of a world point."""
        if self.cam is not None:
            K = np.asarray(self.cam["K"], float)
            E = np.asarray(self.cam["extrinsic"], float)
            pc = E[:3, :3] @ np.asarray(p, float) + E[:3, 3]
            if pc[2] > 1e-6:
                return float(K[0, 0] * pc[0] / pc[2] + K[0, 2])
        return float(p[1]) * 100.0  # camera at +x looking -x: image right = +y

    def _assign(self, objs: list[_Obj]):
        for o in sorted(objs, key=lambda o: (o.u, o.name)):
            if o.pid is not None:
                continue
            if self._cat_n.get(o.category, 1) <= 1:
                o.pid = o.category
            else:
                k = self._cat_assigned.get(o.category, 0)
                self._cat_assigned[o.category] = k + 1
                o.pid = f"{o.category}_{chr(ord('A') + k)}"

    def pid(self, name: str | None) -> str | None:
        if name is None:
            return None
        o = self.objs.get(name)
        if o is None:
            c = object_category(self.task, name)
            if c is None:
                return _token(name)
            o = self.objs[name] = _Obj(name, c)
            self._cat_n[c] = self._cat_n.get(c, 0) + 1
        if o.pid is None:
            self._assign([o])
        return o.pid

    # ------------------------------------------------------------------ update
    def update(self, snap: dict, t: float) -> None:
        self.t = float(t)
        if self.cam is None and snap.get("cam") is not None:
            self.cam = snap["cam"]
        poses = snap["poses"]
        self._last_poses = poses
        if not self.objs:
            self._init_objects(poses)
        self.attrs.update(snap.get("attrs", {}))
        vis = snap.get("vis_px", {})
        newly = []
        for o in self.objs.values():
            p = poses.get(o.name)
            if p is None:
                continue
            far = abs(p[0]) > 5 or abs(p[1]) > 5  # teleported out of the scene
            seen = (not far) and vis.get(o.name, 0) >= VIS_PX
            if not far:
                o.u = self._u(p)
            if seen:
                if o.first_t is None:
                    o.first_t = t
                    newly.append(o)
                o.last_seen_t = t
            o.visible = seen
        self._assign(newly)
        self._update_slots(poses)
        # robot
        q = snap.get("qpos", [])
        if len(q) > 7:
            self.gripper = "open" if float(q[7]) > 0.035 else "closed"
        g = [n for n in snap.get("grasped", []) if n in self.objs]
        self.holding = self.pid(g[0]) if g else None
        # sub-goal completions
        ti = int(snap.get("task_index", 0))
        while self.task_index < ti and self.task_index < len(self.task_list):
            self._on_complete(self.task_list[self.task_index], snap)
            self.task_index += 1
        self._task_state(snap)
        self._relations(poses)
        held_cat = self.objs[g[0]].category if g else None
        if held_cat != "container":
            self._rel_free = {k: r.obj for k, r in self.relations.items() if r.kind == "under"}

    def _update_slots(self, poses):
        by_cat: dict[str, list[_Obj]] = {}
        for o in self.objs.values():
            p = poses.get(o.name)
            if p is not None and abs(p[0]) <= 5 and abs(p[1]) <= 5 and o.first_t is not None:
                by_cat.setdefault(o.category, []).append(o)
        for cat, lst in by_cat.items():
            n_total = self._cat_n.get(cat, 1)
            if n_total <= 1:
                for o in lst:
                    z = poses.get(o.name, [0, 0, 0])[2]
                    o.last_slot = "lifted" if z > 0.08 and cat not in ("button",) else "table"
                    if o.first_slot is None:
                        o.first_slot = o.last_slot
                continue
            ranked = sorted(lst, key=lambda o: o.u)
            names = (["left", "right"] if len(ranked) == 2 else
                     ["left", "middle", "right"] if len(ranked) == 3 else
                     ["table"] if len(ranked) == 1 else
                     [f"pos{i + 1}" for i in range(len(ranked))])
            for o, s in zip(ranked, names):
                if o.visible or o.category in FIXTURES:
                    o.last_slot = s
                    if o.first_slot is None:
                        o.first_slot = s

    # ------------------------------------------------------------------ events
    def _add_interaction(self, verb, target, dest=None):
        k = len(self.interactions) + 1
        self.interactions.append(Interaction(k, verb, target, self.t, dest, Provenance(self.t, SRC)))

    def _on_complete(self, task: dict, snap: dict) -> None:
        name = str(task.get("name") or "")
        lname = name.lower()
        seg = _seg(task.get("segment"))
        demo = bool(task.get("demonstration"))
        tok = None
        if lname in ("static", "no record", "remain static") or lname.startswith("move to the top of the button"):
            return
        ev = {"t": self.t, "name": name, "demo": demo, "segment": seg}
        if lname.startswith("pick up the container"):
            tgt = self.pid(seg)
            self.rel_at_pick.append({"t": self.t, "picked": tgt, "relations": dict(self._rel_free)})
            self.held = tgt
            self._add_interaction("picked", tgt)
            ev.update(verb="picked", target=tgt)
        elif lname.startswith("put down the container"):
            self._add_interaction("placed", self.held or "container", "table")
            ev.update(verb="placed", target=self.held)
        elif re.match(r"pick up the peg by grasping the (far|near) end", lname):
            end = re.search(r"(far|near) end", lname).group(1)
            peg = self.pid(self._peg_of(seg))
            self.held = peg
            self._add_interaction("picked", peg)
            tok = f"grasp_{end}_end"
            ev.update(verb="picked", target=peg)
            if demo:
                self._designate("grasp", peg, "video")
        elif re.match(r"insert the peg from the (left|right) side", lname):
            side = re.search(r"(left|right) side", lname).group(1)
            self._add_interaction("inserted", self.held or "peg", "box")
            tok = f"insert_{side}_side"
            ev.update(verb="inserted", target=self.held)
        elif lname == "pick up the peg":
            self.held = self.pid("peg")
            self._add_interaction("picked", self.held)
            tok = "picked_peg"
        elif lname.startswith("hook the cube"):
            self._add_interaction("pushed", self.pid("fixed_cube"), self.pid("goal_site"))
            tok = "hooked_cube"
        elif lname.startswith("close the gripper and push"):
            self._add_interaction("pushed", self.pid("fixed_cube"), self.pid("goal_site"))
            tok = "pushed_cube_with_gripper"
        elif self.task == "MoveCube" and lname == "pick up the cube":
            self.held = self.pid("fixed_cube")
            self._add_interaction("picked", self.held)
            tok = "picked_cube"
        elif self.task == "MoveCube" and lname.startswith("place the cube onto the target"):
            self._add_interaction("placed", self.pid("fixed_cube"), self.pid("goal_site"))
            tok = "placed_cube"
        elif lname.startswith("pick up"):
            tgt = self.pid(seg) if seg else self.holding
            self.held = tgt
            self._add_interaction("picked", tgt)
            ev.update(verb="picked", target=tgt)
            if self.task == "VideoRepick" and demo:
                self._designate("repick", tgt, "video")
        elif lname.startswith("put it into the bin"):
            self._add_interaction("placed", self.held, "bin")
            if self.held:
                self._binned.add(self.held)
            ev.update(verb="placed", target=self.held, dest="bin")
        elif re.match(r"(place|drop) the .*onto (the )?(correct )?target|place the .* onto the target", lname):
            dest = self.pid(seg)
            self._add_interaction("placed", self.held, dest)
            ev.update(verb="placed", target=self.held, dest=dest)
            if demo and self.task in ("VideoPlaceButton", "VideoPlaceOrder"):
                self._demo_seq_add(dest)
        elif re.match(r"(drop the cube on(to)? (the )?table|put it down|place the cube onto the table|put the .* on the table)",
                      lname):
            self._add_interaction("placed", self.held, "table")
            ev.update(verb="placed", target=self.held, dest="table")
        elif lname.startswith("press"):
            btn = self.pid("button_cap")
            self._add_interaction("pressed", btn)
            self._press_count += 1
            ev.update(verb="pressed", target=btn)
            if demo and self.task in ("VideoPlaceButton", "VideoPlaceOrder"):
                self._demo_seq_add("button")
            if self.task == "StopCube":
                ev.update(passes=self._stop_passes)
        elif lname.startswith("move to the top of the"):
            side = "right" if "right-side" in lname else "left"
            tgt = self.pid(seg)
            self._add_interaction("reached", tgt)
            ev.update(verb="reached", target=tgt, side=side)
        elif re.match(r"move to the nearest (left|right) target by circling around the stick (clockwise|counterclockwise)",
                      lname):
            m = re.match(r"move to the nearest (left|right) target by circling around the stick (clockwise|counterclockwise)",
                         lname)
            tok = f"{m.group(1)}_{'cw' if m.group(2) == 'clockwise' else 'ccw'}"
        elif re.match(r"^move [a-z \-]+$", lname) and self.task == "PatternLock":
            tok = _token(lname[len("move "):])
        else:
            ev.update(verb="unmapped")
        if tok is not None:
            (self.demo_tokens if demo else self.exec_tokens).append(tok)
            ev.update(token=tok)
            if demo and self.task in ("MoveCube", "InsertPeg", "RouteStick", "PatternLock"):
                role = {"MoveCube": "demo", "InsertPeg": "demo", "RouteStick": "route", "PatternLock": "moves"}[self.task]
                self.sequences[role] = Sequence(role, list(self.demo_tokens), "video", self.t, Provenance(self.t, SRC))
        self.events.append(ev)

    def _peg_of(self, seg: str | None) -> str:
        p = self.attrs0.get("peg")
        return p[1:] if isinstance(p, str) and p.startswith("@") else "peg_0"

    def _designate(self, role: str, target: str, via: str):
        self.designations[(role, target)] = Designation(role, target, via, self.t, Provenance(self.t, SRC))

    def _demo_seq_add(self, item: str):
        seq = self.sequences.get("demo")
        items = (list(seq.items) if seq else []) + [item]
        self.sequences["demo"] = Sequence("demo", items, "video", self.t, Provenance(self.t, SRC))
        targets = [x for x in items if x != "button"]
        if self.task == "VideoPlaceOrder":
            k = _instruction_ordinal(self.instruction)
            if k is not None and len(targets) >= k and ("place", targets[k - 1]) not in self.designations:
                self._designate("place", targets[k - 1], "video")
        elif self.task == "VideoPlaceButton" and "button" in items:
            i = items.index("button")
            before = [x for x in items[:i] if x != "button"]
            after = [x for x in items[i + 1:] if x != "button"]
            if "right before" in self.instruction and before:
                if not any(r == "place" for r, _ in self.designations):
                    self._designate("place", before[-1], "video")
            if "right after" in self.instruction and after:
                if not any(r == "place" for r, _ in self.designations):
                    self._designate("place", after[0], "video")

    # ------------------------------------------------------------------ task state
    def _task_state(self, snap):
        a = self.attrs
        task = self.task
        if task == "PickXtimes":
            n = sum(1 for r in self.interactions if r.verb == "picked")
            tgt = self._first_pick_target() or self._target_cube_pid()
            if tgt:
                self.counters[("picked", tgt)] = Counter("picked", tgt, n, a.get("num_repeats"), Provenance(self.t, SRC))
        elif task == "SwingXtimes":
            n = sum(1 for e in self.events if e.get("verb") == "reached" and e.get("side") == "left")
            tgt = self._target_cube_pid()
            if tgt:
                self.counters[("swung", tgt)] = Counter("swung", tgt, n, a.get("num_repeats"), Provenance(self.t, SRC))
        elif task == "BinFill":
            for c in ("red", "green", "blue"):
                total = a.get(f"{c}_cubes_target_number")
                if not total:
                    continue
                n = int(a.get(f"{c}_cubes_in_bin", 0) or 0)
                self.counters[("binned", f"{c}_cube")] = Counter("binned", f"{c}_cube", n, int(total),
                                                                 Provenance(self.t, SRC))
        elif task == "StopCube":
            cube, tgt = snap["poses"].get("target_cube"), snap["poses"].get("target")
            if cube is not None and tgt is not None:
                dx, dy = cube[0] - tgt[0], cube[1] - tgt[1]
                over = math.hypot(dx, dy) < 0.02
                # a fast leg can jump over the 2 cm disk between frames: also count a sign change of the offset
                # along the motion axis (fixed line through the target) while not already "over"
                prev = getattr(self, "_stop_prev", None)
                crossed = False
                if prev is not None and not self._stop_over and not over:
                    ax = np.array([cube[0] - prev[0], cube[1] - prev[1]])
                    if np.linalg.norm(ax) > 1e-6:
                        p0 = (prev[0] - tgt[0]) * ax[0] + (prev[1] - tgt[1]) * ax[1]
                        p1 = dx * ax[0] + dy * ax[1]
                        perp = abs(dx * ax[1] - dy * ax[0]) / np.linalg.norm(ax)
                        crossed = p0 < 0 <= p1 and perp < 0.02
                if (over and not self._stop_over) or crossed:
                    self._stop_passes += 1
                self._stop_over = over
                self._stop_prev = (cube[0], cube[1])
            self.counters[("reached", "target")] = Counter("reached", self.pid("target_cube") or "cube",
                                                           self._stop_passes, a.get("stop_time"),
                                                           Provenance(self.t, SRC))
        elif task == "VideoRepick":
            des = [d.target for (r, _), d in self.designations.items() if r == "repick"]
            if des:
                n = sum(1 for e in self.events if e.get("verb") == "picked" and not e["demo"])
                self.counters[("picked", des[0])] = Counter("picked", des[0], n, a.get("num_repeats"),
                                                            Provenance(self.t, SRC))
        elif task == "PickHighlight":
            # targets are highlighted from execution step 10 on; designate when the highlight is first visible
            names = a.get("target_cube_names") or []
            if snap.get("elapsed", -1) >= 10 and not snap.get("is_demo"):
                for n in names:
                    pid = self.pid(n)
                    if ("pick", pid) not in self.designations:
                        self._designate("pick", pid, "highlight")
        elif task in ("PatternLock",):
            hs = a.get("highlight_starts") or {}
            if snap.get("is_demo"):
                for k in hs:
                    if k not in self._hl_seen:
                        self._hl_seen.add(k)
                        self._cells_demo.append(self._cell_name(int(k), snap["poses"]))
                        self.sequences["trace"] = Sequence("trace", list(self._cells_demo), "video", self.t,
                                                           Provenance(self.t, SRC))

    def _cell_name(self, k: int, poses) -> str:
        cells = sorted([(n, p) for n, p in poses.items() if n.startswith("target_")], key=lambda x: x[0])
        xs = sorted({round(p[0], 3) for _, p in cells})
        ys = sorted({round(p[1], 3) for _, p in cells})
        p = poses.get(f"target_{k}")
        if p is None:
            return f"cell_{k}"
        r = int(np.argmin([abs(p[0] - x) for x in xs]))
        c = int(np.argmin([abs(p[1] - y) for y in ys]))
        return f"cell_r{r}c{c}"

    def _target_cube_pid(self):
        tc = self.attrs0.get("target_cube")
        if isinstance(tc, str) and tc.startswith("@"):
            return self.pid(tc[1:])
        return None

    def _first_pick_target(self):
        for r in self.interactions:
            if r.verb == "picked":
                return r.target
        return None

    # ------------------------------------------------------------------ relations
    def _relations(self, poses):
        conts = [o for o in self.objs.values() if o.category == "container"]
        if not conts:
            return
        for o in self.objs.values():
            if not o.category.endswith("_cube"):
                continue
            pid = o.pid
            if pid is None:
                continue
            if o.visible:
                if pid in self.relations and self.relations[pid].kind == "under":
                    self.relations.pop(pid, None)
                continue
            if pid in self.relations:
                continue  # hidden: the relation persists with the container identity (swaps move containers)
            p = poses.get(o.name)
            if p is None or abs(p[0]) > 5:
                continue
            best, bd = None, 1e9
            for c in conts:
                q = poses.get(c.name)
                if q is None or abs(q[0]) > 5:
                    continue
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d < bd and q[2] > p[2] - 0.005:
                    best, bd = c, d
            if best is not None and bd < 0.04:
                cid = best.pid or self.pid(best.name)
                self.relations[pid] = Relation("under", pid, cid, self.t, Provenance(self.t, SRC))

    # ------------------------------------------------------------------ graph
    def graph(self) -> BeliefGraph:
        objs = []
        for o in sorted(self.objs.values(), key=lambda o: o.pid or "~"):
            if o.pid is None:
                continue
            if o.first_t is None:
                vis = "unknown"
            elif o.visible or o.category in FIXTURES:
                vis = "visible"
            else:
                vis = "hidden"
            ev = o.last_seen_t if o.last_seen_t is not None else self.t
            if o.pid in self._binned:
                continue  # in the bin: accounted for by the binned counter
            objs.append(ObjectNode(o.pid, o.category, o.last_slot, vis, o.last_seen_t, Provenance(min(ev, self.t), SRC),
                                   o.first_slot))
        return BeliefGraph(
            t=self.t, objects=objs,
            robot=RobotNode(self.gripper, self.holding, Provenance(self.t, SRC)),
            relations=list(self.relations.values()),
            interactions=list(self.interactions),
            counters=list(self.counters.values()),
            designations=list(self.designations.values()),
            sequences=list(self.sequences.values()))


# ----------------------------------------------------------------------------- offline episode
def build_episode(rec: dict, task: str, n_data: int, max_interactions: int = 4) -> tuple[list[dict], OracleBuilder]:
    """rec = replay pickle (snaps, task_list, data_to_snap). Returns one row per data frame + the final builder."""
    snaps = rec["snaps"]
    d2s = rec.get("data_to_snap") or list(range(min(n_data, len(snaps))))
    if len(d2s) < n_data:
        raise ValueError(f"replay maps {len(d2s)} of {n_data} data frames")
    # observation time of every snapshot: the data time of the first data frame at or after it
    obs_t = np.full(len(snaps), np.nan)
    j = n_data - 1
    for i in range(len(snaps) - 1, -1, -1):
        while j > 0 and d2s[j - 1] >= i:
            j -= 1
        if d2s[j] >= i:
            obs_t[i] = j / FPS
    cam = next((s["cam"] for s in snaps if s.get("cam")), None)
    b = OracleBuilder(task, rec["task_list"], rec.get("instruction", ""), dict(snaps[0].get("attrs", {})), cam)
    rows = []
    si = 0
    for f in range(n_data):
        target = d2s[f]
        while si <= target:
            t_obs = obs_t[si] if not np.isnan(obs_t[si]) else f / FPS
            b.update(snaps[si], min(float(t_obs), f / FPS))
            si += 1
        b.t = f / FPS
        g = b.graph()
        g.validate()
        rows.append({"frame_index": f, "bg_json": g.to_json(),
                     "bg_mnemonic": serialize(g, "mnemonic", max_interactions),
                     "bg_full": serialize(g, "full", max_interactions)})
    return rows, b


# ----------------------------------------------------------------------------- online tracker (evaluation)
class OracleBGTracker:
    """Same builder fed from a live evaluation env (privileged state each step). Use ``attach_online_trackers``."""

    def __init__(self, task: str, bg_filter: str = "mnemonic", max_interactions: int = 4):
        self.task = task
        self.bg_filter = bg_filter
        self.max_interactions = max_interactions
        self.builder: OracleBuilder | None = None
        self.n_frames = 0

    def on_snapshot(self, snap: dict, task_list: list[dict], instruction: str):
        if self.builder is None:
            self.builder = OracleBuilder(self.task, task_list, instruction, dict(snap.get("attrs", {})), snap.get("cam"))
        self.builder.update(snap, self.n_frames / FPS)
        self.n_frames += 1

    def graph(self) -> BeliefGraph:
        return self.builder.graph()

    def text(self) -> str:
        return serialize(self.builder.graph(), self.bg_filter, self.max_interactions)


def attach_online_trackers(pool, specs, bg_filter: str = "mnemonic", max_interactions: int = 4) -> dict:
    """Hook every env service of an InProcSimPool so that each recorded frame (reset demo frames with "NO RECORD"
    dropped, as evaluation does, then every step) feeds an OracleBGTracker. Returns {slot: tracker}."""
    from simplememvla.bg.robomme_replay import ReplayRecorder

    trackers = {}
    for slot, spec in enumerate(specs):
        if slot >= len(pool.services):
            break
        svc = pool.services[slot]
        tr = OracleBGTracker(spec["task"], bg_filter, max_interactions)
        trackers[slot] = tr
        _hook_service(svc, tr, ReplayRecorder)
    return trackers


def _hook_service(svc, tracker, recorder_cls):
    def ensure_env(task_name):
        svc._ensure_env_orig(task_name)
        rce = svc.env
        if getattr(rce, "_bg_hooked", False):
            return
        orig_env_reset = rce.reset

        def env_reset(episode_idx):
            rec = recorder_cls.__new__(recorder_cls)
            rec.snaps, rec._prev_attrs, rec._depth, rec._in_reset, rec._cam_done = [], {}, 0, True, False
            rce.close()
            env = rce.builder.make_env_for_episode(int(episode_idx), include_maniskill_obs=True)
            rce.env = env
            w = rec._install_hook(env)
            obs, info = env.reset()
            rec._in_reset = False
            snaps = [s for s in rec.snaps if s["subgoal_rec"].strip() != "NO RECORD"]
            if snaps:
                snaps[-1]["is_demo"] = False
            tl = [dict(name=t.get("name"), demonstration=bool(t.get("demonstration", False)),
                       segment=rec._seg_name(t.get("segment"))) for t in w.unwrapped.task_list]
            instr = info.get("task_goal") if isinstance(info, dict) else ""
            instr = instr[0] if isinstance(instr, (list, tuple)) and instr else str(instr or "")
            tracker.builder, tracker.n_frames = None, 0
            for s in snaps:
                tracker.on_snapshot(s, tl, instr)
            rec.snaps = []
            tracker._rec, tracker._tl, tracker._instr = rec, tl, instr
            rce._status = str(info.get("status", "ongoing")) if isinstance(info, dict) else "ongoing"
            rce._done = False
            return obs, info

        orig_step = rce.step_one

        def step_one(action):
            out = orig_step(action)
            rec = tracker._rec
            for s in rec.snaps:
                tracker.on_snapshot(s, tracker._tl, tracker._instr)
            rec.snaps = []
            return out

        rce.reset = env_reset
        rce.step_one = step_one
        rce._bg_hooked = True

    svc._ensure_env_orig = svc._ensure_env
    svc._ensure_env = ensure_env
