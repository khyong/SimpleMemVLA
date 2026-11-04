"""Belief Graph internal representation (JSON-serializable).

A ``BeliefGraph`` is the belief at one decision time ``t`` (seconds since the episode start, LeRobot
``frame_index / fps``). Every fact carries a ``Provenance``: ``evidence_t`` is the time of the latest observation
the fact rests on and must satisfy ``evidence_t <= t`` (causality, checked by ``BeliefGraph.validate``).

Node / edge types
  ObjectNode    persistent id, category, last position slot, visibility (visible|hidden|unknown), last_seen_t
  RobotNode     gripper (open|closed|unknown), held object id
  Relation      subject --(in|on|under)--> object, ``since_t`` = time the relation started
  Interaction   k-th robot interaction (verb, target, optional destination), time ``t``
  Counter       progress counter, e.g. picked(green_cube) = 2 / 3
  Designation   a target designated earlier (highlight, conditioning video, button) for a role
  Sequence      an ordered list observed earlier (e.g. demo placement order)

Persistent ids are assigned at first observation and never re-derived from the current position: same-looking
objects get ``<category>_A``, ``<category>_B`` ... in the left-to-right order of first observation
(``assign_persistent_ids``).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

VISIBILITY = ("visible", "hidden", "unknown")
GRIPPER = ("open", "closed", "unknown")
RELATIONS = ("in", "on", "under")
SOURCES = ("oracle", "unsup", "manual")

TIME_QUANTUM = 0.5  # seconds; all serialized times are multiples of this


def quantize_time(seconds: float) -> float:
    """Deterministic rounding to TIME_QUANTUM (half-up, no banker's rounding)."""
    if seconds < 0:
        raise ValueError(f"negative elapsed time {seconds}")
    q = math.floor(seconds / TIME_QUANTUM + 0.5 + 1e-9)
    return q * TIME_QUANTUM


@dataclass(frozen=True)
class Provenance:
    evidence_t: float
    source: str = "oracle"
    confidence: float = 1.0

    def __post_init__(self):
        if self.source.split(":")[0] not in SOURCES:
            raise ValueError(f"unknown provenance source {self.source!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} not in [0, 1]")


@dataclass
class ObjectNode:
    id: str
    category: str
    slot: str | None = None          # coarse position slot, e.g. "left", "bin_B", "table"
    visibility: str = "unknown"
    last_seen_t: float | None = None
    prov: Provenance = field(default_factory=lambda: Provenance(0.0))
    first_slot: str | None = None    # slot at first observation (the id was named by it)

    def __post_init__(self):
        if self.visibility not in VISIBILITY:
            raise ValueError(f"visibility {self.visibility!r}")


@dataclass
class RobotNode:
    gripper: str = "unknown"
    holding: str | None = None
    prov: Provenance = field(default_factory=lambda: Provenance(0.0))

    def __post_init__(self):
        if self.gripper not in GRIPPER:
            raise ValueError(f"gripper {self.gripper!r}")


@dataclass
class Relation:
    kind: str
    subj: str
    obj: str
    since_t: float
    prov: Provenance = field(default_factory=lambda: Provenance(0.0))

    def __post_init__(self):
        if self.kind not in RELATIONS:
            raise ValueError(f"relation kind {self.kind!r}")


@dataclass
class Interaction:
    k: int                 # 1-based order among all interactions of the episode
    verb: str              # picked | placed | pressed | swung | touched | inserted | opened | ...
    target: str
    t: float
    dest: str | None = None
    prov: Provenance = field(default_factory=lambda: Provenance(0.0))


@dataclass
class Counter:
    verb: str
    target: str
    value: int
    total: int | None = None
    prov: Provenance = field(default_factory=lambda: Provenance(0.0))


@dataclass
class Designation:
    role: str              # e.g. "pick", "repick", "place", "press"
    target: str
    via: str               # "highlight" | "video" | "instruction" | "button"
    t: float
    prov: Provenance = field(default_factory=lambda: Provenance(0.0))


@dataclass
class Sequence:
    role: str              # e.g. "order"
    items: list[str]
    via: str               # "video" | ...
    t: float
    prov: Provenance = field(default_factory=lambda: Provenance(0.0))


@dataclass
class BeliefGraph:
    t: float
    objects: list[ObjectNode] = field(default_factory=list)
    robot: RobotNode = field(default_factory=RobotNode)
    relations: list[Relation] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    counters: list[Counter] = field(default_factory=list)
    designations: list[Designation] = field(default_factory=list)
    sequences: list[Sequence] = field(default_factory=list)

    # ---------------------------------------------------------------- json
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BeliefGraph":
        def prov(x):
            return Provenance(**x) if x is not None else Provenance(0.0)

        def mk(kls, x):
            x = dict(x)
            x["prov"] = prov(x.get("prov"))
            return kls(**x)

        return cls(
            t=float(d["t"]),
            objects=[mk(ObjectNode, o) for o in d.get("objects", [])],
            robot=mk(RobotNode, d.get("robot", {"gripper": "unknown"})),
            relations=[mk(Relation, r) for r in d.get("relations", [])],
            interactions=[mk(Interaction, r) for r in d.get("interactions", [])],
            counters=[mk(Counter, r) for r in d.get("counters", [])],
            designations=[mk(Designation, r) for r in d.get("designations", [])],
            sequences=[mk(Sequence, r) for r in d.get("sequences", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> "BeliefGraph":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------ checks
    def all_provenances(self) -> Iterable[tuple[str, Provenance]]:
        for o in self.objects:
            yield f"object:{o.id}", o.prov
        yield "robot", self.robot.prov
        for r in self.relations:
            yield f"rel:{r.subj}:{r.kind}:{r.obj}", r.prov
        for r in self.interactions:
            yield f"int:{r.k}", r.prov
        for r in self.counters:
            yield f"count:{r.verb}:{r.target}", r.prov
        for r in self.designations:
            yield f"des:{r.role}", r.prov
        for r in self.sequences:
            yield f"seq:{r.role}", r.prov

    def validate(self) -> None:
        """Raise if any fact rests on evidence after ``t`` (or carries a time after ``t``)."""
        eps = 1e-6
        for name, p in self.all_provenances():
            if p.evidence_t > self.t + eps:
                raise ValueError(f"{name}: evidence_t {p.evidence_t} > t {self.t}")
        for r in self.interactions:
            if r.t > self.t + eps:
                raise ValueError(f"interaction #{r.k} at {r.t} > t {self.t}")
        for r in self.relations:
            if r.since_t > self.t + eps:
                raise ValueError(f"relation since {r.since_t} > t {self.t}")
        for r in self.designations + self.sequences:  # type: ignore[operator]
            if r.t > self.t + eps:
                raise ValueError(f"{type(r).__name__} at {r.t} > t {self.t}")
        ids = [o.id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate object ids {ids}")
        ks = [r.k for r in self.interactions]
        if ks != sorted(ks) or len(ks) != len(set(ks)):
            raise ValueError(f"interaction order not strictly increasing: {ks}")


def assign_persistent_ids(first_seen: list[tuple[str, float, float]]) -> list[str]:
    """Persistent ids for objects given (category, first_seen_t, first_seen_x) in any order.

    Objects whose category is unique keep the category as id. Same-category objects get suffixes A, B, ... in
    the order (first_seen_t, left-to-right x) of their FIRST observation; later motion never renames them.
    Returns ids aligned with the input order.
    """
    by_cat: dict[str, list[int]] = {}
    for i, (cat, _, _) in enumerate(first_seen):
        by_cat.setdefault(cat, []).append(i)
    ids = [""] * len(first_seen)
    for cat, idx in by_cat.items():
        if len(idx) == 1:
            ids[idx[0]] = cat
            continue
        order = sorted(idx, key=lambda i: (round(first_seen[i][1], 6), first_seen[i][2], i))
        for rank, i in enumerate(order):
            ids[i] = f"{cat}_{chr(ord('A') + rank)}"
    return ids
