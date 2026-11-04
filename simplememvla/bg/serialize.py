"""Deterministic BG text serialization, filtering, truncation and parsing.

Text format (one line, clauses joined by " ; ", fixed clause order):

    Belief: <clause> ; <clause> ; ...
    Belief: none                                   (nothing to remember)

Format version v2 (token budget: Qwen3.5 splits digits, so times are written as compactly as possible).
Clause grammar (``<id>`` = persistent object id, ``<x>`` = elapsed seconds, multiple of 0.5, written "31" or
"12.5" -- no trailing ".0"):

    count <verb>(<id>)=<n>/<N>      counter with known total   e.g. count picked(green_cube)=2/3
    count <verb>(<id>)=<n>          counter, total unknown
    target(<role>)=<id> via <how>                               e.g. target(pick)=red_cube via video
    order(<role>)=<id>><id>>... via <how>                       e.g. order(demo)=target_A>button>target_B via video
    #<k> <verb>(<id>) <x>s ago                                  e.g. #1 picked(green_cube) 31s ago
    #<k> <verb>(<id>,<id>) <x>s ago                             interaction with destination
    <id> <in|on|under> <id> since <x>s ago                      relation, e.g. red_cube under cover_B since 20s ago
    <id> hidden at <slot> since <x>s ago                        hidden object without a relation
    <id> now at <slot>                                          current slot of a visible object whose identity
                                                                cannot be read from the frame (same-looking
                                                                objects, e.g. containers after a swap) and that
                                                                another kept clause refers to
  full filter only:
    <id> at <slot> visible
    holding(<id>) | gripper <open|closed>

Clause order: counters, targets, orders, interactions (by k), relations (by subject), hidden objects, current
slots of ambiguous objects, then the full-filter clauses. Within a group the sort key is fixed, so equal beliefs always give byte-identical text.

Mnemonic filter (default): keep only facts that the current frame cannot decide — counters, designations,
orders, past interactions, and the location/relations of objects that are hidden now. Visible object positions,
relations of visible objects and the gripper state are dropped. Two debounce rules keep transient states out:
a hidden object without a relation is kept only after HIDDEN_MIN_S of continuous hiding (brief occlusion by the
arm is not memory), and "now at" is kept only when the object has left the slot it was named by (first_slot).

Truncation: the last ``max_interactions`` interactions (by k) are kept, everything else of the filtered set is
kept (counters always survive).
"""
from __future__ import annotations

import re
from typing import Iterable

from simplememvla.bg import BG_PREFIX
from simplememvla.bg.schema import BeliefGraph, quantize_time

DEFAULT_MAX_INTERACTIONS = 4
FORMAT_VERSION = "v3"
HIDDEN_MIN_S = 2.0
_ID = r"[a-z][A-Za-z0-9_]*"
_X = r"\d+(?:\.5)?"


def _ago(now: float, then: float) -> str:
    q = quantize_time(max(0.0, now - then))
    return f"{int(q)}" if float(q).is_integer() else f"{q:.1f}"


# ----------------------------------------------------------------------------- canonical facts
def facts(bg: BeliefGraph, bg_filter: str = "mnemonic",
          max_interactions: int = DEFAULT_MAX_INTERACTIONS) -> list[tuple]:
    """Canonical, ordered fact tuples that the text of ``bg`` must encode (after filter + truncation)."""
    if bg_filter not in ("mnemonic", "full"):
        raise ValueError(f"bg_filter {bg_filter!r}")
    now = bg.t
    hidden = {o.id for o in bg.objects if o.visibility == "hidden"}
    out: list[tuple] = []
    for c in sorted(bg.counters, key=lambda c: (c.verb, c.target)):
        out.append(("count", c.verb, c.target, int(c.value), None if c.total is None else int(c.total)))
    for d in sorted(bg.designations, key=lambda d: (d.role, d.target)):
        out.append(("target", d.role, d.target, d.via))
    for s in sorted(bg.sequences, key=lambda s: s.role):
        out.append(("order", s.role, tuple(s.items), s.via))
    inter = sorted(bg.interactions, key=lambda r: r.k)
    if max_interactions is not None and max_interactions >= 0:
        inter = inter[max(0, len(inter) - max_interactions):] if max_interactions else []
    for r in inter:
        out.append(("int", int(r.k), r.verb, r.target, r.dest, _ago(now, r.t)))
    rel_subjects = set()
    for r in sorted(bg.relations, key=lambda r: (r.subj, r.kind, r.obj)):
        if bg_filter == "mnemonic" and r.subj not in hidden:
            continue
        rel_subjects.add(r.subj)
        out.append(("rel", r.subj, r.kind, r.obj, _ago(now, r.since_t)))
    for o in sorted(bg.objects, key=lambda o: o.id):
        if o.visibility == "hidden" and o.id not in rel_subjects and o.slot:
            since = o.last_seen_t if o.last_seen_t is not None else o.prov.evidence_t
            if bg_filter == "full" or now - since >= HIDDEN_MIN_S:
                out.append(("hidden", o.id, o.slot, _ago(now, since)))
    # identity of same-looking visible objects is memory, not perception: keep their slot when referenced
    cat_count: dict[str, int] = {}
    for o in bg.objects:
        cat_count[o.category] = cat_count.get(o.category, 0) + 1
    referenced = set()
    for f in out:
        for x in f[1:]:
            if isinstance(x, str):
                referenced.add(x)
            elif isinstance(x, tuple):
                referenced.update(x)
    for o in sorted(bg.objects, key=lambda o: o.id):
        if (o.visibility == "visible" and o.slot and cat_count.get(o.category, 0) >= 2
                and o.id in referenced and (o.first_slot is None or o.slot != o.first_slot)):
            out.append(("now", o.id, o.slot))
    if bg_filter == "full":
        for o in sorted(bg.objects, key=lambda o: o.id):
            if o.visibility == "visible" and o.slot:
                out.append(("visible", o.id, o.slot))
        if bg.robot.holding:
            out.append(("holding", bg.robot.holding))
        elif bg.robot.gripper != "unknown":
            out.append(("gripper", bg.robot.gripper))
    return out


# ----------------------------------------------------------------------------- text
def _clause(f: tuple) -> str:
    kind = f[0]
    if kind == "count":
        _, verb, tgt, n, total = f
        return f"count {verb}({tgt})={n}" + (f"/{total}" if total is not None else "")
    if kind == "target":
        _, role, tgt, via = f
        return f"target({role})={tgt} via {via}"
    if kind == "order":
        _, role, items, via = f
        return f"order({role})={'>'.join(items)} via {via}"
    if kind == "int":
        _, k, verb, tgt, dest, ago = f
        args = tgt if dest is None else f"{tgt},{dest}"
        return f"#{k} {verb}({args}) {ago}s ago"
    if kind == "rel":
        _, subj, rel, obj, ago = f
        return f"{subj} {rel} {obj} since {ago}s ago"
    if kind == "hidden":
        _, oid, slot, ago = f
        return f"{oid} hidden at {slot} since {ago}s ago"
    if kind == "now":
        return f"{f[1]} now at {f[2]}"
    if kind == "visible":
        return f"{f[1]} at {f[2]} visible"
    if kind == "holding":
        return f"holding({f[1]})"
    if kind == "gripper":
        return f"gripper {f[1]}"
    raise ValueError(f"unknown fact kind {kind!r}")


def facts_to_text(fs: Iterable[tuple]) -> str:
    clauses = [_clause(f) for f in fs]
    return f"{BG_PREFIX} " + (" ; ".join(clauses) if clauses else "none")


def serialize(bg: BeliefGraph, bg_filter: str = "mnemonic",
              max_interactions: int = DEFAULT_MAX_INTERACTIONS) -> str:
    return facts_to_text(facts(bg, bg_filter=bg_filter, max_interactions=max_interactions))


_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("count", re.compile(rf"^count ({_ID})\(({_ID})\)=(\d+)(?:/(\d+))?$")),
    ("target", re.compile(rf"^target\(({_ID})\)=({_ID}) via ({_ID})$")),
    ("order", re.compile(rf"^order\(({_ID})\)=({_ID}(?:>{_ID})*) via ({_ID})$")),
    ("int", re.compile(rf"^#(\d+) ({_ID})\(({_ID})(?:,({_ID}))?\) ({_X})s ago$")),
    ("rel", re.compile(rf"^({_ID}) (in|on|under) ({_ID}) since ({_X})s ago$")),
    ("hidden", re.compile(rf"^({_ID}) hidden at ({_ID}) since ({_X})s ago$")),
    ("now", re.compile(rf"^({_ID}) now at ({_ID})$")),
    ("visible", re.compile(rf"^({_ID}) at ({_ID}) visible$")),
    ("holding", re.compile(rf"^holding\(({_ID})\)$")),
    ("gripper", re.compile(r"^gripper (open|closed)$")),
]


class BGParseError(ValueError):
    pass


def parse(text: str) -> list[tuple]:
    """Inverse of ``facts_to_text``. Raises ``BGParseError`` on any clause that does not match the grammar."""
    text = text.strip()
    if not text.startswith(BG_PREFIX):
        raise BGParseError(f"missing {BG_PREFIX!r} prefix: {text[:40]!r}")
    body = text[len(BG_PREFIX):].strip()
    if body == "none":
        return []
    out: list[tuple] = []
    for clause in body.split(" ; "):
        clause = clause.strip()
        for kind, pat in _PATTERNS:
            m = pat.match(clause)
            if not m:
                continue
            g = m.groups()
            if kind == "count":
                out.append(("count", g[0], g[1], int(g[2]), None if g[3] is None else int(g[3])))
            elif kind == "target":
                out.append(("target", g[0], g[1], g[2]))
            elif kind == "order":
                out.append(("order", g[0], tuple(g[1].split(">")), g[2]))
            elif kind == "int":
                out.append(("int", int(g[0]), g[1], g[2], g[3], g[4]))
            elif kind == "rel":
                out.append(("rel", g[0], g[1], g[2], g[3]))
            elif kind == "hidden":
                out.append(("hidden", g[0], g[1], g[2]))
            elif kind == "now":
                out.append(("now", g[0], g[1]))
            elif kind == "visible":
                out.append(("visible", g[0], g[1]))
            elif kind == "holding":
                out.append(("holding", g[0]))
            elif kind == "gripper":
                out.append(("gripper", g[0]))
            break
        else:
            raise BGParseError(f"unparseable clause {clause!r}")
    return out


def split_answer(answer: str) -> tuple[str, str | None]:
    """Split a generated assistant answer into (sub-task, BG text or None)."""
    idx = answer.find(BG_PREFIX)
    if idx < 0:
        return answer.strip(), None
    return answer[:idx].strip(), answer[idx:].strip()


def try_parse_answer(answer: str) -> tuple[str, list[tuple] | None, str | None]:
    """(sub-task, facts or None, error or None) for a generated answer; never raises."""
    sub, bg = split_answer(answer)
    if bg is None:
        return sub, None, "no BG span"
    try:
        return sub, parse(bg), None
    except BGParseError as e:
        return sub, None, str(e)
