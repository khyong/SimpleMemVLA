"""UnsupBG vs BG-Oracle metrics (report 03).

Facts are compared without their elapsed times (timing is reported separately) and without interaction order
numbers (an extra or missing early event would otherwise shift every #k):
    counter      (verb, target, value, total)
    designation  (role, target)
    order        (role, items)
    interaction  (verb, target, dest)            multiset
    relation     (subj, kind, obj)               hidden subjects only (mnemonic)
    hidden       (id)                            hidden object without relation
    slot         (id, slot)                      "now at" of ambiguous same-looking objects
Per fact type: precision / recall over frames sampled every ``stride`` frames of execution and demo.
Re-observation agreement: at every oracle frame where a hidden cube becomes visible again, does the unsup belief of
the previous frame put it under / in the same object as the oracle?
Goal consistency: the final designation (Reference) / final counter (Counting) / last "under" relation of the
instruction's cube (Permanence) / demo order (Imitation) equals the oracle's.

ID-aligned variant (``align=True``, reported next to the exact one, never used to build labels): object IDs are
arbitrary names, so one spurious early track shifts every later letter (container_B for container_A) and a naming
difference (blue_cube_C for cube_C) zeroes an otherwise identical structure. Per episode, ONE fixed one-to-one map
from unsup object IDs to oracle object IDs is fit (Hungarian on co-occurrence of the IDs at the same position of
facts with the same ID-free skeleton, over the sampled frames). Only category-compatible pairs may be matched: same
base category after removing a colour prefix, and no conflicting colours (red_cube never maps to green_cube).
Unmatched unsup IDs are renamed so they can never coincide with an oracle ID. The map is fit with the oracle, so the
aligned scores are an optimistic bound on structure agreement, not a label-quality number by themselves.
"""
from __future__ import annotations

from collections import Counter as MC

import numpy as np

from simplememvla.bg.schema import BeliefGraph
from simplememvla.bg.serialize import facts

TYPES = ("counter", "designation", "order", "interaction", "relation", "hidden", "slot")


def canon(bg: BeliefGraph) -> dict[str, MC]:
    out = {k: MC() for k in TYPES}
    for f in facts(bg, "mnemonic", 10_000):
        k = f[0]
        if k == "count":
            out["counter"][(f[1], f[2], f[3], f[4])] += 1
        elif k == "target":
            out["designation"][(f[1], f[2])] += 1
        elif k == "order":
            out["order"][(f[1], tuple(f[2]))] += 1
        elif k == "int":
            out["interaction"][(f[2], f[3], f[4])] += 1
        elif k == "rel":
            out["relation"][(f[1], f[2], f[3])] += 1
        elif k == "hidden":
            out["hidden"][(f[1],)] += 1
        elif k == "now":
            out["slot"][(f[1], f[2])] += 1
    return out


COLOURS = ("red", "green", "blue", "yellow", "white", "black", "orange", "purple", "pink", "brown", "gray", "grey")


def _cat(category: str) -> tuple[str | None, str]:
    """(colour or None, base category) of an object category such as 'blue_cube' or 'container'."""
    head, _, rest = category.partition("_")
    if head in COLOURS and rest:
        return head, rest
    return None, category


def _compatible(cu: str, co: str) -> bool:
    (ku, bu), (ko, bo) = _cat(cu), _cat(co)
    return bu == bo and (ku is None or ko is None or ku == ko)


def _map_fact(fact, m: dict):
    if isinstance(fact, tuple):
        return tuple(_map_fact(x, m) for x in fact)
    return m.get(fact, fact) if isinstance(fact, str) else fact


def _flat(fact) -> list:
    out = []
    for x in fact:
        out.extend(_flat(x) if isinstance(x, tuple) else [x])
    return out


def _skeleton(fact, ids: set):
    if isinstance(fact, tuple):
        return tuple(_skeleton(x, ids) for x in fact)
    return "*" if isinstance(fact, str) and fact in ids else fact


def fit_id_map(oracle_bgs: list[BeliefGraph], unsup_bgs: list[BeliefGraph]) -> dict[str, str]:
    """One fixed one-to-one unsup-ID -> oracle-ID map per episode (see module docstring)."""
    from scipy.optimize import linear_sum_assignment

    cat_o, cat_u = {}, {}
    for g in oracle_bgs:
        for ob in g.objects:
            cat_o.setdefault(ob.id, ob.category)
    for g in unsup_bgs:
        for ob in g.objects:
            cat_u.setdefault(ob.id, ob.category)
    U, O = sorted(cat_u), sorted(cat_o)
    if not U or not O:
        return {}
    iu, io = {x: i for i, x in enumerate(U)}, {x: i for i, x in enumerate(O)}
    S = np.zeros((len(U), len(O)))
    for go, gu in zip(oracle_bgs, unsup_bgs):
        co, cu = canon(go), canon(gu)
        for k in TYPES:
            by_sk: dict = {}
            for fo, n in co[k].items():
                by_sk.setdefault(_skeleton(fo, set(cat_o)), ([], []))[0].append((fo, n))
            for fu, n in cu[k].items():
                sk = _skeleton(fu, set(cat_u))
                if sk in by_sk:
                    by_sk[sk][1].append((fu, n))
            for sk, (los, lus) in by_sk.items():
                if not los or not lus:
                    continue
                w = 1.0 / (len(los) * len(lus))  # spread one unit of evidence per skeleton over its candidate pairs
                pos = [i for i, x in enumerate(_flat(sk)) if x == "*"]
                for fo, no in los:
                    ao = _flat(fo)
                    for fu, nu in lus:
                        au = _flat(fu)
                        for i in pos:
                            if au[i] in iu and ao[i] in io:
                                S[iu[au[i]], io[ao[i]]] += w * min(no, nu)
    for a, x in enumerate(U):
        for b, y in enumerate(O):
            if not _compatible(cat_u[x], cat_o[y]):
                S[a, b] = -1e9
    r, c = linear_sum_assignment(-S)
    m = {U[a]: O[b] for a, b in zip(r, c) if S[a, b] > -1e8}
    for x in U:  # unmatched unsup IDs must never coincide with an oracle ID
        m.setdefault(x, "u:" + x)
    return m


def _map_bg_canon(c: dict[str, MC], m: dict) -> dict[str, MC]:
    return {k: MC({_map_fact(f, m): n for f, n in v.items()}) for k, v in c.items()}


def compare_episode(oracle_json: list[str], unsup_json: list[str], stride: int = 10, align: bool = False) -> dict:
    """Exact-ID comparison; with ``align=True`` the unsup IDs are first renamed by ``fit_id_map`` and the result
    additionally carries ``id_map``."""
    agg = {k: {"tp": 0, "pred": 0, "gold": 0} for k in TYPES}
    n = min(len(oracle_json), len(unsup_json))
    m: dict = {}
    if align:
        fr = range(0, n, stride)
        m = fit_id_map([BeliefGraph.from_json(oracle_json[f]) for f in fr],
                       [BeliefGraph.from_json(unsup_json[f]) for f in fr])
    for f in range(0, n, stride):
        o = canon(BeliefGraph.from_json(oracle_json[f]))
        u = _map_bg_canon(canon(BeliefGraph.from_json(unsup_json[f])), m)
        for k in TYPES:
            agg[k]["tp"] += sum((o[k] & u[k]).values())
            agg[k]["pred"] += sum(u[k].values())
            agg[k]["gold"] += sum(o[k].values())
    # re-observation agreement
    reob, reob_ok = 0, 0
    prev = None
    for f in range(n):
        og = BeliefGraph.from_json(oracle_json[f])
        vis = {o.id for o in og.objects if o.visibility == "visible"}
        if prev is not None:
            pg_o, pg_u = prev
            hid_before = {r.subj: r.obj for r in pg_o.relations}
            for oid, obj in hid_before.items():
                if oid in vis:
                    reob += 1
                    u_rel = {m.get(r.subj, r.subj): m.get(r.obj, r.obj) for r in pg_u.relations}
                    reob_ok += int(u_rel.get(oid) == obj)
        prev = (og, BeliefGraph.from_json(unsup_json[f]))
    last_o = canon(BeliefGraph.from_json(oracle_json[n - 1]))
    last_u = _map_bg_canon(canon(BeliefGraph.from_json(unsup_json[n - 1])), m)
    goal = None
    for k in ("designation", "counter", "order"):
        if last_o[k]:
            goal = last_o[k] == last_u[k]
            break
    out = {"types": agg, "reobs": reob, "reobs_ok": reob_ok, "goal_consistent": goal, "frames": n}
    if align:
        out["id_map"] = {k: v for k, v in m.items() if k != v}
    return out


def summarize(results: list[dict]) -> dict:
    tot = {k: {"tp": 0, "pred": 0, "gold": 0} for k in TYPES}
    reob = reob_ok = 0
    goal = []
    for r in results:
        for k in TYPES:
            for f in ("tp", "pred", "gold"):
                tot[k][f] += r["types"][k][f]
        reob += r["reobs"]
        reob_ok += r["reobs_ok"]
        if r["goal_consistent"] is not None:
            goal.append(r["goal_consistent"])
    out = {}
    for k, v in tot.items():
        out[k] = {"precision": v["tp"] / v["pred"] if v["pred"] else None,
                  "recall": v["tp"] / v["gold"] if v["gold"] else None, "n_gold": v["gold"], "n_pred": v["pred"]}
    out["reobservation_agreement"] = reob_ok / reob if reob else None
    out["n_reobservations"] = reob
    out["goal_consistency"] = sum(goal) / len(goal) if goal else None
    return out
