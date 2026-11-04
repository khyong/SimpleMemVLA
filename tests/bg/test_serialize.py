"""[CPU] Test 2: serialization round trip and determinism on random Belief Graphs.

    python -m tests.bg.test_serialize [--n 5000]
"""
from __future__ import annotations

import argparse
import json
import random

from simplememvla.bg.schema import (BeliefGraph, Counter, Designation, Interaction, ObjectNode, Provenance,
                                    Relation, RobotNode, Sequence)
from simplememvla.bg.serialize import facts, parse, serialize

IDS = ["red_cube", "green_cube", "blue_cube", "container_A", "container_B", "container_C", "button", "target_A",
       "target_B", "peg", "bin"]
VERBS = ["picked", "placed", "pressed", "swung", "touched", "inserted"]


def random_bg(rng: random.Random) -> BeliefGraph:
    t = rng.choice([0.0, 0.5, 3.0, 17.25, 41.0, 63.3])
    def tt():
        return round(rng.uniform(0, t), 2)
    objs = []
    for oid in rng.sample(IDS, rng.randint(0, 6)):
        ev = tt()
        objs.append(ObjectNode(oid, oid.split("_")[-1], rng.choice(["left", "middle", "right", "table", None]),
                               rng.choice(["visible", "hidden", "unknown"]), ev, Provenance(ev)))
    ids = [o.id for o in objs] or ["red_cube"]
    rels = []
    for subj in rng.sample(ids, min(len(ids), rng.randint(0, 2))):
        rels.append(Relation(rng.choice(["in", "on", "under"]), subj, rng.choice(IDS), tt()))
    times = sorted(tt() for _ in range(rng.randint(0, 7)))
    inter = [Interaction(k + 1, rng.choice(VERBS), rng.choice(IDS), x, rng.choice([None, None, "bin", "target_A"]))
             for k, x in enumerate(times)]
    cnt = [Counter(v, rng.choice(IDS), rng.randint(0, 5), rng.choice([None, 3, 5]))
           for v in rng.sample(VERBS, rng.randint(0, 2))]
    des = [Designation(r, rng.choice(IDS), rng.choice(["video", "highlight"]), tt())
           for r in rng.sample(["pick", "place", "press"], rng.randint(0, 2))]
    seq = [Sequence("order", rng.sample(IDS, rng.randint(1, 4)), "video", tt()) for _ in range(rng.randint(0, 1))]
    robot = RobotNode(rng.choice(["open", "closed", "unknown"]), rng.choice([None, "red_cube"]))
    return BeliefGraph(t, objs, robot, rels, inter, cnt, des, seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = random.Random(0)
    fails = []
    for i in range(a.n):
        bg = random_bg(rng)
        bg.validate()
        for flt in ("mnemonic", "full"):
            for k in (0, 2, 4, 100):
                s1 = serialize(bg, flt, k)
                s2 = serialize(BeliefGraph.from_json(bg.to_json()), flt, k)  # determinism through JSON
                if s1 != s2:
                    fails.append((i, flt, k, "nondeterministic"))
                if parse(s1) != facts(bg, flt, k):
                    fails.append((i, flt, k, "roundtrip", s1))
    # truncation rule: exactly the last K interactions (by k) survive, for every K and history length
    for n_int in range(0, 9):
        for K in range(0, 7):
            bg = BeliefGraph(t=100.0, interactions=[Interaction(k + 1, "picked", "red_cube", float(k)) for k in range(n_int)])
            ks = [f[1] for f in facts(bg, "mnemonic", K) if f[0] == "int"]
            if ks != list(range(max(1, n_int - K + 1), n_int + 1)):
                fails.append(("truncation", n_int, K, ks))
    rep = {"graphs": a.n, "checks": a.n * 8, "failures": len(fails), "examples": fails[:5], "pass": not fails}
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
