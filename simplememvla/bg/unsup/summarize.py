"""Summarize UnsupBG-vs-Oracle metrics (metrics.jsonl of unsup/pilot.py) per suite and fact type (report 03).

    python -m simplememvla.bg.unsup.summarize <out>/metrics.jsonl [--md table.md]
"""
import argparse
import json

from simplememvla.bg.unsup.metrics import TYPES, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--md", default=None)
    ap.add_argument("--aligned", action="store_true", help="use the ID-aligned scores of unsup/rescore.py")
    a = ap.parse_args()
    rs = {}
    for line in open(a.metrics):
        r = json.loads(line)
        if a.aligned:
            r = {**r, **r["aligned"]} if "aligned" in r else {}
        if "types" in r:
            rs[r["ep"]] = r  # last record per episode wins
    by = {}
    for r in rs.values():
        by.setdefault(r["suite"], []).append(r)
    lines = ["| suite | n ep | " + " | ".join(f"{t} P/R (gold)" for t in TYPES) + " | re-obs agree | goal consist. |",
             "|---" * (len(TYPES) + 4) + "|"]
    out = {}
    for s in sorted(by):
        sm = summarize(by[s])
        out[s] = sm
        cells = []
        for t in TYPES:
            v = sm[t]
            p = "-" if v["precision"] is None else f"{v['precision']:.2f}"
            rc = "-" if v["recall"] is None else f"{v['recall']:.2f}"
            cells.append(f"{p}/{rc} ({v['n_gold']})")
        ro = "-" if sm["reobservation_agreement"] is None else f"{sm['reobservation_agreement']:.2f} ({sm['n_reobservations']})"
        gc = "-" if sm["goal_consistency"] is None else f"{sm['goal_consistency']:.2f}"
        lines.append(f"| {s} | {len(by[s])} | " + " | ".join(cells) + f" | {ro} | {gc} |")
    txt = "\n".join(lines)
    print(txt)
    per_ep = ["", "| ep | task | tracks | sec | " + " | ".join(f"{t} tp/pred/gold" for t in TYPES) + " |",
              "|---" * (len(TYPES) + 4) + "|"]
    for r in sorted(rs.values(), key=lambda r: (r["suite"], r["task"], r["ep"])):
        per_ep.append(f"| {r['ep']} | {r['task']} | {r.get('n_tracks')} | {r.get('seconds')} | " + " | ".join(
            f"{r['types'][t]['tp']}/{r['types'][t]['pred']}/{r['types'][t]['gold']}" for t in TYPES) + " |")
    print("\n".join(per_ep))
    if a.md:
        open(a.md, "w").write(txt + "\n" + "\n".join(per_ep) + "\n")
        json.dump(out, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
