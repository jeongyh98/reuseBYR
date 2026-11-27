"""Table 5 (SimplerEnv-Bridge, pi0-Bridge): pass@1 / +retrieval / +selector / oracle.

Reads the client logs
  outputs/bridge/selector_<task>_s<seed>.log    (--score_library; MSCORE lines)
  outputs/bridge/retrieval_<task>_s<seed>.log   (server --retrieval; RESULT line, optional)
for tasks carrot / spoon / stack and noise seeds 42 / 43 / 44.

Usage: python scripts/compute_bridge.py [--root outputs/bridge]
"""
import argparse
import os
import re

TASKS = ["carrot", "spoon", "stack"]
SEEDS = [42, 43, 44]
MSCORE = re.compile(r"MSCORE \w+ ep(\d+) ([\d.]+) success=(True|False)")
RESULT = re.compile(r"RESULT task=\w+ SR=(\d+)/(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/bridge")
    root = ap.parse_args().root
    rows = []
    print(f"{'task':8} {'n':>3} {'pass@1':>7} {'+retrieval':>15} {'+selector':>15} {'oracle':>7}")
    for task in TASKS:
        eps = {}
        for seed in SEEDS:
            for line in open(f"{root}/selector_{task}_s{seed}.log"):
                m = MSCORE.match(line)
                if m:
                    eps.setdefault(int(m[1]), {})[seed] = (float(m[2]), m[3] == "True")
        ids = sorted(e for e in eps if len(eps[e]) == len(SEEDS))
        p1 = sum(ok for e in ids for _, ok in eps[e].values()) / (len(SEEDS) * len(ids))
        sel = sum(max(eps[e].values())[1] for e in ids) / len(ids)      # highest s(tau)
        orc = sum(any(ok for _, ok in eps[e].values()) for e in ids) / len(ids)
        retr = None
        files = [f"{root}/retrieval_{task}_s{s}.log" for s in SEEDS]
        if all(os.path.exists(f) for f in files):
            hits = [RESULT.search(open(f).read()) for f in files]
            if all(hits):
                retr = sum(int(h[1]) for h in hits) / sum(int(h[2]) for h in hits)
        rows.append((p1, retr, sel, orc))
        r = f"{100 * retr:5.1f} ({100 * (retr - p1):+5.1f})" if retr is not None else "n/a"
        print(f"{task:8} {len(ids):3d} {100 * p1:7.1f} {r:>15} {100 * sel:7.1f} ({100 * (sel - p1):+5.1f}) {100 * orc:7.1f}")
    P1, SEL, ORC = (sum(r[i] for r in rows) / len(rows) for i in (0, 2, 3))
    R = sum(r[1] for r in rows) / len(rows) if all(r[1] is not None for r in rows) else None
    r = f"{100 * R:5.1f} ({100 * (R - P1):+5.1f})" if R is not None else "n/a"
    print(f"{'overall':8} {'':3} {100 * P1:7.1f} {r:>15} {100 * SEL:7.1f} ({100 * (SEL - P1):+5.1f}) {100 * ORC:7.1f}"
          f"   capture {100 * (SEL - P1) / (ORC - P1):.0f}%")


if __name__ == "__main__":
    main()
