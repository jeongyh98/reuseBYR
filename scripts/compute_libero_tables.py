"""LIBERO tables from the outputs of libero_eval/run_libero.sh.

Reads outputs/libero/<backbone>_<variant>_<suite>_s<seed>/{eval_info.json, manifold_scores.tsv}.

  pass@1     mean success of the `selector` runs (frozen policy; scores are only logged)
  +Selector  per episode, the success of the seed with the highest score s(tau)
  oracle     per episode, success in any seed (pass@N); headroom H_N = oracle - pass@1
  +Retrieval mean success of the `retrieval` runs
  Retr+Sel   selector applied to the `retrieval_selector` runs

Scores and successes are always paired from the same run.

Usage: python scripts/compute_libero_tables.py [--root outputs/libero]
"""
import argparse
import json
import os

import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
BACKBONES = ["pi0", "smolvla", "pi05"]


def successes(root, bb, variant, suite, seed):
    p = f"{root}/{bb}_{variant}_{suite}_s{seed}/eval_info.json"
    if not os.path.exists(p):
        return None
    tasks = sorted(json.load(open(p))["per_task"], key=lambda r: r["task_id"])
    return np.array([s for r in tasks for s in r["metrics"]["successes"]], bool)


def scores(root, bb, variant, suite, seed):
    p = f"{root}/{bb}_{variant}_{suite}_s{seed}/manifold_scores.tsv"
    if not os.path.exists(p):
        return None
    return {int(q[0]): float(q[1]) for q in (ln.split("\t") for ln in open(p)) if len(q) >= 2}


def selector(root, bb, variant, suite, seeds):
    """(pass@1, selector, oracle) in % for one cell, or None if a run is missing."""
    S = [successes(root, bb, variant, suite, s) for s in seeds]
    M = [scores(root, bb, variant, suite, s) for s in seeds]
    if any(x is None for x in S + M):
        return None
    n = min(len(s) for s in S)
    succ = np.stack([s[:n] for s in S])
    sc = np.full(succ.shape, -np.inf)
    for i, m in enumerate(M):
        for e, v in m.items():
            if e < n:
                sc[i, e] = v
    return (succ.mean() * 100, succ[sc.argmax(0), np.arange(n)].mean() * 100, succ.any(0).mean() * 100)


def mean_success(root, bb, variant, suite, seeds):
    S = [successes(root, bb, variant, suite, s) for s in seeds]
    return None if any(x is None for x in S) else np.mean([s.mean() for s in S]) * 100


def fmt(x, sign=False):
    return "  n/a" if x is None or np.isnan(x) else (f"{x:+5.1f}" if sign else f"{x:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/libero")
    root = ap.parse_args().root
    seeds3 = [1000, 2000, 3000]

    print("Table 1 / Table 2 - LIBERO (base suites, N=3 seeds); gains over pass@1")
    print(f"{'backbone':8} {'suite':15} {'pass@1':>6} {'+Retr':>6} {'+Sel':>6} {'H_3':>6}")
    hx, hy = [], []
    for bb in BACKBONES:
        rows = []
        for su in SUITES:
            c = selector(root, bb, "selector", f"{su}_base", seeds3)
            r = mean_success(root, bb, "retrieval", f"{su}_base", seeds3)
            if c is None:
                continue
            p1, sel, orc = c
            rows.append((p1, np.nan if r is None else r - p1, sel - p1, orc - p1))
            hx.append(orc - p1); hy.append(sel - p1)
            print(f"{bb:8} {su:15} {fmt(rows[-1][0])} {fmt(rows[-1][1], 1)} {fmt(rows[-1][2], 1)} {fmt(rows[-1][3])}")
        if rows:
            m = np.array(rows)
            print(f"{bb:8} {'mean':15} {fmt(m[:, 0].mean())} {fmt(np.nanmean(m[:, 1]) if not np.isnan(m[:, 1]).all() else np.nan, 1)}"
                  f" {fmt(m[:, 2].mean(), 1)} {fmt(m[:, 3].mean())}")
    if len(hx) >= 2:
        print(f"\nFig. 1 - headroom H_3 vs selector gain over {len(hx)} cells: r = {np.corrcoef(hx, hy)[0, 1]:.3f}")

    print("\nTable 4 - selection vs. resampling (4-suite mean success, %)")
    print(f"{'backbone':8} {'pass@1':>6} {'rand-of-3':>9} {'sel N=2':>7} {'sel N=3':>7} {'oracle':>6}")
    rng = np.random.default_rng(0)
    for bb in BACKBONES:
        rows = []
        for su in SUITES:
            c3 = selector(root, bb, "selector", f"{su}_base", seeds3)
            c2 = selector(root, bb, "selector", f"{su}_base", seeds3[:2])
            if c3 is None or c2 is None:
                continue
            S = [successes(root, bb, "selector", f"{su}_base", s) for s in seeds3]
            n = min(len(s) for s in S)
            succ = np.stack([s[:n] for s in S])
            rand = succ[rng.integers(0, len(seeds3), n), np.arange(n)].mean() * 100   # keep one at random
            rows.append((c3[0], rand, c2[1], c3[1], c3[2]))
        if rows:
            m = np.array(rows).mean(0)
            print(f"{bb:8} {m[0]:6.1f} {m[1]:9.1f} {m[2]:7.1f} {m[3]:7.1f} {m[4]:6.1f}")

    print("\nTable 3 - retrieval x selector composition (pi0, base suites, N=3); gains over pass@1")
    print(f"{'suite':15} {'pass@1':>6} {'+Retr':>6} {'+Sel':>6} {'Retr+Sel':>8}")
    acc = []
    for su in SUITES:
        c = selector(root, "pi0", "selector", f"{su}_base", seeds3)
        rs = selector(root, "pi0", "retrieval_selector", f"{su}_base", seeds3)
        r = mean_success(root, "pi0", "retrieval", f"{su}_base", seeds3)
        if c is None:
            continue
        p1, sel, _ = c
        row = (p1, np.nan if r is None else r - p1, sel - p1, np.nan if rs is None else rs[1] - p1)
        acc.append(row)
        print(f"{su:15} {fmt(row[0])} {fmt(row[1], 1)} {fmt(row[2], 1)} {fmt(row[3], 1):>8}")
    if acc:
        m = np.array(acc)
        print(f"{'mean':15} {fmt(m[:, 0].mean())} {fmt(m[:, 1].mean(), 1)} {fmt(m[:, 2].mean(), 1)} {fmt(m[:, 3].mean(), 1):>8}")

    print("\nTable 7 - LIBERO-Occ (N=2 seeds); gains over occluded pass@1")
    seeds2 = [1000, 2000]
    for bb in ["pi0", "pi05"]:
        acc = []
        for su in SUITES:
            c = selector(root, bb, "selector", f"{su}_occluded", seeds2)
            r = mean_success(root, bb, "retrieval", f"{su}_occluded", seeds2)
            if c is None:
                continue
            p1, sel, _ = c
            acc.append((p1, np.nan if r is None else r - p1, sel - p1))
            print(f"{bb:8} {su:15} {fmt(acc[-1][0])} {fmt(acc[-1][1], 1)} {fmt(acc[-1][2], 1)}")
        if acc:
            m = np.array(acc)
            print(f"{bb:8} {'mean':15} {fmt(m[:, 0].mean())} {fmt(m[:, 1].mean(), 1)} {fmt(m[:, 2].mean(), 1)}")


if __name__ == "__main__":
    main()
