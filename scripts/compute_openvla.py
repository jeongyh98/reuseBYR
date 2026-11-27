"""Table 6 (OpenVLA-7B, SimplerEnv coke-can, 25 initial states): sample@1 / episode-level
best-of-3 (selector) / oracle pass@3.

Reads, for sampling seeds 101 / 102 / 103,
  outputs/openvla/sample_s<seed>.log    stdout of simpler_openvla/run_openvla.sh
  outputs/openvla/scores_s<seed>.tsv    MANIFOLD_FILE of the same run
The final 'success' of each episode is the last info dict printed before the next
"pick coke can" instruction line.

Usage: python scripts/compute_openvla.py [--root outputs/openvla]
"""
import argparse
import re

SEEDS = [101, 102, 103]
SUCC = re.compile(r"'lifted_object_significantly': \w+, 'success': (True|False)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/openvla")
    root = ap.parse_args().root
    runs = {}
    for seed in SEEDS:
        score = {int(e): float(s) for e, s in (l.split()[:2] for l in open(f"{root}/scores_s{seed}.tsv") if l.strip())}
        chunks = open(f"{root}/sample_s{seed}.log").read().split("pick coke can\n")[1:]
        succ = {e: SUCC.findall(c)[-1] == "True" for e, c in enumerate(chunks)}
        runs[seed] = {e: (score[e], succ[e]) for e in score if e in succ}
    eps = sorted(set.intersection(*(set(r) for r in runs.values())))
    p1 = sum(runs[s][e][1] for s in SEEDS for e in eps) / (len(SEEDS) * len(eps))
    sel = sum(runs[max(SEEDS, key=lambda s: runs[s][e][0])][e][1] for e in eps) / len(eps)
    orc = sum(any(runs[s][e][1] for s in SEEDS) for e in eps) / len(eps)
    for s in SEEDS:
        print(f"seed {s}: success {sum(runs[s][e][1] for e in eps) / len(eps):.2f}")
    print(f"n={len(eps)}  sample@1 {p1:.2f} | episode bo3 {sel:.2f} | oracle pass@3 {orc:.2f}")


if __name__ == "__main__":
    main()
