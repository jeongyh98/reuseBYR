"""Train the action MLP (image embedding -> z-normalized 50x7 action chunk) on the agentview
library. Its prediction is the action key of the retrieval score.

Usage:
  python libero_eval/train_action_mlp.py --library data/libero_library_agentview.npz \
      --out data/action_mlp_libero.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mlp import MLP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    lib = np.load(args.library)
    img, act_pred, epi = lib["img"], lib["act_pred"], lib["episode_index"]

    # 70/30 split by episode
    eps = np.unique(epi)
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    tr_eps = set(eps[:int(len(eps) * 0.7)].tolist())
    train = np.array([e in tr_eps for e in epi])
    Xtr, Ytr = (torch.from_numpy(a[train]).to(args.device) for a in (img, act_pred))
    Xev, Yev = (torch.from_numpy(a[~train]).to(args.device) for a in (img, act_pred))

    model = MLP(img.shape[1], args.hidden, act_pred.shape[1], depth=2).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = torch.from_numpy(rng.choice(len(Xtr), 512)).to(args.device)
        loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                eval_mse = ((model(Xev) - Yev) ** 2).mean().item()
            print(f"step {step:4d}  train {loss.item():.4f}  eval {eval_mse:.5f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"d_in": img.shape[1], "hidden": args.hidden, "d_out": act_pred.shape[1],
                           "depth": 2, "H": 50, "action_dim": 7},
                "eval_mse": float(eval_mse)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
