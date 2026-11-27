"""Build the LIBERO demonstration library.

Samples 50 parquet shards of HuggingFaceVLA/libero (fixed seed) and stores, per frame:
  img        (M, 1536)    DINOv2+SigLIP embedding of the chosen camera view
  act_chunk  (M, 50, 7)   raw delta-action chunk (zero-padded at the episode tail)
  act_pred   (M, 350)     z-normalized flattened chunk (action key of the retrieval score)
  state      (M, 8)       raw robot state
  episode_index, frame_index

Two libraries are built from the same frames:
  --view agentview  -> used to train the action MLP (train_action_mlp.py)
  --view wrist      -> used for retrieval and for the selector score at evaluation time

Usage:
  python libero_eval/build_library.py --view agentview --out data/libero_library_agentview.npz
  python libero_eval/build_library.py --view wrist     --out data/libero_library_wrist.npz
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from encoder import DinoSigLIPEncoder  # noqa: E402
from libero_data import decode_image, fetch, library_parquets  # noqa: E402

H_CHUNK = 50
VIEWS = {"agentview": "observation.images.image", "wrist": "observation.images.image2"}


def build_action_chunks(action_seq: np.ndarray, H: int) -> np.ndarray:
    """(T, d) -> (T, H, d) delta-from-current action chunks, zero-padded at the tail."""
    T, d = action_seq.shape
    out = np.zeros((T, H, d), np.float32)
    for t in range(T):
        end = min(t + H, T)
        out[t, :end - t] = action_seq[t:end] - action_seq[t:t + 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=list(VIEWS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_parquets", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    img_col = VIEWS[args.view]

    files = library_parquets(args.n_parquets, args.seed)
    enc = DinoSigLIPEncoder(device=args.device)
    out = {k: [] for k in ("img", "act_chunk", "state", "episode_index", "frame_index")}
    t0, total = time.time(), 0
    for i, f in enumerate(files):
        rows = pq.read_table(fetch(f), columns=[img_col, "action", "observation.state",
                                                "episode_index", "frame_index"]).to_pylist()
        by_ep: dict[int, list] = {}
        for r in rows:
            by_ep.setdefault(r["episode_index"], []).append(r)
        for ei, frames in by_ep.items():
            frames.sort(key=lambda r: r["frame_index"])
            T = len(frames)
            embs = [enc([decode_image(frames[j][img_col]) for j in range(b, min(b + args.batch, T))]).cpu()
                    for b in range(0, T, args.batch)]
            out["img"].append(np.concatenate([e.numpy() for e in embs]))
            out["act_chunk"].append(build_action_chunks(np.array([r["action"] for r in frames], np.float32), H_CHUNK))
            out["state"].append(np.array([r["observation.state"] for r in frames], np.float32))
            out["episode_index"].append(np.full(T, ei, np.int64))
            out["frame_index"].append(np.array([r["frame_index"] for r in frames], np.int64))
            total += T
        print(f"[{i + 1}/{len(files)}] {f}: {total} frames, {(time.time() - t0) / 60:.1f} min", flush=True)

    d = {k: np.concatenate(v, 0) for k, v in out.items()}
    flat = d["act_chunk"].reshape(len(d["act_chunk"]), -1)
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-6
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, img=d["img"].astype(np.float32), act_chunk=d["act_chunk"].astype(np.float32),
             act_pred=((flat - mu) / sd).astype(np.float32), act_norm_mu=mu, act_norm_sd=sd,
             state=d["state"], episode_index=d["episode_index"], frame_index=d["frame_index"])
    print(f"saved {args.out}: {len(d['img'])} frames ({args.view} view)")


if __name__ == "__main__":
    main()
