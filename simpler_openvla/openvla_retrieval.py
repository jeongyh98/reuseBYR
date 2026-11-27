"""Demonstration library access for the OpenVLA SimplerEnv policy (patches/SimplerEnv).

The library (DEMO_LIBRARY, built by collect_openvla_library.py) provides
  - the selector score: max cosine similarity of the current frame to the library frames
  - per-step re-ranking: the image-nearest library frame's immediate action, used to pick
    among N sampled OpenVLA actions
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from encoder import DinoSigLIPEncoder  # noqa: E402

_LIB = os.environ.get("DEMO_LIBRARY", "data/cokecan_library.npz")
_enc = _img = _act = None


def _load():
    global _enc, _img, _act
    if _enc is not None:
        return
    _enc = DinoSigLIPEncoder(device="cuda")
    d = np.load(_LIB, allow_pickle=True)
    img = d["img"].astype(np.float32)
    img /= (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
    _img = torch.from_numpy(img).cuda()                 # (N, 1536)
    _act = d["act_chunk"][:, 1, :].astype(np.float32)   # (N, 7) immediate action
    print(f"[openvla_retrieval] library: {_img.shape[0]} frames", flush=True)


@torch.no_grad()
def retrieve(pil_image, k: int = 1):
    """Immediate action (7,) of the image-nearest library frame(s)."""
    _load()
    sims = (_enc([np.asarray(pil_image)]) @ _img.T)[0]
    if k == 1:
        return _act[int(sims.argmax())]
    return _act[torch.topk(sims, k).indices.cpu().numpy()].mean(0)


def select(candidates, retrieved):
    """Candidate whose 6-d motion direction best matches the retrieved action (cosine),
    with gripper-sign agreement as a tiebreak."""
    r = np.asarray(retrieved, np.float32)
    rd, rn = r[:6], np.linalg.norm(r[:6]) + 1e-8
    best, best_s = None, -1e9
    for a in candidates:
        a = np.asarray(a, np.float32)
        s = float(a[:6] @ rd) / (np.linalg.norm(a[:6]) + 1e-8) / rn + 0.1 * float(np.sign(a[6]) == np.sign(r[6]))
        if s > best_s:
            best_s, best = s, a
    return best
