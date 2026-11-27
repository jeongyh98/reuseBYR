"""Event-schema retrieval (paper Sec. 3.4, supplementary Secs. 2-3).

Candidates are restricted to event-start frames of the demonstration library (episode start,
gripper open/close transition, or a sharp change of end-effector motion direction). Each
candidate is scored by a sum of per-query z-normalized L2 distances over
    image embedding  +  predicted action chunk  (+ robot state, when available)
and the action chunks of the top-K candidates are averaged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

K_TOP = 5


@dataclass
class RetrievalLibrary:
    img: np.ndarray                 # (N, 1536)  L2-normalized image embeddings
    act_pred: np.ndarray            # (N, H*d_a) z-normalized action chunks (retrieval key)
    action_chunks_raw: np.ndarray   # (N, H, d_a) raw delta-action chunks (what is injected)
    episode_index: np.ndarray       # (N,)
    frame_index: np.ndarray         # (N,)
    state: Optional[np.ndarray] = None   # (N, d_s) robot state (optional)
    _state_mu: Optional[np.ndarray] = field(default=None, init=False)
    _state_sd: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        assert len(self.img) == len(self.act_pred) == len(self.action_chunks_raw) == len(self.episode_index)
        if self.state is not None:
            assert len(self.state) == len(self.img)
            self._state_mu = self.state.mean(0, keepdims=True).astype(np.float32)
            self._state_sd = (self.state.std(0, keepdims=True) + 1e-6).astype(np.float32)

    @property
    def N(self) -> int:
        return len(self.img)


@dataclass
class RetrievalResult:
    indices: np.ndarray             # (K,) library indices, best first
    action_chunks: np.ndarray       # (K, H, d_a) raw chunks
    scores: np.ndarray              # (K,) fused distance (lower = closer)


def build_event_starts(lib: RetrievalLibrary, grip_dim: int = 6, cos_thresh: float = 0.3,
                       mag_min: float = 0.02, grip_thresh: float = 1.0) -> np.ndarray:
    """Mark event-start frames: episode start, gripper transition, or motion-direction change.
    Uses the step-1 entry of each delta-action chunk (step 0 is the zero delta-from-self)."""
    ac = lib.action_chunks_raw
    g_delta = ac[:, 1, grip_dim]
    mot = ac[:, 1, :3].astype(np.float64)
    ep = lib.episode_index
    order = np.lexsort((lib.frame_index, ep))
    starts = np.zeros(lib.N, bool)
    prev_e, prev_mot = None, None
    for idx in order:
        e = ep[idx]
        if e != prev_e:
            starts[idx] = True
            prev_e, prev_mot = e, mot[idx]
            continue
        if abs(g_delta[idx]) > grip_thresh:
            starts[idx] = True
        else:
            m, pm = mot[idx], prev_mot
            nm, npm = np.linalg.norm(m), np.linalg.norm(pm)
            if nm > mag_min and npm > mag_min and (m @ pm) / (nm * npm) < cos_thresh:
                starts[idx] = True
        prev_e, prev_mot = e, mot[idx]
    return starts


def espr_retrieve(lib: RetrievalLibrary, img_q: np.ndarray, act_q: np.ndarray,
                  state_q: Optional[np.ndarray] = None, *, event_starts: np.ndarray,
                  k: int = K_TOP, lambda_image: float = 1.0, lambda_action: float = 1.0,
                  lambda_state: float = 1.0) -> RetrievalResult:
    valid = event_starts.copy()
    d_img = np.linalg.norm(lib.img - img_q, axis=1)
    d_act = np.linalg.norm(lib.act_pred - act_q, axis=1)
    d_state = None
    if state_q is not None and lib.state is not None and lambda_state > 0:
        sz_q = (state_q - lib._state_mu[0]) / lib._state_sd[0]
        sz = (lib.state - lib._state_mu) / lib._state_sd
        d_state = np.linalg.norm(sz - sz_q, axis=1)
    for d in (d_img, d_act):
        d[~valid] = np.inf
    if d_state is not None:
        d_state[~valid] = np.inf

    def z(d):
        v = d[valid]
        return (d - v.mean()) / (v.std() + 1e-6)

    score = lambda_image * z(d_img) + lambda_action * z(d_act)
    if d_state is not None:
        score = score + lambda_state * z(d_state)
    score[~valid] = np.inf
    kk = min(k, int(valid.sum()))
    topk = np.argpartition(score, kk - 1)[:kk]
    topk = topk[np.argsort(score[topk])]
    return RetrievalResult(indices=topk, action_chunks=lib.action_chunks_raw[topk],
                           scores=score[topk].astype(np.float32))


def aggregate_chunks(result: RetrievalResult) -> np.ndarray:
    """Top-K action chunks -> one (H, d_a) chunk (mean)."""
    return result.action_chunks.mean(axis=0)
