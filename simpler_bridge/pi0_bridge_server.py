"""pi0-Bridge policy server (lerobot env).

POST /act takes a normalized observation (224x224 image in [0,1], 7-d bound-normalized state,
instruction) and returns the action chunk (4, 7) in normalized space. POST /reset marks an
episode boundary. The SimplerEnv client (pi0_bridge_client.py) handles proprioception,
normalization, and action conversion.

The flow-matching noise is seeded per call from (--noise_seed, episode, call), so each noise
seed gives one reproducible stochastic rollout per episode; the selector uses seeds 42/43/44.

With --retrieval, event-schema retrieval over the Bridge library warm-starts the flow-matching
sampler: x_{t*} = t* z + (1 - t*) r with t* = 1 - alpha, then the remaining round(t* * steps)
Euler steps are integrated.

Usage:
  python simpler_bridge/pi0_bridge_server.py --ckpt data/pi0_bridge --noise_seed 42
  python simpler_bridge/pi0_bridge_server.py --ckpt data/pi0_bridge --noise_seed 42 --retrieval \
      --espr_library data/bridge_espr_library.npz --action_mlp data/bridge_action_mlp.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="data/pi0_bridge")
ap.add_argument("--noise_seed", type=int, default=42)
ap.add_argument("--retrieval", action="store_true")
ap.add_argument("--alpha", type=float, default=0.3)
ap.add_argument("--espr_library", default="data/bridge_espr_library.npz")
ap.add_argument("--action_mlp", default="data/bridge_action_mlp.pt")
ap.add_argument("--port", type=int, default=5007)
args = ap.parse_args()

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks  # noqa: E402

pol = PI0Policy.from_pretrained(args.ckpt, strict=False)
try:
    lm = pol.model.paligemma_with_expert.paligemma
    lm.model.language_model.embed_tokens.weight = lm.lm_head.weight
except Exception:
    pass
pol = pol.to("cuda").eval()
pre, post = make_pre_post_processors(pol.config, pretrained_path=args.ckpt)
print(f"[server] ready: chunk={pol.config.chunk_size} retrieval={args.retrieval} alpha={args.alpha}", flush=True)

TASK_KW = {"carrot": 0, "spoon": 1, "stack": 2, "cube": 2}


def task_of(instr: str) -> int:
    s = instr.lower()
    for kw, tid in TASK_KW.items():
        if kw in s:
            return tid
    return -1


_S = {}
if args.retrieval:
    from encoder import DinoSigLIPEncoder
    from mlp import load_mlp
    from retrieval import RetrievalLibrary, espr_retrieve
    _S["enc"] = DinoSigLIPEncoder(device="cuda")
    e = np.load(args.espr_library)
    _S["lib"] = RetrievalLibrary(img=e["emb"].astype(np.float32), act_pred=e["act_pred"].astype(np.float32),
                                 action_chunks_raw=e["chunk"].astype(np.float32),
                                 episode_index=e["episode_index"].astype(np.int64),
                                 frame_index=e["frame_index"].astype(np.int64))
    _S["ev"] = e["event_starts"].astype(bool)
    _S["task"] = e["task"].astype(np.int64)
    _S["amlp"] = load_mlp(args.action_mlp, "cuda")
    print(f"[server] retrieval library {_S['lib'].N} frames, {int(_S['ev'].sum())} event starts", flush=True)

    _orig_sample = pol.model.sample_actions

    def sample_warmstart(images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=None, **kw):
        m = pol.model
        rt7 = getattr(m, "_rt", None)
        if rt7 is None:
            return _orig_sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise, num_steps=num_steps, **kw)
        if num_steps is None:
            num_steps = m.config.num_inference_steps
        bsize, device = state.shape[0], state.device
        shape = (bsize, m.config.chunk_size, m.config.max_action_dim)
        z = m.sample_noise(shape, device) if noise is None else noise
        rt = torch.zeros(shape, device=device, dtype=z.dtype)
        rt[:, :, :rt7.shape[-1]] = rt7.to(device, z.dtype)
        t_star = 1.0 - max(0.0, min(1.0, args.alpha))
        x_t = t_star * z + (1.0 - t_star) * rt
        n_rem = max(0, int(round(num_steps * t_star)))
        orig_dim = m.config.action_feature.shape[0] if hasattr(m.config, "action_feature") else 7
        if n_rem == 0:
            return x_t[:, :, :orig_dim]
        pe, ppm, pam = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        p2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        p4d = m._prepare_attention_masks_4d(p2d)
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_kv = m.paligemma_with_expert.forward(attention_mask=p4d, position_ids=pos, past_key_values=None,
                                                     inputs_embeds=[pe, None], use_cache=True)
        dt = -t_star / n_rem
        for step in range(n_rem):
            tt = torch.tensor(t_star + step * dt, dtype=torch.float32, device=device).expand(bsize)
            v = m.denoise_step(state=state, prefix_pad_masks=ppm, past_key_values=past_kv, x_t=x_t, timestep=tt).float()
            x_t = x_t + dt * v
        return x_t
    pol.model.sample_actions = sample_warmstart


@torch.no_grad()
def retrieve_chunk(img_hwc01: np.ndarray, instr: str) -> torch.Tensor:
    emb = _S["enc"]([(np.asarray(img_hwc01) * 255).astype(np.uint8)]).float()
    emb_n = torch.nn.functional.normalize(emb, dim=1)
    img_q = emb_n[0].cpu().numpy().astype(np.float32)
    act_q = _S["amlp"](emb_n)[0].cpu().numpy().astype(np.float32)
    ev = _S["ev"].copy()
    tid = task_of(instr)
    if tid >= 0 and (ev & (_S["task"] == tid)).any():   # same-task events only
        ev = ev & (_S["task"] == tid)
    res = espr_retrieve(_S["lib"], img_q, act_q, event_starts=ev, k=5, lambda_state=0.0)
    return torch.tensor(res.action_chunks.mean(0)[:, :7], dtype=torch.float32, device="cuda")


app = Flask(__name__)
_SEED = {"ep": 0, "call": 0}


@app.route("/health")
def health():
    return jsonify(ok=True, chunk=pol.config.chunk_size)


@app.route("/reset", methods=["POST"])
def reset():
    pol.reset()
    _SEED["ep"] += 1
    _SEED["call"] = 0
    return jsonify(ok=True)


@app.route("/act", methods=["POST"])
def act():
    d = request.get_json()
    img = np.asarray(d["image"], dtype=np.float32)
    if img.shape[-1] == 3:                       # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
    batch = {"observation.images.top": torch.from_numpy(img).float(),
             "observation.state": torch.from_numpy(np.asarray(d["state"], dtype=np.float32)),
             "task": d["instruction"]}
    with torch.no_grad():
        if args.retrieval:
            pol.model._rt = retrieve_chunk(np.transpose(img, (1, 2, 0)), d["instruction"])
        torch.manual_seed(args.noise_seed * 100000 + _SEED["ep"] * 1000 + _SEED["call"])
        _SEED["call"] += 1
        chunk = post(pol.predict_action_chunk(pre(batch)))
    chunk = np.asarray(chunk.detach().float().cpu())
    if chunk.ndim == 3:
        chunk = chunk[0]
    return jsonify(action_chunk=chunk[:, :7].tolist())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=args.port, threaded=False)
