"""Demonstration-fit error (DFE) probe.

For chunk-start states of the LIBERO demonstrations, draw K action chunks from the frozen
policy and measure the MSE to the demonstrated chunk (raw action space). The minimum over K
indicates whether the demonstrated action is reachable by the policy's sampler.

Usage:
  python libero_eval/demo_fit_probe.py --backbone pi0 [--n_chunks 400 --k 3 --out dfe_pi0.npz]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_data import decode_image, fetch, parquet_files  # noqa: E402

MODELS = {"pi0": "lerobot/pi0_libero_finetuned_v044", "pi05": "lerobot/pi05_libero_finetuned_v044",
          "smolvla": "lerobot/smolvla_libero"}
H = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(MODELS), required=True)
    ap.add_argument("--n_chunks", type=int, default=400)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda"

    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
    # run the sampler eagerly (pi0.5 checkpoints request torch.compile + CUDA graphs)
    torch.compile = (lambda model=None, *a, **k: (model if model is not None else (lambda m: m)))

    from huggingface_hub import snapshot_download
    from lerobot.policies.factory import make_pre_post_processors
    if args.backbone == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as PolicyCls
    elif args.backbone == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as PolicyCls
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PolicyCls

    model_id = MODELS[args.backbone]
    pol = PolicyCls.from_pretrained(model_id, strict=False)
    try:  # re-tie PaliGemma embeddings (pi0 / pi0.5)
        lm = pol.model.paligemma_with_expert.paligemma
        lm.model.language_model.embed_tokens.weight = lm.lm_head.weight
    except Exception:
        pass
    pol = pol.to(dev).eval()
    pre, post = make_pre_post_processors(pol.config, pretrained_path=snapshot_download(model_id))

    tasks = {r["task_index"]: r["__index_level_0__"]
             for r in pq.read_table(fetch("meta/tasks.parquet")).to_pylist()}

    @torch.no_grad()
    def sample_chunk(agv, wrist, state8, lang):
        to_t = lambda im: torch.from_numpy(im).float().permute(2, 0, 1)[None] / 255.0
        if args.backbone == "smolvla":
            batch = {"observation.images.camera1": to_t(agv), "observation.images.camera2": to_t(wrist)}
        else:
            batch = {"observation.images.image": to_t(agv), "observation.images.image2": to_t(wrist),
                     "observation.images.empty_camera_0": torch.zeros(1, 3, 224, 224)}
        batch.update({"observation.state": torch.from_numpy(state8).float()[None], "task": [lang]})
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        ch = post(pol.predict_action_chunk(pre(batch)))
        return np.asarray(ch.detach().float().cpu())[0]          # (chunk, 7) raw

    rng = np.random.default_rng(0)
    parqs = parquet_files()
    rng.shuffle(parqs)
    mse_min, mse_mean, done = [], [], 0
    for pf in parqs:
        if done >= args.n_chunks:
            break
        by_ep = {}
        for r in pq.read_table(fetch(pf)).to_pylist():
            by_ep.setdefault(r["episode_index"], []).append(r)
        for rows in by_ep.values():
            if done >= args.n_chunks:
                break
            rows = sorted(rows, key=lambda r: r["frame_index"])
            acts = np.array([r["action"] for r in rows], np.float32)
            for start in range(0, max(1, len(rows) - H), H):
                if done >= args.n_chunks:
                    break
                r = rows[start]
                demo = acts[start:start + H]
                if len(demo) < 5:
                    continue
                agv, wrist = decode_image(r["observation.images.image"]), decode_image(r["observation.images.image2"])
                state, lang = np.array(r["observation.state"], np.float32), tasks[r["task_index"]]
                errs = []
                for _ in range(args.k):
                    chk = sample_chunk(agv, wrist, state, lang)
                    L = min(len(demo), len(chk))
                    errs.append(float(((chk[:L] - demo[:L]) ** 2).mean()))
                mse_min.append(min(errs)); mse_mean.append(float(np.mean(errs)))
                done += 1
                if done % 50 == 0:
                    print(f"{done}/{args.n_chunks}  minMSE={np.mean(mse_min):.4f}", flush=True)

    print(f"DFE [{args.backbone}] n={done} K={args.k}: minMSE {np.mean(mse_min):.4f} +- {np.std(mse_min):.4f}"
          f" | meanMSE {np.mean(mse_mean):.4f}")
    if args.out:
        np.savez(args.out, mse_min=np.array(mse_min), mse_mean=np.array(mse_mean))


if __name__ == "__main__":
    main()
