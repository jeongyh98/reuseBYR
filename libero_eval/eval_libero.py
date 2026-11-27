"""LIBERO evaluation of a frozen flow-matching VLA (pi0 / pi0.5 / SmolVLA) through lerobot-eval.

Variants (--variant):
  vanilla             the frozen policy alone
  selector            the frozen policy alone, plus per-episode logging of the
                      demonstration-manifold score s(tau) = mean over action-chunk queries of the
                      max cosine similarity between the wrist-view embedding and the library.
                      Run with N seeds; scripts/compute_libero_tables.py then keeps, per
                      episode, the seed with the highest s(tau).
  retrieval           event-schema retrieval: the flow-matching start noise becomes
                      alpha * retrieved_chunk + (1 - alpha) * N(0, I), alpha = 0.3
  retrieval_selector  retrieval plus score logging (retrieval x selector composition)

The policy always receives its normal observations; only the retrieval query and the score
read the wrist camera.

Example:
  python libero_eval/eval_libero.py --backbone pi0 --suite libero_object_base --variant selector \
      --seed 1000 --n_episodes 50 --library data/libero_library_wrist.npz \
      --action_mlp data/action_mlp_libero.pt --output_dir outputs/libero/pi0_selector_libero_object_base_s1000
"""
from __future__ import annotations
import argparse
import atexit
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from encoder import DinoSigLIPEncoder  # noqa: E402
from mlp import load_mlp  # noqa: E402
from retrieval import RetrievalLibrary, aggregate_chunks, build_event_starts, espr_retrieve  # noqa: E402

BACKBONES = {
    "pi0": dict(model="lerobot/pi0_libero_finetuned_v044",
                normalizer="policy_preprocessor_step_5_normalizer_processor.safetensors",
                needs_state_arg=True, wrist_key="observation.images.image2"),
    "pi05": dict(model="lerobot/pi05_libero_finetuned_v044",
                 normalizer="policy_preprocessor_step_2_normalizer_processor.safetensors",
                 needs_state_arg=False, wrist_key="observation.images.image2"),
    "smolvla": dict(model="lerobot/smolvla_libero",
                    normalizer="policy_preprocessor_step_5_normalizer_processor.safetensors",
                    needs_state_arg=True, wrist_key="observation.images.camera2"),
}
VARIANTS = ("vanilla", "selector", "retrieval", "retrieval_selector")
SCORE_FILE = "manifold_scores.tsv"


def load_library(path: str, backbone: str) -> RetrievalLibrary:
    """Load the wrist library; robot state is normalized with the policy's own state statistics,
    so it lives in the same space as the (already normalized) query state."""
    import safetensors.torch as st
    from huggingface_hub import hf_hub_download
    bc = BACKBONES[backbone]
    stats = st.load_file(hf_hub_download(bc["model"], bc["normalizer"]))
    mu = stats["observation.state.mean"].cpu().numpy().astype(np.float32)
    sd = stats["observation.state.std"].cpu().numpy().astype(np.float32)
    d = np.load(path)
    state = ((d["state"].astype(np.float32) - mu) / (sd + 1e-6)).astype(np.float32)
    lib = RetrievalLibrary(img=d["img"].astype(np.float32), act_pred=d["act_pred"].astype(np.float32),
                           action_chunks_raw=d["act_chunk"].astype(np.float32),
                           episode_index=d["episode_index"].astype(np.int64),
                           frame_index=d["frame_index"].astype(np.int64), state=state)
    lib._state_mu = np.zeros((1, state.shape[1]), np.float32)   # state is already normalized
    lib._state_sd = np.ones((1, state.shape[1]), np.float32)
    return lib


def policy_class(backbone: str):
    if backbone == "pi05":
        from lerobot.policies.pi05 import modeling_pi05 as m
        return m.PI05Policy, m
    if backbone == "pi0":
        from lerobot.policies.pi0 import modeling_pi0 as m
        return m.PI0Policy, m
    from lerobot.policies.smolvla import modeling_smolvla as m
    return m.SmolVLAPolicy, m


def install_patch(lib, action_mlp, encoder, backbone: str, alpha: float, score_path: str | None,
                  device: str = "cuda"):
    """Replace <Policy>.predict_action_chunk so that every action chunk is sampled from
    alpha * retrieved + (1 - alpha) * noise (alpha = 0 keeps the policy unchanged) and,
    optionally, the per-episode selector score is logged."""
    PolicyCls, mod = policy_class(backbone)
    ACTION = mod.ACTION
    needs_state_arg = BACKBONES[backbone]["needs_state_arg"]
    img_key = BACKBONES[backbone]["wrist_key"]
    event_starts = build_event_starts(lib, cos_thresh=0.3, mag_min=0.02, grip_thresh=1.0)
    print(f"[retrieval] {int(event_starts.sum())}/{lib.N} event-start frames", flush=True)
    chunk_scores: list[float] = []
    episode = {"i": 0}

    def flush_score():
        if chunk_scores:
            Path(score_path).parent.mkdir(parents=True, exist_ok=True)
            with open(score_path, "a") as f:
                f.write(f"{episode['i']}\t{sum(chunk_scores) / len(chunk_scores):.5f}\n")
            episode["i"] += 1
        chunk_scores.clear()

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        raw_img = batch[img_key]
        if raw_img.dim() == 5:          # (B, T, 3, H, W) -> last frame
            raw_img = raw_img[:, -1]
        elif raw_img.dim() == 3:
            raw_img = raw_img.unsqueeze(0)
        img_emb = encoder(raw_img)
        emb_np = img_emb[0].cpu().numpy()
        chunk_size, max_dim = self.config.chunk_size, self.config.max_action_dim

        noise = torch.zeros((1, chunk_size, max_dim), device=device)
        if alpha > 0:
            act_q = action_mlp(img_emb)[0].cpu().numpy()
            state_q = None
            if "observation.state" in batch:
                sq = batch["observation.state"]
                state_q = (sq[0] if sq.dim() > 1 else sq).cpu().numpy().astype(np.float32)
            res = espr_retrieve(lib, emb_np, act_q, state_q, event_starts=event_starts, k=5)
            retrieved = aggregate_chunks(res)                    # (50, 7) raw delta actions
            noise_np = np.zeros((1, chunk_size, max_dim), np.float32)
            noise_np[0, :, :retrieved.shape[1]] = retrieved[:chunk_size]
            noise = torch.from_numpy(noise_np).to(device)
        if score_path is not None:
            chunk_scores.append(float((lib.img @ emb_np).max()))
        if alpha < 1.0:
            noise = alpha * noise + (1 - alpha) * torch.randn_like(noise)

        if backbone == "smolvla":
            self.eval()
            return self._get_action_chunk(batch, noise=noise)
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[mod.OBS_LANGUAGE_TOKENS]
        masks = batch[mod.OBS_LANGUAGE_ATTENTION_MASK]
        out_dim = self.config.output_features[ACTION].shape[0]
        if needs_state_arg:
            a = self.model.sample_actions(images, img_masks, tokens, masks, self.prepare_state(batch), noise=noise)
        else:
            a = self.model.sample_actions(images, img_masks, tokens, masks, noise=noise)
        return a[:, :, :out_dim]

    PolicyCls.predict_action_chunk = predict_action_chunk

    if score_path is not None:
        _orig_reset = PolicyCls.reset

        def reset(self, *a, **kw):
            flush_score()               # episode boundary: write the finished episode's score
            return _orig_reset(self, *a, **kw)
        PolicyCls.reset = reset
        atexit.register(flush_score)    # last episode

    if backbone == "smolvla":
        # SmolVLA's stock select_action calls _get_action_chunk directly; route it through
        # predict_action_chunk while keeping the observation-queue handling identical.
        from lerobot.policies.utils import populate_queues

        @torch.no_grad()
        def select_action(self, batch, noise=None, **kwargs):
            self.eval()
            batch = self._prepare_batch(batch)
            self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
            if self._check_get_actions_condition():
                actions = self.predict_action_chunk(batch)
                self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
            return self._queues[ACTION].popleft()
        PolicyCls.select_action = select_action


def patch_pi0_loading():
    """lerobot's PI0Policy.from_pretrained is strict and fails on the tied PaliGemma embedding;
    load non-strictly and re-tie embed_tokens to lm_head."""
    from lerobot.policies.pi0 import modeling_pi0 as m
    orig = m.PI0Policy.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, *a, **kw):
        kw["strict"] = False
        model = orig(cls, *a, **kw)
        lm = model.model.paligemma_with_expert.paligemma
        if hasattr(lm, "language_model") and hasattr(lm.language_model, "embed_tokens") and hasattr(lm, "lm_head"):
            lm.language_model.embed_tokens.weight = lm.lm_head.weight
        return model
    m.PI0Policy.from_pretrained = from_pretrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--suite", required=True, help="libero_{spatial,object,goal,10}_{base,occluded}")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--n_episodes", type=int, default=50, help="episodes per task")
    ap.add_argument("--task_ids", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episode_length", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.3, help="retrieval strength (retrieval variants)")
    ap.add_argument("--library", default="data/libero_library_wrist.npz")
    ap.add_argument("--action_mlp", default="data/action_mlp_libero.pt")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    if args.suite.endswith("_occluded"):
        import register_libero_occ  # noqa: F401
    elif args.suite.endswith("_base"):
        import register_libero_base  # noqa: F401
    if args.backbone == "pi0":
        patch_pi0_loading()

    device = "cuda"
    if args.variant != "vanilla":
        alpha = args.alpha if args.variant.startswith("retrieval") else 0.0
        score_path = str(Path(args.output_dir) / SCORE_FILE) if args.variant.endswith("selector") else None
        if score_path and os.path.exists(score_path):
            os.remove(score_path)
        lib = load_library(args.library, args.backbone)
        action_mlp = load_mlp(args.action_mlp, device) if alpha > 0 else None
        enc = DinoSigLIPEncoder(device=device)
        enc._load()
        install_patch(lib, action_mlp, enc, args.backbone, alpha, score_path, device)

    task_ids = "[" + ",".join(x.strip() for x in args.task_ids.split(",")) + "]"
    argv = ["lerobot-eval", f"--policy.path={BACKBONES[args.backbone]['model']}", f"--policy.device={device}",
            "--env.type=libero", f"--env.task={args.suite}", f"--env.task_ids={task_ids}",
            f"--env.episode_length={args.episode_length}", f"--eval.n_episodes={args.n_episodes}",
            "--eval.batch_size=1", f"--seed={args.seed}", f"--output_dir={args.output_dir}"]
    if args.backbone == "smolvla":   # SmolVLA was finetuned with renamed camera keys
        argv.append('--rename_map={"observation.images.image": "observation.images.camera1",'
                    ' "observation.images.image2": "observation.images.camera2"}')
    sys.argv = argv
    from lerobot.scripts.lerobot_eval import eval_main
    eval_main()


if __name__ == "__main__":
    main()
