"""Build the coke-can demonstration library from successful greedy OpenVLA rollouts.

Runs OpenVLA-7B (patched SimplerEnv policy, COLLECT=1) on the coke-can task over 3 can
orientations x 5 x 5 object positions, keeps the successful episodes, and stores per frame:
  img        (N, 1536)  DINOv2+SigLIP embedding
  act_chunk  (N, 50, 7) cumulative future actions (entry 1 = the frame's own action)
  episode_index, frame_index

Usage (SimplerEnv env, from the SimplerEnv checkout so relative asset paths resolve):
  PYTHONPATH=<repo>/simpler_openvla python <repo>/simpler_openvla/collect_openvla_library.py \
      --out <repo>/data/cokecan_library.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ["COLLECT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from encoder import DinoSigLIPEncoder  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import run_maniskill2_eval_single_episode  # noqa: E402
from simpler_env.policies.openvla.openvla_model import OpenVLAInference  # noqa: E402
from simpler_env.utils.env.env_builder import get_robot_control_mode  # noqa: E402

H = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_dir", default="outputs/openvla/collect_videos")
    args = ap.parse_args()

    model = OpenVLAInference(saved_model_path="openvla/openvla-7b", policy_setup="google_robot")
    base = dict(model=model, ckpt_path="openvla/openvla-7b", robot_name="google_robot_static",
                env_name="GraspSingleOpenedCokeCanInScene-v0", scene_name="google_pick_coke_can_1_v4",
                robot_init_x=0.35, robot_init_y=0.20, robot_init_quat=[0, 0, 0, 1],
                control_mode=get_robot_control_mode("google_robot_static", "openvla"),
                rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
                control_freq=3, sim_freq=501, max_episode_steps=80, enable_raytracing=False,
                additional_env_save_tags=None, obs_camera_name=None, logging_dir=args.log_dir)

    imgs, acts, epi, n_tot = [], [], [], 0
    for opt in [{"lr_switch": True}, {"upright": True}, {"laid_vertically": True}]:
        for ox in np.linspace(-0.35, -0.12, 5):
            for oy in np.linspace(-0.02, 0.42, 5):
                ok = run_maniskill2_eval_single_episode(obj_init_x=float(ox), obj_init_y=float(oy),
                                                        additional_env_build_kwargs={**opt, "urdf_version": None}, **base)
                n_tot += 1
                if ok and len(model._traj_img) >= 3:
                    e = len(set(epi))
                    imgs += model._traj_img; acts += model._traj_act; epi += [e] * len(model._traj_img)
                print(f"[collect] {n_tot} episodes, {len(set(epi))} successful, {len(imgs)} frames", flush=True)

    enc = DinoSigLIPEncoder(device="cuda")
    epi, acts = np.array(epi), np.array(acts, np.float32)
    IMG, CH, EPI, FI = [], [], [], []
    for e in np.unique(epi):
        idx = np.where(epi == e)[0]
        ea, T = acts[idx], len(idx)
        embs = enc([imgs[i] for i in idx]).cpu().numpy().astype(np.float32)
        for t in range(T):
            ch = np.zeros((H, 7), np.float32)
            for k in range(1, H):
                ch[k] = ch[k - 1] + (ea[t + k - 1] if (t + k - 1) < T else 0.0)
            IMG.append(embs[t]); CH.append(ch); EPI.append(int(e)); FI.append(t)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, img=np.array(IMG, np.float32), act_chunk=np.array(CH, np.float32),
             episode_index=np.array(EPI, np.int64), frame_index=np.array(FI, np.int64))
    print(f"saved {args.out}: {len(IMG)} frames from {len(np.unique(EPI))} successful episodes")


if __name__ == "__main__":
    main()
