"""SimplerEnv-Bridge (WidowX) evaluation client for the pi0-Bridge server (SimplerEnv env).

Follows INT-ACT's Bridge adapter: p01/p99 bound normalization, proprioception = EEF pose in
the robot base frame (top-down rpy + gripper openness), actions denormalized (except the
gripper) and converted to axis-angle, gripper [0,1] -> [-1,1].

Options:
  --score_library L  print, per episode, the demonstration-manifold score
                     "MSCORE <task> ep<i> <s(tau)> success=<bool>": mean over steps of the max
                     cosine similarity to the same-task frames of library L.
  --collect F        save, for successful episodes, one frame per action-chunk query together
                     with that chunk to F (input for build_bridge_library.py).

Usage:
  python simpler_bridge/pi0_bridge_client.py --task carrot --score_library data/bridge_library.npz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from transforms3d.euler import euler2axangle, mat2euler
from transforms3d.quaternions import quat2mat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import importlib.util  # noqa: E402

# ManiSkill2_real2sim reads its asset root at import time; default it to the SimplerEnv checkout
SIMPLER_ROOT = Path(importlib.util.find_spec("simpler_env").origin).resolve().parents[1]
os.environ.setdefault("MS2_ASSET_DIR", str(SIMPLER_ROOT / "ManiSkill2_real2sim/data"))

import simpler_env  # noqa: E402,F401
import simpler_env.evaluation.maniskill2_evaluator as _ev  # noqa: E402
import simpler_env.utils.env.observation_utils as _ou  # noqa: E402

TASK_ID = {"carrot": 0, "spoon": 1, "stack": 2}
TASKS = {
    "carrot": "PutCarrotOnPlateInScene-v0",
    "spoon": "PutSpoonOnTableClothInScene-v0",
    "stack": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
}
DEFAULT_ROT = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])   # Bridge top-down frame
IMG_SIZE = (224, 224)


def normalize_bound(x, lo, hi, eps=1e-8):
    return np.clip(2 * (x - lo) / (hi - lo + eps) - 1, -1, 1)


def denormalize_bound(x, lo, hi):
    return (x + 1) / 2 * (hi - lo) + lo


# keep a handle on the env to read the EEF pose (SimplerEnv only passes images to the policy)
_ENV = {"env": None}
_orig_get_image = _ou.get_image_from_maniskill2_obs_dict


def _get_image(env, obs, camera_name=None):
    _ENV["env"] = env
    return _orig_get_image(env, obs, camera_name=camera_name)


_ou.get_image_from_maniskill2_obs_dict = _get_image
_ev.get_image_from_maniskill2_obs_dict = _get_image


class PI0BridgeInference:
    def __init__(self, server, stats, record):
        self.server, self.record = server, record
        self.pro_lo, self.pro_hi = (np.array(stats["proprio"][k], np.float64) for k in ("p01", "p99"))
        self.act_lo, self.act_hi = (np.array(stats["action"][k], np.float64) for k in ("p01", "p99"))
        self.task_description = None

    def reset(self, task_description):
        self.task_description = task_description
        self.queue = []
        self.ep_imgs = []                          # every step (selector score)
        self.col_imgs, self.col_chunks = [], []    # one frame per action-chunk query (library)
        try:
            requests.post(f"{self.server}/reset", timeout=30)
        except Exception:
            pass

    def _proprio(self):
        env = _ENV["env"].unwrapped
        ee = env.agent.robot.pose.inv() * env.tcp.pose
        rpy = np.array(mat2euler(quat2mat(np.asarray(ee.q, np.float64)) @ DEFAULT_ROT.T, axes="sxyz"))
        openness = 1.0 - float(env.agent.get_gripper_closedness())
        return np.concatenate([np.asarray(ee.p, np.float64), rpy, [openness]])

    def step(self, image, task_description=None, *a, **k):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        state = normalize_bound(self._proprio(), self.pro_lo, self.pro_hi)
        img224 = cv2.resize(np.asarray(image, np.uint8), IMG_SIZE, interpolation=cv2.INTER_AREA)
        if not self.queue:
            img = img224.astype(np.float32) / 255.0
            r = requests.post(f"{self.server}/act", json={"image": img.tolist(), "state": state.tolist(),
                                                          "instruction": self.task_description}, timeout=120)
            chunk = np.asarray(r.json()["action_chunk"], np.float64)
            if self.record:
                self.col_imgs.append(img224.astype(np.uint8))
                self.col_chunks.append(chunk.astype(np.float32))
            for na in chunk:
                raw = denormalize_bound(na[:6], self.act_lo[:6], self.act_hi[:6])
                ax, ang = euler2axangle(raw[3], raw[4], raw[5])
                self.queue.append((raw[:3], ax * ang, 2.0 * (na[6] > 0.5) - 1.0))
        if self.record:
            self.ep_imgs.append(img224.astype(np.uint8))
        wv, rax, grip = self.queue.pop(0)
        raw_action = {"world_vector": wv, "rotation_delta": rax, "open_gripper": np.array([grip])}
        action = {"world_vector": wv, "rot_axangle": rax, "gripper": np.array([grip]),
                  "terminate_episode": np.array([0.0])}
        return raw_action, action

    def visualize_epoch(self, *a, **k):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--n_episodes", type=int, default=24)
    ap.add_argument("--server", default="http://127.0.0.1:5007")
    ap.add_argument("--stats", default="data/bridge_statistics.json")
    ap.add_argument("--score_library", default=None)
    ap.add_argument("--collect", default=None)
    ap.add_argument("--log_dir", default="outputs/bridge/videos")
    args = ap.parse_args()

    from simpler_env.evaluation.maniskill2_evaluator import run_maniskill2_eval_single_episode
    from simpler_env.utils.env.env_builder import get_robot_control_mode
    overlay = SIMPLER_ROOT / "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
    model = PI0BridgeInference(args.server, json.load(open(args.stats)),
                               record=bool(args.collect or args.score_library))
    base = dict(model=model, ckpt_path="pi0bridge", robot_name="widowx", env_name=TASKS[args.task],
                scene_name="bridge_table_1_v1", robot_init_x=0.147, robot_init_y=0.028,
                robot_init_quat=[0, 0, 0, 1], control_mode=get_robot_control_mode("widowx", "openvla"),
                rgb_overlay_path=str(overlay), control_freq=5, sim_freq=500, max_episode_steps=60,
                enable_raytracing=False, additional_env_save_tags=None, obs_camera_name=None,
                logging_dir=args.log_dir)

    if args.score_library:
        import torch
        from encoder import DinoSigLIPEncoder
        enc = DinoSigLIPEncoder(device="cuda")
        lib = np.load(args.score_library, allow_pickle=True)
        memb = lib["emb"].astype(np.float32)[lib["task"] == TASK_ID[args.task]]
        memb /= (np.linalg.norm(memb, axis=1, keepdims=True) + 1e-8)
        memb_t = torch.from_numpy(memb).cuda()

    succ, col_img, col_chunk, col_epi = 0, [], [], []
    for ep in range(args.n_episodes):
        ok = bool(run_maniskill2_eval_single_episode(additional_env_build_kwargs={}, obj_episode_id=ep, **base))
        succ += int(ok)
        if args.score_library and model.ep_imgs:
            with torch.no_grad():
                sims = [(enc(model.ep_imgs[j:j + 32]) @ memb_t.T).max(dim=1).values.float().cpu().numpy()
                        for j in range(0, len(model.ep_imgs), 32)]
            print(f"MSCORE {args.task} ep{ep} {float(np.concatenate(sims).mean()):.5f} success={ok}", flush=True)
        if args.collect and ok and len(model.col_imgs) >= 2:
            col_img += model.col_imgs
            col_chunk += model.col_chunks
            col_epi += [len(set(col_epi))] * len(model.col_imgs)
        print(f"[eval] {args.task} ep{ep} success={ok} ({succ}/{ep + 1})", flush=True)
    print(f"RESULT task={args.task} SR={succ}/{args.n_episodes}={succ / args.n_episodes * 100:.1f}%", flush=True)
    if args.collect and col_img:
        os.makedirs(os.path.dirname(args.collect) or ".", exist_ok=True)
        np.savez(args.collect, img=np.asarray(col_img, np.uint8), chunk=np.asarray(col_chunk, np.float32),
                 episode_index=np.asarray(col_epi, np.int64))
        print(f"collected {len(col_img)} frames -> {args.collect}", flush=True)


if __name__ == "__main__":
    main()
