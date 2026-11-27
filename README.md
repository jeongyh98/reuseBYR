# Reuse Before You Retrieve

Code for **"Reuse Before You Retrieve: Diagnosing Headroom and Complementarity for Test-Time
Augmentation of Embodied Multimodal Policies"** (Workshop on Embodied Multimodal Reasoning in
Physical Environments, ECCV 2026).

Two measurable axes govern test-time augmentation of a **frozen** VLA policy:

| Axis | Question | Probe | Intervention |
|---|---|---|---|
| Recoverable headroom `H_N = pass@N − pass@1` | Does the policy already succeed in some of its stochastic rollouts? | repeated rollouts | **episode-level selector**: keep the rollout closest to the demonstration manifold |
| Retrieval complementarity | Does the policy lack the demonstrated action prior? | **DFE** (demonstration-fit error) | **event-schema retrieval** (flow-matching warm start, α = 0.3) |

The selector score of a rollout τ is `s(τ)`: the mean over queries of the maximum cosine
similarity between the frozen DINOv2+SigLIP embedding of the current frame and the frames of
the demonstration library. For each episode the selector keeps the rollout (seed) with the
highest `s(τ)`.

## Repository layout

```
common/            shared modules: DINOv2+SigLIP encoder, event-schema retrieval, action MLP
libero_eval/       LIBERO / LIBERO-Occ (Tables 1-4, 7, Fig. 1)
  build_library.py      demonstration library from HuggingFaceVLA/libero
  train_action_mlp.py   action MLP (action key of the retrieval score)
  eval_libero.py        evaluation through lerobot-eval (variants below)
  run_libero.sh         one backbone x variant x suite over seeds
  demo_fit_probe.py     DFE probe
  register_libero_{base,occluded}.py   suite registration
simpler_bridge/    SimplerEnv-Bridge with pi0-Bridge (Table 5): policy server + SimplerEnv client
simpler_openvla/   SimplerEnv coke-can with OpenVLA-7B (Table 6)
patches/           patches for SimplerEnv (OpenVLA policy) and LIBERO-Plus (NumPy 2)
scripts/           compute the tables from the evaluation outputs
```

## Installation

Two Python environments are used, because lerobot and SimplerEnv/OpenVLA need different
library versions. The versions below are the ones used for the paper.

**lerobot environment** (LIBERO, the Bridge policy server, library building)

- Python 3.12, `lerobot==0.4.4` (torch 2.10, transformers 4.53.3), robosuite 1.4.0, mujoco 3.8.1
- `pip install flask pyarrow huggingface_hub safetensors`
- LIBERO-Plus ([sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) @ `4976dc3`):
  ```bash
  git clone https://github.com/sylvestf/LIBERO-plus && cd LIBERO-plus && git checkout 4976dc3
  git apply <this repo>/patches/LIBERO-plus/env_wrapper_numpy2.patch && pip install -e .
  ```
- LIBERO-Occ (Li et al., arXiv:2606.10862): copy the `libero_{spatial,object,goal,10}_occluded`
  folders of its release into `libero/libero/bddl_files/` and `libero/libero/init_files/` of
  LIBERO-Plus. The clean `*_base` suites take their task list from these folders, so this step
  is needed for all LIBERO experiments.

**SimplerEnv environment** (Bridge client, OpenVLA)

- Python 3.10, torch 2.2.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10,
  flash-attn 2.5.5, tensorflow 2.15.0, sapien 2.2.2, `pip install requests transforms3d opencv-python`
- SimplerEnv ([simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) @ `06accac`, with
  its ManiSkill2_real2sim submodule), installed following its README, plus the OpenVLA policy:
  ```bash
  cd SimplerEnv && git checkout 06accac && git apply <this repo>/patches/SimplerEnv/openvla_policy.patch
  ```

Models (`lerobot/pi0_libero_finetuned_v044`, `lerobot/pi05_libero_finetuned_v044`,
`lerobot/smolvla_libero`, `openvla/openvla-7b`, `juexzz/INTACT-pi0-finetune-bridge`,
`facebook/dinov2-base`, `google/siglip-base-patch16-224`) and the LIBERO dataset are downloaded
from the Hugging Face Hub on first use. The pi0 models need access to `google/paligemma-3b-pt-224`.

All commands below are run from the repository root. Generated files go to `data/` and
`outputs/`.

## LIBERO (Tables 1-4, 7, Fig. 1)

**1. Demonstration library and action MLP** (lerobot env)

```bash
python libero_eval/build_library.py --view agentview --out data/libero_library_agentview.npz
python libero_eval/train_action_mlp.py --library data/libero_library_agentview.npz --out data/action_mlp_libero.pt
python libero_eval/build_library.py --view wrist --out data/libero_library_wrist.npz
```

The library is a fixed random subset of the LIBERO demonstrations (50 of the 377 parquet
shards of `HuggingFaceVLA/libero`, 36,560 frames). Retrieval and the selector score use the
wrist-camera library; the action MLP is trained on the agentview library.

**2. Evaluation.** `eval_libero.py --variant` selects:

| variant | what runs | used for |
|---|---|---|
| `selector` | frozen policy; logs `s(τ)` per episode to `manifold_scores.tsv` | pass@1, +Selector, oracle pass@N |
| `retrieval` | event-schema retrieval warm start (α = 0.3) | +Retrieval |
| `retrieval_selector` | retrieval, plus `s(τ)` logging | retrieval × selector composition |
| `vanilla` | frozen policy, unmodified | sanity check |

The policy always receives its normal camera inputs; only the retrieval query and `s(τ)` use
the wrist camera. `run_libero.sh` runs one configuration over the seeds (50 episodes × 10 tasks
per seed on the base suites, 25 × 10 on the occluded suites):

```bash
# Tables 1, 2 and Fig. 1: 3 backbones x 4 suites, seeds 1000/2000/3000
for B in pi0 smolvla pi05; do for S in spatial object goal 10; do
  BACKBONE=$B VARIANT=selector  SUITE=libero_${S}_base bash libero_eval/run_libero.sh
  BACKBONE=$B VARIANT=retrieval SUITE=libero_${S}_base bash libero_eval/run_libero.sh
done; done

# Table 3: retrieval x selector composition (pi0)
for S in spatial object goal 10; do
  BACKBONE=pi0 VARIANT=retrieval_selector SUITE=libero_${S}_base bash libero_eval/run_libero.sh
done

# Table 7: LIBERO-Occ, seeds 1000/2000
for B in pi0 pi05; do for S in spatial object goal 10; do for V in selector retrieval; do
  BACKBONE=$B VARIANT=$V SUITE=libero_${S}_occluded SEEDS="1000 2000" bash libero_eval/run_libero.sh
done; done; done

python scripts/compute_libero_tables.py    # Tables 1-4, 7 and the Fig. 1 correlation
```

**3. DFE probe** (400 demonstration states, K = 3 samples each)

```bash
for B in pi0 smolvla pi05; do python libero_eval/demo_fit_probe.py --backbone $B; done
```

## SimplerEnv-Bridge (Table 5)

The policy runs in a server in the lerobot environment; the SimplerEnv client sends
observations to it. Each server process uses one noise seed.

**1. Checkpoint and normalization statistics**

```bash
python simpler_bridge/convert_intact_checkpoint.py --out data/pi0_bridge       # lerobot env
curl -L -o data/bridge_statistics.json \
  https://raw.githubusercontent.com/ai4ce/INT-ACT/66ae3ed4719c68cb2f3f22868015a7a171b04e1f/config/dataset/bridge_statistics.json
```

**2. Demonstration library.** The Bridge library holds the successful episodes of one
rollout pass of the base policy (24 episodes per task, a noise seed not used for evaluation),
one frame per action-chunk query (15 per episode; 510 frames in the paper):

```bash
python simpler_bridge/pi0_bridge_server.py --ckpt data/pi0_bridge --noise_seed 0        # terminal 1, lerobot env
for T in carrot spoon stack; do                                                        # terminal 2, SimplerEnv env
  python simpler_bridge/pi0_bridge_client.py --task $T --collect data/bridge_collect_$T.npz
done
python simpler_bridge/build_bridge_library.py --collected data/bridge_collect_*.npz --out_dir data   # lerobot env
```

**3. Evaluation**, for noise seeds 42, 43, 44. Start a fresh server for each seed and run
the three tasks in this order (the per-call noise seed counts episodes since server start):

```bash
mkdir -p outputs/bridge
# selector: base policy + s(tau) per episode
python simpler_bridge/pi0_bridge_server.py --ckpt data/pi0_bridge --noise_seed 42             # terminal 1
for T in carrot spoon stack; do                                                               # terminal 2
  python simpler_bridge/pi0_bridge_client.py --task $T --score_library data/bridge_library.npz \
    | tee outputs/bridge/selector_${T}_s42.log
done

# retrieval
python simpler_bridge/pi0_bridge_server.py --ckpt data/pi0_bridge --noise_seed 42 --retrieval # terminal 1
for T in carrot spoon stack; do                                                               # terminal 2
  python simpler_bridge/pi0_bridge_client.py --task $T | tee outputs/bridge/retrieval_${T}_s42.log
done

python scripts/compute_bridge.py
```

## SimplerEnv coke-can with OpenVLA (Table 6)

Run in the SimplerEnv environment with `SIMPLER_ROOT` pointing to the patched SimplerEnv
checkout.

```bash
export SIMPLER_ROOT=/path/to/SimplerEnv REPO=$PWD
mkdir -p outputs/openvla data

# demonstration library: successful greedy rollouts (75 initial states)
(cd $SIMPLER_ROOT && PYTHONPATH=$SIMPLER_ROOT:$REPO/simpler_openvla \
  python $REPO/simpler_openvla/collect_openvla_library.py --out $REPO/data/cokecan_library.npz)

# greedy decoding
bash simpler_openvla/run_openvla.sh | tee outputs/openvla/greedy.log

# per-step re-ranking among N sampled actions
RERANK_N=4 bash simpler_openvla/run_openvla.sh | tee outputs/openvla/rerank.log

# three sampled runs with s(tau) logging -> sample@1 / episode-level best-of-3 / oracle
for s in 101 102 103; do
  SAMPLE_T=1.0 NOISE_SEED=$s MANIFOLD_LOG=1 MANIFOLD_FILE=$PWD/outputs/openvla/scores_s$s.tsv \
    bash simpler_openvla/run_openvla.sh | tee outputs/openvla/sample_s$s.log
done
python scripts/compute_openvla.py
```

## Notes

- **Score-success pairing.** `s(τ)` is logged by the run that produced the rollout, and the
  table scripts always read success from that same run. Pairing scores with another run's
  successes breaks the selector.
- **Seeds.** LIBERO seeds are lerobot-eval seeds (environment and policy). On Bridge the server
  seeds the flow-matching noise per call from (seed, episode, call). OpenVLA uses `NOISE_SEED`
  with temperature sampling.
- **Hyperparameters.** Retrieval: top-K = 5 event-start candidates, equal weights on the
  image, action and state distances (z-normalized per query), α = 0.3. Event starts: episode
  start, gripper transition, or a motion-direction cosine below 0.3 (minimum motion 0.02).

## License

The code in this repository is released under the [MIT License](LICENSE). Models, datasets
and third-party code used with it (LIBERO / LIBERO-Plus, LIBERO-Occ, SimplerEnv, lerobot,
OpenVLA, INT-ACT) are covered by their own licenses.
