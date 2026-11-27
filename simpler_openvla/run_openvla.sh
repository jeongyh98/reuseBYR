#!/bin/bash
# OpenVLA-7B on the SimplerEnv coke-can task (25 object positions, upright can).
# Requires SimplerEnv with patches/SimplerEnv/openvla_policy.patch applied.
#   SIMPLER_ROOT   path to the SimplerEnv checkout (required)
#   DEMO_LIBRARY   coke-can library (for MANIFOLD_LOG / RERANK_N), default <repo>/data/cokecan_library.npz
# Behaviour is selected with the policy's environment variables (see openvla_model.py), e.g.
#   greedy:                      bash simpler_openvla/run_openvla.sh
#   sampled run with scores:     SAMPLE_T=1.0 NOISE_SEED=101 MANIFOLD_LOG=1 \
#                                MANIFOLD_FILE=$PWD/outputs/openvla/scores_s101.tsv bash simpler_openvla/run_openvla.sh
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
SIMPLER_ROOT=${SIMPLER_ROOT:?set SIMPLER_ROOT to the SimplerEnv checkout}
export DEMO_LIBRARY=${DEMO_LIBRARY:-$REPO/data/cokecan_library.npz}
export PYTHONPATH=$SIMPLER_ROOT:$REPO/simpler_openvla:$PYTHONPATH TF_CPP_MIN_LOG_LEVEL=3
if [ -n "$MANIFOLD_FILE" ]; then mkdir -p "$(dirname "$MANIFOLD_FILE")"; rm -f "$MANIFOLD_FILE"; fi
cd "$SIMPLER_ROOT"
python simpler_env/main_inference.py \
  --policy-model openvla --ckpt-path openvla/openvla-7b \
  --robot google_robot_static --policy-setup google_robot \
  --control-freq 3 --sim-freq 501 --max-episode-steps 80 \
  --env-name GraspSingleOpenedCokeCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True urdf_version=None
