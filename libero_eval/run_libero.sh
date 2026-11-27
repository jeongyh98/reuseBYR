#!/bin/bash
# Run one backbone x variant x suite over several seeds.
#   BACKBONE  pi0 | pi05 | smolvla
#   VARIANT   vanilla | selector | retrieval | retrieval_selector
#   SUITE     libero_{spatial,object,goal,10}_base | libero_{spatial,object,goal,10}_occluded
#   SEEDS     default "1000 2000 3000"
#   NEPS      episodes per task; default 50 (base, n=500) or 25 (occluded, n=250)
# Outputs: outputs/libero/<backbone>_<variant>_<suite>_s<seed>/{eval_info.json,manifold_scores.tsv}
#
# Example: BACKBONE=pi0 VARIANT=selector SUITE=libero_object_base bash libero_eval/run_libero.sh
set -e
BACKBONE=${BACKBONE:?}; VARIANT=${VARIANT:?}; SUITE=${SUITE:?}
SEEDS=${SEEDS:-"1000 2000 3000"}
if [[ $SUITE == *_occluded ]]; then NEPS=${NEPS:-25}; else NEPS=${NEPS:-50}; fi
OUT_ROOT=${OUT_ROOT:-outputs/libero}
cd "$(dirname "$0")/.."
for SEED in $SEEDS; do
  OUT=$OUT_ROOT/${BACKBONE}_${VARIANT}_${SUITE}_s${SEED}
  if [ -f "$OUT/eval_info.json" ]; then echo "skip $OUT"; continue; fi
  python libero_eval/eval_libero.py --backbone "$BACKBONE" --variant "$VARIANT" --suite "$SUITE" \
    --seed "$SEED" --n_episodes "$NEPS" --output_dir "$OUT"
done
