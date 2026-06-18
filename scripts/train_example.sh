#!/usr/bin/env bash
# Single-GPU example: register the example identity's expressions.
set -e
python train.py \
  --idname antelope_toy \
  --neutral halfo_m_o_e \
  --n_views 25 \
  --log 0000 \
  --deform_fc --normalize_mesh --view_independent --use_loss_n
