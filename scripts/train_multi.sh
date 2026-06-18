#!/usr/bin/env bash
# Multi-GPU example via torchrun. Processes all identities under --data_root,
# split across ranks. Set --nproc_per_node to your GPU count.
set -e
torchrun --standalone --nproc_per_node=8 train_multi.py \
  --data_root assets \
  --neutral halfo_m_o_e \
  --n_views 25 \
  --run_log 0000 \
  --deform_fc --normalize_mesh --use_loss_n
