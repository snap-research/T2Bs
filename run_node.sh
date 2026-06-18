torchrun --standalone --nproc_per_node=8 train_multi.py \
  --n_views 25 --run_log 0000 --deform_fc --normalize_mesh \
  --neutral halfo_m_o_e --use_loss_n

torchrun --standalone --nproc_per_node=8 train_multi.py \
  --n_views 25 --run_log 0001 --normalize_mesh \
  --neutral halfo_m_o_e --use_loss_n

torchrun --standalone --nproc_per_node=8 train_multi.py --n_views 25 --run_log 0050 --normalize_mesh --neutral halfo_m_o_e --view_independent