#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/nfs/usr/jluo2/trellis2_gpt_textured_test"
DST_ROOT="/nfs/usr/jluo2/trellis2_gpt_textured_test_t2bs"
# SRC_ROOT="/nfs/usr/jluo2/trellis2_gpt_5kto10k_textured_test"
# DST_ROOT="/nfs/usr/jluo2/trellis2_gpt_5kto10k_textured_test_t2bs"

# Loop over identities
for id_dir in "$SRC_ROOT"/*; do
  [ -d "$id_dir" ] || continue
  identity="$(basename "$id_dir")"

  # Loop over expressions
  for exp_dir in "$id_dir"/*; do
    [ -d "$exp_dir" ] || continue
    expression="$(basename "$exp_dir")"

    dst="$DST_ROOT/$identity/obj/$expression"
    mkdir -p "$dst"

    # Copy all files/folders inside the expression folder into .../obj
    cp -a "$exp_dir"/. "$dst"/
  done
done

echo "Done."