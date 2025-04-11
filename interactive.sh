#!/bin/bash

# Request resources
srun -p vision-pulkitag-3090 \
    -q vision-pulkitag-debug \
    -t 02:00:00 \
    -w improbablex[001-009] \
    -N 1 \
    --mem=32G \
    --gres=gpu:1 \
    --pty bash --rcfile <(cat <<EOF
source /data/scratch/rileyis/.bashrc
export HOME=/data/scratch/rileyis/
cd /data/scratch/rileyis/agent-rl/
# mamba activate agent-rl
export PATH="$HOME/.local/bin:$PATH"
EOF
)
