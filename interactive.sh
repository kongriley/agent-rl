#!/bin/bash

# Request resources
srun -p vision-pulkitag-3090 \
    -q vision-pulkitag-debug \
    -t 02:00:00 \
    -w improbablex[001-009] \
    -N 1 \
    --mem=32G \
    --gres=gpu:1 \
    --pty bash -c '
    # Set HOME path
    export HOME=/data/scratch/rileyis/agent-rl/
    
    # Source bashrc
    source /data/scratch/rileyis/.bashrc
    
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    mamba activate agent-rl
    
    # Start interactive shell
    exec bash -i
    '
