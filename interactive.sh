#!/bin/bash

srun -p vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090 \
    -q vision-pulkitag-debug \
    -t 02:00:00 \
    -N 1 \
    --mem=200G \
    --gres=gpu:1 \
    --pty bash -i

source /data/scratch/rileyis/.bashrc
cd /data/scratch/rileyis/agent-rl/
mamba activate agent-rl-new