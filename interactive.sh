srun -p vision-pulkitag-3090 \
    -q vision-pulkitag-debug \
    -t 02:00:00 \
    -w improbablex[001-009] \
    -N 1 \
    --mem=32G \
    --gres=gpu:1 \
    --pty bash -i

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl

# Navigate to your project directory
export HOME=/data/scratch/rileyis/agent-rl/