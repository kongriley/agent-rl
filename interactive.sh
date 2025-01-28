srun -p vision-pulkitag-3090,vision-pulkitag-2080 \
    -q vision-pulkitag-debug \
    -t 02:00:00 \
    --mem 32G \
    --gres=gpu:1 \
    --pty bash -i

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl

# Navigate to your project directory
export HOME=/data/scratch/rileyis/agent-rl/