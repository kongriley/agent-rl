#!/bin/bash
#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090       # specify the partition
#SBATCH -q vision-pulkitag-debug         # specify the QoS
#SBATCH -t 02:00:00                   # job time
#SBATCH -n 1                       # number of tasks
#SBATCH --gres=gpu:1                  # request GPU resource
#SBATCH --mem=64G                     # total memory
#SBATCH --cpus-per-task=8             # number of CPUs per task
#SBATCH --output=out/%x.%j.out

# GPUS=2

export HOME=/data/scratch/rileyis

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl

# Navigate to your project directory
cd /data/scratch/rileyis/agent-rl/

PORT=8000

echo "starting vllm"
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --port $PORT

# Function to check if vllm is up
function wait_for_vllm {
    iters=50
    for i in $(seq 1 $iters); do
        if lsof -i :$PORT > /dev/null; then
            echo "vllm is up and running!"
            return 0
        fi
        echo "Waiting for vllm to start... ($i/$iters)"
        sleep 5
    done
    echo "vllm failed to start within expected time."
    return 1
}

# Wait for vllm to be ready
wait_for_vllm
if [ $? -ne 0 ]; then
    echo "Exiting due to vllm startup failure."
    exit 1
fi

python train_grpo.py