#!/bin/bash
#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000       # specify the partition
#SBATCH -q vision-pulkitag-debug         # specify the QoS
#SBATCH -t 2:00:00                   # job time
#SBATCH -n 1                       # number of tasks
#SBATCH --gres=gpu:2                  # request GPU resource
#SBATCH --mem=200G
#SBATCH --output=out/%x.%j.out

export HOME=/data/scratch/rileyis

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl-new

# Navigate to your project directory
cd /data/scratch/rileyis/agent-rl/

PORT=8000

# Function to check if vllm is up
function wait_for_vllm {
    iters=$1
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

# echo "starting vllm"
# trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --port $PORT --gpu_memory_utilization 0.45 &

# # Wait for vllm to be ready
# wait_for_vllm 100
# if [ $? -ne 0 ]; then
#     echo "Exiting due to vllm startup failure."
#     exit 1
# fi

accelerate launch --config_file grpo_config.yaml train_grpo.py