#!/bin/bash
#SBATCH -p csail-shared       # specify the partition
#SBATCH -q lab-free      # specify the QoS
#SBATCH -t 02:00:00                   # job time
#SBATCH --gres=gpu:1                  # request GPU resource
#SBATCH --mem=32G                     # total memory
#SBATCH --cpus-per-task=8             # number of CPUs per task
#SBATCH --output=out/%x.%j.out

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl

# Navigate to your project directory
export HOME=/data/scratch/rileyis/
cd /data/scratch/rileyis/agent-rl/

# Run your Python script 
vllm serve Qwen/Qwen2.5-3B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --port 8000 &
sleep 30

# Function to check if vllm is up
function wait_for_vllm {
    for i in {1..24}; do
        if lsof -i :8000 > /dev/null; then
            echo "vllm is up and running!"
            return 0
        fi
        echo "Waiting for vllm to start... ($i/24)"
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

python redteaming.py
