#!/bin/bash
#SBATCH -p vision-pulkitag-a100,vision-pulkitag-h100        # specify the partition
#SBATCH -q vision-pulkitag-main       # specify the QoS
#SBATCH -t 2-00:00:00                 # job time
#SBATCH --gres=gpu:1                  # request GPU resource
#SBATCH --mem=48G                     # total memory
#SBATCH --cpus-per-task=8             # number of CPUs per task

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl

# Navigate to your project directory
export HOME=/data/scratch/rileyis/
cd /data/scratch/rileyis/

# Run your Python script 
vllm serve Qwen/Qwen2.5-3B-Instruct --enable-auto-tool-choice --tool-call-parser hermes &