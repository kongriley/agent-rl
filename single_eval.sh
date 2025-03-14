#!/bin/bash
#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090       # specify the partition
#SBATCH -q vision-pulkitag-debug         # specify the QoS
#SBATCH -t 02:00:00                   # job time
#SBATCH -n 1                       # number of tasks
#SBATCH --gres=gpu:1                  # request GPU resource
#SBATCH --mem=64G                     # total memory
#SBATCH --cpus-per-task=8             # number of CPUs per task
#SBATCH --output=out/%x.%j.out

export HOME=/data/scratch/rileyis

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl

# Navigate to your project directory
cd /data/scratch/rileyis/agent-rl/

python refusal_eval.py /data/scratch/rileyis/agent-rl/results/api_bank/zero-shot/log_1741363144.4690404.txt
