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

# Default values
REDTEAM_SCALE=1.5
VICTIM_SCALE=1.5

# Parse named parameters
while getopts "r:v:" opt; do
  case $opt in
    r) REDTEAM_SCALE=$OPTARG ;;
    v) VICTIM_SCALE=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

python base_testing.py --n-iters 500 --suite-name slack --redteam-scale $REDTEAM_SCALE --victim-scale $VICTIM_SCALE --mode zero-shot

