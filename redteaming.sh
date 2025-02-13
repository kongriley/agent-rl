#!/bin/bash
#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090       # specify the partition
#SBATCH -q vision-pulkitag-debug         # specify the QoS
#SBATCH -t 02:00:00                   # job time
#SBATCH -n 1                       # number of tasks
#SBATCH --gres=gpu:2                  # request GPU resource
#SBATCH --mem=64G                     # total memory
#SBATCH --cpus-per-task=8             # number of CPUs per task
#SBATCH --output=out/%x.%j.out

GPUS=2

# Source your bashrc
source /data/scratch/rileyis/.bashrc

# Activate your conda environment
mamba activate agent-rl

# Navigate to your project directory
export HOME=/data/scratch/rileyis/
cd /data/scratch/rileyis/agent-rl/

# Default values
PORT=8000
REDTEAM_SCALE=1.5
VICTIM_SCALE=1.5

# Parse named parameters
while getopts "p:r:v:" opt; do
  case $opt in
    p) PORT=$OPTARG ;;
    r) REDTEAM_SCALE=$OPTARG ;;
    v) VICTIM_SCALE=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

if ! lsof -i :$PORT > /dev/null; then
    echo "starting up vllm"
    if [[ $REDTEAM_SCALE == $VICTIM_SCALE ]]; then
        vllm serve Qwen/Qwen2.5-${REDTEAM_SCALE}B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --port $PORT --enforce_eager &
    else
        vllm serve Qwen/Qwen2.5-${REDTEAM_SCALE}B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --port $PORT --gpu-memory-utilization 0.4 --tensor-parallel-size $GPUS &
        vllm serve Qwen/Qwen2.5-${VICTIM_SCALE}B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --port $(($PORT + 1)) --gpu-memory-utilization 0.4 --tensor-parallel-size $GPUS &
    fi
    sleep 60
fi

# Function to check if vllm is up
function wait_for_vllm {
    for i in {1..50}; do
        if lsof -i :$PORT > /dev/null && { [[ $REDTEAM_SCALE == $VICTIM_SCALE ]] || lsof -i :$(($PORT + 1)) > /dev/null; }; then
            echo "vllm is up and running!"
            return 0
        fi
        echo "Waiting for vllm to start... ($i/50)"
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

python base_testing.py --n-iters 500 --suite-name workspace --redteam-scale $REDTEAM_SCALE --victim-scale $VICTIM_SCALE

