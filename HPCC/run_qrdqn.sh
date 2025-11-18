#!/bin/bash
#SBATCH --job-name=qrdqn
#SBATCH --output=/home/wamoody/DRLIDS/results/main/output.log
#SBATCH --error=/home/wamoody/DRLIDS/results/main/error.log
#SBATCH --time=12:00:00
#SBATCH --partition=nocona       # CPU-only partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Reduced for faster scheduling
#SBATCH --mem=64G                # Plenty for RL training
# NO GPU REQUEST HERE

# Ensure results directory exists
mkdir -p /home/wamoody/DRLIDS/results/main

# Activate environment
source ~/miniconda3/bin/activate tf1

export PYTHONUNBUFFERED=1

cd /home/wamoody/DRLIDS

python -u mainHpcc.py
