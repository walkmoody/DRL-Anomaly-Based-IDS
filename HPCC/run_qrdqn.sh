#!/bin/bash
#SBATCH --job-name=qrdqn
#SBATCH --output=/home/wamoody/DRLIDS/results/main/output.log
#SBATCH --error=/home/wamoody/DRLIDS/results/main/error.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=toreador

# Ensure results directory exists
mkdir -p /home/wamoody/DRLIDS/results/main

# Activate environment with all dependencies
source ~/miniconda3/bin/activate tf1

# Force Python to flush output immediately (no buffering)
export PYTHONUNBUFFERED=1

# Navigate to project folder
cd /home/wamoody/DRLIDS

# Run your script
python -u mainHpcc.py
