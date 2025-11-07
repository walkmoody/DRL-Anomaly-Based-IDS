#!/bin/bash

#SBATCH --job-name=Main_IDS_experiment
#SBATCH --partition=t4v2,rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-main-%j.out
#SBATCH --error=slurm-main-%j.err

# prepare your environment here
# Simulated SLURM run on Windows

# Activate your Anaconda env
conda activate tf1

# Run your script
python "C:\DRL-Anomaly-Based-IDS\main.py"

# Copy results
# Copy-Item -Recurse ".\results\main" "C:\Users\walke\results_backup"
