# ================================================
# Local simulation of SLURM run for DRLIDS project
# ================================================

# 1️⃣ Activate your conda environment
# (Make sure you've already done: conda init powershell)
conda activate tf1

# 2️⃣ Define paths
$project = "C:\DRL-Anomaly-Based-IDS"
$results = "$project\Local\results\mainHpcc"
$logFile = "$results\output.log"
$backup  = "C:\Users\walke\results_backup"

# Ensure results and backup folders exist
New-Item -ItemType Directory -Force -Path $results | Out-Null
New-Item -ItemType Directory -Force -Path $backup | Out-Null

# 3️⃣ Run the main script
Write-Host "Starting DRLIDS local test run..."
python "$project\Local\mainHpcc.py" *> $logFile

# 4️⃣ Optional: copy log or results to backup
Copy-Item -Recurse -Force $results $backup

Write-Host "✅ Run complete. Output and logs saved to:"
Write-Host "   Results: $results"
Write-Host "   Backup : $backup"
