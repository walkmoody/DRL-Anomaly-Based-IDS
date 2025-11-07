# Simulated SLURM run on Windows

# 1️⃣ Activate your Anaconda environment
# PowerShell scripts don’t automatically recognize `conda activate`
# unless you initialize conda for PowerShell first. 
# Run this once manually in PowerShell if you haven't yet:
#     conda init powershell
# Then restart PowerShell and the following works:
conda activate tf1

# 2️⃣ Define paths
$project = "C:\DRL-Anomaly-Based-IDS"
$results = "$project\results\main"
$backup  = "C:\Users\walke\results_backup"

New-Item -ItemType Directory -Force -Path $results | Out-Null

python "$project\main2.py" *> "$results\output.log"

Write-Host "Run complete. Output saved to $results and copied to $backup"

# powershell -ExecutionPolicy Bypass -File "C:\DRL-Anomaly-Based-IDS\run_main.ps1"