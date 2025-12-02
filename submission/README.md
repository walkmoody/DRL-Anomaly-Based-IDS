# DRL-Based Anomaly Detection System (QR-DQN)

This project implements a Deep Reinforcement Learning anomaly detection model using the NSL-KDD dataset. The upgraded QR-DQN architecture improves stability, handles class imbalance, and achieves significantly higher anomaly recall than the baseline code from the public GitHub repository.

## Environment Setup
(Refer to the video walkthrough if provided for a full setup demonstration.)

This project requires **Python 3.7**.  
Below is the recommended Conda setup:

conda create -n drl python=3.7
conda activate drl

pip install numpy==1.21.6 tensorflow==2.10.1 matplotlib==3.5.3 scikit-learn==1.0.2 seaborn==0.11.2 gym==0.26.2 pandas==1.3.5 scipy==1.7.3 tqdm scapy

running the project

python main.py