import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from common import IDSEnvironment, ReplayBuffer, QRDQNAgent
from QRDQNHpcc import train_qr_dqn_agent_batch, test

# ------------------------------
# HPC-specific paths
# ------------------------------
BASEDIR = "/home/wamoody/DRLIDS"
RESULTS_DIR = f"{BASEDIR}/results/mainHpcc"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------
# TensorFlow / GPU setup
# ------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("NOT using GPU")

# ------------------------------
# Dataset processing
# ------------------------------
def process_dataset():
    train_data_arff, _ = arff.loadarff(f"{BASEDIR}/NSL-KDD/KDDTrain+.arff")
    test_data_arff, _ = arff.loadarff(f"{BASEDIR}/NSL-KDD/KDDTest+.arff")
    
    train_data = pd.DataFrame(train_data_arff)
    test_data = pd.DataFrame(test_data_arff)

    # Decode byte-strings
    for df in [train_data, test_data]:
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].str.decode('utf-8')
    
    # Drop missing values
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # Normalize numerical features
    numerical_cols = [0,4,5,7,8,9,10,12,13,14,15,16,17,18,19,
                      22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    scaler = StandardScaler()
    train_data.iloc[:, numerical_cols] = scaler.fit_transform(train_data.iloc[:, numerical_cols])
    test_data.iloc[:, numerical_cols] = scaler.transform(test_data.iloc[:, numerical_cols])

    # One-hot encode categorical features
    categorical_cols = [1,2,3,6,11,20,21,41]
    train_data = pd.get_dummies(train_data, columns=train_data.columns[categorical_cols])
    test_data = pd.get_dummies(test_data, columns=test_data.columns[categorical_cols])
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

    return train_data, test_data

train_data, test_data = process_dataset()
# ------------------------------
# Plotting
# ------------------------------
def visualize_training_results(rewards, save_path):
    moving_avg = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    plt.plot(moving_avg, label='Moving Avg (100)', color='red')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ------------------------------
# Main HPC training/testing
# ------------------------------
if __name__ == '__main__':
    # Train

    print("Start Env")
    train_env = IDSEnvironment(train_data, train=True)
    start_time = time.time()
    training_rewards, agent = train_qr_dqn_agent_batch(train_env)
    print(f"Training finished in {time.time() - start_time:.2f}s")

    visualize_training_results(training_rewards, save_path=f"{RESULTS_DIR}/qr_dqn_train_rewards.png")

    # Save model
    save_path = f"{RESULTS_DIR}/qrdqn_modelHpcc"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.model.save(save_path)

    # Test
    test_env = IDSEnvironment(test_data, train=False)
    results, test_rewards = test(agent, test_env, num_episodes=50)
    visualize_training_results(test_rewards, save_path=f"{RESULTS_DIR}/qr_dqn_test_rewards.png")

    # Compute metrics manually for first 20 episodes
    y_true, y_pred = [], []
    for episode in range(100):
        state = test_env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, _, done, _ = test_env.step(action)
            idx = test_env.current_data_pointer - 1
            if idx < 0: idx = 0
            true_label = test_env.dataset.iloc[idx, -1]
            y_true.append(1 if str(true_label).lower() == 'anomaly' else 0)
            y_pred.append(action)
            state = next_state

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred))
    print("Finished successfully.")