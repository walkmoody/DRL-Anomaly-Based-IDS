import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

import seaborn as sns
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from common import IDSEnvironment, ReplayBuffer, QRDQNAgent
from QRDQNHpcc import train_qr_dqn_agent_batch, test
from tqdm import tqdm


# ------------------------------
# HPC-specific paths
# ------------------------------
BASEDIR = "/home/wamoody/DRLIDS"
BASEDIR2 = "/home/wamoody/DRLIDS"
RESULTS_DIR = f"{BASEDIR}/results/mainHpcc"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_PATH = f"{RESULTS_DIR}/qrdqn_modelHpcc"
TEST_DATA_PATH = f"{RESULTS_DIR}/test_data_scaled.pkl"
TRAIN_COLS_PATH = f"{RESULTS_DIR}/train_columns.pkl"
SCALER_PATH = f"{RESULTS_DIR}/scaler.pkl"


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

# Assume your data is in a DataFrame called df
# and the last column is 'class' with values 'normal' or 'anomaly'

def process_dataset(normalize_ratio=True, warmup_rows=1000):
    train_data_arff, _ = arff.loadarff(f"{BASEDIR2}/NSL-KDD/KDDTrain+.arff")
    test_data_arff, _ = arff.loadarff(f"{BASEDIR2}/NSL-KDD/KDDTest+.arff")
    train_df = pd.DataFrame(train_data_arff)
    test_df = pd.DataFrame(test_data_arff)

    # Decode byte-strings
    train_df = train_df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    test_df = test_df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Drop missing values
    y_train = train_df['class']
    y_test = test_df['class']
    X_train = train_df.drop(columns=['class'])
    X_test = test_df.drop(columns=['class'])

    categorical_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Align test columns with training (some services may differ)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Add label column back
    X_train_scaled['class'] = y_train.values
    X_test_scaled['class'] = y_test.values

    # ------------------------
    # Normalize first `warmup_rows` rows: 25% anomalies / 75% normal
    # ------------------------
    if normalize_ratio:
        subset = X_train_scaled.head(warmup_rows)

        normal_df = subset[subset['class'] == 'normal']
        anomaly_df = subset[subset['class'] != 'normal']

        n_anomaly = len(anomaly_df)
        n_normal = int(n_anomaly * 3)  # 75% normal, 25% anomaly

        normal_sampled = normal_df.sample(n=min(n_normal, len(normal_df)), random_state=42)
        normalized_subset = pd.concat([normal_sampled, anomaly_df]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Replace the first `warmup_rows` with normalized subset
        X_train_scaled.iloc[:len(normalized_subset)] = normalized_subset

    print("Class distribution after normalization (subset):")
    print(X_train_scaled.head(warmup_rows)['class'].value_counts())
    print(X_train_scaled.head(3).iloc[:, -3:])  # Last 3 columns

    return X_train_scaled, X_test_scaled, scaler


# Run dataset processing
train_data, test_data, scaler = process_dataset()


# Save both the scaler and the one-hot encoded column structure
joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))
joblib.dump(train_data.columns.tolist(), os.path.join(RESULTS_DIR, "train_columns.pkl"))
joblib.dump(test_data, os.path.join(RESULTS_DIR, "test_data_scaled.pkl"))
print("Saved test_data_scaled.pkl")

print(f"Saved scaler and train_columns to {RESULTS_DIR}")

# ------------------------------
# Plotting
# ------------------------------
def visualize_training_results(rewards, save_path):
    moving_avg = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    plt.plot(moving_avg, label='Moving Avg', color='red')
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

    print("Warming up replay buffer with random actions...")

    replay_buffer = ReplayBuffer(capacity=10000)
    for _ in range(1000):  # 1000 random transitions
        state = train_env.reset()
        action = train_env.action_space.sample()
        next_state, reward, done, _ = train_env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        if done:
            train_env.reset()
            
    print("Replay buffer warm-up complete.")

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

    print("Finished successfully.")

'''

if __name__ == '__main__':
    
    print("Starting Test-only Run", flush=True)

    # Initialize test environment
    print("Loading test dataset and scaler...")
    test_data = joblib.load(TEST_DATA_PATH)
    train_columns = joblib.load(TRAIN_COLS_PATH)
    scaler = joblib.load(SCALER_PATH)

    y_test = test_data['class']
    X_test = test_data.drop(columns=['class'])
    X_test = X_test.reindex(columns=[c for c in train_columns if c != 'class'], fill_value=0)

    test_data = X_test.copy()
    test_data['class'] = y_test.values

    print(f"Test data shape: {test_data.shape}")


    test_env = IDSEnvironment(test_data, train=False)

    # Load trained agent
    print(f"Loading model from: {MODEL_PATH}")
    agent = QRDQNAgent(state_size=test_env.num_features, action_size=test_env.action_space.n)
    agent.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    agent.target_model = agent.model  # sync target network

    # Run test episodes
    NUM_EPISODES = 50
    print(f"Running {NUM_EPISODES} test episodes...")
    results, test_rewards = test(agent, test_env, num_episodes=NUM_EPISODES)

    # Visualize rewards
    visualize_training_results(test_rewards,
                               save_path=os.path.join(RESULTS_DIR, "qr_dqn_test_rewards.png"))

    # Compute accuracy / precision / recall
    print("Computing accuracy metrics...")
    y_true, y_pred = [], []
    NUM_EVAL_EPISODES = 20
    for episode in tqdm(range(NUM_EVAL_EPISODES), desc="Evaluating"):
        state = test_env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = test_env.step(action)
            idx = test_env.current_data_pointer - 1
            if idx < 0:
                idx = 0

            # Handle both encoded (numeric) or string-based labels
            true_label = test_env.dataset.iloc[idx, -1]
            if isinstance(true_label, (float, int)):
                y_true.append(int(true_label))
            else:
                y_true.append(1 if str(true_label).lower() == "anomaly" else 0)

            y_pred.append(action)
            state = next_state

    print("\nEvaluation Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred))
    print("Test-only run complete.")

    '''