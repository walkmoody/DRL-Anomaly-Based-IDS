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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from commonIQN import train_iqn_agent, test_iqn_agent, IDSEnvironment, IQNAgent
from tqdm import tqdm


# ------------------------------
# HPC-specific paths
# ------------------------------
BASEDIR = "/home/wamoody/DRLIDS"
BASEDIR2 = "/home/wamoody/DRLIDS"
RESULTS_DIR = f"{BASEDIR}/results/mainHpcc"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_PATH = f"{RESULTS_DIR}/iqn_modelHpcc"
TEST_DATA_PATH = f"{RESULTS_DIR}/test_data_scaled.pkl"
TRAIN_COLS_PATH = f"{RESULTS_DIR}/train_columns.pkl"
SCALER_PATH = f"{RESULTS_DIR}/scaler.pkl"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("Using GPU")
    # Enable memory growth to avoid TF pre-allocating all GPU memory
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
else:
    print("NOT using GPU")

# ------------------------------
# Dataset processing
# ------------------------------

# Assume your data is in a DataFrame called df
# and the last column is 'class' with values 'normal' or 'anomaly'

def process_dataset():
    train_data_arff, _ = arff.loadarff(f"{BASEDIR2}/NSL-KDD/KDDTrain+.arff")
    test_data_arff, _ = arff.loadarff(f"{BASEDIR2}/NSL-KDD/KDDTest+.arff")
    train_df = pd.DataFrame(train_data_arff)
    test_df = pd.DataFrame(test_data_arff)

    # Decode bytes
    train_df = train_df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    test_df = test_df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    y_train = train_df['class']
    y_test = test_df['class']
    X_train = train_df.drop(columns=['class'])
    X_test = test_df.drop(columns=['class'])

    categorical_cols = ['protocol_type', 'service', 'flag', 'land',
                        'logged_in', 'is_host_login', 'is_guest_login']

    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Add labels back before shuffle
    X_train_scaled['class'] = y_train.values
    X_test_scaled['class'] = y_test.values

    print(X_train_scaled['class'].value_counts())
    print(X_train_scaled.head(3).iloc[:, -3:])

    # ✔️ Shuffle AFTER adding labels
    train_df_shuffled = X_train_scaled.sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df_shuffled, X_test_scaled, scaler


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

if __name__ == "__main__":
    # ------------------------------
    # Dataset and environment setup
    # ------------------------------
    print("Loading and processing dataset...")
    train_data, test_data, scaler = process_dataset()

    # Save scaler and train column structure
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))
    joblib.dump(train_data.columns.tolist(), os.path.join(RESULTS_DIR, "train_columns.pkl"))
    joblib.dump(test_data, os.path.join(RESULTS_DIR, "test_data_scaled.pkl"))
    print("Scaler, train columns, and test data saved.")

    # Initialize environments
    train_env = IDSEnvironment(train_data, train=True)
    test_env = IDSEnvironment(test_data, train=False)

    # ------------------------------
    # Train IQN Agent
    # ------------------------------
    print("Starting training...")
    start_time = time.time()
    training_rewards, agent = train_iqn_agent(train_env)
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    # Save training rewards plot
    visualize_training_results(training_rewards, save_path=f"{RESULTS_DIR}/IQN_train_rewards.png")

    # Save trained model
    try:
        agent.model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print("Warning: failed to save model:", e)

    # ------------------------------
    # Test IQN Agent
    # ------------------------------
    agent.epsilon = 0.0  # Fully greedy for testing
    print("Starting testing...")
    test_metrics = test_iqn_agent(agent, test_env, num_episodes=50)

    # Save test rewards plot if present
    if "rewards" in test_metrics:
        visualize_training_results(
            test_metrics["rewards"],
            save_path=f"{RESULTS_DIR}/IQN_test_rewards.png"
        )

    # Print test metrics
    print("===== Test Metrics =====")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print("Confusion Matrix:\n", test_metrics["confusion_matrix"])
    print("===== Run Complete =====")


'''
# TEST-ONLY MODE (active)
if __name__ == '__main__':
    print("===== Starting Test-Only Mode =====", flush=True)

    # Initialize test environment
    print("Initializing test environment...")
    test_env = IDSEnvironment(test_data, train=False)
    
    # Load trained IQN agent
    tf.keras.backend.clear_session()
    print(f"Loading IQN model from: {MODEL_PATH}")
    agent = IQNAgent(state_size=test_env.num_features, action_size=test_env.action_space.n)
    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        agent.model = loaded_model
        # Use a separate target model (copy weights) for safety
        try:
            agent.target_model = tf.keras.models.clone_model(loaded_model)
            agent.target_model.set_weights(loaded_model.get_weights())
        except Exception:
            agent.target_model = loaded_model
        print("Model loaded successfully.")
    except Exception as e:
        print("Warning: failed to load model:", e)
        print("Proceeding with untrained agent.")
    
    # Evaluate agent with greedy policy
    agent.epsilon = 0.0  # Fully greedy for testing
    print("Starting testing...")
    test_metrics = test_iqn_agent(agent, test_env, num_episodes=50)

    # Print test metrics
    print("===== Test Metrics =====")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print("Confusion Matrix:\n", test_metrics["confusion_matrix"])
    print("===== Test-Only Run Complete =====")

'''