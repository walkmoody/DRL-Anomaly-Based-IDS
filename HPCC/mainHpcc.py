# main_qrdqn.py
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from common import IDSEnvironment
from QRDQNHpcc import train_qr_dqn_agent, test, visualize_training_results

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print(
    "Using GPU"
    if tf.config.list_physical_devices("GPU")
    else "NOT using GPU"
)


# ------------------------------
# Dataset processing
# ------------------------------
def process_dataset():
    train_data_arff, _ = arff.loadarff(f"{BASEDIR2}/NSL-KDD/KDDTrain+.arff")
    test_data_arff, _ = arff.loadarff(f"{BASEDIR2}/NSL-KDD/KDDTest+.arff")

    train_df = pd.DataFrame(train_data_arff)
    test_df = pd.DataFrame(test_data_arff)

    # Decode bytes to str
    train_df = train_df.applymap(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )
    test_df = test_df.applymap(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    y_train = train_df["class"]
    y_test = test_df["class"]
    X_train = train_df.drop(columns=["class"])
    X_test = test_df.drop(columns=["class"])

    categorical_cols = [
        "protocol_type",
        "service",
        "flag",
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
    ]

    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    # add labels back
    X_train_scaled["class"] = y_train.values
    X_test_scaled["class"] = y_test.values

    # shuffle train
    train_df_shuffled = (
        X_train_scaled.sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return train_df_shuffled, X_test_scaled, scaler


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # ---------- Data ----------
    print("Processing dataset...")
    train_data, test_data, scaler = process_dataset()

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(train_data.columns.tolist(), TRAIN_COLS_PATH)
    joblib.dump(test_data, TEST_DATA_PATH)
    print(
        f"Saved scaler, train_columns, and test_data_scaled.pkl to {RESULTS_DIR}"
    )

    # ---------- Train ----------
    print("Initializing training environment...")
    train_env = IDSEnvironment(train_data, train=True)

    print("Beginning QRDQN training...")
    start_time = time.time()
    training_rewards, agent = train_qr_dqn_agent(
        train_env,
        num_episodes=150,
        batch_size=64,
        gamma=0.99,
        warmup_size=2000,
        train_every=2,
        update_target_every=1000,
    )
    print(f"Training finished in {time.time() - start_time:.2f}s")

    visualize_training_results(
        training_rewards,
        save_path=f"{RESULTS_DIR}/QRDQN_train_rewards.png",
    )

    # ---------- Save Model (Correct for QRDQN) ----------
    print("Saving QRDQN agent...")

    # Save weights
    agent.model.save_weights(f"{RESULTS_DIR}/qrdqn_weights.h5")
    agent.target_model.save_weights(f"{RESULTS_DIR}/qrdqn_target_weights.h5")

    # Save metadata (architecture info)
    model_meta = {
        "state_size": train_env.num_features,
        "action_size": train_env.action_space.n,
        "num_quantiles": agent.num_quantiles,
    }
    joblib.dump(model_meta, f"{RESULTS_DIR}/qrdqn_meta.pkl")

    print("Saved: qrdqn_weights.h5, qrdqn_target_weights.h5, qrdqn_meta.pkl")


    # ---------- Test ----------
    print("Initializing test environment...")
    test_env = IDSEnvironment(test_data, train=False)

    print("Evaluating QRDQN agent...")
    results, test_rewards = test(agent, test_env, num_episodes=50)
    visualize_training_results(
        test_rewards,
        save_path=f"{RESULTS_DIR}/QRDQN_test_rewards.png",
    )

    print("Finished successfully.")
