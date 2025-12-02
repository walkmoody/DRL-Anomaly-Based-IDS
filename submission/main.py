# main_qrdqn.py
import os
import numpy as np
import tensorflow as tf
import joblib
from scipy.io import arff
import pandas as pd

from common import IDSEnvironment
from QRDQN import QRDQNAgent, test

# PATHS
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES_DIR = os.path.join(BASEDIR, "dependencies")

SCALER_PATH = os.path.join(DEPENDENCIES_DIR, "scaler.pkl")
TRAIN_COLS_PATH = os.path.join(DEPENDENCIES_DIR, "train_columns.pkl")
TEST_ARFF_PATH = os.path.join(DEPENDENCIES_DIR, "KDDTest+ copy.arff")

# load data

def process_dataset():

    test_raw, _ = arff.loadarff(TEST_ARFF_PATH)
    test_df = pd.DataFrame(test_raw)

    # Decode byte strings
    test_df = test_df.applymap(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    # Normal vs anomaly
    test_df["class"] = test_df["class"].apply(
        lambda x: "normal" if str(x).lower().startswith("normal") else "anomaly"
    )

    y = test_df["class"]
    X = test_df.drop(columns=["class"])

    categorical = [
        "protocol_type", "service", "flag",
        "land", "logged_in", "is_host_login", "is_guest_login"
    ]

    X = pd.get_dummies(X, columns=categorical)

    # Align columns
    train_columns = joblib.load(TRAIN_COLS_PATH)
    train_columns = [c for c in train_columns if c != "class"]

    X = X.reindex(columns=train_columns, fill_value=0)

    scaler = joblib.load(SCALER_PATH)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=train_columns)

    # Add class column back
    X_scaled["class"] = y.values

    return X_scaled

# Main
if __name__ == "__main__":
    print("QRDQN TEST-ONLY MODE")

    print("Loading test dataset...")
    test_data = process_dataset()

    print("Initializing test environment...")
    test_env = IDSEnvironment(test_data, train=False)

    print(f"State size = {test_env.num_features}")
    print(f"Action size = {test_env.action_space.n}")

    tf.keras.backend.clear_session()

    print("Loading QRDQN weights...")
    meta = joblib.load(os.path.join(DEPENDENCIES_DIR, "qrdqn_meta.pkl"))
    num_quantiles = meta["num_quantiles"]

    # Rebuild agent architecture
    agent = QRDQNAgent(
        state_size=test_env.num_features,
        action_size=test_env.action_space.n,
        num_quantiles=num_quantiles,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
    )

    # Build model
    dummy = np.zeros((1, test_env.num_features), dtype=np.float32)
    agent.model(dummy)
    agent.target_model(dummy)

    # Load weights
    agent.model.load_weights(os.path.join(DEPENDENCIES_DIR, "qrdqn_weights.h5"))
    agent.target_model.load_weights(os.path.join(DEPENDENCIES_DIR, "qrdqn_target_weights.h5"))

    print("Weights loaded successfully!")

    agent.epsilon = 0.0

    results, rewards = test(agent, test_env, num_episodes=1)


    print("\nTest-Only Run Complete")
