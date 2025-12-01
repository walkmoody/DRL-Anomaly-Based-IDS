# main_qrdqn.py
import os
import numpy as np
import tensorflow as tf
import joblib
from scipy.io import arff
import pandas as pd

from common import IDSEnvironment, QRDQNAgent
from QRDQNHpcc import test, visualize_training_results


# PATHS
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES_DIR = os.path.join(BASEDIR, "dependencies")

MODEL_PATH = os.path.join(DEPENDENCIES_DIR, "qrdqn_modelHpcc")
SCALER_PATH = os.path.join(DEPENDENCIES_DIR, "scaler.pkl")
TRAIN_COLS_PATH = os.path.join(DEPENDENCIES_DIR, "train_columns.pkl")
TEST_ARFF_PATH = os.path.join(DEPENDENCIES_DIR, "KDDTest+.arff")
RESULTS_DIR = os.path.join(BASEDIR, "results")

# LOAD + PREPROCESS TEST SET (only)

def load_test_dataset():

    test_raw, _ = arff.loadarff(TEST_ARFF_PATH)
    test_df = pd.DataFrame(test_raw)

    # Decode bytes
    test_df = test_df.applymap(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    # Convert all attack names -> anomaly
    test_df["class"] = test_df["class"].apply(
        lambda x: "normal" if str(x).lower().startswith("normal") else "anomaly"
    )

    y = test_df["class"]
    X = test_df.drop(columns=["class"])

    categorical = [
        "protocol_type","service","flag",
        "land","logged_in","is_host_login","is_guest_login"
    ]

    X = pd.get_dummies(X, columns=categorical)

    # Ensure same columns as training
    train_columns = joblib.load(TRAIN_COLS_PATH)
    train_columns = [c for c in train_columns if c != "class"]

    X = X.reindex(columns=train_columns, fill_value=0)

    scaler = joblib.load(SCALER_PATH)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=train_columns)

    # Add class column back
    X_scaled["class"] = y.values

    return X_scaled

if __name__ == "__main__":
    print("QRDQN TEST-ONLY MODE")

    print("Loading test dataset...")
    test_data = load_test_dataset()

    print("Initializing test environment...")
    test_env = IDSEnvironment(test_data, train=False)

    print(f"State size = {test_env.num_features}")
    print(f"Action size = {test_env.action_space.n}")

    
    print("Rebuilding QRDQN agent architecture...")
    agent = QRDQNAgent(
        state_size=test_env.num_features,
        action_size=test_env.action_space.n,
        num_quantiles=51,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97
    )
    print("Agent architecture rebuilt.")
    
    tf.keras.backend.clear_session()
    print(f"Loading trained QRDQN model from {MODEL_PATH} ...")

    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")

        # overwrite agent's internal model
        agent.model = loaded_model

        # Copy weights into target_model
        try:
            agent.target_model = tf.keras.models.clone_model(loaded_model)
            agent.target_model.set_weights(loaded_model.get_weights())
        except:
            agent.target_model = loaded_model

    except Exception as e:
        print("ERROR loading model:", e)
        exit()

    # -------------------------------------------------
    # GREEDY EVALUATION
    # -------------------------------------------------
    print("Running greedy evaluation...")
    agent.epsilon = 0.0

    results, rewards = test(agent, test_env, num_episodes=1)

    # -------------------------------------------------
    # SAVE REWARD PLOT
    # -------------------------------------------------
    visualize_training_results(
        rewards,
        save_path=os.path.join(RESULTS_DIR, "QRDQN_test_rewards.png"),
    )

    print("\n===== FINAL METRICS =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n===== Test-Only Run Complete =====")
