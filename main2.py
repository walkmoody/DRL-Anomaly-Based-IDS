import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from typing import List, Optional, Dict, Any, Tuple
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff
import pandas as pd
from QRDQN import QRDQNAgent, train_qr_dqn_agent, test, train_qr_dqn_agent_batch
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sys
import time
from tqdm import tqdm

BASEDIR = r"C:\DRL-Anomaly-Based-IDS"

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU")
else:
    print("TensorFlow is NOT using GPU")


pd.set_option('display.max_rows', None)  # show all rows
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)  # don't wrap lines
pd.set_option('display.max_colwidth', None)  # full text in cells


def discretize(value, bins):
    return np.digitize(value, bins, right=True)

def process_dataset():
    train_data_arff, meta_train = arff.loadarff(fr'{BASEDIR}\NSL-KDD\KDDTrain+.arff') # train data
    test_data_arff, meta_test = arff.loadarff(fr'{BASEDIR}\NSL-KDD/KDDTest+.arff')

    # Convert ARFF data to pandas DataFrame
    train_data = pd.DataFrame(train_data_arff)
    test_data = pd.DataFrame(test_data_arff)

    # Convert byte-strings to strings for all columns
    for df in [train_data, test_data]:
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].str.decode('utf-8')

    # 1. Handling Missing Values
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # 2. Normalize Numerical Features
    numerical_cols = [0, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16,
                      17, 18, 19, 22, 23, 24, 25, 26, 27, 28,
                      29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    scaler = StandardScaler()
    train_data.iloc[:, numerical_cols] = scaler.fit_transform(train_data.iloc[:, numerical_cols])
    test_data.iloc[:, numerical_cols] = scaler.transform(test_data.iloc[:, numerical_cols])

    # 3. Convert Categorical Features
    categorical_cols = [1, 2, 3, 6, 11, 20, 21, 41]
    train_data = pd.get_dummies(train_data, columns=train_data.columns[categorical_cols])
    test_data = pd.get_dummies(test_data, columns=test_data.columns[categorical_cols])

    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

    return (train_data, test_data)

train_data, test_data = process_dataset()

class IDSEnvironment(gym.Env):

    def __init__(self, train = True):
        print(f'IDSEnvironment INIT (train={train})')
        super().__init__()
        
        self.train = train
        self.dataset = train_data if train else test_data
        self.num_features = self.dataset.shape[1] - 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.current_data_pointer = 0
        self.state = self.dataset.iloc[self.current_data_pointer, :-1].values

    def discretize_state(self, state):
        # Define bins for each feature
        return np.clip((state*10).astype(int), 0, 9) # already normalized data so this workrs

    def step(self, curr_action):
        intrusion = self.dataset.iloc[self.current_data_pointer, -1]
        self.state = self.discretize_state(self.dataset.iloc[self.current_data_pointer, :-1].values)
        
        correct_action = 1 if intrusion == 'anomaly' else 0
        
        if self.train:
            reward = 1.0 if curr_action == correct_action else -0.2
            curr_reward = np.random.normal(loc=reward, scale=0.1) # randomness during training
        else:
            curr_reward = 1.0 if curr_action == correct_action else -0.2  # deterministic during testing

        self.current_data_pointer += 1
        done = self.current_data_pointer >= min(100, len(self.dataset))

        return self.state, curr_reward, done, {}


    def reset(self, *args, **kwargs):

        self.state = self.discretize_state(self.dataset.iloc[0, :-1].values)

        self.current_data_pointer = 0

        seed = kwargs.get('seed', None)
        options = kwargs.get('options', None)

        if seed is not None:
            np.random.seed(seed)

        # Handle the options as needed...

        return self.state

    def render(self, mode='human'):
        if mode == 'human':
            print("Current State:", self.state)
        elif mode == 'ansi':
            return "Current State: " + str(self.state)
        else:
            raise ValueError("Unsupported render mode: " + mode)

    def close(self):
        pass

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def visualize_training_results(rewards, save_path):

    # Calculate moving average with window size of 100
    moving_avg = [np.mean(rewards[max(0, i - 100):i + 1]) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))

    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    plt.plot(moving_avg, label='Moving Average (100 episodes)', color='red')

    plt.title("Training Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")

'''
if __name__ == '__main__':
    # Train
    print("Env Setup", flush=True)
    train_env = IDSEnvironment(train=True)
    print("Start Train", flush=True)

    start_time = time.time()
    training_rewards, agent = train_qr_dqn_agent_batch(train_env)

    print(f"Training done in {time.time() - start_time:.2f} seconds", flush=True)
    visualize_training_results(training_rewards, save_path=fr'{BASEDIR}\results\main\qr_dqn_train_rewards.png')

    save_path = fr'{BASEDIR}\results\main\qrdqn_model'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.model.save(save_path)
    print(f"Model saved to {save_path}")

    # Test 
    print("Start Test")
    test_env = IDSEnvironment(train=False)
    results, test_rewards = test(agent, test_env, num_episodes=50)
    visualize_training_results(test_rewards, save_path=fr'{BASEDIR}\results\main\qr_dqn_test_rewards.png')

    # Test loop
    y_true, y_pred = [], []

    for episode in range(10):
        state = test_env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = test_env.step(action)

            # Get the true label from the dataset (subtract 1 because pointer was incremented in step)
            idx = test_env.current_data_pointer - 1
            if idx < 0:
                idx = 0
            true_label = test_env.dataset.iloc[idx, -1]

            y_true.append(1 if str(true_label).lower() == 'anomaly' else 0)
            y_pred.append(action)

            state = next_state

    # Print metrics
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))

'''
if __name__ == '__main__':
    # Skip training, just load saved model
    print("Start Test", flush=True)

    # Initialize test environment
    test_env = IDSEnvironment(train=False)

    # Load trained model
    model_path = fr'{BASEDIR}\results\main\qrdqn_model'
    agent = QRDQNAgent(state_size=test_env.num_features, action_size=test_env.action_space.n)
    agent.model = tf.keras.models.load_model(model_path, compile=False)
    agent.target_model = agent.model 
    test_env = IDSEnvironment(train=False)

    # Run tests
    results, test_rewards = test(agent, test_env, num_episodes=50)

    # Optional: visualize test rewards
    visualize_training_results(test_rewards, save_path=fr'{BASEDIR}\results\main\qr_dqn_test_rewards.png')

    # Evaluate some metrics manually
    y_true, y_pred = [], []
    num_episodes = 20
    for episode in tqdm(range(num_episodes), desc="Testing"):
        state = test_env.reset()
        q_quantiles = agent.predict_quantiles(state[None, :])
        print(np.sum(q_quantiles, axis=2)) 
        done = False
        while not done:
            action = agent.act(state)  # use loaded model to select action
            next_state, reward, done, _ = test_env.step(action)
            # AFTER
            idx = test_env.current_data_pointer - 1
            if idx < 0:
                idx = 0
            true_label = test_env.dataset.iloc[idx, -1]

            y_true.append(1 if true_label == 'anomaly' else 0)
            y_pred.append(action)
            state = next_state

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
