import gym
from gym import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Dict, Any, Tuple
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff
import pandas as pd
from QRDQN import QRDQNAgent, train_qr_dqn_agent, test




pd.set_option('display.max_rows', None)  # show all rows
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)  # don't wrap lines
pd.set_option('display.max_colwidth', None)  # full text in cells


def discretize(value, bins):
    return np.digitize(value, bins, right=True)

def process_dataset():
    train_data_arff, meta_train = arff.loadarff("NSL-KDD/KDDTrain+.arff")
    test_data_arff, meta_test = arff.loadarff("NSL-KDD/KDDTest+.arff")

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

class IDSEnvironment(gym.Env):

    def __init__(self, dataset_path="KDDTrain+.txt"):
        print('IDSEnvironment INIT')
        super(IDSEnvironment, self).__init__()

        # Load and preprocess the dataset
        self.train_data, self.test_data = process_dataset()
        self.num_features = self.train_data.shape[1] - 1  # exclude label column
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features,),
                                            dtype=np.float32)
        self.current_data_pointer = 0

        # Define action space (binary decision for now: alert vs. no alert)
        self.action_space = spaces.Discrete(2)
        self.state_space = 148
        self.state = self.train_data.iloc[self.current_data_pointer, :-1].values

    def discretize_state(self, state):
        # Define bins for each feature
        return np.clip((state*10).astype(int), 0, 9) # already normalized data so this workrs

    def step(self, curr_action):

        self.state = self.discretize_state(self.train_data.iloc[self.current_data_pointer, :-1].values)

        # Determine reward
        intrusion = self.train_data.iloc[self.current_data_pointer, -1]
        correct_action = 1 if intrusion == 'anomaly' else 0
        reward = 1 if curr_action == correct_action else -1
        curr_reward = np.random.normal(loc=reward, scale=0.1) # randomness needed
        

        self.current_data_pointer += 1
        done = self.current_data_pointer >= 100 # len(self.train_data) // keep at 100 to run

        return self.state, curr_reward, done, {}

    def reset(self, *args, **kwargs):
        self.state = self.discretize_state(self.train_data.iloc[0, :-1].values)

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

if __name__ == '__main__':
    env = IDSEnvironment()

    training_rewards, agent = train_qr_dqn_agent(env, num_episodes=50)
    visualize_training_results(training_rewards, save_path="results/main/qr_dqn_rewards.png")

    results, test_rewards = test(agent, env, num_episodes=50)
    print(results)

    num_episodes = 50
    rewards = []

    # Run 50 episodes // start with 50 SSS
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get the current label
            current_index = env.current_data_pointer
            true_label = env.train_data.iloc[current_index, -1]

            # Map labels to action: 0 = normal, 1 = anomaly
            action = 1 if true_label == 'anomaly' else 0

            # Step environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        print(f"Episode {episode+1} finished — total reward: {total_reward}")

    visualize_training_results(rewards, save_path="results/main/training_rewards.png")
