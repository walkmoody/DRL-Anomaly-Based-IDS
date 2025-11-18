# common.py
import numpy as np
import pandas as pd
import gym
from gym import spaces
from collections import deque
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import random
import tensorflow as tf

class IDSEnvironment(gym.Env):

    def __init__(self, dataset, train=True):
        print(f'IDSEnvironment INIT (train={train})')
        super().__init__()
        
        self.train = train
        self.dataset = dataset
        self.num_features = self.dataset.shape[1] - 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.current_data_pointer = 0
        self.state = self.dataset.iloc[self.current_data_pointer, :-1].values.astype(np.float32)



    def discretize_state(self, state):
        # Define bins for each feature
        return np.clip((state*10).astype(int), 0, 9) # already normalized data so this workrs

    def step(self, curr_action):
        intrusion = self.dataset.iloc[self.current_data_pointer, -1]
        self.state = self.dataset.iloc[self.current_data_pointer, :-1].values.astype(np.float32)
        # self.state = self.discretize_state(state)
        
        if intrusion == 'anomaly':
            reward = 3.0 if curr_action == 1 else -6.0
        else:
            reward = 2.0 if curr_action == 0 else -4.0


        done = (self.current_data_pointer + 1) >= 50
        self.current_data_pointer += 1

        return self.state, reward, done, {}

    def reset(self, *args, **kwargs):
        
        self.state = self.dataset.iloc[0, :-1].values.astype(np.float32)


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

class QRDQNAgent:
    def __init__(self, state_size, action_size, num_quantiles=51, learning_rate=1e-4, gamma=.99,
                 epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 update_target_every=1000):

        self.state_size = state_size
        self.action_size = action_size
        self.num_quantiles = num_quantiles
        self.tau = tf.constant(np.linspace(0.0, 1.0, num_quantiles, dtype=np.float32))

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.update_target_every = update_target_every
        self.train_step = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network(hard=True)

    # --------------------------------------
    # MODEL
    # --------------------------------------
    def _build_model(self):
        inp = tf.keras.Input(shape=(self.state_size,), dtype=tf.float32)
        x = tf.keras.layers.Dense(64, activation='relu')(inp)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        out = tf.keras.layers.Dense(self.action_size * self.num_quantiles)(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss='mse')
        return model

    # --------------------------------------
    # QUANTILE PREDICTION
    # --------------------------------------
    def predict_quantiles(self, states, use_target=False):
        model = self.target_model if use_target else self.model
        preds = model.predict(states, verbose=0)
        return preds.reshape(-1, self.action_size, self.num_quantiles)

    # --------------------------------------
    # EPSILON-GREEDY ACTION
    # --------------------------------------
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state, dtype=np.float32).reshape(1, -1)
        q = self.predict_quantiles(state)
        q_means = np.mean(q, axis=2)   # FIXED: mean, not sum
        return int(np.argmax(q_means[0]))
    
        # --------------------------------------
    # BATCH ACTION SELECTION (restored)
    # --------------------------------------
    def act_batch(self, states):
        """
        states: np.array shape (B, state_size)
        Returns: np.array of greedy actions shape (B,)
        """

        states = np.asarray(states, dtype=np.float32)

        # Predict quantile distributions for all states
        q_quantiles = self.predict_quantiles(states, use_target=False)  # (B, A, Q)

        # Mean across quantiles gives Q-values
        q_means = np.mean(q_quantiles, axis=2)  # (B, A)

        # Greedy action per batch element
        actions = np.argmax(q_means, axis=1)

        return actions.astype(np.int32)


    # --------------------------------------
    # TARGET NET UPDATE
    # --------------------------------------
    def update_target_network(self, hard=False, tau=0.005):
        if hard:
            self.target_model.set_weights(self.model.get_weights())
        else:
            mw = self.model.get_weights()
            tw = self.target_model.get_weights()
            self.target_model.set_weights(
                [(1 - tau) * t + tau * m for t, m in zip(tw, mw)]
            )

    # --------------------------------------
    # TRAINING (PATCHED)
    # --------------------------------------
    def train(self, experiences):

        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.vstack(states).astype(np.float32)
        next_states = np.vstack(next_states).astype(np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, np.float32)
        dones = np.array(dones, np.float32)

        # --------------------------
        # Compute target quantiles
        # --------------------------
        next_q = self.predict_quantiles(next_states, use_target=True)
        next_q_means = np.mean(next_q, axis=2)
        next_actions = np.argmax(next_q_means, axis=1)
        batch = np.arange(len(actions))
        next_q_a = next_q[batch, next_actions, :]

        targets = rewards[:, None] + (1 - dones[:, None]) * self.gamma * next_q_a

        # --------------------------
        # Compute loss
        # --------------------------
        with tf.GradientTape() as tape:
            pred = self.model(states)
            pred = tf.reshape(pred, (-1, self.action_size, self.num_quantiles))

            # mask for selected actions
            mask = tf.one_hot(actions, self.action_size, dtype=tf.float32)
            pred_a = tf.reduce_sum(pred * mask[:, :, None], axis=1)

            diff = targets - pred_a
            huber = tf.where(tf.abs(diff) <= 1.0,
                             0.5 * diff ** 2,
                             tf.abs(diff) - 0.5)

            tau = self.tau[None, :]
            quantile_loss = tf.abs(tau - tf.cast(diff < 0, tf.float32)) * huber
            loss = tf.reduce_mean(tf.reduce_sum(quantile_loss, axis=1))

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # --------------------------
        # Update target net
        # --------------------------
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_network(hard=True)

