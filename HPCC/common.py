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
        #self.state = self.discretize_state(self.state)
        
        if intrusion == 'anomaly':
            reward = 1.0 if curr_action == 1 else -1.0
        else:
            reward = 1.0 if curr_action == 0 else -1.0

        self.current_data_pointer += 1

        done = self.current_data_pointer >= len(self.dataset)
        label = 1 if intrusion == 'anomaly' else 0
        return self.state, reward, done,  {"label": label}

    def reset(self, episode_num=0, *args, **kwargs):

        # If we're within 1000 rows of end → wrap around
        if self.current_data_pointer >= len(self.dataset):
            self.current_data_pointer = 0

        # Every 10 episodes, jump to a new part of the dataset
        if episode_num % 10 == 1:
            jump_point = np.random.randint(0, len(self.dataset) - 10000)
            self.current_data_pointer = jump_point

        # Load state from current pointer
        self.state = (
            self.dataset.iloc[self.current_data_pointer, :-1].values.astype(np.float32)
        )

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

    def add(self, state, action, reward, next_state, done, label):
        """
        label must be 0 (normal) or 1 (anomaly)
        """
        entry = (state, action, reward, next_state, done, label)

        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Only sample real entries (no None)
        return random.sample(self.buffer, batch_size)

    def sample_balanced(self, batch_size, min_anom=8):
        """
        Balanced sampling of anomalies vs normal entries.
        Requires label in entry[5].
        """

        # split buffer into anomalies and normals
        anomalies = [e for e in self.buffer if e[5] == 1]
        normals   = [e for e in self.buffer if e[5] == 0]

        # not enough anomalies yet → fallback
        if len(anomalies) < min_anom:
            return self.sample(batch_size)

        # sample anomalies + fill rest with normals
        anom_sample = random.sample(anomalies, min(min_anom, len(anomalies)))
        normal_needed = batch_size - len(anom_sample)

        # in case normals are too few
        normal_sample = random.sample(normals, min(normal_needed, len(normals)))

        combined = anom_sample + normal_sample

        # fallback: if we still don't have enough samples
        if len(combined) < batch_size:
            combined += random.sample(self.buffer, batch_size - len(combined))

        return combined

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

        states, actions, rewards, next_states, dones, labels = zip(*experiences)
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

class IQNAgent:

    def __init__(self, state_size, action_size,
                 num_tau_samples=32, embedding_dim=64, num_quantiles=51,
                 gamma=0.99, learning_rate=1e-4,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.97,
                 batch_size=64, update_target_every=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.num_tau_samples = num_tau_samples
        self.embedding_dim = embedding_dim
        self.num_quantiles = num_quantiles

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.train_step = 0

        # Build online & target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network(hard=True)

    def _build_model(self):
        # Inputs
        states_input = tf.keras.Input(shape=(self.state_size,), dtype=tf.float32)
        taus_input = tf.keras.Input(shape=(self.num_tau_samples, 1), dtype=tf.float32)

        # Cosine embedding for taus
        i_pi = tf.constant(np.arange(1, self.embedding_dim + 1) * np.pi, dtype=tf.float32)
        cos_tau = tf.cos(tf.matmul(taus_input, i_pi[None, :]))  # (B, num_tau_samples, embedding_dim)
        cos_tau = tf.keras.layers.Dense(128, activation='relu')(cos_tau)  # project to match state

        # State embedding
        x = tf.keras.layers.Dense(128, activation='relu')(states_input)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.expand_dims(x, axis=1)  # (B,1,128)

        # Combine state + tau embeddings
        x = tf.keras.layers.Multiply()([x, cos_tau])  # (B, num_tau_samples, 128)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        quantiles = tf.keras.layers.Dense(self.action_size)(x)  # (B, num_tau_samples, A)

        model = tf.keras.Model(inputs=[states_input, taus_input], outputs=quantiles)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=self.quantile_huber_loss)
        return model


    def sample_taus(self, batch_size):
        return np.random.uniform(0, 1, size=(batch_size, self.num_tau_samples, 1)).astype(np.float32)

    def quantile_huber_loss(self, y_true, y_pred, kappa=1.0):
        delta = y_true - y_pred
        huber_loss = tf.where(tf.abs(delta) <= kappa,
                              0.5 * tf.square(delta),
                              kappa * (tf.abs(delta) - 0.5 * kappa))
        tau = tf.linspace(0.0, 1.0, self.num_tau_samples)
        tau = tf.reshape(tau, (1, self.num_tau_samples, 1))
        loss = tf.abs(tau - tf.cast(delta < 0, tf.float32)) * huber_loss
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        taus = self.sample_taus(1)
        q_values = self.model.predict([state, taus], verbose=0)  # (1, num_tau_samples, A)
        q_mean = np.mean(q_values, axis=1)
        return int(np.argmax(q_mean[0]))

    def train(self, experiences):
        states, actions, rewards, next_states, dones, _ = zip(*experiences)
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        batch_size = len(states)

        taus = self.sample_taus(batch_size)
        next_taus = self.sample_taus(batch_size)

        # Compute target quantiles
        next_q = self.target_model.predict([next_states, next_taus], verbose=0)
        next_q_mean = np.mean(next_q, axis=1)
        next_actions = np.argmax(next_q_mean, axis=1)
        next_q_selected = next_q[np.arange(batch_size), :, next_actions]

        targets = rewards[:, None] + (1 - dones[:, None]) * self.gamma * next_q_selected

        # Predict current quantiles
        current_pred = self.model.predict([states, taus], verbose=0)

        # Mask to update only chosen actions
        mask = np.zeros_like(current_pred)
        for i, a in enumerate(actions):
            mask[i, :, a] = 1
        y_true = mask * targets[:, :, None] + (1 - mask) * current_pred

        # Train
        self.model.train_on_batch([states, taus], y_true)

        # Update target
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_network(hard=True)

    def update_target_network(self, hard=False, tau=0.005):
        if hard:
            self.target_model.set_weights(self.model.get_weights())
        else:
            mw = self.model.get_weights()
            tw = self.target_model.get_weights()
            self.target_model.set_weights([(1 - tau) * t + tau * m for t, m in zip(tw, mw)])
