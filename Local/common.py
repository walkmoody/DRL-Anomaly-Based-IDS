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
        self.state = self.dataset.iloc[self.current_data_pointer, :-1].values


    def discretize_state(self, state):
        # Define bins for each feature
        return np.clip((state*10).astype(int), 0, 9) # already normalized data so this workrs

    def step(self, curr_action):
        intrusion = self.dataset.iloc[self.current_data_pointer, -1]
        state = self.dataset.iloc[self.current_data_pointer, :-1].values
        self.state = self.discretize_state(state)
        
        correct_action = 1 if intrusion == 'anomaly' else 0
        reward = 1.0 if curr_action == correct_action else -1.0

        done = (self.current_data_pointer + 1) >= 50
        self.current_data_pointer += 1

        return self.state, reward, done



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

class QRDQNAgent:
    def __init__(self, state_size, action_size, num_quantiles=51 , learning_rate=1e-4, gamma=.99,
                epsilon_start = 1.0, epsilon_min=.05, epsilon_decay=0.995,
                update_target_every=1000):
        self.state_size = state_size
        self.action_size = action_size

        self.num_quantiles = num_quantiles
        self.tau = tf.constant(np.linspace(0.0, 1.0, num_quantiles, dtype=np.float32), dtype =tf.float32)

        self.memory = []

        self.gamma = gamma  # discount rate
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = 64
        self.update_target_every = update_target_every
        self.train_step = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network(hard=True)


    def _build_model(self):
        inputs = tf.keras.Input(shape =(self.state_size,), dtype=tf.float32)
        x = tf.keras.layers.Dense(64,activation ='relu')(inputs)
        x = tf.keras.layers.Dense(64,activation ='relu')(x)
        x = tf.keras.layers.Dense(self.action_size * self.num_quantiles, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self._dummy_loss) 
        return model
    
    def _dummy_loss(self, y_true, y_pred):

        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def predict_quantiles(self, states, use_target=False):
        model = self.target_model if use_target else self.model
        preds = model.predict(states, verbose=0)
        return preds.reshape(-1, self.action_size, self.num_quantiles)

    def act_batch(self, states):
        """
        states: np.array of shape (N, state_size)
        returns: array of actions of length N
        """
        q_quantiles = self.predict_quantiles(states, use_target=False)  # (N, A, Q)
        q_means = np.sum(q_quantiles, axis=2)  # (N, A)
        actions = np.argmax(q_means, axis=1)
        
        # Add epsilon-greedy
        random_mask = np.random.rand(len(states)) < self.epsilon
        random_actions = np.random.randint(0, self.action_size, size=len(states))
        actions[random_mask] = random_actions[random_mask]
        
        return actions

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Ensure state is 2D for model.predict (batch dimension)
        state = np.array(state, dtype=np.float32).reshape(1, -1)  # shape (1, state_size)
        
        # Predict quantiles for the batch (here batch=1)
        q_quantiles = self.predict_quantiles(state, use_target=False)  # shape (1, A, Q)
        q_means = np.mean(q_quantiles, axis=2)  # average across quantiles -> shape (1, A)
        
        action = int(np.argmax(q_means[0]))  # pick best action
        return action



    def update_target_network(self, hard=False, tau=0.005):
        if hard:
            self.target_model.set_weights(self.model.get_weights())
        else:
            # soft update
            w = self.model.get_weights()
            tw = self.target_model.get_weights()
            new_w = [(1 - tau) * tw_i + tau * w_i for tw_i, w_i in zip(tw, w)]
            self.target_model.set_weights(new_w)

    def train(self, experiences):
        # experiences is list of tuples (s, a, r, s2, done)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.vstack(states).astype(np.float32).reshape(-1, self.state_size)
        next_states = np.vstack(next_states).astype(np.float32).reshape(-1, self.state_size)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Predict quantiles for next_states using target_model
        next_q_quantiles = self.predict_quantiles(next_states, use_target=True)  # shape (N, A, Q)
        next_q_means = np.sum(next_q_quantiles, axis=2)  # (N, A)
        next_actions = np.argmax(next_q_means, axis=1)  # (N,)

        # Select the quantiles of chosen next actions
        batch_idx = np.arange(len(next_actions))
        next_quantiles_for_action = next_q_quantiles[batch_idx, next_actions, :]  # (N, Q)

        # Build target quantiles: r + gamma * (1 - done) * next_quantiles
        targets = rewards[:, None] + (1.0 - dones[:, None]) * self.gamma * next_quantiles_for_action  # (N, Q)

        # Expand targets for all actions so we can compute loss only on chosen actions
        # We'll create y_true shaped same as model output (N, A * Q)
        current_q_quantiles = self.predict_quantiles(states, use_target=False)  # (N, A, Q)
        y_true = np.copy(current_q_quantiles)  # start from current predictions

        # Put the targets in the selected action slots
        y_true[batch_idx, actions, :] = targets

        # Flatten y_true to shape (N, A*Q)
        y_true_flat = y_true.reshape(-1, self.action_size * self.num_quantiles).astype(np.float32)

        # Train via train_on_batch using our custom loss computed as quantile huber loss
        # We'll perform one gradient step manually for stability
        with tf.GradientTape() as tape:
            preds = self.model(states, training=True)  # (N, A*Q)
            preds_reshaped = tf.reshape(preds, (-1, self.action_size, self.num_quantiles))  # (N,A,Q)
            # Build mask for chosen actions
            # Compute quantile huber loss between y_true (targets) and preds_reshaped
            err = tf.expand_dims(y_true, -1) - tf.expand_dims(preds_reshaped, -2)  # shape (N,A,Q,1) - (N,A,1,Q) -> broadcast -> (N,A,Q,Q)
            # But simpler and robust approach: compute huber between preds_reshaped and y_true for each (N,A,Q)
            # We'll directly compute err_reshaped = y_true - preds_reshaped
            err_reshaped = tf.cast(y_true - preds_reshaped, tf.float32)  # (N,A,Q)
            # Huber:
            huber = tf.where(tf.abs(err_reshaped) <= 1.0, 0.5 * tf.square(err_reshaped), tf.abs(err_reshaped) - 0.5)
            # tau shape (Q,)
            tau = tf.reshape(self.tau, (1, 1, self.num_quantiles))  # (1,1,Q)
            inv = tf.cast(err_reshaped < 0.0, tf.float32)
            quantile_loss = tf.abs(tau - inv) * huber
            loss = tf.reduce_mean(tf.reduce_sum(quantile_loss, axis=-1))  # average over N and A
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # update target network periodically (or do soft update)
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_network(hard=True)
