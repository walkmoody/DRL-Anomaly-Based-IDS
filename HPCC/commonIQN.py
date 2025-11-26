# commonIQN.py

import numpy as np
import pandas as pd
import gym
from gym import spaces
from collections import deque
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix
)
import random
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
from collections import Counter


# =====================================================
#                  ENVIRONMENT
# =====================================================

class IDSEnvironment(gym.Env):
    """
    IDS environment for NSL-KDD style data.
    Assumes final column is 'class' with values 'normal' or 'anomaly'.
    """

    def __init__(self, dataset, train=True):
        print(f'IDSEnvironment INIT (train={train})')
        super().__init__()

        self.train = train
        self.dataset = dataset
        self.num_features = self.dataset.shape[1] - 1  # all but 'class'

        # Data is standardized (StandardScaler), so allow real-valued observations
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0 = normal, 1 = anomaly

        self.current_data_pointer = 0
        self.state = self.dataset.iloc[self.current_data_pointer, :-1].values.astype(np.float32)

    def discretize_state(self, state):
        # Not used, but kept if you ever want discrete features again.
        return np.clip((state * 10).astype(int), 0, 9)

    def step(self, curr_action):
        """
        Reward design:
          - action 0 = predict "normal"
          - action 1 = predict "anomaly"

        We treat anomalies as more important, but not insanely lopsided:

          If true = anomaly:
              correct (1) -> +10
              wrong   (0) -> -10

          If true = normal:
              correct (0) -> +2
              wrong   (1) -> -5
        """
        intrusion = self.dataset.iloc[self.current_data_pointer, -1]
        self.state = self.dataset.iloc[self.current_data_pointer, :-1].values.astype(np.float32)

        is_anom = (intrusion == 'anomaly')

        if is_anom:
            reward = 10.0 if curr_action == 1 else -10.0
        else:
            reward = 2.0 if curr_action == 0 else -5.0

        # Optional: scale rewards to stabilize RL (uncomment if needed)
        # reward = reward / 10.0

        self.current_data_pointer += 1
        done = self.current_data_pointer >= len(self.dataset)

        return self.state, reward, done, {'label': 1 if is_anom else 0}

    def reset(self, episode_num=0, *args, **kwargs):
        """
        Basic reset with occasional jumps to different parts of the dataset
        during training, so the agent doesn't always see the same prefix.
        """
        # Wrap around if at end
        if self.current_data_pointer >= len(self.dataset):
            self.current_data_pointer = 0

        # Every 10 episodes, jump somewhere else (only if big enough dataset)
        if episode_num % 10 == 1 and len(self.dataset) > 10000:
            jump_point = np.random.randint(0, len(self.dataset) - 10000)
            self.current_data_pointer = jump_point

        self.state = self.dataset.iloc[self.current_data_pointer, :-1].values.astype(np.float32)
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


# =====================================================
#                  REPLAY BUFFER
# =====================================================

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done, label):
        """
        label: 0 (normal) or 1 (anomaly), only used for balanced sampling.
        """
        entry = (state, action, reward, next_state, done, label)

        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry

        self.position = (self.position + 1) % self.capacity

    def sample_balanced(self, batch_size=128, anomaly_ratio=0.4):
        """
        Return a batch with approximately `anomaly_ratio` anomalies
        and (1 - anomaly_ratio) normals. Falls back gracefully if
        there aren't enough of either class.
        """
        if len(self.buffer) < batch_size:
            # Not enough to sample, just random
            return random.sample(self.buffer, len(self.buffer))

        anomalies = [e for e in self.buffer if e[5] == 1]
        normals   = [e for e in self.buffer if e[5] == 0]

        target_anom = int(batch_size * anomaly_ratio)
        target_norm = batch_size - target_anom

        # sample anomalies
        if len(anomalies) > 0:
            anom_sample = random.sample(anomalies, min(len(anomalies), target_anom))
        else:
            anom_sample = []

        # sample normals
        if len(normals) > 0:
            norm_sample = random.sample(normals, min(len(normals), target_norm))
        else:
            norm_sample = []

        batch = anom_sample + norm_sample

        # If still short (early training), fill remaining from entire buffer
        if len(batch) < batch_size:
            batch += random.sample(self.buffer, batch_size - len(batch))

        random.shuffle(batch)
        return batch

    def __len__(self):
        return len(self.buffer)


# =====================================================
#                  IQN AGENT
# =====================================================

class IQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        num_tau_samples=32,
        embedding_dim=64,
        num_quantiles=32,
        gamma=0.99,
        learning_rate=1e-4,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        batch_size=128,
        update_target_every=500
    ):
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
        """
        A beefed-up IQN network:
          - Two 128-unit dense layers for state embedding
          - Tau embedding with cosine features -> 128
          - Combine via element-wise multiplication
          - Two more 128-unit dense layers
          - Output quantiles for each action
        """
        states_input = tf.keras.Input(shape=(self.state_size,), dtype=tf.float32)
        taus_input = tf.keras.Input(shape=(self.num_tau_samples, 1), dtype=tf.float32)

        # ----------------------------
        # State embedding
        # ----------------------------
        x = tf.keras.layers.Dense(128, activation='relu')(states_input)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.expand_dims(x, axis=1)  # (B, 1, 128)

        # ----------------------------
        # Tau (quantile) embedding
        # ----------------------------
        i_pi = tf.constant(
            np.arange(1, self.embedding_dim + 1) * np.pi, dtype=tf.float32
        )  # (embedding_dim,)
        cos_tau = tf.cos(tf.matmul(taus_input, i_pi[None, :]))  # (B, num_tau_samples, embedding_dim)
        cos_tau = tf.keras.layers.Dense(128, activation='relu')(cos_tau)

        # ----------------------------
        # Combine state & quantile
        # ----------------------------
        x = tf.keras.layers.Multiply()([x, cos_tau])  # (B, num_tau_samples, 128)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        quantiles = tf.keras.layers.Dense(self.action_size)(x)  # (B, num_tau_samples, A)

        model = tf.keras.Model(inputs=[states_input, taus_input], outputs=quantiles)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=self.quantile_huber_loss
        )
        return model

    def sample_taus(self, batch_size):
        return np.random.uniform(
            0, 1, size=(batch_size, self.num_tau_samples, 1)
        ).astype(np.float32)

    def quantile_huber_loss(self, y_true, y_pred, kappa=1.0):
        """
        Standard IQN quantile Huber loss using a fixed grid of taus.
        y_true, y_pred: (B, num_tau_samples, A)
        """
        delta = y_true - y_pred  # (B, N, A)

        # Huber loss part
        abs_delta = tf.abs(delta)
        huber_loss = tf.where(
            abs_delta <= kappa,
            0.5 * tf.square(delta),
            kappa * (abs_delta - 0.5 * kappa)
        )

        # quantile regression term
        tau = tf.linspace(0.0, 1.0, self.num_tau_samples + 1)[1:]  # avoid 0
        tau = tf.reshape(tau, (1, self.num_tau_samples, 1))  # (1, N, 1)
        tau = tf.cast(tau, tf.float32)

        indicator = tf.cast(delta < 0.0, tf.float32)
        loss = tf.abs(tau - indicator) * huber_loss  # (B, N, A)

        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    def act(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state, dtype=np.float32).reshape(1, -1)
        taus = self.sample_taus(1)
        q_values = self.model.predict([state, taus], verbose=0)  # (1, N, A)
        q_mean = np.mean(q_values, axis=1)  # (1, A)
        return int(np.argmax(q_mean[0]))

    def train_step_batch(self, experiences):
        """
        One gradient step on a batch of experiences.
        """
        states, actions, rewards, next_states, dones, _ = zip(*experiences)
        states = np.vstack(states).astype(np.float32)
        next_states = np.vstack(next_states).astype(np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        batch_size = len(states)

        taus = self.sample_taus(batch_size)
        next_taus = self.sample_taus(batch_size)

        # Compute target quantiles
        next_q = self.target_model.predict([next_states, next_taus], verbose=0)  # (B, N, A)
        next_q_mean = np.mean(next_q, axis=1)  # (B, A)
        next_actions = np.argmax(next_q_mean, axis=1)  # (B,)

        next_q_selected = next_q[np.arange(batch_size), :, next_actions]  # (B, N)

        targets = rewards[:, None] + (1.0 - dones[:, None]) * self.gamma * next_q_selected  # (B, N)

        # Current quantiles
        current_pred = self.model.predict([states, taus], verbose=0)  # (B, N, A)

        # Build y_true with only chosen actions updated
        y_true = np.copy(current_pred)
        for i, a in enumerate(actions):
            y_true[i, :, a] = targets[i]

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
            self.target_model.set_weights([
                (1 - tau) * t + tau * m for t, m in zip(tw, mw)
            ])


# =====================================================
#            TRAINING & TESTING FUNCTIONS
# =====================================================

def train_iqn_agent(
    env,
    num_episodes=1000,
    batch_size=128,
    warmup_size=10000,
    train_updates_per_step=1,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.1,
    buffer_capacity=50000,
    save_path=None,
    validate_env=None,
    validate_every=100
):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = IQNAgent(
        state_size,
        action_size,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        batch_size=batch_size,
        update_target_every=500
    )
    memory_buffer = ReplayBuffer(capacity=buffer_capacity)
    rewards = []

    # ----------------------------
    # Warm-up
    # ----------------------------
    print(f"Starting warm-up with {warmup_size} random steps...")
    for _ in range(warmup_size):
        state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        label = info.get("label", 0)
        memory_buffer.add(state, action, reward, next_state, float(done), label)
        if done:
            env.reset()

    print(f"Warmup complete. Buffer size: {len(memory_buffer)}")
    anoms = sum(1 for e in memory_buffer.buffer if e[5] == 1)
    print(f"Warmup anomalies in buffer: {anoms}")

    print("Training IQN Agent...")

    try:
        for episode in range(num_episodes):

            # Normal episode logic
            state = env.reset(episode_num=episode)
            total_reward = 0.0
            done = False
            step_count = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                label = info.get("label", 0)
                memory_buffer.add(state, action, reward, next_state, float(done), label)

                state = next_state
                total_reward += reward
                step_count += 1

                if step_count > 500:
                    done = True

                # Training
                if len(memory_buffer) >= batch_size:
                    for _ in range(train_updates_per_step):
                        experiences = memory_buffer.sample_balanced(batch_size=batch_size)
                        agent.train_step_batch(experiences)

            # Logging
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            rewards.append(total_reward)
            print(f"Episode {episode+1}/{num_episodes}  Reward:{total_reward:.1f}  Epsilon:{agent.epsilon:.3f}")

            # -----------------------------------------------------
            # 🔥 MEMORY CLEANUP EVERY EPISODE (NOT EVERY 50 EPISODES)
            # -----------------------------------------------------
            print("  -> Memory cleanup...")
            main_weights  = agent.model.get_weights()
            target_weights = agent.target_model.get_weights()

            tf.keras.backend.clear_session()
            gc.collect()

            # Rebuild fresh models
            agent.model = agent._build_model()
            agent.model.set_weights(main_weights)

            agent.target_model = agent._build_model()
            agent.target_model.set_weights(target_weights)

            gc.collect()

    except Exception as e:
        print("Training aborted with exception:", e)
        try:
            if save_path is not None:
                agent.model.save(save_path)
                print("Saved checkpoint after abort to:", save_path)
        except Exception:
            pass
        raise
    finally:
        try:
            del memory_buffer
            gc.collect()
        except Exception:
            pass

    return rewards, agent


def test_iqn_agent(agent, env, num_episodes=100, max_steps_per_episode=200):
    all_true_labels = []
    all_predicted_labels = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        while not done and step_count < max_steps_per_episode:
            action = agent.act(state)
            next_state, _, done, info = env.step(action)
            all_true_labels.append(info["label"])
            all_predicted_labels.append(action)
            state = next_state
            step_count += 1

    # Compute metrics
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels)
    recall = recall_score(all_true_labels, all_predicted_labels)
    confusion = confusion_matrix(all_true_labels, all_predicted_labels)

    pred_counts = Counter(all_predicted_labels)
    print("Predicted action counts:", dict(pred_counts))

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print("Confusion Matrix:\n", confusion)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion
    }
