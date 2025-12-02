# common.py
import numpy as np
import gym
from gym import spaces
import random
import tensorflow as tf


class IDSEnvironment(gym.Env):
    """
    Environment for NSL-KDD intrusion detection.
    Assumes `dataset` is a pandas DataFrame where:
      - all feature columns are numeric
      - the last column is 'class' with 'normal' or 'anomaly'
    """

    def __init__(self, dataset, train=True):
        super().__init__()
        print(f"IDSEnvironment INIT (train={train})")

        self.train = train
        self.dataset = dataset
        self.num_features = dataset.shape[1] - 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.max_steps = len(self.dataset)
        self.current_data_pointer = 0
        self.state = self.dataset.iloc[0, :-1].values.astype(np.float32)

    def step(self, action):
        """Take action on current row and move forward by 1."""
        idx = self.current_data_pointer

        # Current state & label for this row
        state = self.dataset.iloc[idx, :-1].values.astype(np.float32)
        label_str = self.dataset.iloc[idx, -1]
        label = 1 if str(label_str).lower() == "anomaly" else 0

        # Reward: correct +1, incorrect -1
        reward = 1.0 if action == label else -1.0

        # Move to next row
        self.current_data_pointer += 1
        done = self.current_data_pointer >= self.max_steps

        if not done:
            next_state = self.dataset.iloc[self.current_data_pointer, :-1].values.astype(np.float32)
        else:
            next_state = np.zeros_like(state)

        self.state = next_state
        return next_state, reward, done, {"label": label}

    def reset(self, episode_num: int = 0):
        # stays at the start of the dataset
        self.current_data_pointer = 0
        self.state = self.dataset.iloc[0, :-1].values.astype(np.float32)
        return self.state

    def render(self, mode="human"):
        print("Current State:", self.state)

    def close(self):
        pass


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done, label):
        """
        label must be 0 (normal) or 1 (anomaly).
        """
        entry = (state, action, reward, next_state, done, label)

        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def sample_balanced(self, batch_size: int, min_anom: int = 8):
        """
        Balanced sampling of anomalies vs normal entries.
        Requires label in entry[5].
        """
        anomalies = [e for e in self.buffer if e[5] == 1]
        normals = [e for e in self.buffer if e[5] == 0]

        if len(anomalies) < min_anom:
            return self.sample(batch_size)

        anom_sample = random.sample(anomalies, min(min_anom, len(anomalies)))
        normal_needed = batch_size - len(anom_sample)
        normal_sample = random.sample(normals, min(normal_needed, len(normals)))

        combined = anom_sample + normal_sample

        if len(combined) < batch_size:
            combined += random.sample(self.buffer, batch_size - len(combined))

        return combined

    def __len__(self):
        return len(self.buffer)


class QRDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        num_quantiles=51,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
        update_target_every=1000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.num_quantiles = num_quantiles
        self.tau = tf.constant(
            np.linspace(0.0, 1.0, num_quantiles, dtype=np.float32)
        )

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

    # MODEL

    def _build_model(self):
        inp = tf.keras.Input(shape=(self.state_size,), dtype=tf.float32)
        x = tf.keras.layers.Dense(64, activation="relu")(inp)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        out = tf.keras.layers.Dense(self.action_size * self.num_quantiles)(x)

        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse",  # unused for QR loss but required by compile
        )
        return model

    # QUANTILE PREDICTION

    def predict_quantiles(self, states, use_target=False):
        model = self.target_model if use_target else self.model
        preds = model.predict(states, verbose=0)
        return preds.reshape(-1, self.action_size, self.num_quantiles)

    # EPSILON-GREEDY ACTION

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state, dtype=np.float32).reshape(1, -1)
        q = self.predict_quantiles(state)
        q_means = np.mean(q, axis=2)  # (1, action_size)
        return int(np.argmax(q_means[0]))

    def act_batch(self, states):
        """
        states: np.array shape (B, state_size)
        Returns: np.array of greedy actions shape (B,)
        """
        states = np.asarray(states, dtype=np.float32)
        q_quantiles = self.predict_quantiles(states, use_target=False)
        q_means = np.mean(q_quantiles, axis=2)  # (B, A)
        actions = np.argmax(q_means, axis=1)
        return actions.astype(np.int32)

    def update_target_network(self, hard=False, tau=0.005):
        if hard:
            self.target_model.set_weights(self.model.get_weights())
        else:
            mw = self.model.get_weights()
            tw = self.target_model.get_weights()
            self.target_model.set_weights(
                [(1 - tau) * t + tau * m for t, m in zip(tw, mw)]
            )

    #  TRAINING STEP
    def train(self, experiences):
        states, actions, rewards, next_states, dones, labels = zip(*experiences)
        states = np.vstack(states).astype(np.float32)
        next_states = np.vstack(next_states).astype(np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, np.float32)
        dones = np.array(dones, np.float32)

        # Target quantiles
        next_q = self.predict_quantiles(next_states, use_target=True)  # (B, A, Q)
        next_q_means = np.mean(next_q, axis=2)  # (B, A)
        next_actions = np.argmax(next_q_means, axis=1)  # (B,)
        idx = np.arange(len(actions))
        next_q_a = next_q[idx, next_actions, :]  # (B, Q)

        targets = rewards[:, None] + (1.0 - dones[:, None]) * self.gamma * next_q_a

        # Loss
        with tf.GradientTape() as tape:
            pred = self.model(states)  # (B, A*Q)
            pred = tf.reshape(
                pred, (-1, self.action_size, self.num_quantiles)
            )  # (B, A, Q)

            mask = tf.one_hot(actions, self.action_size, dtype=tf.float32)
            pred_a = tf.reduce_sum(pred * mask[:, :, None], axis=1)  # (B, Q)

            diff = targets - pred_a  # (B, Q)
            huber = tf.where(
                tf.abs(diff) <= 1.0,
                0.5 * diff**2,
                tf.abs(diff) - 0.5,
            )

            tau = self.tau[None, :]  # (1, Q)
            quantile_loss = tf.abs(tau - tf.cast(diff < 0, tf.float32)) * huber
            loss = tf.reduce_mean(tf.reduce_sum(quantile_loss, axis=1))

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights)
        )

        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_network(hard=True)
