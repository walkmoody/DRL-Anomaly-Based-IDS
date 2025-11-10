import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import random
import matplotlib.pyplot as plt
from main import IDSEnvironment, ReplayBuffer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

#trained with this one

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


def train_qr_dqn_agent(env, num_episodes=200, batch_size=64, gamma=0.99):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = QRDQNAgent(state_size, action_size,
                    num_quantiles=51,
                    learning_rate=1e-4,
                    gamma=gamma,
                    epsilon_start=1.0,
                    epsilon_min=0.05,
                    epsilon_decay=0.995,
                    update_target_every=1000)
    memory_buffer = ReplayBuffer(capacity=50000)

    rewards = []
    print("training")

    for episode in range(num_episodes):
        curr_state = env.reset()
        total_reward = 0
        done = False

        TRAIN_EVERY = 10
        step_count = 0

        print("episode: ", episode)
        while not done:
            action = agent.act(curr_state)
            nxt_state, reward, done, _ = env.step(action)
            # store raw state vectors (not reshaped)
            memory_buffer.add(curr_state, action, reward, nxt_state, float(done))
            if len(memory_buffer) > batch_size and step_count % TRAIN_EVERY == 0:
                experiences = memory_buffer.sample(batch_size)
                agent.train(experiences)
            curr_state = nxt_state
            total_reward += reward
        rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} -- total_reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")
    return rewards, agent

def train_qr_dqn_agent_batch(env, num_episodes=200, batch_size=64, gamma=0.99, train_every=10):
    """
    Training loop that uses batch action selection for speed.
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = QRDQNAgent(state_size, action_size,
                       num_quantiles=51,
                       learning_rate=1e-4,
                       gamma=gamma,
                       epsilon_start=1.0,
                       epsilon_min=0.05,
                       epsilon_decay=0.995,
                       update_target_every=1000)
    
    memory_buffer = ReplayBuffer(capacity=50000)
    rewards = []

    print("training with batch actions...")

    for episode in range(num_episodes):
        curr_state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        states_batch = []
        while not done:
            # collect batch of states for prediction
            states_batch.append(curr_state)
            
            # batch predict actions every TRAIN_EVERY steps
            if len(states_batch) == train_every or done:
                states_array = np.array(states_batch, dtype=np.float32)
                actions_batch = agent.act_batch(states_array)
                for idx, action in enumerate(actions_batch):
                    s = states_batch[idx]
                    nxt_state, reward, done_flag, _ = env.step(action)
                    memory_buffer.add(s, action, reward, nxt_state, float(done_flag))
                    total_reward += reward
                    curr_state = nxt_state
                    step_count += 1
                    done = done_flag
                    # Train if memory has enough samples
                    if len(memory_buffer) > batch_size:
                        experiences = memory_buffer.sample(batch_size)
                        agent.train(experiences)
                states_batch = []  # reset batch

        rewards.append(total_reward)
        # decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} -- total_reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")

    return rewards, agent

def test(agent, env, num_episodes=100):
    """
    Test a DQNAgent on a given environment and compute classification metrics.

    :param agent: The DQNAgent to be tested.
    :param env: The environment to test the agent on.
    :param num_episodes: Number of test episodes.
    :return: A dictionary containing average reward and classification metrics.
    """
    total_rewards = []
    all_true = []
    all_pred = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(state)
            nxt_state, reward, done, _ = env.step(action)
            # true label must be pulled from env.dataset
            idx = env.current_data_pointer - 1
            if idx < 0:
                idx = 0
            true_label = env.dataset.iloc[idx, -1]
            true_bin = 1 if str(true_label).lower() == 'anomaly' else 0
            all_true.append(true_bin)
            all_pred.append(action)
            ep_reward += reward
            state = nxt_state
        total_rewards.append(ep_reward)
    avg_reward = np.mean(total_rewards)
    # compute metrics using binary labels
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    from sklearn.metrics import f1_score, confusion_matrix
    f1 = f1_score(all_true, all_pred, zero_division=0)
    conf = confusion_matrix(all_true, all_pred)
    results = {
        'Average Reward': avg_reward,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf
    }
    for k, v in results.items():
        print(f"{k}: {v}")
    return results, total_rewards


def visualize_training_results(rewards):
    """
    Visualizes the training results.

    Args:
    - rewards (list): A list of rewards received at each episode.
    """

    # Calculate moving average with window size of 100
    moving_avg = [np.mean(rewards[max(0, i - 100):i + 1]) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))

    plt.plot(rewards, label='QRDQN Episode Reward', alpha=0.6)
    plt.plot(moving_avg, label='QRDQN Moving Average (100 episodes)', color='red')

    plt.title("QRDQN Training Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    env = IDSEnvironment()
    training_rewards, agent = train_qr_dqn_agent(env)
    visualize_training_results(training_rewards)
    results, test_rewards = test(agent,env)
    # 1. Bar Plot for Metrics
    metrics = ['Average Reward', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [results[metric] for metric in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylabel('Value')
    plt.title('QRDQN Metrics Visualization')
    plt.ylim([0, 1])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    # 2. Heatmap for Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['Confusion Matrix'], annot=True, cmap="YlGnBu", fmt='g')
    plt.title('QRDQN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    visualize_training_results(training_rewards)
    print(results)



