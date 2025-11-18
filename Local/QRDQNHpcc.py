import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns


#classes
from common import IDSEnvironment, ReplayBuffer, QRDQNAgent

def train_qr_dqn_agent(env, num_episodes=100, batch_size=64, gamma=0.99):

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Recommended exploration settings for IDS
    agent = QRDQNAgent(
        state_size, action_size,
        num_quantiles=51,
        learning_rate=1e-4,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,        # better for anomaly detection
        epsilon_decay=0.995,     # decay ONCE per episode (slow & stable)
        update_target_every=1000
    )

    memory_buffer = ReplayBuffer(capacity=50000)
    rewards = []

    print("Training...")

    for episode in range(num_episodes):

        curr_state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        TRAIN_EVERY = 10

        print(f"Episode {episode+1}/{num_episodes}")

        while not done:

            # --- choose action with epsilon-greedy ---
            action = agent.act(curr_state)

            # --- environment step ---
            nxt_state, reward, done, _ = env.step(action)

            # store transition
            memory_buffer.add(curr_state, action, reward, nxt_state, float(done))

            # --- train periodically ---
            if len(memory_buffer) > batch_size and step_count % TRAIN_EVERY == 0:
                experiences = memory_buffer.sample(batch_size)
                agent.train(experiences)

            curr_state = nxt_state
            total_reward += reward
            step_count += 1

        # store episode reward
        rewards.append(total_reward)

        # --- DECAY EPSILON *ONCE PER EPISODE* ---
        agent.epsilon = max(agent.epsilon_min,
                            agent.epsilon * agent.epsilon_decay)

        # progress print every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} "
                  f"-- total_reward = {total_reward:.2f}, "
                  f"epsilon = {agent.epsilon:.3f}")

    return rewards, agent


def train_qr_dqn_agent_batch(env, num_episodes=100, batch_size=64, gamma=0.99, train_every=10):
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
                       epsilon_min=0.1,
                       epsilon_decay=0.999,
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

        print(f"Episode {episode+1}/{num_episodes} -- total_reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")

    return rewards, agent

def train_dqn_agent_optimized(env, num_episodes=100, batch_size=32, train_every=32):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = QRDQNAgent(state_size, action_size)
    memory_buffer = ReplayBuffer(capacity=5000)  # big enough for 5000 steps

    rewards = []

    for episode in range(num_episodes):
        curr_state = env.reset()  # already shape (1, state_size)
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            curr_action = agent.act(curr_state)
            nxt_state, reward, done, _ = env.step(curr_action)
            nxt_state = np.expand_dims(nxt_state, axis=0)

            memory_buffer.add(curr_state, curr_action, reward, nxt_state, done)
            total_reward += reward
            curr_state = nxt_state
            step_count += 1

            # Train every `train_every` steps
            if len(memory_buffer) >= batch_size and step_count % train_every == 0:
                experiences = memory_buffer.sample(batch_size)
                agent.train(experiences)

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

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
    plt.savefig("training_metrics.png")
    plt.close()
    

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

    # 2. Heatmap for Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['Confusion Matrix'], annot=True, cmap="YlGnBu", fmt='g')
    plt.title('QRDQN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()
    visualize_training_results(training_rewards)
    print(results)