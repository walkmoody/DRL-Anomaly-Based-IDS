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

def train_qr_dqn_agent_batch(env, num_episodes=150, batch_size=64, gamma=0.99, train_every=10):
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
                       epsilon_decay=0.97,
                       update_target_every=1000)
    
    memory_buffer = ReplayBuffer(capacity=10000)
    rewards = []
    cycle_count = 10
    print("training with batch actions...")
    for episode in range(num_episodes):
        
        curr_state = env.reset(episode_num=episode)
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
                    nxt_state, reward, done_flag, _ = env.step(action)
                    memory_buffer.add(states_batch[idx], action, reward, nxt_state, float(done_flag))
                    total_reward += reward
                    curr_state = nxt_state
                    step_count += 1
                    if done_flag or step_count > 1000:
                        done = True
                        break  # Stop processing the batch if done

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

def train_qr_dqn_agent_soft_target(env, num_episodes=100, batch_size=64, gamma=0.99, tau=0.005):

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = QRDQNAgent(
        state_size, action_size,
        num_quantiles=51,
        learning_rate=1e-4,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
        update_target_every=1_000  # will be used for hard updates if needed
    )

    memory_buffer = ReplayBuffer(capacity=50_000)
    rewards = []

    print("Training with soft-target updates...")

    for episode in range(num_episodes):
        curr_state = env.reset(episode_num=episode)
        total_reward = 0
        done = False
        step_count = 0
        TRAIN_EVERY = 10

        while not done:
            # --- choose action ---
            action = agent.act(curr_state)

            # --- environment step ---
            nxt_state, reward, done, _ = env.step(action)

            # --- store transition ---
            memory_buffer.add(curr_state, action, reward, nxt_state, float(done))

            # --- train periodically ---
            if len(memory_buffer) > batch_size and step_count % TRAIN_EVERY == 0:
                experiences = memory_buffer.sample(batch_size)
                agent.train(experiences)
                # Soft update target network after every train step
                agent.update_target_network(hard=False, tau=tau)

            if agent.train_step % agent.update_target_every == 0:
                agent.update_target_network(hard=True)

            curr_state = nxt_state
            total_reward += reward
            step_count += 1

            # Safety limit per episode
            if step_count > 1000:
                done = True
                break

        rewards.append(total_reward)

        # Decay epsilon once per episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)


        print(f"Episode {episode+1}/{num_episodes} -- total_reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")

    return rewards, agent

'''

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
        epsilon_decay=0.97,     # decay ONCE per episode (slow & stable)
        update_target_every=1000
    )

    memory_buffer = ReplayBuffer(capacity=50000)
    rewards = []

    print("Training...")

    for episode in range(num_episodes):

        curr_state = env.reset(episode_num=episode)
        total_reward = 0
        done = False
        step_count = 0

        TRAIN_EVERY = 10

        while not done:
            action = agent.act(curr_state)
            nxt_state, reward, done, _ = env.step(action)
            memory_buffer.add(curr_state, action, reward, nxt_state, float(done))

            # --- train periodically ---
            if len(memory_buffer) > batch_size and step_count % TRAIN_EVERY == 0:
                experiences = memory_buffer.sample(batch_size)
                agent.train(experiences)

            curr_state = nxt_state
            total_reward += reward
            step_count += 1

            if step_count > 1000:
                done = True
                print(step_count)
                break  # Stop processing the batch if done

        # store episode reward
        rewards.append(total_reward)

        # --- DECAY EPSILON *ONCE PER EPISODE* ---
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)


        print(f"Episode {episode+1}/{num_episodes} -- total_reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")

    return rewards, agent
'''

def train_qr_dqn_agent(env, num_episodes=100, batch_size=64, gamma=0.99,
                      replay_buffer=None, warmup_size=2000, train_every=2,
                      update_target_every=1000):

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = QRDQNAgent(
        state_size, action_size,
        num_quantiles=51,
        learning_rate=1e-4,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
        update_target_every=1000
    )

    if replay_buffer is None:
        memory_buffer = ReplayBuffer(capacity=12000)  # SAFER
    else:
        memory_buffer = replay_buffer

    rewards = []
    print("Training...")

    for episode in range(num_episodes):

        curr_state = env.reset(episode_num=episode)
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.act(curr_state)
            nxt_state, reward, done, info = env.step(action)
            label = info["label"]
            memory_buffer.add(curr_state, action, reward, nxt_state, float(done), label)

            if len(memory_buffer) > max(batch_size, warmup_size) and step_count % train_every == 0:
                experiences = memory_buffer.sample_balanced(batch_size)
                agent.train(experiences)

            curr_state = nxt_state
            total_reward += reward
            step_count += 1

            if step_count > 1000:
                done = True
                break

        rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        print(f"Episode {episode+1}/{num_episodes} -- total_reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")

        # ---------------------------
        # 🔥 OOM PREVENTION EVERY 20 EPOCHS
        # ---------------------------
        if (episode + 1) % 20 == 0:
            print("Performing periodic TF cleanup...")
            main_w = agent.model.get_weights()
            target_w = agent.target_model.get_weights()

            tf.keras.backend.clear_session()
            gc.collect()

            agent.model = agent._build_model()
            agent.model.set_weights(main_w)

            agent.target_model = agent._build_model()
            agent.target_model.set_weights(target_w)
            gc.collect()

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
        state = env.reset(episode)
        done = False
        ep_reward = 0.0
        step = 0
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
            step += 1
            if step > 1000:
                done = True
        total_rewards.append(ep_reward)
        print(f"Episode {episode+1}/{num_episodes} -- total_reward = {total_rewards:.2f}")

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



