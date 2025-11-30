# QRDQNHpcc.py
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import gc

from common import IDSEnvironment, ReplayBuffer, QRDQNAgent


# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================
def train_qr_dqn_agent(
    env,
    num_episodes=100,
    batch_size=64,
    gamma=0.99,
    replay_buffer=None,
    warmup_size=2000,
    train_every=2,
    update_target_every=1000,
):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = QRDQNAgent(
        state_size,
        action_size,
        num_quantiles=51,
        learning_rate=1e-4,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
        update_target_every=update_target_every,
    )

    if replay_buffer is None:
        memory_buffer = ReplayBuffer(capacity=50000)
    else:
        memory_buffer = replay_buffer

    rewards = []
    print("Training QRDQN...")

    for episode in range(num_episodes):
        state = env.reset(episode_num=episode)
        total_reward = 0.0
        done = False
        step_count = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            label = info["label"]

            memory_buffer.add(
                state, action, reward, next_state, float(done), label
            )

            # Train after warmup
            if (
                len(memory_buffer) > max(batch_size, warmup_size)
                and step_count % train_every == 0
            ):
                experiences = memory_buffer.sample_balanced(batch_size)
                agent.train(experiences)

            state = next_state
            total_reward += reward
            step_count += 1

            if step_count > 1000:
                done = True

        rewards.append(total_reward)
        agent.epsilon = max(
            agent.epsilon_min, agent.epsilon * agent.epsilon_decay
        )

        print(
            f"Episode {episode+1}/{num_episodes} -- reward={total_reward:.2f}, eps={agent.epsilon:.3f}"
        )

        # Optional cleanup every 25 episodes
        if (episode + 1) % 25 == 0:
            gc.collect()

    return rewards, agent


# ================================================================
# TEST FUNCTION
# ================================================================
def test(agent, env, num_episodes=50):
    total_rewards = []
    all_true = []
    all_pred = []

    # Use greedy policy during eval
    orig_eps = agent.epsilon
    agent.epsilon = 0.0

    print("Testing QRDQN...")

    for episode in range(num_episodes):
        state = env.reset(episode)
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            action = agent.act(state)  # now purely greedy
            next_state, reward, done, info = env.step(action)
            label = info["label"]

            all_true.append(label)
            all_pred.append(action)

            ep_reward += reward
            state = next_state
            step += 1

            if step > 1000:
                done = True

        total_rewards.append(ep_reward)
        print(
            f"Test Episode {episode+1}/{num_episodes} -- reward={ep_reward:.2f}"
        )

    # restore epsilon
    agent.epsilon = orig_eps

    avg_reward = np.mean(total_rewards)
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    f1 = f1_score(all_true, all_pred, zero_division=0)
    conf = confusion_matrix(all_true, all_pred)

    results = {
        "Average Reward": avg_reward,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": conf,
    }

    print("===== QRDQN Evaluation Metrics =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results, total_rewards


# ================================================================
# PLOTTING
# ================================================================
def visualize_training_results(rewards, save_path="training_metrics.png"):
    moving_avg = [
        np.mean(rewards[max(0, i - 100) : i + 1]) for i in range(len(rewards))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.6)
    plt.plot(moving_avg, label="Moving Average (100)", color="red")
    plt.title("QRDQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
