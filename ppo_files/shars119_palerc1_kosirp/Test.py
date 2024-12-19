import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
from torch.distributions import Categorical
from Train import PolicyNet, preprocess
import matplotlib.pyplot as plt

"""
THIS CODE WILL RUN FOR A NUMBER OF EPISODES AND PLOT THE EVALUATION METRICS
"""

NUM_FRAMES = 2  # Number of frames to stack
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)  # Auto select device to be either NVIDIA Graphics Card, Metal Performance or CPU


def evaluate_model(model_path="Model.pth", episodes=100):
    """
    Load the saved model and evaluate it for a given number of episodes.
    Print the average reward over these evaluation episodes and plot results.
    """
    env = gym.make("PongDeterministic-v4", render_mode="rgb_array")
    env = gym.wrappers.FrameStackObservation(env, NUM_FRAMES)
    state, _ = env.reset()
    env.render()
    state = preprocess(state)

    input_size = np.prod(state.shape)
    n_actions = env.action_space.n

    # Load model
    policy_net = PolicyNet(input_size, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    eval_rewards = []

    for ep in range(episodes):
        # Optionally record a video of the first episode
        if ep == 0:
            env = gym.wrappers.RecordVideo(
                env, "recordings", episode_trigger=lambda x: x == 0
            )
            state, _ = env.reset()
        else:
            state, _ = env.reset()

        state = preprocess(state)
        done = False
        ep_reward = 0

        while not done:
            s_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, _ = policy_net(s_tensor)
            dist = Categorical(policy)
            action = dist.sample().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess(next_state)
            ep_reward += reward
            state = next_state

        eval_rewards.append(ep_reward)
        print(f"Episode: {ep} Reward: {ep_reward}")

    avg_eval_reward = np.mean(eval_rewards)
    print(f"Average Evaluation Reward over {episodes} episodes: {avg_eval_reward}")

    env.close()

    # -------------------------------------
    # Plotting Evaluation Metrics
    # -------------------------------------
    # Compute running average (e.g., window of 10)
    window_size = 10
    if len(eval_rewards) >= window_size:
        running_avg_rewards = [
            np.mean(eval_rewards[max(0, i - window_size + 1) : i + 1])
            for i in range(len(eval_rewards))
        ]
    else:
        # If fewer than 10 episodes, just copy eval_rewards
        running_avg_rewards = eval_rewards.copy()

    plt.figure(figsize=(12, 6))

    # Plot episode rewards
    plt.plot(eval_rewards, label="Episode Rewards", color="blue")
    # Plot running average rewards
    plt.plot(running_avg_rewards, label=f"Running Avg ({window_size})", color="orange")

    plt.title("Evaluation Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
