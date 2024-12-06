import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
from torch.distributions import Categorical
from train import PolicyNet, preprocess

NUM_FRAMES = 2  # Number of frames to stack
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)  # Auto select device to be either NVIDIA Graphics Card, Metal Performance or CPU


def evaluate_model(model_path="mlp_ppo_pong_model_2.pth", episodes=10):
    """
    Load the saved model and evaluate it for a given number of episodes.
    Print the average reward over these evaluation episodes.
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
        avg_eval_reward = np.mean(eval_rewards)
    print(f"Average Evaluation Reward over {episodes} episodes: {avg_eval_reward}")
    env.close()


if __name__ == "__main__":

    evaluate_model()
