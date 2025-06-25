import gymnasium as gym
from DQN import DQN
from replay_buffer import ReplayBuffer
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import ale_py
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils as utils
from collections import deque


class Agent:

    def __init__(self):

        pass


class DQN(nn.Module):
    def __init__(self, action_space):  # Input image, must undergo downsampling
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
        )

    def forward(self, x):

        return self.layers(x / 255)


ACTION_MAP = {  # DISCRETE ACTION STATES
    0: "NOOP",  # DO NOTHING
    # 1: "FIRE",  # FIRE BUTTON WITHOUT UPDATING JOYSTICK POSITION
    2: "RIGHT",  # APPLY A DELTA UPWARDS ON JOYSTICK
    3: "LEFT",  # APPLY A DELTA LEFTWARDS ON JOYSTICK
    # 4: "RIGHTFIRE",  # EXECUTE RIGHT AND FIRE
    # 5: "LEFTFIRE",  # EXECUTE LEFT AND FIRE
}  # We ignore rightfire, leftfire and fire because they don't matter in pong


def main():

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    # env = gym.wrappers.FrameStackObservation(env, 4)
    env.reset()

    ACTION_MAP = {  # DISCRETE ACTION STATES
        0: "NOOP",  # DO NOTHING
        # 1: "FIRE",  # FIRE BUTTON WITHOUT UPDATING JOYSTICK POSITION
        2: "RIGHT",  # APPLY A DELTA UPWARDS ON JOYSTICK
        3: "LEFT",  # APPLY A DELTA LEFTWARDS ON JOYSTICK
        # 4: "RIGHTFIRE",  # EXECUTE RIGHT AND FIRE
        # 5: "LEFTFIRE",  # EXECUTE LEFT AND FIRE
    }  # We ignore rightfire, leftfire and fire because they don't matter in pong

    epochs = 1000

    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available() else "cpu"
    # )

    device = "cpu"

    model = DQN(env.action_space.n).to(device)

    MEAN_GOAL_REWARD = 19.5  # Could set higher depending on training restrictions
    N = 10_000
    M = 1_00_000
    gamma = 1e-3
    batch_size = 32
    min_replay_size = 10_000
    learning_rate = 1e-4
    sync_frames = 1000

    epsilon_start = 1.0
    epsilon_final = 0.02
    epsilon_decay_frames = 10**5

    rb = ReplayBuffer(capacity=N)  # Initialize Replay buffer to a capacity value

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    # A = {1,....K} action se passed ot the emulator modifying internal state and game score
    # Environment is stochastic, agent observes image
    # vector of raw pixel values representing current screen
    # receives reward r
    # game score depend on whole prior sequence of actions and observations
    # feedback about an action may only be received after many thousands of time-steps have elapsed
    # We must understand the game from a sequence of actions
    # This creates a MDP, each sequence is a distinct state
    # we can apply RL Methods for MDPs
    # use complete sequence s_t as state representation at time t
    # make choices that maximiie the future rewards, discount by a factor of gamma per time-step
    # future discounted reward -> R_t = \sum_{t'=t}^{T} \gamma^{t'-t}r_{t'}
    # T is the timestep at which the game terminates
    # Optimal action value function Q*(s,a) as maximum expected return achievable by following any strategy
    # after seeing some sequence s and then taking some action a, Q*(s,a) = max_{\pi}\mathbb{E}[R_t|s_t = s, a_t = a, \pi]
    # \pi is a policy mapping sequences to actions or distributions over actions
    # This obeys the bellman equation
    env.reset()
    best_reward = -float("inf")
    actions_rewards_counts = {key: [] for key in ACTION_MAP.keys()}
    for episode in range(1, 100):
        for t in range(1, N):
            action = np.random.choice(list(ACTION_MAP.keys()))
            obs, reward, terminated, truncated, info = env.step(action)
            stacked_frames = np.stack(frame_stack, axis=0)
            if terminated:
                env.reset()
                break
            actions_rewards_counts[action].append(reward)
            plt.imshow(stacked_frames[-1], cmap="gray")
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        if terminated:
            break

    # action_reward_df = pd.DataFrame.from_dict(
    #     actions_rewards_counts, orient="index"
    # ).transpose()

    # Plotting action vs reward
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (action, rewards) in enumerate(actions_rewards_counts.items()):
        axes[idx].plot(rewards)
        axes[idx].set_title(f"Action: {ACTION_MAP[action]}")
        axes[idx].set_xlabel("Step")
        axes[idx].set_ylabel("Reward")
    plt.tight_layout()
    plt.show()

    env.close()
    return


def training(rb: ReplayBuffer, model, optimizer, batch_size, gamma, device, env):

    states, actions, rewards, next_states, dones = rb.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    M = 1_000_000
    T = 10_000

    for episode in range(1, M):
        # sequence initialization
        states, actions, rewards, next_states, done = env.reset()

        for t in range(1, T):

            action = np.random.choice(list(ACTION_MAP.keys()))
            states, actions, rewards, next_states, done = env.step(action)

    return


def preprocess(observation):
    """

    Extract important details from the simulation environment


    Args:
        observation ([TODO:type]): observation
    """
    observation = observation[34:-30, :, :]  # Crops the image
    observation = observation[::2, ::2, 0]  # Downsample by factor of 2
    observation[observation == 144] = 0  # Remove colours
    observation[observation == 72] = 0  # Remove colours
    observation[observation != 0] = 255  # Set all remaining colours to white

    return observation  # Return preprocessed observation


def record_best_performance():
    """[TODO:summary]
    Need a function to output the best result (save a gif)

    [TODO:description]
    """

    pass


def frame_stack():



if __name__ == "__main__":
    # env = gym.make("ALE/Pong-v5", render_mode="human").env
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayscaleObservation(env)
    # env = gym.wrappers.FrameStackObservation(env, 4)

    main()
