import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ale_py
import random
from collections import deque

# Map actions in the discrete environment
ACTION_MAP = {
    0: "NOOP",
    # 1: "FIRE",
    # 2: "RIGHT",
    # 3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}
MEAN_GOAL_REWARD = 19.5  # Final score goal over evaluations
MAX_EPISODES = 10_000  # Max number of episodes to iterate over
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32  # Batch Size
LEARNING_RATE = 1e-4  # Learning Rate
RANDOM_SEED = 42  # RANDOM SEED TO ENSURE CONSISTENCY
TRAIN = False  # Global flag to enable training
AGENT_HISTORY_LENGTH = 4  # Number of frames to stack as input to network


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        experience = (
            state,
            action,
            reward,
            next_state,
            done,
        )
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.stack(states), actions, rewards, np.stack(next_states), dones


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # Adjust kernel size
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),  # Use adaptive pooling to reduce output size dynamically
            nn.Flatten(),
            nn.Linear(64, 512),  # Match the flattened size of AdaptiveAvgPool2d output
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.layers(x / 255.0)


class Agent:  # The actual player for our RL environment
    def __init__(self, env, experience_buffer, n_actions, device):
        self.env = env
        self.eb = experience_buffer
        self.device = device
        self.epsilon = 0.2

        # Q Networks
        self.policy = DQN(n_actions).to(device)  # Policy network
        self.target = DQN(n_actions).to(
            device
        )  # TARGET NETWORK , we only update this every few runs, stabilizes our updates
        # Optimizer for the network
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=LEARNING_RATE)

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def take_action(self, state):
        # Greedy epsilon algorithm
        if np.random.random() < self.epsilon:
            return np.random.choice(list(ACTION_MAP.keys()))
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy(state)
            _, action = q_values.max(1)
            return action.item()

        # return np.argmax(q_value)

    def optimize(self):
        states, actions, rewards, next_states, dones = self.eb.sample(BATCH_SIZE)

        pass


def preprocess(observation):

    observation = observation[34:-16, :, :]  # Crops the image
    observation = observation[::2, ::2]  # Downsample by factor of 2
    observation = np.mean(observation, axis=2).astype(np.uint8)  # Convert to grayscale
    observation[observation == 144] = 0  # Remove colours
    observation[observation == 72] = 0  # Remove colours
    observation[observation != 0] = 255  # Set all remaining colours to white
    observation = np.expand_dims(observation, axis=0)  # Add channel dimension

    return observation


def main():

    env = gym.make("Pong-v4", render_mode="rgb_array")
    eb = ExperienceBuffer(capacity=50000)
    agent = Agent(
        env,
        eb,
        n_actions=env.action_space.n,
        device="cpu",
    )

    state, _ = env.reset()
    for episode in range(1, MAX_EPISODES):

        # Implicit loop for t in range(1, T) -> T=> Timestep where it terminates

        t = 0
        done = False

        state = preprocess(state)
        while not t == 10:
            # stacked_frames = deque(  # Frame stacker
            #     [state] * AGENT_HISTORY_LENGTH,
            #     maxlen=AGENT_HISTORY_LENGTH,
            # )
            action = agent.take_action(state)
            print(action)
            next_state, reward, done, _, _ = env.step(action)
            state = preprocess(next_state)
            t += 1  # Add to the timestep count

        state, _ = env.reset()

        state = preprocess(state)

    # while not TRAIN:
    #     states, actions = env.reset()
    #     action = random.choice(list(ACTION_MAP.keys()))
    #     next_state, reward, done, _, _ = env.step(action)

    #     # plt.imshow(next_state)
    #     plt.imshow(preprocess(next_state), cmap="gray")  # Show grayscale image
    #     plt.show()

    #     # print(next_state)
    #     # print(reward)

    return


if __name__ == "__main__":

    main()
