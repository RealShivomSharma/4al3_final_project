import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import ale_py
import numpy as np
import pandas as pd
from collections import deque

RENDER_MODE_TRAIN = "rgb_array"
RENDER_MODE_VIEW = "human"
ACTION_MAP = {
    0: "NOOP",  # IDLE Action
    2: "RIGHT",  # Go Right -> Consider replacing with RightFire
    3: "LEFT",  # Go Left -> Consider replacing with LeftFire
}
epochs = 1000
device = "cpu"  # For training
MEAN_GOAL_REWARD = 19.5
N = 10_000  # Capaicty of ReplayBuffer
M = 10_000
gamma = 1e-3  # Discount rate
batch_size = 32
min_replay_size = 10_000
learning_rate = 1e-4
epsilon_start = 1.0
epsilon_end = 0.2
epsilon_decay_frames = 10**5


class ReplayBuffer:
    def __init__(self, capacity=N):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, data):
        self.buffer.append(data)

    def sample(self):
        return self.buffer[0]  # Need to grab a random sample from the experiences


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

        return self.layers(x / 255.0)


class Agent:
    def __init__(self):
        pass


def trainloop(model, optimizer, loss_fn):

    return


def preprocess(observation):

    observation = observation[34:-30, :, :]  # Crops the image
    observation = observation[::2, ::2, 0]  # Downsample by factor of 2
    observation[observation == 144] = 0  # Remove colours
    observation[observation == 72] = 0  # Remove colours
    observation[observation != 0] = 255  # Set all remaining colours to white

    return observation


def main():

    env = gym.make("ALE/Pong-v5", render_mode=RENDER_MODE_TRAIN).env  #
    env = gym.wrappers.FrameStackObservation(env, 4)
    rb = ReplayBuffer()  # Instantiate replay buffer of size N
    model = DQN(env.action_space.n).to(device)  # Create model and push to device
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    trainloop(model, optimizer, loss_fn=None)
    env.reset()

    return


if __name__ == "__main__":

    main()
