import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import ale_py
import numpy as np
import pandas as pd
from collections import deque
import random

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
M = 10_000
T = 10_000  # Capacity of replay buffer
N = 50_000
gamma = 1e-3  # Discount rate
batch_size = 32
min_replay_size = 10_000
learning_rate = 1e-4
epsilon_start = 1.0
epsilon_min = 0.2
epsilon_decay_frames = 10**5
RANDOM_SEED = 42


class ReplayBuffer:
    def __init__(self, capacity=N):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, dones):
        experience = (  # Create a tuple of the experience and add it to the buffer
            state,
            action,
            reward,
            next_state,
            dones,
        )
        self.buffer.append(experience)

    def sample(self, batch_size):
        # indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # states, actions, rewards, dones, next_states = zip(
        #     *[self.buffer[i] for i in indices]
        # )
        # return (
        #     np.array(states) / 255,
        #     np.array(actions),
        #     np.array(rewards, dtype=np.float32),
        #     np.array(dones, dtype=np.uint8),
        #     np.array(next_states) / 255,
        # )
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.stack(states), actions, rewards, np.stack(next_states), dones


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
    def __init__(self, env, replay_memory):
        self.env = env
        self.rb = replay_memory
        self.state = env.reset()
        self.total_reward = 0.0
        self.last_action = 0

    def take_action(self, state):  # Returns the Q Values

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)

            return q_values.argmax(dim=1).item()

    def optimize_model(self):
        if self.rb.__len__() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.rb.sample(self.batch_size)

        # TENSOR CONVERSION

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max[1][0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.target_net.load_state_dict(self.policy_net.state_dict())


def trainloop(agent, model, optimizer, env):

    total_rewards = []
    # experience = tuple(states, actions, rewards, next_states)

    for episode in range(1, M):

        total_reward = 0.0
        state, info = env.reset()
        # states, actions, rewards, dones, next_states = env.reset()
        # state = preprocess(states)
        state_stack = [state] * 4  # 4 frames stacked
        done = False

        # Get sequence
        for t in range(1, T):
            state_input = np.array(state_stack)
            action = agent.take_action(state_input)
            next_frame, reward, done, info = env.step(action)
            next_state = preprocess(next_frame)
            state_stack.append(next_state)
            state_stack.pop(0)

            next_state_input = np.array(state_stack)
            agent.rb.push((state_input, action, reward, next_state_input, done))
            agent.optimizer()

            total_reward += reward

            total_rewards.append(total_reward)

    return total_rewards


def preprocess(observation):

    observation = observation[::2, ::2, 0]  # Downsample by factor of 2
    observation[observation == 144] = 0  # Remove colours
    observation[observation == 72] = 0  # Remove colours
    observation[observation != 0] = 255  # Set all remaining colours to white
    observation = observation.mean(axis=2)  # GRAYSCALE

    return observation


def main():

    env = gym.make("ALE/Pong-v5", render_mode=RENDER_MODE_TRAIN).env
    rb = ReplayBuffer()  # Instantiate replay buffer of size N
    model = DQN(env.action_space.n).to(device)  # Create model and push to device
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # action = np.random.choice(ACTION_MAP.keys())
    # states, actions, rewards, dones, next_states = env.step(action)

    agent = Agent(env, replay_memory=rb)

    trainloop(agent, model, optimizer, env=env)

    return


if __name__ == "__main__":

    main()
