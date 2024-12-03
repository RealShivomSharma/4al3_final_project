import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ale_py
import random
import os
import time
from torch.utils.tensorboard import SummaryWriter
from collections import deque

ACTION_MAP = {
    0: "NOOP",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
} # REDUCED SELECTION THESE SHOULD BE ONLY MOVES REQUIRED TO PLAY (FIRE NEEDED TO RESET/START ATARI ENVIRONMENT)
COMPLETE_ACTION_MAP = {
    0:"NOOP",
    1:"FIRE",
    2:"RIGHT",
    3:"LEFT",
    4:"RIGHTFIRE",
    5:"LEFTFIRE",
}
MEAN_GOAL_REWARD = 19.5  # Final score goal over evaluations
MAX_EPISODES = 10_000  # Max number of episodes to iterate over
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32  # Batch Size
LEARNING_RATE = 1e-4  # Learning Rate
RANDOM_SEED = 42  # RANDOM SEED TO ENSURE CONSISTENCY
TRAIN = False  # Global flag to enable training
AGENT_HISTORY_LENGTH = 4  # Number of frames to stack as input to network
EPSILON_START = 1.0
EPSILON_MIN = 0.2
EPSILON_DECAY_FACTOR = 10**5
MIN_REPLAY_SIZE = 1000  # Define a minimum replay size
TARGET_UPDATE_FREQ = 100
RESUME_TRAINING = True


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        # Need to add a check if the buffer is full, unless there is internal handling
        self.buffer.append(  # append a tuple of experience to buffer
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.stack(states), actions, rewards, np.stack(next_states), dones


class DQN(nn.Module):
    def __init__(self, n_actions):  # Input image, must undergo downsampling
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            # nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):

        return self.layers(x / 255.0)


class Agent:
    def __init__(self, env, replay_memory, n_actions, device, epsilon):
        self.env = env
        self.rm = replay_memory
        self.device = device
        self.epsilon = epsilon
        self.n_actions = n_actions

        # Q NETWORKS
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
        )

    def take_action(self, state):  # Greedy policy

        if np.random.random() < self.epsilon:  # Greedy action
            return np.random.choice(list(ACTION_MAP.keys())) # Reduced action set to the ones that "matter" to us

        with torch.no_grad(): # Without updating the gradients
            state = torch.tensor(state, device=self.device).unsqueeze(0) # Conver the state to a tensor 
            q_values = self.policy_net(state) # pass it to the policy to grab the q value
            return q_values.argmax().item() # Get the max q value from the network

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Copy over the policy to the target network done accordingly with TARGET_UPDATE_FREQ


def preprocess(state):
    state = state[35:195]  # Crop the image
    state = state[::2, ::2, 0]  # Downsample by factor of 2
    state[state == 144] = 0  # Erase background (background type 1)
    state[state == 109] = 0  # Erase background (background type 2)
    state[state != 0] = 1  # Set everything else (paddles, ball) to 1
    return state.astype(np.float32)


def save_checkpoint(agent, episode, checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
    torch.save(
        {
            "episode": episode,
            "policy_net_state_dict": agent.policy_net.state_dict(),
            "target_net_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved at episode {episode}")


def load_checkpoint(agent, checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from {checkpoint_path}")


def main():
    writer = SummaryWriter()
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    trigger = False

    episode_trigger = lambda episode : episode % 100 == 0
    env = gym.wrappers.RecordVideo(
        env, video_folder="./videos", episode_trigger=episode_trigger, disable_logger=True
    )
    device = "cuda"
    rm = ReplayMemory(capacity=50_000)
    # agent = Agent(env, rm, n_actions=env.action_space.n, device="cpu")
    start_episode = 1
    checkpoint_dir = "./checkpoints"
    episode_rewards = []  # Grab the rewards from each episode
    agent = Agent(
        env,
        rm,
        n_actions=env.action_space.n,
        device=device,
        epsilon=EPSILON_START,
    )
    action_counts = {
        "NOOP": 0,
        "FIRE": 0,
        "RIGHT": 0,
        "LEFT": 0,
        "RIGHTFIRE": 0,
        "LEFTFIRE": 0,
    }
    if os.path.exists(checkpoint_dir) and RESUME_TRAINING:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            load_checkpoint(agent, checkpoint_path)
            start_episode = int(latest_checkpoint.split("_")[1].split(".")[0]) + 1
    
    loss_count = 0
    running_loss = 0.0
    for episode in range(start_episode, MAX_EPISODES):
        time_start = time.time()
        state, _ = env.reset()
        state = preprocess(state) / 255.0

        # Initialize the frame stack (stack 4 frames)
        frame_stack = deque(
            [np.zeros_like(state) for _ in range(AGENT_HISTORY_LENGTH)],
            maxlen=AGENT_HISTORY_LENGTH,
        )
        frame_stack.append(state)
        stacked_state = np.stack(frame_stack, axis=0)

        done = False
        episode_reward = 0.0
        epsilon_decay = (EPSILON_START - EPSILON_MIN) / EPSILON_DECAY_FACTOR
        while not done:
            agent.epsilon = max(
                EPSILON_MIN, agent.epsilon - epsilon_decay
            )  # Decay the exploration
            action = agent.take_action(stacked_state)
            action_name = COMPLETE_ACTION_MAP.get(action)
            action_counts[action_name] +=1
            next_state, reward, done, _, _ = env.step(action)
            reward = np.clip(reward, -1, 1)  # Reward clipping
            next_state = preprocess(next_state)
            frame_stack.append(next_state)
            stacked_next_state = np.stack(frame_stack, axis=0)
            agent.rm.push(stacked_state, action, reward, stacked_next_state, done)
            stacked_state = stacked_next_state
            episode_reward += reward

            if len(agent.rm) > MIN_REPLAY_SIZE:
                batch = agent.rm.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                states = torch.tensor(states, device=device,dtype=torch.float32)
                actions = torch.tensor(actions, device=device,dtype=torch.int64)
                rewards = torch.tensor(rewards, device=device,dtype=torch.float32)
                next_states = torch.tensor(next_states, device=device,dtype=torch.float32)
                dones = torch.tensor(dones, device=device, dtype=torch.bool)

                # print(episode_reward)

                # Compute Q values for current states
                q_values = (
                    agent.policy_net(states).gather(1, actions.unsqueeze(1).to(torch.int64)).squeeze(1)
                )

                # Compute Q values for next states
                with torch.no_grad():
                    next_q_values = agent.target_net(next_states).max(1)[0]

                # Compute target Q values
                target_q_values = rewards + (GAMMA * next_q_values * (~dones))

                # Compute loss
                loss = F.mse_loss(q_values, target_q_values)

                running_loss += loss.item()
                loss_count +=1


                # Optimize the model
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

                writer.add_scalar("TargetQ-Value/Action", target_q_values.max().item(), episode) 
                writer.add_scalar("Q-Value/Action", q_values.max().item(), episode)
                writer.add_scalar("Loss/Episode", loss, episode)
                # env.render()
 
        time_end = time.time()
        avg_loss = running_loss / loss_count if loss_count > 0 else 0 
        writer.add_scalar("Epsilon/Episode", agent.epsilon, episode)
        writer.add_scalar("Loss/Running_Avg", avg_loss, episode)
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Time/Episode", time_end-time_start, episode)
        writer.add_scalar("AvgReward/Last100Episodes", np.mean(episode_rewards[-100:]), episode)
        writer.add_scalars("Action/Distribution", action_counts, episode)
        action_counts = {action: 0 for action in action_counts.keys()}  # Reset action counts
        running_loss = 0.0
        loss_count = 0
        writer.flush()
        episode_rewards.append(episode_reward)
        if episode % TARGET_UPDATE_FREQ == 0:
            save_checkpoint(agent, episode, checkpoint_dir)
            agent.update_target_net()  # Copy over the network on a given interval


        writer.close()
        env.close()

    torch.save(agent.policy_net.state_dict(), "./models/model_policy.pth")
    torch.save(agent.target_net.state_dict(), "./models/model_target.pth")
    env.close()

    return


if __name__ == "__main__":
    main()
