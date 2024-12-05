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

# TODO 
# TESTING FUNCTIONS 
# TRAINING (PUT MAIN LOOP STUFF INTO FUNCTION)
# CODE CLEANUP
# ADD LOADING IN FROM MODEL (i.e. COMPLETED MODEL)
# ADD PLOTTING FOR METRICS/PRINTING OF METRICS FOR MILESTONE #2

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
MEAN_GOAL_REWARD = 16  # Final score goal over evaluations
MAX_EPISODES = 10_000  # Max number of episodes to iterate over
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32  # Batch Size
LEARNING_RATE = 1e-4  # Learning Rate
RANDOM_SEED = 42  # RANDOM SEED TO ENSURE CONSISTENCY
TRAIN = False  # Global flag to enable training
AGENT_HISTORY_LENGTH = 4  # Number of frames to stack as input to network
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_FACTOR = 10**5
MIN_REPLAY_SIZE = 10_000  # Define a minimum replay size
TARGET_UPDATE_FREQ = 1000
PRINT_STATS_INTERVAL = 100
RESUME_TRAINING = False



class ReplayMemory:
    def __init__(self, capacity, state_shape, n_actions):
        self.capacity = capacity
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.buffer = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_buffer = np.empty((capacity, *state_shape), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.idx = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done):
        idx = self.idx % self.capacity
        self.buffer[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_buffer[idx] = next_state
        self.dones[idx] = done
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Randomly sample indices
        idxs = np.random.choice(self.size, batch_size, replace=False)
        states = self.buffer[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_buffer[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones



class DQN(nn.Module):
    def __init__(self, n_actions):  # Input image, must undergo downsampling
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), # Hidden layer size (MAYBE MAKE THIS A VARIABLE)
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x / 255.0  # Normalize outside, pass pre-converted tensor
        return self.layers(x)

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
        else:
            with torch.no_grad(): # Without updating the gradients
                state = torch.tensor(state, device=self.device).unsqueeze(0) # Conver the state to a tensor 
                q_values = self.policy_net(state) # pass it to the policy to grab the q value
                return q_values.argmax().item() # Get the max q value from the network

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Copy over the policy to the target network done accordingly with TARGET_UPDATE_FREQ


def preprocess(state):
    # No need to crop or grayscale, as AtariPreprocessing has handled this
    state = state / 255.0  # Normalize the pixel values to [0, 1]
    
    # Apply the binary thresholding step only if needed (depends on environment)
    state[state == 144] = 0  # Erase background type 1
    state[state == 109] = 0  # Erase background type 2
    state[state != 0] = 1  # Set everything else (paddles, ball, etc.) to 1
    
    return state.astype(np.float32)  # Return as float32 for better precision




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

    # epsilons = np.array(epsilons)

    print(f"Checkpoint saved at episode {episode}")


def load_checkpoint(agent, checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from {checkpoint_path}")

def test():

    pass


def train():
    # Setup and training loop
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    # env= gym.wrappers.AtariWrapper(env)
    seed = 31
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) # Dynamically get device 
    print(device)
    rm = ReplayMemory(capacity=10_000,state_shape = env.observation_space.shape , n_actions = env.action_space.n)
    agent = Agent(env, rm, n_actions=env.action_space.n, device=device, epsilon=EPSILON_START)
    episode_rewards = []
    episode_lengths = []
    epsilons = []
    action_counts = {action: 0 for action in COMPLETE_ACTION_MAP.values()}
    # epsilon_decay = EPSILON
    for episode in range(MAX_EPISODES):
        agent.epsilon = max(EPSILON_MIN, agent.epsilon*0.995)
        start_time = time.time()

        state, _ = env.reset()
        # print(state.shape)
        # state = preprocess(state)
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = agent.take_action(state)
            action_name = COMPLETE_ACTION_MAP.get(action)
            action_counts[action_name] += 1
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            # next_state = preprocess(next_state)

            # Store transition in memory
            agent.rm.push(state, action, reward, next_state, done)
            state = next_state
            
            if len(agent.rm) > MIN_REPLAY_SIZE:
                batch = agent.rm.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                states = torch.tensor(states, device=device, dtype=torch.float32)
                actions = torch.tensor(actions, device=device, dtype=torch.long)
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
                next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
                dones = torch.tensor(dones, device=device, dtype=torch.bool)

                # Calculate loss and optimize
                q_values = agent.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = agent.target_net(next_states).max(1)[0]
                    target_q_values = rewards + GAMMA * next_q_values * (~dones)

                loss = F.smooth_l1_loss(q_values, target_q_values)
                agent.optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1)
                loss.backward()
                agent.optimizer.step()

        elapsed_time = time.time() - start_time
        fps = episode_length / elapsed_time
        print(f"Episode {episode}, FPS: {fps:.2f}") 

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilons.append(agent.epsilon)
        # Save and update networks periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            save_checkpoint(agent, episode)
            agent.update_target_net()

        # Print progress periodically
        if episode % PRINT_STATS_INTERVAL == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:])}, Epsilon: {agent.epsilon:.3f}, Episode Length: {np.mean(episode_lengths[-100:])}")

        # Early stopping criteria
        if np.mean(episode_rewards[-100:]) > MEAN_GOAL_REWARD:
            break

def main():
    train()
    # # writer = SummaryWriter()
    # env = gym.make("Pong-v4", render_mode="rgb_array")
    # env = gym.wrappers.FrameStackObservation(env, 4)
    # # episode_trigger = lambda episode: episode % 100 == 0 
    # # env = gym.wrappers.RecordVideo(
    # #     env, video_folder="./videos", episode_trigger=episode_trigger, disable_logger=True
    # # )
    # seed = 31
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) # Dynamically get device 
    # rm = ReplayMemory(capacity=50_000)
    # # agent = Agent(env, rm, n_actions=env.action_space.n, device="cpu")
    # checkpoint_dir = "./checkpoints"
    # episode_rewards = []  # Grab the rewards from each episode
    # best_reward = float('-inf')
    # agent = Agent(
    #     env,
    #     rm,
    #     n_actions=env.action_space.n,
    #     device=device,
    #     epsilon=EPSILON_START,
    # )
    # action_counts = {
    #     "NOOP": 0,
    #     "FIRE": 0,
    #     "RIGHT": 0,
    #     "LEFT": 0,
    #     "RIGHTFIRE": 0,
    #     "LEFTFIRE": 0,
    # }
    # if os.path.exists(checkpoint_dir) and RESUME_TRAINING:
    #     checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    #     if checkpoints:
    #         latest_checkpoint = max(
    #             checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
    #         )
    #         checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    #         load_checkpoint(agent, checkpoint_path)
    #         start_episode = int(latest_checkpoint.split("_")[1].split(".")[0]) + 1
    #         if hasattr(env, 'episode_id'):
    #             env.episode_id = start_episode  # Sync episode_id with the loaded checkpoint
    # else:
    #     start_episode=1
    # print(f"STARTING FROM {start_episode}")
    # loss_count = 0
    # running_loss = 0.0
    # epsilons = []
    # losses = []
    # rewards = []
    # AvgReward = []
    # episode_lengths = []
    # timestep = 0
    # for episode in range(start_episode, MAX_EPISODES):
    #     # print(env.reset())
    #     state, _ = env.reset()
    #     state = preprocess(state)
    #     fig, axes = plt.subplots(1, 4, figsize=(16,4))
    #     for i in range(4):
    #         axes[i].imshow(state[i], cmap='gray')
    #         axes[i].axis('off')
    #         axes[i].set_title(f"Frame {i+1}")
    #     plt.show()
    #     plt.savefig("testplots")
    #     # plt.imshow(state.squeeze(1), cmap='gray') 
    #     # state = preprocess(state) / 255.0

    #     # Initialize the frame stack (stack 4 frames)
    #     # frame_stack = deque(
    #     #     [np.zeros_like(state) for _ in range(AGENT_HISTORY_LENGTH)],
    #     #     maxlen=AGENT_HISTORY_LENGTH,
    #     # )
    #     # frame_stack.append(state)
    #     # stacked_state = np.stack(frame_stack, axis=0)

    #     done = False
    #     episode_reward = 0.0
    #     episode_length = 0.0
    #     epsilon_decay = (EPSILON_START - EPSILON_MIN) / EPSILON_DECAY_FACTOR
    #     while not done:
    #         agent.epsilon = max(
    #             EPSILON_MIN, agent.epsilon - epsilon_decay
    #         )  # Decay the exploration
    #         action = agent.take_action(state)
    #         action_name = COMPLETE_ACTION_MAP.get(action)
    #         action_counts[action_name] +=1
    #         next_state, reward, done, _, _ = env.step(action)
    #         episode_length +=1 
    #         reward = np.clip(reward, -1, 1)  # Reward clipping
    #         next_state = preprocess(next_state)
    #         # frame_stack.append(next_state)
    #         # stacked_next_state = np.stack(frame_stack, axis=0)
    #         agent.rm.push(state, action, reward, next_state, done)
    #         state = next_state 
    #         episode_reward += reward

    #         if len(agent.rm) > MIN_REPLAY_SIZE:
    #             batch = agent.rm.sample(BATCH_SIZE)
    #             states, actions, rewards, next_states, dones = batch

    #             states = torch.tensor(states, device=device,dtype=torch.float32)
    #             actions = torch.tensor(actions, device=device,dtype=torch.int64)
    #             rewards = torch.tensor(rewards, device=device,dtype=torch.float32)
    #             next_states = torch.tensor(next_states, device=device,dtype=torch.float32)
    #             dones = torch.tensor(dones, device=device, dtype=torch.bool)

    #             # print(episode_reward)

    #             # Compute Q values for current states
    #             q_values = (
    #                 agent.policy_net(states).gather(1, actions.unsqueeze(1).to(torch.int64)).squeeze(1)
    #             )

    #             # Compute Q values for next states
    #             with torch.no_grad():
    #                 next_q_values = agent.target_net(next_states).max(1)[0]

    #             # Compute target Q values
    #             target_q_values = rewards + (GAMMA * next_q_values * (~dones))

    #             # Compute loss
    #             loss = F.smooth_l1_loss(q_values, target_q_values) # Huber loss


    #             # Optimize the model
    #             agent.optimizer.zero_grad()
    #             torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1)
    #             loss.backward()
    #             agent.optimizer.step()

    #             timestep+=1


    #             if episode % 1 == 0:
    #                 print(f"Episode {episode} , Timestep {timestep}, Average Total Reward / 100 Episodes {np.mean(episode_rewards[-100:]):.1f}, Episode Length = {np.mean(episode_lengths[-100:]):.1f} , Loss {loss.item()})")
    #                 print(pd.DataFrame_from_records(action_counts))

    #     episode_lengths.append(episode_length) 
    #     epsilons.append(agent.epsilon)
    #     action_counts = {action: 0 for action in action_counts.keys()}  # Reset action counts
    #     running_loss = 0.0
    #     loss_count = 0
    #     episode_rewards.append(episode_reward)
    #     if episode % TARGET_UPDATE_FREQ == 0:
    #         save_checkpoint(agent, episode, checkpoint_dir)
    #         agent.update_target_net()  # Copy over the network on a given interval
        
    #     if np.mean(episode_rewards[-100:]) > MEAN_GOAL_REWARD:
    #         env.close()
    #         break

    #         # reward reaches 19 18 etc 
        
    #     env.close()

    # torch.save(agent.policy_net.state_dict(), "./models/model_policy.pth")
    # torch.save(agent.target_net.state_dict(), "./models/model_target.pth")
    # env.close()

    return


if __name__ == "__main__":
    main()
