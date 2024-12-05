import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import numpy as np

# PPO Actor-Critic Network
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, output_dim)  # Action probabilities
        self.value_head = nn.Linear(hidden_dim, 1)  # Value prediction
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_dist = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy_dist, value

# Advantage computation
def compute_advantage(rewards, values, gamma, lamda, dones):
    advantage = []
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(rewards) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_advantage = delta + gamma * lamda * (1 - dones[t]) * last_advantage
        advantage.insert(0, last_advantage)
    return advantage

# PPO update function
def ppo_update(policy_net, value_net, optimizer, states, actions, rewards, dones, old_log_probs, gamma=0.99, lamda=0.95, epsilon=0.2):
    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)

    # Get new policy and value predictions
    new_policy_dist, new_values = policy_net(states)
    new_log_probs = torch.log(new_policy_dist.gather(1, actions.unsqueeze(-1)))

    # Compute advantage
    advantages = compute_advantage(rewards, new_values.detach(), gamma, lamda, dones)

    # PPO objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # Value function loss
    value_loss = F.mse_loss(new_values.squeeze(-1), rewards)

    # Total loss
    loss = surrogate_loss + 0.5 * value_loss  # Added weight to value loss

    # Update the policy network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize environment and policy
env = gym.make("Pong-v0")  # Choose your environment
observation_space = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]  # Flattened image
action_space = env.action_space.n  # Number of possible actions

policy_net = PPOActorCritic(input_dim=observation_space, hidden_dim=64, output_dim=action_space)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)

num_epochs = 1000  # Set the number of training epochs
gamma = 0.99
lamda = 0.95
epsilon = 0.2
max_timesteps = 1000  # Max timesteps per episode

# Training loop
for epoch in range(num_epochs):
    states, actions, rewards, dones, old_log_probs = [], [], [], [], []

    # Get the initial state (handling tuple return)
    state, _ = env.reset()  # Reset and only use the first element (state)
    done = False
    while not done:
        # Convert state to grayscale, flatten it, and normalize
        state = np.array(state)  # Ensure it's a numpy array
        state = np.mean(state, axis=2)  # Convert to grayscale (average over RGB channels)
        state = state.flatten()  # Flatten the image to 1D
        
        # Normalize the state (optional, recommended for stability)
        state = (state - np.mean(state)) / np.std(state)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        policy_dist, value = policy_net(state_tensor)
        action = torch.multinomial(policy_dist, 1).item()  # Sample action
        log_prob = torch.log(policy_dist[action])  # Log probability of chosen action
        
        next_state, reward, done, _ = env.step(action)  # Step in environment
        
        # Store experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(log_prob.item())
        
        state = next_state

        if len(states) >= max_timesteps:  # End episode if max timesteps reached
            break

    # Update PPO policy
    ppo_update(policy_net, policy_net, optimizer, states, actions, rewards, dones, old_log_probs)
