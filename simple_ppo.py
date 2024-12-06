import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ale_py

# -------------------------------
# Hyperparameters (simplified)
# -------------------------------
MAX_EPISODES = 500
MAX_STEPS_PER_EPISODE = 1000

# PPO parameters
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 1e-4
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
EPOCHS = 3
MINI_BATCH_SIZE = 256
CLIP_RANGE = 0.2

device = torch.device("mps")  # Change to "cuda" or "mps" if available.


# -------------------------------
# Preprocessing: crop and downsample
# -------------------------------
def preprocess(frames):
    """
    Preprocessing Pong frames:
    1. Convert to numpy
    2. Crop image
    3. Downsample by factor of 2
    4. Remove background and binarize
    Resulting shape ~ (stacked_frames, 80, 80)
    """
    state = np.array(frames)  # shape (stack, H, W, C)
    state = state[:, 35:195]  # crop
    state = state[:, ::2, ::2, 0]  # downsample and take single channel
    state[state == 144] = 0
    state[state == 109] = 0
    state[state != 0] = 1
    return state.astype(np.float32)


# -------------------------------
# Simple MLP Policy-Value Network
# π(a|s), V(s)
# -------------------------------
class SimpleMLP(nn.Module):
    """
    A simple two-layer MLP that outputs:
    - π(a|s) as a categorical distribution over actions
    - V(s) as a scalar value.

    Let input s be flattened images.
    """

    def __init__(self, input_size, n_actions):
        super(SimpleMLP, self).__init__()
        # A single hidden layer MLP
        self.fc1 = nn.Linear(input_size, 256)
        self.fc_policy = nn.Linear(256, n_actions)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Policy (as probabilities)
        policy_logits = self.fc_policy(x)
        policy = F.softmax(policy_logits, dim=-1)
        # Value
        value = self.fc_value(x)
        return policy, value


# -------------------------------
# PPO Formula Functions
# -------------------------------


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    """
    Compute Generalized Advantage Estimator (GAE):
    A_t = sum_{l=0}^{∞} (γλ)^l δ_{t+l},
    where δ_t = r_t + γV_{t+1} - V_t

    Here we do a finite sum since we have one episode.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    advantage = 0.0

    # We append the last value again for bootstrap
    values = np.append(values, values[-1])

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantage = delta + gamma * lam * (1 - dones[t]) * advantage
        advantages[t] = advantage
        returns[t] = advantage + values[t]
    return advantages, returns


def ppo_ratio(new_log_probs, old_log_probs):
    """
    ratio = exp(log π_new(a|s) - log π_old(a|s))
    """
    return torch.exp(new_log_probs - old_log_probs)


def ppo_clipped_objective(ratio, advantages, clip_range=CLIP_RANGE):
    """
    L^{CLIP} = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε)*A ) ]
    """
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return torch.min(ratio * advantages, clipped_ratio * advantages).mean()


def value_loss_function(values, returns):
    """
    L^V = (V(s) - R_t)^2
    """
    return (values.squeeze() - returns).pow(2).mean()


def entropy_of_dist(dist):
    """
    Entropy to encourage exploration:
    H(π) = -Σ π(a|s) log π(a|s)
    """
    return dist.entropy().mean()


# -------------------------------
# Training Loop
# -------------------------------
def main():
    # Create Pong environment
    env = gym.make("PongDeterministic-v4", render_mode="rgb_array")
    env = gym.wrappers.FrameStackObservation(env, 2)

    state, _ = env.reset()
    state = preprocess(state)
    # Flatten state: 2 frames * 80 * 80 = 12800
    input_size = np.prod(state.shape)
    n_actions = env.action_space.n

    # Initialize network and optimizer
    policy_net = SimpleMLP(input_size, n_actions).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    total_rewards = []

    for episode in range(MAX_EPISODES):
        states = []
        actions = []
        old_log_probs = []
        values = []
        rewards = []
        dones_list = []

        # Run one episode
        state, _ = env.reset()
        state = preprocess(state)

        for step in range(MAX_STEPS_PER_EPISODE):
            s_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            policy, value = policy_net(s_tensor)
            dist = Categorical(policy)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            action_item = action.item()

            next_state, reward, done, _, _ = env.step(action_item)
            next_state = preprocess(next_state)

            # Store transition
            states.append(state.flatten())
            actions.append(action_item)
            old_log_probs.append(log_prob.item())
            values.append(value.item())
            rewards.append(reward)
            dones_list.append(done)

            state = next_state
            if done:
                break

        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(rewards, values, dones_list)

        # Convert to torch
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(np.array(actions)).to(device)
        old_log_probs_t = torch.FloatTensor(np.array(old_log_probs)).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        # PPO update
        dataset = torch.utils.data.TensorDataset(
            states_t, actions_t, old_log_probs_t, returns_t, advantages_t
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=MINI_BATCH_SIZE, shuffle=True
        )

        for _ in range(EPOCHS):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in dataloader:
                # Forward pass
                policy, values_pred = policy_net(batch_states)
                dist = Categorical(policy)
                new_log_probs = dist.log_prob(batch_actions)

                # ratio = exp(new_log_probs - old_log_probs)
                ratio = ppo_ratio(new_log_probs, batch_old_log_probs)

                # Policy loss = - L^{CLIP}
                policy_loss = -ppo_clipped_objective(ratio, batch_advantages)

                # Value loss
                v_loss = value_loss_function(values_pred, batch_returns)

                # Entropy
                ent = entropy_of_dist(dist)

                # Total PPO loss
                loss = policy_loss + VALUE_LOSS_COEF * v_loss - ENTROPY_COEF * ent

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

        ep_reward = sum(rewards)
        total_rewards.append(ep_reward)
        avg_reward = np.mean(total_rewards[-100:])
        print(f"Episode {episode}, Reward: {ep_reward}, Avg(100): {avg_reward}")

        # Optional early stopping
        if avg_reward >= 5:
            print("Solved Pong!")
            break

    env.close()


if __name__ == "__main__":
    main()
