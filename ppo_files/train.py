import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ale_py
import matplotlib.pyplot as plt

# ------------------------------
# Hyperparameters
# ------------------------------
MAX_EPISODES = 1500
MAX_STEPS_PER_EPISODE = 1000

# PPO parameters
GAMMA = 0.99  # Discount Factor
LAMBDA = 0.95  # GAE Param => Controls Bias-Variance Trade-off
LEARNING_RATE = 1e-4
VALUE_LOSS_WEIGHT = 0.5  # Measures how much value loss matters in total loss
ENTROPY_COEF = 0.01  # Entropy bonus for exploring stochastic situations in the policy
EPOCHS = 3  # How many times data gets passed through the forward/backward passes
MINI_BATCH_SIZE = 256  # Mini batch size experimented with 32, 64, 128 and 256
CLIP_RANGE = 0.2  #

# Environment Settings
NUM_FRAMES = 2  # Number of frames to stack

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)  # Auto select device to be either NVIDIA Graphics Card, Metal Performance or CPU


# -------------------------------
# Preprocessing: crop and downsample
# -------------------------------
def preprocess(frames, display=False):
    """
    Preprocessing Pong frames:
    1. Convert to numpy
    2. Crop image
    3. Downsample by factor of 2
    4. Remove background and binarize
    Resulting shape ~ (stacked_frames, 80, 80)
    """
    state = np.array(frames)  # shape (stack, H, W, C)

    if display:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(state[0], cmap="gray")
        axes[0].set_title("Original Frame")

    state = state[:, 35:195]  # crop

    if display:
        axes[1].imshow(state[0], cmap="gray")
        axes[1].set_title("Cropped Frame")

    state = state[:, ::2, ::2, 0]  # downsample and take single channel

    if display:
        axes[2].imshow(state[0], cmap="gray")
        axes[2].set_title("Downsampled Frame")

    state[state == 144] = 0  # Remove Background
    state[state == 109] = 0  # Remove Background
    state[state != 0] = 1  # Set Paddles and Ball to 1 (Binary)

    if display:
        axes[3].imshow(state[0], cmap="gray")
        axes[3].set_title("Binarized Frame")

        for i in range(NUM_FRAMES):
            axes[4].imshow(state[i], cmap="gray")
            axes[4].set_title(f"Final Frame {i+1}")
        plt.show()

    return state.astype(np.float32)


# -------------------------------
# Simple MLP Policy-Value Network
# π(a|s), V(s)
# -------------------------------
class PolicyNet(nn.Module):
    """
    A simple two-layer MLP that outputs:
    - π(a|s) as a categorical distribution over actions
    - V(s) as a scalar value.
    Our input s are the flattened images preprocessed (2x80x80).
    This model makes use of no convolutional layers to conduct our experiment
    and is successful in outputting the policy distribution alongside the value
    """

    def __init__(self, input_size, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc_policy = nn.Linear(256, n_actions)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Get the log probabilities
        policy_logits = self.fc_policy(x)
        # Run through softmax activation function to introduce non-linearity
        policy = F.softmax(policy_logits, dim=-1)
        # Get value as a scalar
        value = self.fc_value(x)
        return policy, value


# -------------------------------
# PPO Formula Functions
# -------------------------------
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    """
    Compute Generalized Advantage Estimator (GAE)
    A_t = δ_t + (γ * λ) * δ_{t+1} + ... + (γ * λ)^{T-t+1} * δ_{T-1}
    where δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    """
    # Initialize advantages
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    advantage = 0.0

    # We append the last value again for bootstrap
    values = np.append(values, values[-1])
    # We use this to improve the estimate of the current state's value based on our next state

    # Advantage is computed in reversed order as it depends on previous states
    for t in reversed(range(len(rewards))):
        # Compute delta
        # we add a 1 - dones[t] to ensure that the states that end up ending the episode aren't considered
        # gamme + values[t+1] is the discounted value of our next state
        # We subtract the value of the current state from our immediate reward and the discounted value to get
        # The difference between our predicted value andthe actual return
        # This is the delta_t -> temporal difference error
        delta_t = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        # Compute advantage based on our parameters
        advantage = delta_t + gamma * lam * (1 - dones[t]) * advantage
        # Store the advantage
        advantages[t] = advantage
        # Our return is considered the total advantage + the value of our current state
        # R_t = A_t + V(s_t)
        returns[t] = advantage + values[t]
    return advantages, returns


def ppo_ratio(new_log_probs, old_log_probs):
    """
    ratio = exp(log π_new(a|s) - log π_old(a|s))
    This is a ratio since the log ratio is taking a subtraction
    """
    return torch.exp(new_log_probs - old_log_probs)


def ppo_clipped_objective(ratio, advantages, clip_range=CLIP_RANGE):
    """
    L^{CLIP} = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε)*A ) ]
    """
    # We clip everything to 1 or 0, to stabilize our advantage
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return torch.min(ratio * advantages, clipped_ratio * advantages).mean()


def value_loss_function(values, returns):
    """
    L^V = (V(s) - R_t)^2
    Value is a tensor containing our predicted value of states, estimated by the value net
    Returns is a tensor
    """
    return (values.squeeze() - returns).pow(2).mean()


def entropy_of_dist(dist):
    """
    H(π) = -Σ π(a|s) log π(a|s)
    We calculate the entropy of our Categorical distribution,taken over our
    batch of states/actions. The final value is the average uncertainty in our
    in the action selection of our policy. This allows us to improve our exploration
    by factoring in our entropy bonus
    """
    return dist.entropy().mean()


def train():
    """Function containing the core logic of training
    NOTES FOR TRAINING, TRAINING THE MODEL FROM SCRATCH WITH THE
    CURRENT HYPERPARAMETERS TAKES APPROXIMATELY 1.5-2.5 hrs on
    M3 PRO, and Similar on a RTX 3070 and RTX 3080
    """
    # Create Pong environment
    env = gym.make("PongDeterministic-v4", render_mode="rgb_array")
    env = gym.wrappers.FrameStackObservation(env, NUM_FRAMES)

    state, _ = env.reset()  # Reset and grab the state observation
    state = preprocess(state)  # Preprocess the initial state
    input_size = np.prod(state.shape)  # Flatten the state's shape
    n_actions = env.action_space.n  # Number of available actions => 6

    # Initialize network and optimizer
    policy_net = PolicyNet(input_size, n_actions).to(device)  # Initialize policy_net
    optimizer = torch.optim.Adam(
        policy_net.parameters(), lr=LEARNING_RATE
    )  # Initialize optimizer

    # Metrics to log
    total_rewards = []  # Total rewards across episodes
    policy_losses_per_episode = []  # Policy losses per episode
    value_losses_per_episode = []
    entropies_per_episode = []
    total_losses_per_episode = []
    average_rewards_per_episode = []

    for episode in range(MAX_EPISODES):
        # Trajectories of actions to be stored in here
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
            # Convert state to tensor
            s_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            # Get policy and value output
            policy, value = policy_net(s_tensor)
            # Get distribution of the policy and sample action
            dist = Categorical(policy)
            action = dist.sample()
            # Get log probability of the action
            log_prob = dist.log_prob(action)
            action_item = action.item()
            # Take a step with this action
            next_state, reward, done, _, _ = env.step(action_item)
            # Crop, Downsample and Binarize next state observation
            next_state = preprocess(next_state)

            # Store transition
            states.append(state.flatten())
            actions.append(action_item)
            old_log_probs.append(log_prob.item())
            values.append(value.item())
            rewards.append(reward)
            dones_list.append(done)

            state = next_state  # Update our state
            if done:  # If game reached a terminal state we are done
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
        )  # Avoids division by zero and over-estimation
        # Also aids in converging faster

        # We'll track sums for this episode
        ep_policy_loss = 0.0
        ep_value_loss = 0.0
        ep_entropy = 0.0
        ep_total_loss = 0.0
        update_count = 0

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

                # ratio
                ratio = ppo_ratio(new_log_probs, batch_old_log_probs)

                # Policy loss = - L^{CLIP}
                policy_loss = -ppo_clipped_objective(ratio, batch_advantages)

                # Value loss
                v_loss = value_loss_function(values_pred, batch_returns)

                # Entropy
                ent = entropy_of_dist(dist)

                # Total PPO loss
                loss = policy_loss + VALUE_LOSS_WEIGHT * v_loss - ENTROPY_COEF * ent

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                # Accumulate losses for logging
                ep_policy_loss += policy_loss.item()
                ep_value_loss += v_loss.item()
                ep_entropy += ent.item()
                ep_total_loss += loss.item()
                update_count += 1

        # Compute average losses for this episode
        if update_count > 0:
            ep_policy_loss /= update_count
            ep_value_loss /= update_count
            ep_entropy /= update_count
            ep_total_loss /= update_count

        ep_reward = sum(rewards)
        total_rewards.append(ep_reward)
        avg_reward = np.mean(total_rewards[-100:])
        print(
            f"Episode {episode}, Reward: {ep_reward}, Avg(100): {avg_reward}, "
            f"Policy Loss: {ep_policy_loss:.4f}, Value Loss: {ep_value_loss:.4f}, "
            f"Entropy: {ep_entropy:.4f}, Total Loss: {ep_total_loss:.4f}"
        )

        # Log metrics
        policy_losses_per_episode.append(ep_policy_loss)
        value_losses_per_episode.append(ep_value_loss)
        entropies_per_episode.append(ep_entropy)
        total_losses_per_episode.append(ep_total_loss)
        average_rewards_per_episode.append(avg_reward)

        # Optional early stopping
        if avg_reward >= 5:
            print("Solved Pong!")
            torch.save(policy_net.state_dict(), "Model.pth")
            break

    else:
        # If loop completes without break:
        torch.save(policy_net.state_dict(), "Model.pth")

    env.close()
    # -----------------------
    # Plotting
    # -----------------------
    # Episode Reward Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(total_rewards, label="Episode Reward")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Policy Loss Plot
    plt.subplot(2, 3, 2)
    plt.plot(policy_losses_per_episode, label="Policy Loss")
    plt.title("Policy Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    # Value Loss Plot
    plt.subplot(2, 3, 3)
    plt.plot(value_losses_per_episode, label="Value Loss")
    plt.title("Value Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    # Entropy Plot
    plt.subplot(2, 3, 4)
    plt.plot(entropies_per_episode, label="Entropy")
    plt.title("Policy Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")

    # Total PPO Loss
    plt.subplot(2, 3, 5)
    plt.plot(total_losses_per_episode, label="Total PPO Loss")
    plt.title("Total PPO Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    # 100 Episode Average Reward
    plt.subplot(2, 3, 6)
    plt.plot(average_rewards_per_episode, label="Avg(100) Reward")
    plt.title("100-Episode Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (100)")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

    return


if __name__ == "__main__":

    train()
