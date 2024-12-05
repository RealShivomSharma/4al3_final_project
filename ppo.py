import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ale_py
from torch.distributions import Categorical


MAX_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 3000
LAMBDA = 0.95
GAMMA = 0.99
LEARNING_RATE = 1e-4
VALUE_LOSS = 0.5
ENTROPY_COEF = 0.01
EPOCHS = 3
MINI_BATCH_SIZE = 64
CLIP_RANGE = 0.2
torch.autograd.set_detect_anomaly(True)


class Agent:
    def __init__(
        self,
        input_shape,
        n_actions,
        learning_rate,
        clip_range,
        value_loss_coef,
        entropy_coef,
        epochs,
        mini_batch_size,
        device,
    ):

        self.device = device
        self.policy_net = PolicyNet(input_shape, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

    def take_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.policy_net(state)
            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().item(), log_prob, value

    def compute_advantage(self, rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
        values = np.array(values + [values[-1]])
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        advantage = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + gamma * lam * (1 - dones[t]) * advantage
            advantages[t] = advantage
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(old_log_probs),
            torch.FloatTensor(returns),
            torch.FloatTensor(advantages),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.mini_batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_old_log_probs = batch_old_log_probs.to(self.device)
                batch_returns = batch_returns.to(self.device)
                batch_advantages = batch_advantages.to(self.device)

                # Compute log probabilities and ratio
                policy, values = self.policy_net(batch_states)
                dist = Categorical(policy)
                log_probs = dist.log_prob(batch_actions)

                # Clamp log ratio values before applying exp
                log_ratio = log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, min=-10, max=10)  # Prevent overflow
                ratio = torch.exp(log_ratio)

                # Check for NaN in ratio
                assert not torch.isnan(ratio).any(), f"NaN detected in ratio: {ratio}"

                # Apply clipping
                ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

                # Compute the policy loss
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * batch_advantages,
                ).mean()

                # Compute value loss
                value_loss = (
                    self.value_loss_coef
                    * (values.squeeze() - batch_returns).pow(2).mean()
                )

                # Compute entropy loss
                entropy_loss = -self.entropy_coef * dist.entropy().mean()

                # Print and check for NaN values in loss components
                print(
                    f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Entropy Loss: {entropy_loss.item()}"
                )
                assert not torch.isnan(policy_loss).any(), "Policy loss contains NaNs"
                assert not torch.isnan(value_loss).any(), "Value loss contains NaNs"
                assert not torch.isnan(entropy_loss).any(), "Entropy loss contains NaNs"

                # Total loss
                loss = policy_loss + value_loss + entropy_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), max_norm=1.0
                )

                # Update model parameters
                self.optimizer.step()


class PolicyNet(nn.Module):

    def __init__(self, input_shape, n_actions):

        super(PolicyNet, self).__init__()

        # Feature extraction convolutional layers
        self.conv1 = nn.Conv2d(
            input_shape[0],
            32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=4,
            stride=2,
        )
        self.conv3 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=1,
        )
        convw = get_convolutional_output_size(
            get_convolutional_output_size(
                get_convolutional_output_size(input_shape[1], 8, 4), 4, 2
            ),
            3,
            1,
        )
        convh = get_convolutional_output_size(
            get_convolutional_output_size(
                get_convolutional_output_size(input_shape[2], 8, 4), 4, 2
            ),
            3,
            1,
        )

        linear_input_size = convw * convh * 64

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(linear_input_size, 512)

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))

        policy = F.softmax(self.policy_head(x), dim=1)
        policy = torch.clamp(policy, min=1e-8, max=1 - 1e-8)
        value = self.value_head(x)

        return policy, value


def preprocess(state):
    state = np.array(state)  # Convert the list of frames to a numpy array
    state = state[:, 35:195]  # Crop the image
    state = state[:, ::2, ::2, 0]  # Downsample by factor of 2
    state[state == 144] = 0  # Erase background (background type 1)
    state[state == 109] = 0  # Erase background (background type 2)
    state[state != 0] = 1  # Set everything else (paddles, ball) to 1
    return state.astype(np.float32)


def get_convolutional_output_size(size, kernel, stride):
    return (size - (kernel - 1) - 1) // stride + 1


def train():

    env = gym.make("Pong-v4", render_mode="rgb_array")
    env = gym.wrappers.FrameStackObservation(env, 4)  # Stack 4 observations

    state, _ = env.reset()

    agent = Agent(
        input_shape=preprocess(state).shape,
        n_actions=env.action_space.n,
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE,
        value_loss_coef=VALUE_LOSS,
        entropy_coef=ENTROPY_COEF,
        epochs=EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        device=torch.device("mps"),
    )

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        state = preprocess(state)

        episode_rewards = []
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        for step in range(MAX_STEPS_PER_EPISODE):
            action, log_prob, value = agent.take_action(state)

            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess(next_state)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.cpu().item())
            values.append(value.cpu().item())
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_rewards.append(reward)

            if done:
                break

            advantages, returns = agent.compute_advantage(rewards, values, dones)

            agent.update(
                states=np.array(states),
                actions=np.array(actions),
                old_log_probs=np.array(log_probs),
                returns=returns,
                advantages=advantages,
            )

            print(f"Episode {episode}: Total Reward = {sum(episode_rewards)}")
            if episode % 100 == 0:
                env.render()

            env.close()

    return


def main():

    train()

    pass


if __name__ == "__main__":

    main()
