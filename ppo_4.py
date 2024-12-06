import random
import gymnasium as gym
import ale_py
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.gamma = 0.99
        self.eps_clip = 0.1

        self.layers = nn.Sequential(
            nn.Linear(6000, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def state_to_tensor(self, I):
        """Preprocess 210x160x3 uint8 frame into 6000 (75x80) 1D float vector."""
        if I is None:
            return torch.zeros(1, 6000)
        I = I[35:185]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (type 1)
        I[I == 109] = 0  # erase background (type 2)
        I[I != 0] = 1  # paddles and ball set to 1
        return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)

    def pre_process(self, x, prev_x):
        return self.state_to_tensor(x) - self.state_to_tensor(prev_x)

    def convert_action(self, action):
        return action + 2

    def forward(
        self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False
    ):
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cpu().numpy())
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob

        # PPO loss calculation
        vs = np.array([[1.0, 0.0], [0.0, 1.0]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])

        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        return torch.mean(loss)


env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0)
env = gym.wrappers.FrameStackObservation(env, 1)

policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

reward_sum_running_avg = None
for it in range(100000):
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for ep in range(10):
        obs, info = env.reset()
        prev_obs = None
        for t in range(190000):
            d_obs = policy.pre_process(obs, prev_obs)
            with torch.no_grad():
                action, action_prob = policy(d_obs)

            prev_obs = obs
            obs, reward, done, trunc, info = env.step(policy.convert_action(action))

            d_obs_history.append(d_obs)
            action_history.append(action)
            action_prob_history.append(action_prob)
            reward_history.append(reward)

            if done or trunc:
                reward_sum = sum(reward_history[-t:])
                reward_sum_running_avg = (
                    0.99 * reward_sum_running_avg + 0.01 * reward_sum
                    if reward_sum_running_avg is not None
                    else reward_sum
                )
                print(
                    f"Iteration {it}, Episode {ep} ({t} timesteps) - "
                    f"last_action: {action}, last_action_prob: {action_prob:.2f}, "
                    f"reward_sum: {reward_sum:.2f}, running_avg: {reward_sum_running_avg:.2f}"
                )
                break

    # Compute advantage
    R = 0
    discounted_rewards = []

    for r in reward_history[::-1]:
        if r != 0:
            R = 0  # scored/lost a point, reset reward sum
        R = r + policy.gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (
        discounted_rewards - discounted_rewards.mean()
    ) / discounted_rewards.std()

    # Update policy
    for _ in range(5):
        n_batch = 24576
        idxs = random.sample(
            range(len(action_history)), min(n_batch, len(action_history))
        )
        d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0)
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
        action_prob_batch = torch.FloatTensor(
            [action_prob_history[idx] for idx in idxs]
        )
        advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs])

        opt.zero_grad()
        loss = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
        loss.backward()
        opt.step()

    if it % 5 == 0:
        torch.save(policy.state_dict(), "params.ckpt")

env.close()
