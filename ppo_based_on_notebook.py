import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ale_py


# CNN to output the probability of going left/right with our paddle
# P(left) = 1 - P(right) # i.e. 50/50 chance if

INPUT_DIMENSION = (2, 80, 80)  # 2 stacked frames from the environment, scaled down
RIGHT = 4
LEFT = 5


class Policy(nn.Module):

    def __init__(self, input_dim=INPUT_DIMENSION, n_actions=1):
        super(Policy, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                input_dim[0],
                4,
                kernel_size=6,
                stride=2,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(4, 16, kernel_size=6, stride=4),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Sigmoid(),  # get the probability
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    """
    convert outputs of a single environment to inputs to pytorch neural net"""
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 4:
        list_of_images = np.expand_dims(list_of_images, 0)
    # subtract bkg and crop
    list_of_images_prepro = (
        np.mean(list_of_images[:, 34:-16:2, ::2] - bkg_color, axis=-1) / 255.0
    )
    batch_input = np.expand_dims(list_of_images_prepro, 0)
    return torch.from_numpy(batch_input).float().to("mps")


def preprocess_single_frame(image, bkg_color=np.array([144, 72, 17])):
    """
    Converts an image from RGB channel to B&W channels.
    Also performs downscale to 80x80. Performs normalization.
    @Param:
    1. image: (array_like) input image. shape = (210, 160, 3)
    2. bkg_color: (np.array) standard encoding for brown in RGB with alpha = 0.0
    @Return:
    - img: (array_like) B&W, downscaled, normalized image of shape (80x80)
    """
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.0
    return img


def collect_trajectories(env, policy, tmax=200, nrand=5):
    """collect trajectories for an environment"""

    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    state, _ = env.reset()

    # perform nrand random steps
    for _ in range(nrand):
        action = np.random.choice([RIGHT, LEFT])
        state, reward, done, _, _ = env.step(action)
        if done:
            state = env.reset()

    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into
        # shape (1, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = (
            torch.tensor(
                preprocess_single_frame(state), dtype=torch.float, device="mps"
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        action = RIGHT if np.random.rand() < probs else LEFT
        probs = probs if action == RIGHT else 1.0 - probs

        # advance the game
        next_state, reward, done, _, _ = env.step(action)

        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        state = next_state

        if done:
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list


def clipped_surrogate(
    policy,
    old_probs,
    states,
    actions,
    rewards,
    reward_discount=0.995,
    epsilon=0.1,
    beta=0.01,
):

    iota = 1.0e-10  # SMALL NUMBER TO AVOID NaN

    discounts = reward_discount ** np.arange(len(rewards))
    discounted_rewards = np.array(rewards) * discounts[:, np.newaxis]
    normalized_rewards = discounted_rewards / (discounted_rewards.std(axis=0) + iota)

    # Prepare tensors
    actions = torch.tensor(actions, dtype=torch.int64, device="mps")
    old_probs = torch.tensor(old_probs, dtype=torch.float, device="mps")
    normalized_rewards = torch.tensor(
        normalized_rewards, dtype=torch.float, device="mps"
    )

    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    new_probs = policy(policy_input).view(states.shape[:-3])

    ratios = new_probs / old_probs
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

    entropy = -(
        new_probs * torch.log(old_probs + iota)
        + (1.0 - new_probs) * torch.log(1.0 - old_probs + iota)
    )

    return torch.mean(
        torch.min(ratios * normalized_rewards, clipped_ratios * normalized_rewards)
        + beta * entropy
    )


def main():
    device = "mps"
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    episode = 1000
    discount_rate = 0.99
    epsilon = 0.1
    beta = 0.01
    tmax = 320
    SGD_epoch = 4

    mean_rewards = []
    env = gym.make("PongDeterministic-v4")
    env = gym.wrappers.FrameStackObservation(env, 2)

    for e in range(episode):
        old_probs, states, actions, rewards = collect_trajectories(
            env, policy, tmax=tmax
        )
        total_rewards = np.sum(rewards, axis=0)
        # for t in range(tmax):
        #     state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        #     action_prob = policy(state)
        #     action = torch.bernoulli(action_prob).item()
        #     next_state, reward, done, _ = env.step(action)

        #     old_probs.append(action_prob.detach().cpu().numpy())
        #     states.append(state.cpu().numpy())
        #     actions.append(action)
        #     rewards.append(reward)

        #     state = next_state
        #     if done:
        #         break

    pass


if __name__ == "__main__":

    main()
