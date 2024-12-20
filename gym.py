import gymnasium as gym
from DQN import DQN
from replay_buffer import ReplayBuffer
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import ale_py
import torch.nn.functional as F
import numpy as np


def main():

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.reset()

    ACTION_MAP = {  # DISCRETE ACTION STATES
        0: "NOOP",  # DO NOTHING
        1: "FIRE",  # FIRE BUTTON WITHOUT UPDATING JOYSTICK POSITION
        2: "RIGHT",  # APPLY A DELTA UPWARDS ON JOYSTICK
        3: "LEFT",  # APPLY A DELTA LEFTWARDS ON JOYSTICK
        4: "RIGHTFIRE",  # EXECUTE RIGHT AND FIRE
        5: "LEFTFIRE",  # EXECUTE LEFT AND FIRE
    }

    epochs = 1000

    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available() else "cpu"
    # )

    device = "cpu"

    model = DQN(env.action_space.n).to(device)

    N = 1_000_000
    M = 1_00_000
    gamma = 1e-3

    rb = ReplayBuffer(capacity=N)  # Initialize Replay buffer to a capacity value

    optimizer = torch.optim.SGD(model.parameters(), lr=1.00e-4)

    # A = {1,....K} action se passed ot the emulator modifying internal state and game score
    # Environment is stochastic, agent observes image
    # vector of raw pixel values representing current screen
    # receives reward r
    # game score depend on whole prior sequence of actions and observations
    # feedback about an action may only be received after many thousands of time-steps have elapsed
    # We must understand the game from a sequence of actions
    # This creates a MDP, each sequence is a distinct state
    # we can apply RL Methods for MDPs
    # use complete sequence s_t as state representation at time t
    # make choices that maximiie the future rewards, discount by a factor of gamma per time-step
    # future discounted reward -> R_t = \sum_{t'=t}^{T} \gamma^{t'-t}r_{t'}
    # T is the timestep at which the game terminates
    # Optimal action value function Q*(s,a) as maximum expected return achievable by following any strategy
    # after seeing some sequence s and then taking some action a, Q*(s,a) = max_{\pi}\mathbb{E}[R_t|s_t = s, a_t = a, \pi]
    # \pi is a policy mapping sequences to actions or distributions over actions
    # This obeys the bellman equation
    env.reset()
    for episode in range(1, 100):
        for t in range(1, N):

            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            plt.imshow((preprocess(obs)))
            plt.draw()
            plt.pause(0.001)
            plt.clf()

    env.close()
    return


def training():

    return


def preprocess(observation):
    """

    Extract important details from the simulation environment


    Args:
        env ([TODO:type]): [TODO:description]
    """

    # observation = observation.mean(axis=2, keepdims=True)  # Convert to grayscale
    observation = observation[34:-30, :, :]  # Crops the image
    # observation = observation[::2, ::2]
    observation[observation == 144] = 0
    observation[observation == 72] = 0
    observation[observation != 0] = 255

    return observation


def record_best_performance():
    """[TODO:summary]
    Need a function to output the best result

    [TODO:description]
    """

    pass


def frame_stack():

    pass


def q_action_value_function(state, action, model, device):
    state = torch.tensor(state, dtype=torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device)
    q_values = model(state)
    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    return q_value


if __name__ == "__main__":
    # env = gym.make("ALE/Pong-v5", render_mode="human").env
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayscaleObservation(env)
    # env = gym.wrappers.FrameStackObservation(env, 4)

    main()
