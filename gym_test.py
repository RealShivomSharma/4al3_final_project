import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py


def gym_test():
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    observation, info = env.reset()
    for _ in range(10000):  # Run for a longer period
        action = env.action_space.sample()  # Sample random action
        observation, reward, done, truncated, info = env.step(action)
        if done or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":

    gym_test()
