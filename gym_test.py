import gymnasium as gym
import matplotlib.pyplot as plt


def gym_test():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()
    for _ in range(1000):  # Run for a bit
        action = env.action_space.sample()  # Sample random action
        observation, reward, done, truncated, info = env.step(action)
        if done or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == "__main__":

    gym_test()
