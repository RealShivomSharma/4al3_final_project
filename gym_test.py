import gymnasium as gym
import ale_py
import time

# Create the environment

env = gym.make("Pong-v4",render_mode='rgb_array')
env = gym.wrappers.FrameStackObservation(env, 4)
episode_trigger = lambda episode: episode % 99 == 0 
env = gym.wrappers.RecordVideo(
    env, video_folder="./videos", episode_trigger=episode_trigger, disable_logger=True
)
# env = gym.make('Pong-v4')

# Reset the environment
observation, info = env.reset()

# Track the observations per second
observations = 0
start_time = time.time()

# Run for a certain number of steps
for episode in range(1000):  # Run for 1000 steps as an example
    action = env.action_space.sample()  # Take a random action
    observation, reward, terminated, truncated, info = env.step(action)  # Step in the environment
    observations += 1
    if terminated or truncated:  # If the episode is done, reset the environment
        observation, info = env.reset()

# Calculate observations per second
elapsed_time = time.time() - start_time
observations_per_second = observations / elapsed_time

print(f"Observations per second: {observations_per_second:.2f}")

# Close the environment
env.close()
