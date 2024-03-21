from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import numpy as np

# Prepare the environment
env = make_atari_env("PongNoFrameskip-v4", n_envs=1)
env = VecFrameStack(env, n_stack=4)
env.metadata['render_fps'] = 25

# Load the pre-trained model
model = PPO.load("models/PPO-TEST-NO-FRAME-FINAL-DONE/1300000.zip")

# Initialize variables
total_reward = 0
episodes = 1  # Number of episodes for evaluation

# Run the evaluation loop
for _ in range(episodes):
    obs = env.reset()

    # Initialize done array for each environment
    done = np.array([False] * env.num_envs)

    # Initialize rewards for each environment
    episode_rewards = np.zeros(env.num_envs)

    while not done.any():
        # Predict actions for all environments
        actions, _ = model.predict(obs)

        # Step the environment with the predicted actions
        obs, rewards, done, info = env.step(actions)

        # Accumulate rewards for each episode
        episode_rewards += rewards

        # render the environment using human mode to graphically see the game
        env.render(mode='human')

    # Print episode rewards for each environment
    print("Episode Rewards:", episode_rewards)
    total_reward += episode_rewards

# Close the environment
env.close()

# Calculate and print the average reward
average_reward = total_reward / episodes
print("Average Reward:", average_reward)
