
# Pre-requisites

Setting up the python environment.

Install [Python 3.12.2](https://www.python.org/downloads/release/python-3122/) as per your OS requirements.

`venv` is a virtual environment setup in python, it is suggested to setup a python virtual environment to isolate the code with all the necessary packages required to run the Pong RL code.

To setup the python virtual environment, execute the following commands.

```bash
pip install stable-baselines3 gym imageio opencv-python torch tensorflow tensorboard tf_keras
# The above command will install all the required python packages
# These packages will be enough to run the Pong RL code
```

# Pong RL Code

```python
from stable_baselines3 import PPO
import warnings
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import os
import sys
import logging

working_dir = os.path.dirname(os.path.realpath(__file__))

log_dir_path = "logs"
model_dir_path = "models"
model_path = "PPO-TEST-NO-FRAME-FINAL-DONE"
OUTPUT_LOG_FILE = 'library_output_ppo_noframe_skip_final_done.log'
logs_dir = os.path.join(working_dir, log_dir_path)
models_dir = os.path.join(working_dir, model_dir_path, model_path)
log_file_path = os.path.join(working_dir, OUTPUT_LOG_FILE)

# Configure logging to write to the file
logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create directories if they don't exist
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Supress warnings
warnings.filterwarnings("ignore")

# Make environment using make_atari_env method
env = make_atari_env("PongNoFrameskip-v4", n_envs=3, seed=0)
env = VecFrameStack(env, n_stack=4)
env.reset()


# while initiating the algorithm, all the hyperparameters can be tweaked e.g., learning rate etc.,
# Define a context manager to capture standard output and redirect it to the logger
class RedirectStdoutToLogger:
    def __init__(self, logger_value):
        self.logger = logger_value
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout

    def write(self, message):
        self.logger.info(message)

    def flush(self):
        pass  # Flush is not needed in this context


# Create a logger
logger = logging.getLogger('PONG')

# Redirect standard output to the logger using the context manager
with RedirectStdoutToLogger(logger):
    # All print statements to the console will be logged to the file

    # Initialize the PPO Algorithm
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logs_dir)
    time_steps = 100000

    # Iterate and same the model for each time_step
    for i in range(1, 30):
        model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=model_path)
        model.save(f"{models_dir}/{time_steps * i}")

# Close the environment
env.close()

```

# SB3 Code breakdown

We will be using the PPO algorithm, from the stable baselines 3 pacakge, to train our policy.

```python
env = make_atari_env("PongNoFrameskip-v4", n_envs=3, seed=0)
```
- `make_atari_env` method creates the environment for Pong with environment id as `PongNoFrameskip-v4`
- `n_envs` creates multiple environments, which in turn is used to create worker threads to speed up the training process.
- `seed` is used for consistency in the training results, even when trained multiple times reducing the randomness.

```python
env = VecFrameStack(env, n_stack=4)
```
`VecFrameStack` method takes `n_stack` as a parameter and its values processes those many frames in a single go, which helps in reducing the time taken for training and results in a `VecEnv`

```python
with RedirectStdoutToLogger(logger):
```
All the model training code is wrapped in the `RedirectStdoutToLogger` class, so all the console logs will be written to a log file defined in the parameters from the code.

```python
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logs_dir)
```
- model is created by initializing a PPO algorithm.
- The beauty of SB3 is it follows plug and play approach, all the code remains the same, we can just plug in different algorithm to create a new model like A2C for example.,
- PPO alogrithm initialises a `CnnPolicy`, which will be used to train the Pong screens from the gym environment.
- `tensorboard_log` is used to refer to the logs directory from which a tensorboard can be constructed with `mean`, `loss`, `average reward` and many other training metrics for comparision.

```python
time_steps=100000
```
`time_steps` are the steps after which each model is saved to the `models_dir`

Iterating this many times creates stable models, which can excel to play the Pong game.


# Evaluation of the PPO algorithm

By choosing the best model with maximum reward, we can evaluate the performance.

We can evaluate the RL algorithm to check whether RL agent scores `21` before the logical agent does.

```python
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
```

```python
env = VecFrameStack(env, n_stack=4)
```
`n_stack` has to be the same value as the value thats provided while training.

We can get the Average rewards, if we increase the number of episodes.

The gameplay using the best model [1300000.zip](https://github.com/astrokathi/PONG-MEDIA/blob/main/1300000.zip) is [pong.mov](https://github.com/astrokathi/PONG-MEDIA/blob/main/pong.mov)