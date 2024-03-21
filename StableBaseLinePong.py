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
