import gymnasium as gym
from gymnasium.wrappers import RecordVideo # Wrapping for recording video
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env # Creates vectorized envs
from stable_baselines3.common.atari_wrappers import WarpFrame # Convert image to grayscale
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder # Stack frames, record video
from stable_baselines3.common.callbacks import CheckpointCallback # Tracking performance
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
import os
import torch
import pandas as pd
import numpy as np
import random
import matplotlib as plt
import platform
from platform import python_version
from importlib.metadata import version

# Print package versions
print(f"Python Version: {python_version()}")
print(f"Torch Version: {version('torch')}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Cuda Version: {torch.version.cuda}")
print(f"Gymnasium Version: {version('gymnasium')}")
print(f"Numpy Version: {version('numpy')}")
print(f"Stable Baselines3 Version: {version('stable_baselines3')}")

env_str = "CarRacing-v3"
log_dir = f"./logs/{env_str}"
os.makedirs(log_dir, exist_ok=True)

gray_scale = True
wrapper_class = WarpFrame if gray_scale else None

# Wind wrapper to simulate lateral wind effects
class WindWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wind_variability = 0.075  # Maximum variability range of wind strength

    def step(self, action):
        """Modify the car's steering based on wind."""
        wind_direction = random.choice([-1, 1])  # Left or Right wind
        current_wind_strength = random.uniform(-self.wind_variability, self.wind_variability)

        # No wind with some probability
        if random.random() < 0.1:
            current_wind_strength = 0

        action[0] = np.clip(action[0] + wind_direction * current_wind_strength, -1, 1)  # Adjust steering
        
        return self.env.step(action)  # Proceed with the modified action

# Function to create an environment with grayscale & wind effects
def make_env():
    env = gym.make(env_str)
    if gray_scale:
        env = WarpFrame(env)  # Convert to grayscale
    env = WindWrapper(env)  # Apply wind effects
    return env

# Create Training Environment
env = make_vec_env(make_env, n_envs=1)  # Apply WindWrapper
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Create evaluation callback
checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=log_dir, name_prefix="ppo_car_racing")

# Initialize PPO model
model = PPO('CnnPolicy', env, verbose=1, ent_coef=0.005)

# Train the model
model.learn(total_timesteps=750000, progress_bar=True, callback=checkpoint_callback)

# Save the model
model.save(os.path.join(log_dir, "ppo_car_racing"))

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close environments
env.close()
env_val.close()