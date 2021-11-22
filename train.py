import os
import sys
import gym
import numpy as np

from stable_baselines3 import SAC

if __name__ == '__main__':
    env_name = "MountainCarContinuous-v0"
    model_save_file = "model.zip"
    if len(sys.argv) < 3:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_save_file>")
        print(" Defaulting to env: " + env_name + ", savefile: " + model_save_file)
    else:
        env_name = sys.argv[1]
        model_save_file = sys.argv[2]

    env = gym.make(env_name)
    env.reset()
    env.render()

    model = SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=25_000)
    model.save(model_save_file)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

