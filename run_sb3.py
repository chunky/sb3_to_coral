import sys
import gym

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
    obs = env.reset()

    model = SAC.load(model_save_file, env)

    for i in range(100000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


