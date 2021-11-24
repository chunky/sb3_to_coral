import sys
import gym

from stable_baselines3 import SAC

if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    model_prefix = 'model'
    if len(sys.argv) < 3:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix>")
        print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
    model_save_file = model_prefix + ".zip"

    env = gym.make(env_name)
    obs = env.reset()

    model = SAC.load(model_save_file, env)

    for i in range(100000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


