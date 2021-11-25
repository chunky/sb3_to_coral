import sys
import gym

from stable_baselines3 import SAC

# This is here so as to generate a model.zip file; I don't tune the parameters or even expect it to
#  generate an agent that works usefully, I just want a complete-and-intact saved model.
# At the very least, increase number of timesteps to model.learn if you need something useful

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
    env.reset()
    env.render()

    model = SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=25)
    # model.learn(total_timesteps=250_000)
    model.save(model_save_file)
