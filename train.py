import sys
import gym

from stable_baselines3 import SAC

# This is here so as to generate a model.zip file; I don't tune the parameters or even expect it to
#  generate an agent that works usefully, I just want a complete-and-intact saved model.
# At the very least, increase number of timesteps to model.learn if you need something useful

if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    model_prefix = 'model'
    n_hidden_layers = 4
    n_nodes_per_layer = 64
    if len(sys.argv) < 3:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix> [<n_hidden_layers> <n_nodes_per_layer>]")
        print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
        if len(sys.argv) >= 5:
            n_hidden_layers = int(sys.argv[3])
            n_nodes_per_layer = int(sys.argv[4])

    model_save_file = model_prefix + ".zip"
    env = gym.make(env_name)
    env.reset()
    env.render()

    nn = [n_nodes_per_layer for i in range(n_hidden_layers)]
    print("nn: {}".format(nn))
    # "pi=[]" is an array of widths for the created policy/actor network, qf is for critic
    model = SAC('MlpPolicy', env, verbose=1,
                policy_kwargs=dict(net_arch=dict(pi=nn, qf=[64, 64]))
               )
    model.learn(total_timesteps=250)
    # model.learn(total_timesteps=250_000)
    model.save(model_save_file)

