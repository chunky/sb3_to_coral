import sys
import gym
import torch

from stable_baselines3 import SAC


class OnnxablePolicy(torch.nn.Module):
  def __init__(self,  actor):
      super(OnnxablePolicy, self).__init__()
      self.actor = torch.nn.Sequential(actor.latent_pi, actor.mu)

  def forward(self, observation):
      # NOTE: You may have to process (normalize) observation in the correct
      #       way before using this. See `common.preprocessing.preprocess_obs`
      return self.actor(observation)


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
    onnx_save_file = model_prefix + ".onnx"

    env = gym.make(env_name)
    model = SAC.load(model_save_file, env, verbose=True)
    obs = env.observation_space
    dummy_input = torch.FloatTensor(obs.sample())

    onnxable_model = OnnxablePolicy(model.policy.actor)
    model.policy.to("cpu")
    model.policy.eval()

    torch.onnx.export(onnxable_model, dummy_input, onnx_save_file, opset_version=9, verbose=True)
