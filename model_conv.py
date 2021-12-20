import os.path
import sys
from os import system

import gym
import torch
import torchsummary
import onnx
import onnx_tf.backend
import tensorflow as tf

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

    model_save_file = model_prefix + '.zip'
    onnx_save_file = model_prefix + '.onnx'
    tflite_save_file = model_prefix + '.tflite'
    tflite_quant_save_file = model_prefix + '_quant.tflite'

    print('Creating gym to gather observation sample...')
    env = gym.make(env_name)
    obs = env.observation_space
    # Awkward reshape: https://github.com/onnx/onnx-tensorflow/issues/400
    dummy_input = torch.FloatTensor(obs.sample().reshape(1, -1))

    print('Loading existing SB3 model...')
    model = SAC.load(model_save_file, env, verbose=True)

    print('Exporting to ONNX...')
    onnxable_model = OnnxablePolicy(model.policy.actor)
    model.policy.to("cpu")
    model.policy.eval()
    print(str(onnxable_model.actor))
    # torchsummary.summary(model.policy.actor, input_size=len(dummy_input))

    torch.onnx.export(onnxable_model, dummy_input, onnx_save_file,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=9, verbose=True)

    print('Loading ONNX and checking...')
    onnx_model = onnx.load(onnx_save_file)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    print('Converting ONNX to TF...')
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(model_prefix)

    print('Converting TF to TFLite...')
    converter = tf.lite.TFLiteConverter.from_saved_model(model_prefix)
    tflite_model = converter.convert()
    with open(tflite_save_file, 'wb') as f:
        f.write(tflite_model)

    print('Converting TF to Quantised TFLite...')

    def representative_data_gen():
        global obs
        for i in range(100000):
            yield [obs.sample().reshape(1, -1)]

    converter_quant = tf.lite.TFLiteConverter.from_saved_model(model_prefix)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_data_gen
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.target_spec.supported_types = [tf.int8]
    # Just accept that observations and actions are inherently floaty, let Coral handle that on the CPU
    converter_quant.inference_input_type = tf.float32
    converter_quant.inference_output_type = tf.float32
    tflite_quant_model = converter_quant.convert()
    with open(tflite_quant_save_file, 'wb') as f:
        f.write(tflite_quant_model)

    print('Converting TFLite [nonquant] to Coral...')
    system('edgetpu_compiler --show_operations -o ' + os.path.dirname(model_prefix) + ' ' + tflite_save_file)

    print('Converting TFLite [quant] to Coral...')
    system('edgetpu_compiler --show_operations -o ' + os.path.dirname(model_prefix) + ' ' + tflite_quant_save_file)
