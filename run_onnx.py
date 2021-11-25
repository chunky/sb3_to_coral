import sys
import gym
import onnxruntime as ort

if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    model_prefix = 'model'
    if len(sys.argv) < 3:
        print("Usage: " + str(sys.argv[0]) + " <envname> <model_prefix>")
        print(" Defaulting to env: " + env_name + ", model_prefix: " + model_prefix)
    else:
        env_name = sys.argv[1]
        model_prefix = sys.argv[2]
    model_save_file = model_prefix + ".onnx"

    env = gym.make(env_name)
    obs = env.reset()

    ort_session = ort.InferenceSession(model_save_file)

    for i in range(100000):
        outputs = ort_session.run(
            None,
            {'input': obs.reshape([1, -1])}
        )
        obs, reward, done, info = env.step(outputs[0])
        env.render()
        if done:
            obs = env.reset()


