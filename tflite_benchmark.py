import os.path
import sys
import gym
import time
import gym_rtam
import socket
import re
import tflite_runtime.interpreter as tflite

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: " + str(sys.argv[0]) + " <envname> <tflite model> <csv output file>")
        exit(0)

    env_name = sys.argv[1]
    tflite_model = sys.argv[2]
    output_csv = sys.argv[3]

    dev = os.getenv("EDGETPU_DEVICE", ":0")
    device_description = "CPU"
    if 'edgetpu' in tflite_model:
        from pycoral.utils import edgetpu

        edge_tpus_available = edgetpu.list_edge_tpus()
        print("Coral TPUs available: {}, using {}".format(edge_tpus_available, dev))
        interpreter = edgetpu.make_interpreter(tflite_model, device=dev)
        device_description = "TPU: " + dev
    else:
        interpreter = tflite.Interpreter(model_path=tflite_model)

    # Average over this many inferences
    bench_inference_cnt = 100000
    # Stop benchmarking if it takes longer then this
    max_bench_time_ns = 240 * 1e9

    env = gym.make(env_name)
    obs_space = env.observation_space

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start_time_ns = time.time_ns()
    inference_cnt = 0
    for i in range(bench_inference_cnt):
        # Skip the actual simulation, just grab a random observation
        input_data = obs_space.sample().reshape(1, -1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        inference_cnt += 1
        time_now = time.time_ns()
        if time_now - start_time_ns > max_bench_time_ns:
            break

    add_header = not os.path.exists(output_csv)
    pat = re.compile(r'w(?P<n_nodes_per_layer>[0-9]+)xd(?P<n_hidden_layers>[0-9]+)')
    model_params = pat.search(tflite_model)
    n_nodes_per_layer = model_params.group("n_nodes_per_layer") if model_params is not None else ''
    n_hidden_layers = model_params.group("n_hidden_layers") if model_params is not None else ''
    with open(output_csv, 'a') as out_f:
        if add_header:
            out_f.write("env,file,file_size,inference_cnt,ms_per_inf,inf_per_s,hostname,execute_on,"
                        "is_quantised,n_nodes_per_layer,n_hidden_layers\n")
        model_size = os.path.getsize(tflite_model)
        ns_per_inference = (time_now - start_time_ns) / float(inference_cnt)
        ms_per_inference = ns_per_inference / 1e6
        out_f.write(f"{env_name},{tflite_model},{model_size},{inference_cnt},"
                    f"{ms_per_inference},{1000/ms_per_inference},"
                    f"{socket.gethostname()},{device_description},{'quant' in tflite_model},"
                    f"{n_nodes_per_layer},{n_hidden_layers}\n")
