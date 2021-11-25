# Running SB3 developed agents on TFLite or Coral

## Introduction

I've been using [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
to train agents against some custom Gyms, some of which require fairly
large NNs in order to be effective.

I want those agents to eventually be run on a pi or similar, so I need
to export all the way to [TFLite](https://www.tensorflow.org/lite)
and ideally a [Coral](https://coral.ai/).

## How to use

### Setup

You will need to have configured the Coral system-wide stuff.

Build a venv:

```shell
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Running

This comes with enough defaults to do cradle-to-grave demonstration,
but all the pieces take command-line arguments so I can adjust to taste
for my actual use case.

```shell
# Train an agent with SB3
python3 ./train.py

# Convert model
python3 ./model_conv.py

# Run original SB3 model
python3 ./run_sb3.py
# Run the onnx model
python3 ./run_onnx.py
# Run the TFLite model
python3 ./run_tflite.py
# Run the Coral model ["edgetpu" in the name will attempt to load Coral]
python3 ./run_tflite.py MountainCarContinuous-v0 model_quant_edgetpu
```

Cheers,  
Gary <chunky@icculus.org>

