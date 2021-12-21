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

## Benchmarking

I was curious to explore how the Coral actually performs. bench.sh should
reproduce a file with a variety of NN sizes, then benchmark them all.

A few things about the benchmark:
* For completeness, there's a non-quantised "edgetpu" file built; it
    should perform exactly the same as the CPU non-quantised one [because
    it can't run on the Coral]
* The benchmark simply samples the observation space for pushing through
    TFLite, but doesn't actually execute the Gym. One can imagine perverse
    edge cases here.
* This manufactures NNs, but they aren't trained to completion. One can
    imagine perverse edge cases here, too.
* Simple fully-connected NNs such as these RL models enjoy may not be
    a great use case for the Coral
* The bench.sh script creates some deliberately poorly-dimensioned NNs;
    either they cannot possibly fit on the Coral, or couldn't possibly
    be useful.

## Extras

The full chain, implemented here, to go from SB3 (Torch) to Coral is:
```
Torch => ONNX => Tensorflow => TFLite (normal) => TFLite (quantised) => Coral
```

When this code quantises the network, it explicitly leaves the inputs and
outputs as floats; this means there's some work that gets done on the CPU,
but the observation and action spaces of a gym would mean that work needs
doing, anyways. So although edgetpu\_compiler says that this may be less
efficient when run on the actual device, it's actually not.

The torch-to-ONNX step is a separate beast related to stable-baselines 3, that
warrants discussion; you can find more information on the SB3 docs page, here:
https://stable-baselines3.readthedocs.io/en/master/guide/export.html

Cheers,  
Gary <chunky@icculus.org>

