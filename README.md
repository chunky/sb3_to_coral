# Running SB3 developed agents on a Coral

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
# Train an agent
python3 ./train.py
# Convert model
python3 ./model_conv.py
# Run at the leaf
python3 ./run.py
```

Cheers,  
Gary <chunky@icculus.org>

