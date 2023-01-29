# SMAClite - Starcraft Mulit-Agent Challenge lite
This is a repository for the SMAClite environment. It is a (nearly) pure Python reimplementation of the Starcraft Multi-Agent Challenge, using Numpy and OpenAI Gym.

## Features
The main features of this environment include:
* A fully functional Python implementation of the SMAC environment
* A JSON interface for defining units and scenarios
* Compatibility with the OpenAI Gym API
* (optional) a highly-performant [C++ implementation](https://github.com/micadam/SMAClite-Python-RVO2) of the collision avoidance algorithm

## Available units
The following units are available in this environment:
* baneling
* colossus
* marauder
* marine
* medivac
* spine crawler
* stalker
* zealot
* zergling
## Available scenarios
The following scenarios are available in this environment:
* 10m_vs_11m
* 27m_vs_30m
* 2c_vs_64zg
* 2s3z
* 2s_vs_1sc
* 3s5z
* 3s5z_vs_3s6z
* 3s_vs_5z
* bane_vs_bane
* corridor
* mmm
* mmm2

Note that further scenarios can easily be added by modifying or creating a scenario JSON file.
## Installation
Run
```
pip install .
```
In the SMAClite directory

## Running
As far as we are aware, this project fully adheres to the [OpenAI Gym API](https://www.gymlibrary.dev/), so it can be used with any framework capable of interfacing with Gym-capable environments. We recommend the [ePyMARL](https://github.com/uoe-agents/epymarl) framework, made available in our repository. EPyMARL uses `yaml` files to specify run configurations. To train a model in the `MMM2` scenario using the `MAPPO` algorithm, you can use this example command:
```
python3 src/main.py --config=mappo --env-config=gymma with seed=1 env_args.time_limit=120 env_args.key="smaclite:smaclite/MMM2-v0
```

Note that to use the C++ version of the collision avoidance algorithm, you will have to add the line `use_cpp_rvo2: true` to the `yaml` config file you're referencing, since Sacred does not allow defining new config entries in the command itself.