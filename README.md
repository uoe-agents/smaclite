# SMAClite - Starcraft Mulit-Agent Challenge lite

This is a repository for the SMAClite environment. It is a (nearly) pure Python reimplementation of the Starcraft Multi-Agent Challenge, using Numpy and OpenAI Gym.

## Installation
Run
```
pip install .
```
In the SMAClite directory

## Running
As far as we are aware, this project fully adheres to the [OpenAI Gym API](), so it can be used with any framework capable of interfacing with Gym-capable environments. We recommend the [ePyMARL]() framework, made available in our repository. EPyMARL uses `yaml` files to specify run configurations. To train a model in the `MMM2` scenario using the `MAPPO` algorithm, you can use this example command:
```
python3 src/main.py --config=mappo --env-config=gymma with seed=1 env_args.time_limit=120 env_args.key="smaclite:smaclite/MMM2-v0
```