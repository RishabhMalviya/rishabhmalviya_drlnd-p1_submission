# Overview
This repository serves as my submission for the first Project (p1-navigation) of the Udacity Deep Reinforcement Learning Nanodegree.

The goal of the project is train a DQN agent to navigate a large world and collect yellow bananas, while avoiding blue bananas.

|*Before Training*|*After Training*|
|--------|--------|
|![banana-collector-random-agent](https://github.com/RishabhMalviya/rishabhmalviya_drlnd-p1_submission/blob/master/results/random_agent.gif?raw=true)|![banana-collector-trained-agent](https://github.com/RishabhMalviya/rishabhmalviya_drlnd-p1_submission/blob/master/results/trained_agent.gif?raw=true)|

## Environment Description
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


# Setup
This setup was done on a system with these specifications:
1. **OS**: Windows 10
2. **CUDA Toolkit Version**: 11.2 (Download it from [here](https://developer.nvidia.com/Cuda-downloads))
3. **Python Version**: Python 3.6.8 (You can download the executable installer for Windows 10 from [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe))

Here are the exact steps:
1. Clone this repository with (use [Git Bash for Windows](https://gitforwindows.org/)) `git clone https://github.com/RishabhMalviya/dqn-experiments.git`.
2. In the cloned repository, create a venv by running the following command from Powershell (venv should be installed along with the Python 3.6.8 installation form the link given above): `python -m venv ./venv`. Also, in case you're using Anaconda, you should launch Powershell by searching for "*Anaconda Powershell Prompt*" from Start.
3. Now, activate the venv by running `./venv/Scripts/activate` in Powershell.
4. Upgrade pip with `pip install -U pip`.
5. Install the requirements with `pip install -r requirements.txt`. You should adapt the first three lines of the `requirements.txt` file based on the installation command that the [PyTorch download page](https://pytorch.org/get-started/locally/) recommends for your system.
6. Run `cd ./unity-ml-agents-setup` and `pip install .` to install the Udacity wrapper package for working with Unity ML Agents in the venv.
7. To run the code in the Jupyter Notebook, you will have to first download one of the pre-built environments from below (based on your operating system), then extract the zip and place it in this folder:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
8. Finally, start Jupyter (run `jupyter notebook` from Powershell) from the root of the repo and hack away!   

# User Guide - Quickstart

## Basic Abstractions

### 1. The Environment
This is the [Unity ML Agents Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#banana-collector). 

The environment itself is encapsulated in the zip file you extracted during [setup](#setup), while a simplified wrapper library to interact with it was installed from the `./unity-ml-agents-setup` folder, also during [setup](#setup).

### 2. The Agent
The `Agent` class is defined in the `agent.py` file, and requires a DQN and a set of hyperparameters prior to training:

1. The DQN is defined as a `torch nn.Module` object in the `dqn.py` file. There are two netwrk architectures in that file, one for a standard DQN and the other for a Dueling DQN architecture. In the above results, the standard DQN was used.
2. Although the `Agent` is given a set of default hyperparameters if none are provided, you can instantiate a `DQNHyperparameters` object (the corresponding class is defined in `agent.py`), modify the values, and provide them to the `Agent` if you want. These are the defaults:
    ```
    self.BUFFER_SIZE = int(1e5)  # replay buffer size
    self.GAMMA = 0.99            # discount factor

    self.BATCH_SIZE = 64         # minibatch size
    self.LR = 5e-4               # learning rate 
    self.UPDATE_EVERY = 4        # how often to the current DQN should learn

    self.HARD_UPDATE = False     # to hard update or not (with double DQN, one should)
    self.DOUBLE_DQN = False      # to use double DQN or not
    self.TAU = 1e-3              # for soft update of target parameters
    self.OVERWRITE_EVERY = 128   # how often to clone the local DQN to the target DQN
    ```

### 3. Training Hyperparameters
The hyperparameters used during training are defined in the `TrainingHyperparameters` class in the `train.py` file. These are the defaults:
```
self.EPS_START = 1.0
self.EPS_END = 0.01
self.EPS_DECAY = 0.995
```

## Training
The agent can be set free to interact with the environment and learn using the `train_agent` function from the `train.py` file. This function takes an argument called `completion_criteria`, which is supposed to be a function that takes as an argument a list of the scores from the last 100 episodes (latest first), and returns True or False. For example:
```
completion_criteria=lambda scores_window: np.mean(scores_window) >= 200.0
```

## Visualizing
You can use the functions `run_random_agent` and `run_trained_agent` from the `visualize.py` file.

This won't save a GIF of the interaction, but it will run it in the Unity window. You'll have to then manually record the interaction from your screen with a software like [ScreenToGIF](https://github.com/NickeManarin/ScreenToGif).


# Bibliogarphy

1. **DQN** - [Human-Level Control Through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) - *Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis*
2. **Double DQN** - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) - *Hado van Hasselt and Arthur Guez and David Silver*
3. **Dueling DQN** - [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf) - *Ziyu Wang and Tom Schaul and Matteo Hessel and Hado van Hasselt and Marc Lanctot and Nando de Freitas*
