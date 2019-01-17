# Safe-Actor-Critic
We introduce a novel framework which provides safety in Actor Critic style methods. Here safety is defined as "Prevention from accidents in ML systems due to poor designing on AI systems". Safety can be added in various ways - modification of reward function, constraint based optimization, safety on exploration, etc. In this work, we are basing the notion of safety on constraint based optimization, where constraint is to minimize the variance of the return.

# Some awesome papers on Safety in AI
- Concrete Problems in AI Safety [[Paper]](https://arxiv.org/pdf/1606.06565.pdf)
- A Comprehensive Survey on Safe Reinforcement Learning [[Paper]](http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf)

# Getting Started
The repo contains code for running SAC framework on tabular and Mujoco environments. The code Safe PPO for Mujoco Environments in based on extension of this public [[Github Repo]](https://github.com/sanjaythakur/trpo)

# Prerequisites
```
* Numpy
* OpenAI Gym
* Matplotlib
* Seaborn
* Python 3.5
* Tensorflow
* MuJoCo
```

# Environments Used
* Walker Environment (Mujoco)
* Hopper Environment (Mujoco)
* Ant Environment (Mujoco)
* HalfCheetah Environment (Mujoco)
* Humanoid Environment (Mujoco)

# Training in Mujoco Domain
* ./train.py Hopper-v1 -n 20000
* ./train.py HalfCheetah-v1 -n 4000 -b 5
* ./train.py Ant-v1 -n 70000
* ./train.py Walker2d-v1 -n 20000
* ./train.py Humanoid-v1 -n 60000

# Visualizing the Learning Curve in Mujoco Domain
Use the following PoltingMujocoEnv.ipynb (iPython Notebook) for visualization of return plots for Mujoco Env.

