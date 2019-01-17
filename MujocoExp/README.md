# Safe-Actor-Critic
We introduce a novel framework known as "Safe-Actor-Critic" (S-AC), which provides safety in Actor Critic style methods. Here safety is defined as "Prevention from accidents in ML systems due to poor designing on AI systems". Safety can be added in various ways - modification of reward function, constraint based optimization, safety on exploration, etc. In this work, we are basing the notion of safety on constraint based optimization, where constraint is to minimize the variance of the return.

# Some awesome papers on Safety in AI
- Concrete Problems in AI Safety [[Paper]](https://arxiv.org/pdf/1606.06565.pdf)
- A Comprehensive Survey on Safe Reinforcement Learning [[Paper]](http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf)

# Getting Started
The repo contains code for running SAC framework on tabular and Mujoco environments. The code Safe DPPO for Mujoco Environments in based on extension of this public [[Github Repo]](https://github.com/sanjaythakur/trpo)

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
* Tabular Frozen FourRoom Env : TabularFourRoom -> fourrooms.py
* Walker Environment (Mujoco)
* Hopper Environment (Mujoco)
* Ant Environment (Mujoco)
* HalfCheetah Environment (Mujoco)

# Training Tabular Environment
The following command is used for training Tabular (FrozenFourRoom Environment)
```
python S-AC.py --nruns 50 --nepisodes 4000 --psi 0.25
```
Use deafult parameters in code for best setting

The results of FourRoom would be stored at location "../Results" (change the location to store somewhere else)

# Plotting Return plots
Use the following ReturnPlots.ipynb (iPython Notebook) for visualization of return plots in Frozen FourRoom Env.

# Training in Mujoco Domain
* ./train.py Hopper-v1 -n 20000
* ./train.py HalfCheetah-v1 -n 4000 -b 5
* ./train.py Ant-v1 -n 70000
* ./train.py Walker2d-v1 -n 20000

# Visualizing the Learning Curve in Mujoco Domain
Use the following PoltingLearningCurves.ipynb (iPython Notebook) for visualization of return plots in Frozen FourRoom Env.

