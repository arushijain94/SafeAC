# Safe On-Policy and Off-Policy AC
We introduced a framework to constrain the direct estimate of variance in the return for control setting. By constraining the variance, we limit the abrupt or inconsistent behavior of the agent thus learning safer trajectory. We introduced this notion of safety for both on-policy and off-policy actor critic algorithms. This framework is generic which makes it easy to apply it on top of any existing Policy Gradient approaches. For purpose of this paper, we compare our safe algorithms with baseline actor-critic (AC) in on-policy AC and off-policy AC; proximal policy optimization (PPO) in on-policy AC continuous stat-action cases for Mujoco enviornments.

# Prerequisites
```
* Numpy
* OpenAI Gym
* Matplotlib
* Seaborn
* Python 3.5
```
 For On-Policy AC experiments in Mujoco, refer the Mujoco directory.
 
# Environments Used
* fourroom environment (On Policy AC for discrete state space with 4 actions) (FR)
* puddle-world environment (Off-Policy AC for discrete state space with 8 actions) (PuddleDiscrete)
* puddle-world environment (Off-Policy AC for continuous state space) (PuddleCont) 
* Mujoco environment (On-Policy AC for continuous state-action space) (MujocoExp)

Here following parameters in pseudocode are referred with following names:
* lr_critic: α_w
* lr_theta: α_θ
* lr_sigma: α_z
Keep all parameters as default and change the ones mentioned in supplementary material for optimal performance

# Training for On-Policy AC in Discrete State Space
Go to FR directory
* python SAC_trace_OnP.py --psi 0.1 --lr_critic 0.1 --lr_theta 0.001 --lr_sigma 0.02 --temperature 0.05

# Training for Off-Policy AC in Discrete State Space
Go to PuddleDiscrete directory
* python Puddle_discrete_OffP.py --psi 0.125 --lr_critic 0.1 --lr_theta 1e-4 --lr_sigma 5e-4 --temperature 0.75

# Training for Off-Policy AC in Continuous State Space
Go to PuddleCont directory
* python Puddle_OffPolicy.py --psi 0.005 --lr_critic 0.01 --lr_theta 1e-3 --lr_sigma 5e-3 --temperature 0.75

# Visualizing the Learning Curve
Go to respective .ipynb file (ipython notebook) to visualize the learning curve and trajectories
