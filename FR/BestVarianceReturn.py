## Plot variance in return along episodes
import gym
import numpy as np
from fourrooms import Fourrooms
import math
import os


def GetFrozenStates():
    layout = """\
wwwwwwwwwwwww
w     w     w
w   ffwff   w
w  fffffff  w
w   ffwff   w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

    num_elem = 13
    line_count = 0
    element_count = 0
    frozen_states = []
    state_num = 0
    for line in layout.splitlines():
        for i in range(num_elem):
            if line[i] == "f":
                frozen_states.append(state_num)
            if line[i] != "w":
                state_num += 1
    return frozen_states


dir = "../../Neurips2020Results/Results_AC/BestResult"
names = ["SAC_Psi0.0", "SAC_Psi0.5", "TD_Psi0.01", "MC_Psi0.0", "MC_Psi0.01"]

save_dir = os.path.join(dir, "VariancePerEpisode")
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

env = gym.make('Fourrooms-v0')
n_states = env.observation_space.n
nactions = env.action_space.n
weight_policy_total_list = []

for file_name in names:
    weight_policy_total_list.append(np.load(os.path.join(dir, file_name, "Weights_Policy.npy"))[:,:1000])

num_traj_per_state = 80
goal_state = 62
n_states = 104
max_time = 400
frozen_states = GetFrozenStates()
n_runs = weight_policy_total_list[0].shape[0]
n_episodes = weight_policy_total_list[0].shape[1]
get_variance_after_episode = 20
list_eps = list(np.arange(0, n_episodes, get_variance_after_episode))
list_eps.append(n_episodes -1)
reward_variance = np.zeros((len(names), n_runs, len(list_eps)))
reward_std = np.zeros((len(names), n_runs, len(list_eps)))

for ind,psi in enumerate(names):
    weight_policy_total = weight_policy_total_list[ind]
    for eps_ind,episode in enumerate(list_eps):
        for run in range(n_runs):
            freq_visit = np.zeros(n_states)
            reward_distribution = np.zeros((n_states, num_traj_per_state))

            while(np.sum(freq_visit >= num_traj_per_state)!= n_states-1):
                    start = env.reset()
                    if start == goal_state:
                        continue
                    if freq_visit[start] >= num_traj_per_state:
                        continue
                    freq_visit[start] +=1
                    curr_state = start
                    curr_time = 0
                    d= False
                    gamma=0.99
                    current_dicounting_factor=1.
                    discounted_reward = 0
                    while(curr_state!= 62 and curr_time< max_time and d!=True):
                        action = np.argmax(weight_policy_total[run, episode, curr_state, :])
                        next_state,r,d,_ = env.step(action)
                        if curr_state in frozen_states:
                            r = np.random.normal(0,8.0)
                        discounted_reward += current_dicounting_factor*r
                        curr_state = next_state
                        current_dicounting_factor*=gamma
                        curr_time +=1
                    reward_distribution[start,int(freq_visit[start])-1] = discounted_reward
            reward_variance[ind, run, eps_ind] = np.var(reward_distribution) # flattend array variance
            reward_std[ind, run, eps_ind] = np.std(reward_distribution)

np.save(os.path.join(save_dir, 'RewardDistributionVariance.npy'), reward_variance)
np.save(os.path.join(save_dir, 'RewardDistributionStd.npy'), reward_std)

