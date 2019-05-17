import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
import math
import os
import sys
import datetime
import threading
import csv

#Paper implemented: https://icml.cc/2012/papers/489.pdf
#(Policy Gradients with Variance Related Risk Criteria)

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = 0.5 * np.ones((nfeatures, nactions))  # positive weight initialization
        self.nactions = nactions
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return self.weights[phi, :]
        return self.weights[phi, action]

    def pmf(self, phi):
        v = self.value(phi) / self.temp
        b = v.max()
        new_values = v - b
        y = np.exp(new_values)
        return (y/ y.sum())

    def sample(self, phi):
        prob = self.pmf(phi)
        if prob.sum() >1:
            ind_max = np.argmax(prob)
            prob = np.zeros(self.nactions)
            prob[ind_max] = 1.
        else:
            prob[-1] = 1 - np.sum(prob[:-1])
        return int(self.rng.choice(self.nactions, p=prob))


class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.weight_J = []
        self.weight_V = []
        self.history = []


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
    frozen_states = []
    state_num = 0
    for line in layout.splitlines():
        for i in range(num_elem):
            if line[i] == "f":
                frozen_states.append(state_num)
            if line[i] != "w":
                state_num += 1
    return frozen_states

def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def save_csv(args, file_name, mean_return, last_meansteps,std_return):
    csvData = []
    style = 'a'
    if not os.path.exists(file_name):
        style = 'w'
        csvHeader = ['runs', 'episodes', 'temp', 'lr_p', 'lr_c', 'lambda', 'b', 'mean', 'std', 'steps']
        csvData.append(csvHeader)
    data_row = [args.nruns, args.nepisodes, args.temperature, args.lr_theta, args.lr_critic, args.lam,
                args.b, mean_return, std_return, last_meansteps]
    csvData.append(data_row)
    with open(file_name, style) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def penalty_function_gradient(x):
    #g(x)= (max(0,x))^2
    if x>0:
        return 2*x
    else:
        return 0

def run_agent(outputinfo, nepisodes, frozen_states, nactions, num_states,
              temperature, gamma, lmbda, lr_critic, lr_theta, b, rng):

    history = np.zeros((nepisodes, 2), dtype=np.float32)  # 1. Return 2. Steps
    # storage the weights of the trained model
    weight_policy = np.zeros((nepisodes, num_states, nactions),
                             dtype=np.float32)
    J_overall = np.zeros((nepisodes, num_states), dtype=np.float32) # value fo state J
    V_overall = np.zeros((nepisodes, num_states), dtype=np.float32) # variance of state V

    policy = SoftmaxPolicy(rng, num_states, nactions, temperature)
    # Weights of J and V value of a state
    J = np.zeros(num_states)
    V = np.zeros(num_states)
    env = gym.make('Fourrooms-v0')
    for episode in range(nepisodes):
        R = [] #reward at each time step
        S = [] #phi visited
        A = [] #Action Done
        done= False
        phi = env.reset()
        action = policy.sample(phi)
        while done != True:
            S.append(phi)
            A.append(action)
            phi, reward, done, _ = env.step(action)
            if phi in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)
            R.append(reward)
            action = policy.sample(phi)
        time_duration = len(R)
        G = np.zeros(time_duration)
        G[time_duration-1] = R[time_duration-1]
        for t in range(time_duration-2, -1, -1):
            G[t]= gamma*G[t+1] + R[t]

        for t in range(0, time_duration):
            new_J = J[S[t]] + lr_critic*(G[t] - J[S[t]])
            new_V = V[S[t]]+ lr_critic*(G[t]**2 - J[S[t]]**2 - V[S[t]])
            action_pmf = policy.pmf(S[t])
            policy_update_val = lr_theta* (G[t] - lmbda * penalty_function_gradient(V[S[t]] - b)*(G[t]**2 - 2*J[S[t]]*G[t]))
            policy.weights[S[t], :] -= policy_update_val* action_pmf
            policy.weights[S[t], A[t]] += policy_update_val
            J[S[t]]= new_J
            V[S[t]]= new_V
        J_overall[episode] = J
        V_overall[episode] = V
        weight_policy[episode]= policy.weights
        history[episode, 0]= G[0] #return
        history[episode, 1]= time_duration #steps
        # print("*************episode: ", episode)
        # print("Steps ", time_duration)

    outputinfo.weight_policy.append(weight_policy)
    outputinfo.weight_J.append(J_overall)
    outputinfo.weight_V.append(V_overall)
    outputinfo.history.append(history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='Discount factor for Q value', type=float, default=0.99)
    parser.add_argument('--lr_critic', help="Learning rate for Q value", type=float, default=0.01)
    parser.add_argument('--lr_theta', help="Learning rate for policy parameterization theta", type=float, default=0.001)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=0.1)
    parser.add_argument('--lam', help="Penalty Regularizer lambda", type=float, default=0.)
    parser.add_argument('--b', help="Variance should be less than b", type=float, default=200)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=800)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=50)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)

    args = parser.parse_args()
    now_time = datetime.datetime.now()

    env = gym.make('Fourrooms-v0')
    outer_dir = "Results_BaselineVarUpdated"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "FR_16-05")# + now_time.strftime("%d-%m"))
    if not os.path.exists(outer_dir):        
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Lam" + str(args.lam) + \
               "_LRC" + str(args.lr_critic) + "_LRTheta" + str(args.lr_theta) + "_b" + str(args.b) + \
               "_temp" + str(args.temperature)+"_gam" + str(args.gamma) + "_seed" + str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)

    num_states = env.observation_space.n
    nactions = env.action_space.n
    frozen_states = GetFrozenStates()

    threads = []
    outputinfo = OutputInformation()

    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, args.nepisodes,
                                                     frozen_states, nactions, num_states, args.temperature,
                                                     args.gamma, args.lam, args.lr_critic, args.lr_theta,
                                                     args.b, np.random.RandomState(args.seed + i), ))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    hist = np.asarray(outputinfo.history)
    last_meanreturn = np.mean(hist[:,:-50,0]) #Last 100 episodes mean value of the return
    last_meansteps = np.mean(hist[:,:-50,1]) #Last 100 episodes mean value of the return
    last_stdreturn = np.std(hist[:,:-50,0]) #Last 100 episodes std. dev value of the return


    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'Weights_J.npy'), np.asarray(outputinfo.weight_J))
    np.save(os.path.join(dir_name, 'Weights_V.npy'), np.asarray(outputinfo.weight_V))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
    save_csv(args, os.path.join(outer_dir,"ParamtersDone.csv"), last_meanreturn, last_meansteps,last_stdreturn)

