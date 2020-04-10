import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
import math
import os
import datetime
import threading
import csv


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state, ])

    def __len__(self):
        return self.nstates


class RandomPolicy:
    def __init__(self, nactions):
        self.nactions = nactions

    def sample(self):
        return int(np.random.randint(self.nactions))

    def pmf(self):
        prob_actions = [1. / self.nactions] * self.nactions
        return prob_actions


class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.nactions = nactions
        self.weights = 0.5 * np.ones((nfeatures, nactions))  # positive weight initialization

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi, curr_epsilon):
        if self.rng.uniform() < curr_epsilon:
            return int(self.rng.randint(self.nactions))
        return int(np.argmax(self.value(phi)))


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.random.rand(nfeatures, nactions)  # positive weight initialization
        self.nactions = nactions
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi) / self.temp
        b = v.max()
        new_values = v - b
        y = np.exp(new_values)
        return (y / y.sum())

    def sample(self, phi):
        prob = self.pmf(phi)
        if prob.sum() > 1:
            ind_max = np.argmax(prob)
            prob = np.zeros(self.nactions)
            prob[ind_max] = 1.
        else:
            prob[-1] = 1 - np.sum(prob[:-1])
        return int(self.rng.choice(self.nactions, p=prob))


class StateActionLearning:
    def __init__(self, gamma, lmbda, lr, weights, trace, variance):
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.weights = weights
        self.trace = trace
        self.variance = variance  # binary value (0: its is Q value, 1: sigma(s,a) value)

    def start(self, phi, action):
        self.last_phi = phi
        self.last_action = action
        self.last_value = self.value(phi, action)

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def update(self, phi, action, reward, done):
        # One-step update target
        if self.variance:
            self.trace *= math.pow(self.gamma * self.lmbda, 2.)
        else:
            self.trace *= self.gamma * self.lmbda
        self.trace[self.last_phi, self.last_action] += 1
        self.trace = np.clip(self.trace, -100., 100.)

        update_target = reward
        if not done:
            current_value = self.value(phi, action)
            corrected_curent_value = current_value
            if self.variance:
                corrected_curent_value *= math.pow(self.gamma * self.lmbda, 2.)
            else:
                corrected_curent_value *= self.gamma
            update_target += self.gamma * corrected_curent_value

        # Weight gradient update step
        tderror = update_target - self.last_value
        self.weights += self.lr * tderror * self.trace

        if not done:
            self.last_value = current_value

        self.last_action = action
        self.last_phi = phi

        return tderror


class PolicyGradient:
    def __init__(self, policy, lr, psi, nactions):
        self.lr = lr
        self.policy = policy
        self.psi = psi

    def update(self, phi, action, critic, sigma, current_psi):
        actions_pmf = self.policy.pmf(phi)
        if self.psi != 0.0:  # variance as regularization factor to optimization criterion
            psi = current_psi
            regularization = -self.lr * psi * sigma
            self.policy.weights[phi, :] -= regularization * actions_pmf
            self.policy.weights[phi, action] += regularization

        Q_val = self.lr * critic
        self.policy.weights[phi, :] -= Q_val * actions_pmf
        self.policy.weights[phi, action] += Q_val


class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.weight_Q = []
        self.weight_var = []
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


def save_csv(args, file_name, mean_return, std_return):
    csvData = []
    style = 'a'
    if not os.path.exists(file_name):
        style = 'w'
        csvHeader = ['runs', 'episodes', 'temp', 'lr_p', 'lr_c', 'lr_var', 'psi', 'psi_fixed', 'psi_rate', 'lambda','mean', 'std']
        csvData.append(csvHeader)
    data_row = [args.nruns, args.nepisodes, args.temperature, args.lr_theta, args.lr_critic, args.lr_sigma,
                args.psi, args.psiFixed, args.psiRate, args.lmbda, mean_return, std_return]
    csvData.append(data_row)
    with open(file_name, style) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def run_agent(outputinfo, features, nepisodes,
              frozen_states, nfeatures, nactions, num_states, temperature, gamma_Q, gamma_var,
              lmbda, lr_critic, lr_sigma, lr_theta, psi, rng, psi_fixed, psi_rate):
    history = np.zeros((nepisodes, 3),
                       dtype=np.float32)  # 1. Return 2. Steps 3. TD error 1 norm
    # storage the weights of the trained model
    weight_policy = np.zeros((nepisodes, num_states, nactions),
                             dtype=np.float32)
    weight_Q = np.zeros((nepisodes, num_states, nactions),
                        dtype=np.float32)
    weight_var = np.zeros((nepisodes, num_states, nactions),
                          dtype=np.float32)

    # Using Softmax Policy
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)

    # Action_critic is Q value of state-action pair
    weights_QVal = np.random.rand(nfeatures, nactions)
    trace_Qval = np.zeros_like(weights_QVal)
    action_critic = StateActionLearning(gamma_Q, lmbda, lr_critic, weights_QVal, trace_Qval, 0)

    # Variance is sigma of state-action pair
    weights_var = np.random.rand(nfeatures, nactions)
    trace_var = np.zeros_like(weights_var)
    sigma = StateActionLearning(gamma_var, lmbda, lr_sigma, weights_var, trace_var, 1)
    # Fixing Psi rate
    current_psi = psi
    if not psi_fixed:
        current_psi = 0.0
        psi_inc = float(psi)/psi_rate  # gives the increment in psi for each episode

    # Policy gradient improvement step
    policy_improvement = PolicyGradient(policy, lr_theta, psi, nactions)
    env = gym.make('Fourrooms-v0')
    for episode in range(args.nepisodes):
        return_per_episode = 0
        observation = env.reset()
        phi = features(observation)
        action = policy.sample(phi)

        action_critic.start(phi, action)
        sigma.start(phi, action)

        step = 0
        done = False
        sum_td_error = 0.0
        while done != True:
            old_observation = observation
            old_phi = phi
            old_action = action
            observation, reward, done, _ = env.step(action)

            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)

            phi = features(observation)
            return_per_episode += math.pow(args.gamma_Q, step) * reward
            action = policy.sample(phi)

            # Critic update
            tderror = action_critic.update(phi, action, reward, done)
            sum_td_error += abs(tderror)
            if psi != 0.0:
                try:
                    td_square = math.pow(tderror, 2.0)
                except OverflowError:
                    td_square = tderror * 2
                sigma.update(phi, action, td_square, done)
                sigma_val = sigma.value(old_phi, old_action)
            else:
                sigma_val = 0.0

            critic_val = action_critic.value(old_phi, old_action)
            policy_improvement.update(old_phi, old_action, critic_val, sigma_val, current_psi)

            step += 1
        if current_psi < psi and not psi_fixed :
            current_psi = np.round(current_psi+psi_inc,4)

        history[episode, 0] = step
        history[episode, 1] = return_per_episode
        history[episode, 2] = sum_td_error

        weight_policy[episode, :, :] = policy.weights
        weight_Q[episode, :, :] = action_critic.weights
        weight_var[episode, :, :] = sigma.weights

    outputinfo.weight_policy.append(weight_policy)
    outputinfo.weight_Q.append(weight_Q)
    outputinfo.history.append(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma_Q', help='Discount factor for Q value', type=float, default=0.99)
    parser.add_argument('--gamma_var', help='Discount factor for Variance in return of state-action value', type=float,
                        default=0.99)
    parser.add_argument('--lmbda', help='Lambda', type=float, default=0.6)
    parser.add_argument('--lr_critic', help="Learning rate for Q value", type=float, default=0.05)
    parser.add_argument('--lr_theta', help="Learning rate for policy parameterization theta", type=float, default=0.001)
    parser.add_argument('--lr_sigma', help="Learning rate for sigma variance of return", type=float, default=0.02)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=0.05)
    parser.add_argument('--psi', help="Psi regularizer for Variance in return", type=float, default=0.0)
    parser.add_argument('--psiFixed', help="Psi regularizer is held fixed", type=str2bool,
                        default=True)  # True: fixed psi, False:Variable Psi
    parser.add_argument('--psiRate', help="Num of episodes to reach psi value given", type=int,
                        default=1)  # Num of episodes by which psi value should change from zero to psi value
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)

    args = parser.parse_args()
    now_time = datetime.datetime.now()

    env = gym.make('Fourrooms-v0')
    outer_dir = "../../Neurips2020Results/Results_AC"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "FourRoomSACOnP")
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRTheta" + str(args.lr_theta) + "_LRV" + str(args.lr_sigma) + \
               "_temp" + str(args.temperature) + "_PRate" + str(args.psiRate) + "_lambda" + str(args.lmbda)

    if args.psiFixed:
        dir_name += "_PTypeF"
    else:
        dir_name += "_PTypeV"

    dir_name += "_seed" + str(args.seed)
    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)

    num_states = env.observation_space.n
    nactions = env.action_space.n
    frozen_states = GetFrozenStates()

    threads = []
    features = Tabular(num_states)
    nfeatures = len(features)
    outputinfo = OutputInformation()

    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, features,
                                                     args.nepisodes,
                                                     frozen_states, nfeatures, nactions, num_states, args.temperature,
                                                     args.gamma_Q, args.gamma_var, args.lmbda, args.lr_critic,
                                                     args.lr_sigma, args.lr_theta, args.psi,
                                                     np.random.RandomState(args.seed + i), args.psiFixed,
                                                     args.psiRate,))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    hist = np.asarray(outputinfo.history)
    last_meanreturn = np.round(np.mean(hist[:, :-50, 1]),2)  # Last 100 episodes mean value of the return
    last_stdreturn = np.round(np.std(hist[:, :-50, 1]),2)  # Last 100 episodes std. dev value of the return

    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'Weights_Q.npy'), np.asarray(outputinfo.weight_Q))
    np.save(os.path.join(dir_name, 'Weights_var.npy'), np.asarray(outputinfo.weight_var))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
    save_csv(args, os.path.join(outer_dir, "ParamtersDone.csv"), last_meanreturn, last_stdreturn)

# best till now: c =0.05, theta = 1e-3, temp= 0.05, lam =0.6, psi 0.0
