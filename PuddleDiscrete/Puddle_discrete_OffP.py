import gym
import argparse
import numpy as np
from PuddleDiscrete import PuddleD
import math
from scipy.special import expit
from scipy.misc import logsumexp
import os
import sys
import datetime
import threading


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state, ])

    def __len__(self):
        return self.nstates


class RandomPolicy:
    def __init__(self, nactions, rng):
        self.nactions = nactions
        self.rng = rng
        self.prob_actions = [1. / self.nactions] * self.nactions

    def sample(self):
        return int(self.rng.randint(self.nactions))

    def pmf(self):
        return self.prob_actions


class EgreedyPolicy:
    def __init__(self, rng, nactions, epsilon, weights):
        self.rng = rng
        self.epsilon = epsilon
        self.nactions = nactions
        self.weights = weights

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.nactions))
        return int(np.argmax(self.value(phi)))


class GreedyPolicy:
    def __init__(self, nactions, weights):
        self.nactions = nactions
        self.weights = weights

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        return int(np.argmax(self.value(phi)))


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = 0.5 * np.ones((nfeatures, nactions))  # positive weight initialization
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


class MCEstimate:
    # for getting the true value function of target policy
    def __init__(self, nactions, nfeatures, nruns, features, frozen_states, gamma, weight_policy):
        self.weight_policy = weight_policy
        self.nactions = nactions
        self.nfeatures = nfeatures
        self.nruns = nruns
        self.features = features
        self.frozen_states = frozen_states
        self.gamma = gamma
        self.G_return = np.zeros((nfeatures, nactions))
        self.visit_freq = np.zeros((nfeatures, nactions))
        self.policy = GreedyPolicy(nactions, weight_policy)
        self.return_sum = []

    def run_agent_MC(self):
        env = gym.make('Puddle-v1')
        observation = env.reset()
        states = []
        actions = []
        rewards = []
        phi = self.features(observation)
        done = False
        while done != True:
            old_observation = observation

            action = self.policy.sample(phi)
            observation, reward, done, _ = env.step(action)

            # Frozen state receives a variable uniform reward[-15, 15]
            if old_observation in self.frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)

            states.append(phi)
            actions.append(action)
            rewards.append(reward)
            phi = self.features(observation)
            if done:
                break

        sum_reward = 0
        for i in range(len(rewards) - 1, -1, -1):
            sum_reward += (self.gamma ** i) * rewards[i]
            self.G_return[states[i], actions[i]] += sum_reward
            self.visit_freq[states[i], actions[i]] += 1

    # To get the actual estimate of the Q function under the given theta for the policy by MC method
    def getQEstimates(self):
        threads_internal = []
        for r in range(self.nruns):
            t = threading.Thread(target=self.run_agent_MC)
            threads_internal.append(t)
            t.start()
        for x in threads_internal:
            x.join()

        self.visit_freq[self.visit_freq == 0.] = 1.
        Q = np.divide(self.G_return, self.visit_freq)
        return Q


class StateActionLearning:
    def __init__(self, gamma, lmbda, lr, weights, trace, policy, behavioral_policy, variance):
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.weights = weights
        self.trace = trace
        self.policy = policy
        self.behavioral_policy = behavioral_policy
        self.variance = variance  # binary value (0: its is Q(s,a) value, 1: sigma(s,a) value)

    def start(self, phi, action):
        self.last_phi = phi
        self.last_action = action
        self.last_value = self.value(phi, action)
        # retrace correction factor
        self.last_rho = min(1, (self.policy.pmf(phi)[int(action)] / self.behavioral_policy.pmf()[int(action)]))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    # Update for the parameter of the Q and Sigma value (s,a)
    def update(self, phi, action, reward, done):
        if self.variance:
            self.trace *= math.pow(self.gamma * self.lmbda * self.last_rho, 2.)
        else:
            self.trace *= self.gamma * self.lmbda * self.last_rho
        self.trace[self.last_phi, self.last_action] += 1
        self.trace = np.clip(self.trace, -100., 100.)

        update_target = reward
        if not done:
            current_rho = min(1, (self.policy.pmf(phi)[int(action)] / self.behavioral_policy.pmf()[int(action)]))
            current_value = self.value(phi, action)
            corrected_curent_value = current_value
            if self.variance:
                corrected_curent_value *= math.pow(self.gamma * self.lmbda * current_rho, 2.)
            else:
                corrected_curent_value *= current_rho * self.gamma
            update_target += self.gamma * corrected_curent_value

        # Weight gradient update step
        tderror = update_target - self.last_value
        self.weights += self.lr * tderror * self.trace

        rho_output = self.last_rho

        if not done:
            self.last_value = current_value
            self.last_rho = current_rho

        self.last_action = action
        self.last_phi = phi

        return tderror, rho_output


class PolicyGradient:
    def __init__(self, policy, lr, psi, gamma_Q, gamma_var, lmbda):
        self.lr = lr
        self.policy = policy
        self.psi = psi
        self.trace_Q = np.zeros_like(policy.weights)
        self.trace_var = np.zeros_like(policy.weights)
        self.gamma_Q = gamma_Q
        self.gamma_var = gamma_var
        self.lmbda = lmbda
        self.gamma_var = gamma_var

    # Updation of the theta parameter of the policy
    def update(self, phi, action, critic, sigma, first_time_step, rho, initial_rho, psi):
        actions_pmf = self.policy.pmf(phi)
        if self.psi != 0.0:  # variance as regularization factor to optimization criterion
            self.trace_var *= (self.gamma_var * self.lmbda) ** 2.0
            self.trace_var[phi, :] -= actions_pmf
            self.trace_var[phi, action] += 1.
            self.trace_var *= rho ** 2.0
            # psi = self.psi
            # if first_time_step == 0:
            #     psi = 2 * self.psi
            self.policy.weights -= self.lr * psi * sigma * self.trace_var / initial_rho

        self.trace_Q *= (self.gamma_Q * self.lmbda)
        self.trace_Q[phi, :] -= actions_pmf
        self.trace_Q[phi, action] += 1.
        self.trace_Q *= rho
        self.policy.weights += self.lr * critic * self.trace_Q


class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.weight_Q = []
        self.history = []


def GetFrozenStates():
    layout = """\
wwwwwwwwwwww
w          w
w          w
w          w
w   ffff   w
w   ffff   w
w   ffff   w
w   ffff   w
w          w
w          w
w          w
wwwwwwwwwwww
"""

    num_elem = 12
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


# Get return for target policy
def ReturnTargetPolicy(weights, gamma_Q, frozen_states, features, nactions):
    env = gym.make('Puddle-v1')
    policy = GreedyPolicy(nactions, weights)
    observation = env.reset()
    return_value = 0.0
    phi = features(observation)
    done = False
    current_gamma = 1.0
    step = 0.
    while done != True:
        old_observation = observation

        action = policy.sample(phi)
        observation, reward, done, _ = env.step(action)

        if old_observation in frozen_states:
            reward = np.random.normal(loc=0.0, scale=8.0)

        return_value += current_gamma * reward
        current_gamma *= gamma_Q
        phi = features(observation)
        step += 1
        if done:
            break
    return return_value, step


def run_agent(outputinfo, nepisodes, ksteps,
              frozen_states, temperature, gamma_Q, gamma_var,
              lmbda, lr_critic, lr_sigma, lr_theta, psi, rng, nruns_estimation):
    env = gym.make('Puddle-v1')
    num_states = env.observation_space .n
    features = Tabular(num_states)
    nfeatures = len(features)
    nactions = env.action_space.n

    behavioral_policy = RandomPolicy(nactions, rng)
    storing_arr_dim = int(nepisodes / ksteps)

    history = np.zeros((nepisodes, 3))  # 1. rmse, 2. Return from Target 3. Steps in target 4. Return from behavior policy

    # storage the weights of the trained model
    weight_policy = np.zeros((storing_arr_dim, nfeatures, nactions),
                             dtype=np.float32)
    weight_Q = np.zeros((storing_arr_dim, nfeatures, nactions),
                        dtype=np.float32)

    #Target policy is as oftmax policy
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
    # Action_critic is Q value of state-action pair
    weights_QVal = np.ones((nfeatures, nactions))*0.5  # positive weight initialization
    trace_Qval = np.zeros_like(weights_QVal)
    action_critic = StateActionLearning(gamma_Q, lmbda, lr_critic, weights_QVal, trace_Qval,
                                        policy, behavioral_policy, 0)

    # Variance is sigma of state-action pair
    weights_var = np.zeros((nfeatures, nactions))  #weight initialization
    trace_var = np.zeros_like(weights_var)
    sigma = StateActionLearning(gamma_var, lmbda, lr_sigma, weights_var, trace_var, policy,
                                behavioral_policy, 1)

    # Policy gradient improvement step
    policy_improvement = PolicyGradient(policy, lr_theta, psi, gamma_Q, gamma_var, lmbda)

    first_time_step = 1
    initial_rho = 1.
    n_samples_estimating = 15
    for episode in range(args.nepisodes):
        observation = env.reset()
        phi = features(observation)
        action = behavioral_policy.sample()

        action_critic.start(phi, action)
        sigma.start(phi, action)

        if episode == 0:
            initial_rho = min(1, (policy.pmf(phi)[int(action)] / behavioral_policy.pmf()[int(action)]))

        step = 0
        done = False
        return_behavior_policy = 0.0
        while done != True:

            old_observation = observation
            old_phi = phi
            old_action = action
            observation, reward, done, _ = env.step(action)

            # Frozen state receives a variable normal reward[-8, 8], where reward is given when transition is made
            # out of that state
            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)

            phi = features(observation)
            # Get action from the behavioral policy
            action = behavioral_policy.sample()

            # Critic update
            tderror, rho = action_critic.update(phi, action, reward, done)
            if psi != 0.0:
                try:
                    td_square = math.pow(tderror, 2.0)
                # Just to prevent the overflow error
                except OverflowError:
                    td_square = tderror * 2
                sigma.update(phi, action, td_square, done)
                sigma_val = sigma.value(old_phi, old_action)
            else:
                sigma_val = 0.0

            if episode % ksteps == 0:
                return_behavior_policy += (gamma_Q ** step) * reward

            critic_val = action_critic.value(old_phi, old_action)
            policy_improvement.update(old_phi, old_action, critic_val, sigma_val, first_time_step, rho, initial_rho, psi)
            first_time_step = 0
            step += 1

        if episode % ksteps == 0:
            c_index = int(episode / ksteps)
            weight_policy[c_index] = policy.weights
            weight_Q[c_index] = action_critic.weights

            # Get the return under the target policy
            return_target, step_target = ReturnTargetPolicy(policy.weights, gamma_Q, frozen_states, features, nactions)            
        history[episode, 0] = return_target
        history[episode, 1] = step_target
        history[episode, 2] = return_behavior_policy

    outputinfo.history.append(history)
    outputinfo.weight_Q.append(weight_Q)
    outputinfo.weight_policy.append(weight_policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma_Q', help='Discount factor for Q value', type=float, default=0.99)
    parser.add_argument('--gamma_var', help='Discount factor for Variance in return of state-action value', type=float,
                        default=0.99)
    parser.add_argument('--lmbda', help='Lambda', type=float, default=0.5)
    parser.add_argument('--lr_critic', help="Learning rate for Q value", type=float, default=0.025)
    parser.add_argument('--lr_theta', help="Learning rate for policy parameterization theta", type=float, default=0.0025)
    parser.add_argument('--lr_sigma', help="Learning rate for sigma variance of return", type=float, default=0.006)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=0.25)
    parser.add_argument('--psi', help="Psi regularizer for Variance in return", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=2000)
    parser.add_argument('--ksteps', help="After how many steps evaluate the target policy", type=int, default=40)
    parser.add_argument('--nruns', help="Number of run for target policy", type=int, default=50)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=20)

    args = parser.parse_args()
    now_time = datetime.datetime.now()

    outer_dir = "Results_Puddle_AC_Dec"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "PuddleSACOffP_11-01")# + now_time.strftime("%d-%m"))
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRTheta" + str(args.lr_theta) + "_LRV" + str(args.lr_sigma) + \
               "_temp" + str(args.temperature) + "_seed" + str(args.seed)

    dir_name += "_Policy_" + "S"

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)
    frozen_states = GetFrozenStates()# To get the states where the state is frozen state
    threads = []
    outputinfo = OutputInformation()# Object for storing the information from all threads

    # Threading the multiple runs under different thread to speed up
    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, args.nepisodes, args.ksteps,
                                                     frozen_states, args.temperature,
                                                     args.gamma_Q, args.gamma_var, args.lmbda, args.lr_critic,
                                                     args.lr_sigma, args.lr_theta, args.psi,
                                                     np.random.RandomState(args.seed + i), 1,))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'Weights_Q.npy'), np.asarray(outputinfo.weight_Q))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
