import gym
import argparse
import numpy as np
import math
from puddlesimple import PuddleSimpleEnv
import os
import datetime
import threading
from tiles3 import *

class TileFeature:
    def __init__(self, ntiles, nbins, discrete_states, features_range):
        self.ntiles = ntiles
        self.nbins = nbins
        self.max_discrete_states = discrete_states
        self.iht = IHT(discrete_states)
        self.features_range = features_range
        self.scaling = nbins /features_range

    def __call__(self, input_observation):
        return tiles(self.iht, self.ntiles, input_observation*self.scaling)

    def __len__(self):
        return self.max_discrete_states


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

    def sample(self,phi):
        return int(self.rng.randint(self.nactions))

    def pmf(self,phi):
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

    def pmf(self, phi):
        pmf_array = np.zeros(self.nactions)
        pmf_array[int(np.argmax(self.value(phi)))] = (1. - self.epsilon)
        pmf_array += np.ones(self.nactions)*(self.epsilon/self.nactions)
        return pmf_array


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

    def pmf(self, phi):
        pmf_array = np.zeros(self.nactions)
        pmf_array[int(np.argmax(self.value(phi)))] = 1.
        return pmf_array


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = 0.5 * np.ones((nfeatures, nactions)) # positive weight initialization
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
        y /= y.sum()
        # y = np.round(y, 4)
        y_sum = y.sum()
        if y_sum != 1.0 :
            x = int(np.argmax(y))
            y[x] += 1.0 - y_sum
        return y

    def sample(self, phi):
        prob = self.pmf(phi)
        return int(self.rng.choice(self.nactions, p=prob))


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
        self.last_rho = min(1, (self.policy.pmf(phi)[int(action)] / self.behavioral_policy.pmf(phi)[int(action)]))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    # Update for the parameter of the Q and Sigma value (s,a)
    def update(self, phi, action, reward, done):
        current_rho = min(1, (self.policy.pmf(phi)[int(action)] / self.behavioral_policy.pmf(phi)[int(action)]))
        if self.variance:
            self.trace *= math.pow(self.gamma * self.lmbda * current_rho, 2.)
        else:
            self.trace *= self.gamma * self.lmbda * current_rho
        self.trace[self.last_phi, self.last_action] += 1

        update_target = reward
        if not done:
            current_value = self.value(phi, action)
            corrected_curent_value = current_value
            if self.variance:
                corrected_curent_value *= math.pow(self.gamma * self.lmbda * current_rho, 2.)
            else:
                corrected_curent_value *= current_rho * self.gamma
            update_target += corrected_curent_value

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
    def __init__(self, policy, lr, psi, gamma, lmbda):
        self.lr = lr
        self.policy = policy
        self.psi = psi
        self.trace_Q = np.zeros_like(policy.weights)
        self.trace_var = np.zeros_like(policy.weights)
        self.gamma = gamma
        self.lmbda = lmbda

    # Updation of the theta parameter of the policy
    def update(self, phi, action, critic, sigma, first_time_step, rho, initial_rho):
        actions_pmf = self.policy.pmf(phi)
        if self.psi != 0.0:  # variance as regularization factor to optimization criterion
            self.trace_var *= (self.gamma * self.lmbda) ** 2.0
            self.trace_var[phi, :] -= actions_pmf
            self.trace_var[phi, action] += 1.
            self.trace_var *= rho ** 2.0
            psi = self.psi
            # If not a first step in the episode, new psi = 2*psi, else for first step, new psi is psi. Check algorithm for details.
            if first_time_step == 0:
                psi = 2 * self.psi
            self.policy.weights -= self.lr * psi * sigma * self.trace_var / initial_rho

        self.trace_Q *= (self.gamma * self.lmbda)
        self.trace_Q[phi, :] -= actions_pmf
        self.trace_Q[phi, action] += 1.
        self.trace_Q *= rho
        self.policy.weights += self.lr * critic * self.trace_Q

# Saves weight in the following while running for a thread
class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.weight_Q = []
        self.weight_sigma = []
        self.history = []

# Generates a randlom reward drawn from normal distribution
def tweak_reward_near_puddle(reward):
    noise_mean = 0.0
    noise_sigma = 8.0
    noise = np.random.normal(noise_mean, noise_sigma)
    return reward + noise

# Checks whether the agent has entered puddle zone
def check_if_agent_near_puddle(observation):
    if (observation[0] <= 0.7 and observation[0] >= 0.3):
        if (observation[1] <= 0.7 and observation[1] >= 0.3):
            return True
    return False

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


# Get return for target policy : For evaluating the target policy performance
def get_target_evaluation(policy, gamma, features, env):
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
        if check_if_agent_near_puddle(old_observation):
            reward = tweak_reward_near_puddle(reward)
        return_value += current_gamma * reward
        current_gamma *= gamma
        phi = features(observation)
        step += 1
    return return_value, step

# Runs random behavior policy and use that experience to update target policy (Boltzman distribution)
def run_agent(outputinfo, nepisodes, ksteps,
              temperature, gamma, lmbda, lr_critic, lr_sigma, lr_theta, psi, rng,
              maxDiscreteStates, threadNum, dir_name, ntiles):
    env = gym.make('PuddleEnv-v0')
    features_range = env.observation_space.high - env.observation_space.low
    features = TileFeature(ntiles, 5, maxDiscreteStates, features_range) # 5X5 tiles over joint space
    nfeatures = int(maxDiscreteStates)
    nactions = env.action_space.n
    storing_arr_dim = int(nepisodes / ksteps)
    history = np.zeros((nepisodes, 2))  # 1. Return from Target 2. Steps in target 3. Return from behavior policy

    # storage the weights of the trained model
    weight_policy = np.zeros((storing_arr_dim, nfeatures, nactions))
    weight_Q = np.zeros((storing_arr_dim, nfeatures, nactions))
    weight_sigma = np.zeros((storing_arr_dim, nfeatures, nactions))
    behavioral_policy = RandomPolicy(nactions, rng) # Randomly uniform policy over all the actions

    #Target policy is Boltzmann distribution
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
    # Action_critic is Q value of state-action pair
    weights_QVal = np.zeros((nfeatures, nactions), dtype=np.float32)
    trace_Qval = np.zeros_like(weights_QVal)
    action_critic = StateActionLearning(gamma, lmbda, lr_critic, weights_QVal, trace_Qval,
                                        policy, behavioral_policy, 0)

    # Variance is sigma of state-action pair
    weights_var = np.zeros((nfeatures, nactions), dtype=np.float32)
    trace_var = np.zeros_like(weights_var)
    sigma = StateActionLearning(gamma, lmbda, lr_sigma, weights_var, trace_var, policy,
                                behavioral_policy, 1)
    
    # Policy gradient improvement step
    policy_improvement = PolicyGradient(policy, lr_theta, psi, gamma, lmbda)
    for episode in range(args.nepisodes):
        first_time_step = 1
        observation = env.reset()
        phi = features(observation)
        action = behavioral_policy.sample(phi)
        action_critic.start(phi, action)
        sigma.start(phi, action)
        initial_rho = min(1, (policy.pmf(phi)[int(action)] / behavioral_policy.pmf(phi)[int(action)]))
        step = 0
        done = False
        return_behavior_policy = 0.0
        while done != True:
            old_observation = observation
            old_phi = phi
            old_action = action
            observation, reward, done, _ = env.step(action)
            # modify the reward if agent transitions out of unsafe puddle region, reward ~ N(0,8)
            if check_if_agent_near_puddle(old_observation):
                reward = tweak_reward_near_puddle(reward)

            phi = features(observation)
            # Get action from the behavioral policy
            action = behavioral_policy.sample(phi)

            # Critic update
            tderror, rho = action_critic.update(phi, action, reward, done)
            if psi != 0.0: # updating parameters when the safe version is followed
                try:
                    td_square = math.pow(tderror, 2.0)
                # Just to prevent the overflow error
                except OverflowError:
                    td_square = tderror
                sigma.update(phi, action, td_square, done)
                sigma_val = sigma.value(old_phi, old_action)
            else:
                sigma_val = 0.0

            if episode % ksteps == 0:
                return_behavior_policy += (gamma ** step) * reward

            critic_val = action_critic.value(old_phi, old_action)
            policy_improvement.update(old_phi, old_action, critic_val, sigma_val, first_time_step, rho, initial_rho)
            first_time_step = 0
            step += 1

            
        if episode % ksteps == 0:
            c_index = int(episode / ksteps)
            weight_policy[c_index] = policy.weights
            weight_Q[c_index] = action_critic.weights
            weight_sigma[c_index] = sigma.weights

        # Get the return under the target policy
        return_target, step_target = get_target_evaluation(policy, gamma, features, env)
        history[episode, 0] = return_target
        history[episode, 1] = step_target

        # storing values after every 100 episode to see performance intermittently
        if episode%100 == 0:
            new_name = dir_name+"_thread"+str(threadNum)
            if not os.path.exists(new_name):
                os.makedirs(new_name)
            np.save(os.path.join(new_name,"History.npy"),history)
            np.save(os.path.join(new_name,"PolicyWeight.npy"),weight_policy)
            with open(os.path.join(new_name,'EpisodeDone.txt'), 'w') as file:
                file.write(str(episode))

    outputinfo.history.append(history)
    outputinfo.weight_Q.append(weight_Q)
    outputinfo.weight_policy.append(weight_policy)
    outputinfo.weight_sigma.append(weight_sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Best parameters are mentioned already as default value. Check paper appendix for seeing best parameters.
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lmbda', help='Lambda', type=float, default=0.4)
    parser.add_argument('--lr_critic', help="Learning rate for Q value", type=float, default=0.01)
    parser.add_argument('--lr_theta', help="Learning rate for policy parameterization theta", type=float, default=0.001)
    parser.add_argument('--lr_sigma', help="Learning rate for sigma variance of return", type=float, default=0.005)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=0.75)
    parser.add_argument('--psi', help="Psi regularizer for Variance in return", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=3500)
    parser.add_argument('--ksteps', help="After how many steps evaluate the target policy", type=int, default=50)
    parser.add_argument('--nruns', help="Number of run for target policy", type=int, default=10)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)
    parser.add_argument('--maxDiscreteStates', help="discrete states", type=int, default=1024)
    parser.add_argument('--ntiles', help="tiles", type=int, default=10)


    args = parser.parse_args()
    now_time = datetime.datetime.now()

    outer_dir = "Results"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "PuddleSACOffP_" + now_time.strftime("%d-%m"))
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRTheta" + str(args.lr_theta) + "_LRV" + str(args.lr_sigma) + \
               "_temp" + str(args.temperature) + "_ksteps"+ str(args.ksteps) +"_states" + \
               str(args.maxDiscreteStates) + "_ntile"+str(args.ntiles)+"_seed" + str(args.seed)
    dir_name += "_Policy_" + "S"
    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)
    threads = []
    outputinfo = OutputInformation()# Object for storing the information from all threads

    # Tile coding, therefore diving learning rate by num of tiles
    args.lr_theta /= args.ntiles
    args.lr_critic /= args.ntiles
    args.lr_sigma /= args.ntiles

	# Threading the multiple runs under different thread to speed up
    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, args.nepisodes, args.ksteps,
                                                     args.temperature,
                                                     args.gamma, args.lmbda, args.lr_critic,
                                                     args.lr_sigma, args.lr_theta, args.psi,
                                                     np.random.RandomState(args.seed + i), args.maxDiscreteStates,i,
                                                     dir_name,args.ntiles, ))
        threads.append(t)
        t.start()
    for x in threads:
        x.join()

    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'Weights_Q.npy'), np.asarray(outputinfo.weight_Q))
    np.save(os.path.join(dir_name, 'Weights_Sigma.npy'), np.asarray(outputinfo.weight_sigma))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
