import gym
import argparse
import numpy as np
from PuddleDiscrete import PuddleD
import os
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


class GreedyPolicy:
    def __init__(self, nactions, weights):
        self.nactions = nactions
        self.weights = weights

    def value(self, phi):
        return np.sum(self.weights[phi, :], axis=0)

    def sample(self, phi):
        return int(np.argmax(self.value(phi)))


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.random.rand(nfeatures, nactions)  # positive weight initialization
        self.nactions = nactions
        self.temp = temp

    def value(self, phi):
        return np.sum(self.weights[phi, :], axis=0)

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
    def __init__(self, gamma, lr, weights, policy, behavioral_policy, variance):
        self.lr = lr
        self.gamma = gamma
        self.weights = weights
        self.policy = policy
        self.behavioral_policy = behavioral_policy
        self.variance = variance  # binary value (0: its is Q(s,a) value, 1: sigma(s,a) value)

    def start(self, phi, action):
        self.last_phi = phi
        self.last_action = action
        self.last_value = self.value(phi, action)

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    # Update for the parameter of the Q and Sigma value (s,a)
    def update(self, phi, action, reward, done):
        update_target = reward
        if not done:
            current_rho = min(1, (self.policy.pmf(phi)[int(action)] / self.behavioral_policy.pmf()[int(action)]))
            current_value = self.value(phi, action)
            if self.variance:
                update_target += self.gamma * (current_rho ** 2.0) * current_value
            else:
                update_target += self.gamma * current_rho * current_value

        # Weight gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_action] += self.lr * tderror
        if not done:
            self.last_value = current_value
        self.last_action = action
        self.last_phi = phi
        return tderror


class PolicyGradient:
    def __init__(self, policy, lr):
        self.lr = lr
        self.policy = policy

    # Updation of the theta parameter of the policy
    def update(self, phi, action, critic, sigma, psi, I_Q, I_sigma, rho_Q, rho_sigma):
        actions_pmf = self.policy.pmf(phi)
        if psi != 0.0:  # variance as regularization factor to optimization criterion
            var_constant = -self.lr * psi * I_sigma * rho_sigma * sigma
            self.policy.weights[phi, :] -= var_constant * actions_pmf
            self.policy.weights[phi, action] += var_constant

        Q_constant = self.lr * I_Q * rho_Q * critic
        self.policy.weights[phi, :] -= Q_constant * actions_pmf
        self.policy.weights[phi, action] += Q_constant


class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.history = []
        self.var = []


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
    return return_value, step


# get variance in return for target policy
def VarReturnTargetPolicy(weights, gamma_Q, frozen_states, features, nactions):
    num_rollouts = 800
    env = gym.make('Puddle-v1')
    policy = GreedyPolicy(nactions, weights)
    return_dist = []
    for rollout in range(num_rollouts):
        observation = env.reset()
        return_value = 0.0
        phi = features(observation)
        done = False
        current_gamma = 1.0
        while done != True:
            old_observation = observation
            action = policy.sample(phi)
            observation, reward, done, _ = env.step(action)
            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)
            return_value += current_gamma * reward
            current_gamma *= gamma_Q
            phi = features(observation)
        return_dist.append(return_value)
    return np.var(return_dist)


def run_agent(outputinfo, nepisodes, frozen_states, temperature, gamma_Q, gamma_var,
              lr_critic, lr_sigma, lr_theta, psi, rng):
    env = gym.make('Puddle-v1')
    num_states = env.observation_space.n
    features = Tabular(num_states)
    nfeatures = len(features)
    nactions = env.action_space.n

    get_variance_after_episode = 100
    list_eps = list(np.arange(0, nepisodes, get_variance_after_episode))
    list_eps.append(nepisodes - 1)
    var_return_list = []
    save_index = 0

    behavioral_policy = RandomPolicy(nactions, rng)
    storing_arr_dim = len(list_eps)

    history = np.zeros((nepisodes, 2))  # 1. Return from Target 2. Steps in target
    # storage the weights of the trained model
    weight_policy = np.zeros((storing_arr_dim, nfeatures, nactions),
                             dtype=np.float32)

    # Target policy is as softmax policy
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
    # Action_critic is Q value of state-action pair
    weights_QVal = np.zeros((nfeatures, nactions)) # positive weight initialization
    action_critic = StateActionLearning(gamma_Q, lr_critic, weights_QVal, policy, behavioral_policy, 0)

    # Variance is sigma of state-action pair
    weights_var = np.zeros((nfeatures, nactions))  # weight initialization
    sigma = StateActionLearning(gamma_var, lr_sigma, weights_var, policy,
                                behavioral_policy, 1)

    # Policy gradient improvement step
    policy_improvement = PolicyGradient(policy, lr_theta)

    for episode in range(nepisodes):
        first_time_step = 1
        observation = env.reset()
        phi = features(observation)
        action = behavioral_policy.sample()
        action_critic.start(phi, action)
        sigma.start(phi, action)
        step = 0
        done = False
        I_Q = 1.0
        I_sigma = 1.0
        rho_Q = min(1, (policy.pmf(phi)[int(action)] / behavioral_policy.pmf()[int(action)]))
        rho_sigma = rho_Q
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
            tderror = action_critic.update(phi, action, reward, done)
            if psi != 0.0:
                try:
                    td_square = pow(tderror, 2.0)
                # Just to prevent the overflow error
                except OverflowError:
                    td_square = tderror * 2
                sigma.update(phi, action, td_square, done)
                sigma_val = sigma.value(old_phi, old_action)
            else:
                sigma_val = 0.0

            critic_val = action_critic.value(old_phi, old_action)
            policy_improvement.update(old_phi, old_action, critic_val, sigma_val, psi, I_Q, I_sigma, rho_Q, rho_sigma)
            if first_time_step == 1:
                psi = 2 * psi
                first_time_step = 0
            step += 1
            I_Q *= gamma_Q
            I_sigma *= gamma_var
            rho = min(1, (policy.pmf(phi)[int(action)] / behavioral_policy.pmf()[int(action)]))
            rho_Q *= rho
            rho_sigma *= rho**2

        return_target, step_target = ReturnTargetPolicy(policy.weights, gamma_Q, frozen_states, features, nactions)
        if episode in list_eps:
            weight_policy[save_index] = policy.weights
            var_return_list.append(VarReturnTargetPolicy(policy.weights, gamma_Q, frozen_states, features, nactions))
            save_index += 1

        history[episode, 0] = return_target
        history[episode, 1] = step_target

    outputinfo.history.append(history)
    outputinfo.weight_policy.append(weight_policy)
    outputinfo.var.append(np.asarray(var_return_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lmbda', help='Lambda', type=float, default=0.5)
    parser.add_argument('--lr_critic', help="Learning rate for Q value", type=float, default=0.025)
    parser.add_argument('--lr_theta', help="Learning rate for policy parameterization theta", type=float,
                        default=0.0025)
    parser.add_argument('--lr_sigma', help="Learning rate for sigma variance of return", type=float, default=0.006)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=0.25)
    parser.add_argument('--psi', help="Psi regularizer for Variance in return", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=2000)
    parser.add_argument('--nruns', help="Number of run for target policy", type=int, default=50)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=20)

    args = parser.parse_args()
    outer_dir = "../../Neurips2020Results/Results_Puddle"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "PuddleDiscrete")
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRTheta" + str(args.lr_theta) + "_LRV" + str(args.lr_sigma) + \
               "_temp" + str(args.temperature) + "_lambda" + str(args.lmbda) + "_seed" + str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)
    frozen_states = GetFrozenStates()  # To get the states where the state is frozen state
    threads = []
    outputinfo = OutputInformation()  # Object for storing the information from all threads
    gamma_var = pow(args.gamma * args.lmbda, 2.0)

    # Threading the multiple runs under different thread to speed up
    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, args.nepisodes,
                                                     frozen_states, args.temperature,
                                                     args.gamma, gamma_var, args.lr_critic,
                                                     args.lr_sigma, args.lr_theta, args.psi,
                                                     np.random.RandomState(args.seed + i),))
        threads.append(t)
        t.start()
    for x in threads:
        x.join()
    return_array = np.asarray(outputinfo.history)[:, :, 0]
    step_array = np.asarray(outputinfo.history)[:, :, 1]
    var_return_array = np.asarray(outputinfo.var)

    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
    np.save(os.path.join(dir_name, 'RewardDistributionVarMean.npy'), np.mean(var_return_array, axis=0))
    np.save(os.path.join(dir_name, 'RewardDistributionVarStd.npy'), np.std(var_return_array, axis=0))
    np.save(os.path.join(dir_name, 'RewardDistributionMeanMean.npy'), np.mean(return_array, axis=0))
    np.save(os.path.join(dir_name, 'RewardDistributionMeanStd.npy'), np.std(return_array, axis=0))
    np.save(os.path.join(dir_name, 'RewardDistributionStep.npy'), np.mean(step_array, axis=0))
