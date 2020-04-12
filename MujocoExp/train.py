#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Built on top of Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
from gym_extensions.continuous import box2d
import numpy as np
from policy import Policy
from value_function import NNValueFunction
from variance_function import NNVarianceFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import _pickle as pickle


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, gamma, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                '_MeanReturn': np.mean([discount(t['rewards'], gamma).sum() for t in trajectories]),
                'Steps': total_steps})
    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


# should be called after calculating the value of state: add_value()
def add_delta(trajectories, gamma):
    """ Adds td error at all time steps to the trajectory

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'delta')
    """
    for trajectory in trajectories:
        values = trajectory['values']
        rewards = trajectory['rewards']
        deltas = rewards - values + np.append(values[1:] * gamma, 0)
        trajectory['deltas'] = deltas


# it is discounted reward for variance of states : where reward_var = delta^square
def add_disc_sum_delta_square(trajectories, gamma_var):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma_var: discount of variance: square(gamma* lam)

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_delta_square')
    """
    for trajectory in trajectories:
        deltas = trajectory['deltas']
        deltas_square = np.square(deltas)
        disc_sum_delta_square = discount(deltas_square, gamma_var)
        trajectory['disc_sum_delta_square'] = disc_sum_delta_square


# added variance of return of a state at each point in trajectory
def add_variance(trajectories, var_func):
    """ Adds estimated variance in return of a state to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        var_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'variances')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        variances = var_func.predict(observes)
        trajectory['variances'] = variances


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


# adding advantage function calculator for variance of state
def add_gae_var(trajectories, gamma_var, lam):
    """ Add generalized advantage estimator for finding advantage of  variance:  sum(gamma*lambda)^k * delta_var

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted delta square - Var_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        # td error
        deltas = trajectory['deltas']
        variances = trajectory['variances']
        deltas_square = np.square(deltas)
        tds_var = deltas_square + np.append(variances[1:] * gamma_var, 0) - variances
        advantages_var = discount(tds_var, gamma_var * lam)
        trajectory['advantages_var'] = advantages_var


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    disc_sum_delta_square = np.concatenate([t['disc_sum_delta_square'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    advantages_var = np.concatenate([t['advantages_var'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    advantages_var = (advantages_var - advantages_var.mean()) / (advantages_var.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew, advantages_var, disc_sum_delta_square


def log_batch_stats(observes, actions, advantages, disc_sum_rew, advantages_var, disc_sum_delta_square, logger, episode,
                    seed):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_mean_adv_var': np.mean(advantages_var),
                '_min_adv_var': np.min(advantages_var),
                '_max_adv_var': np.max(advantages_var),
                '_std_adv_var': np.var(advantages_var),
                '_mean_rew_var': np.mean(disc_sum_delta_square),
                '_min_rew_var': np.min(disc_sum_delta_square),
                '_max_rew_var': np.max(disc_sum_delta_square),
                '_std_rew_var': np.var(disc_sum_delta_square),
                '_Episode': episode,
                '_Seed': seed
                })


def main(env_name, num_episodes, seed, gamma, lam, psi, kl_targ, batch_size, hid1_mult, policy_logvar):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    killer = GracefulKiller()
    name_folder = env_name + "_Psi" + str(psi)  # + "_" + datetime.utcnow().strftime("%b-%d_%H:%M:%S")

    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    # now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    now = "Seed" + str(seed)
    logger = Logger(logname=name_folder, now=now)
    # aigym_path = os.path.join('/tmp', name_folder, now)
    # env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    var_func = NNVarianceFunction(obs_dim, hid1_mult)
    policy = Policy(env_name, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, psi, name_folder, seed)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, gamma, logger, episodes=5)
    gamma_var = gamma * lam
    episode = 0
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, gamma, logger, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_delta(trajectories, gamma)
        add_variance(trajectories, var_func)  # add estimated variances to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_disc_sum_delta_square(trajectories,
                                  gamma_var)  # calculated discounted sum of delta square for estimating variance
        add_gae(trajectories, gamma, lam)  # calculate advantage
        add_gae_var(trajectories, gamma_var, lam)  # calculate advantage for variance
        # concatenate all episodes into single NumPy arrays

        observes, actions, advantages, disc_sum_rew, advantages_var, disc_sum_delta_square = build_train_set(
            trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, advantages_var, disc_sum_delta_square, logger,
                        episode, seed)
        policy.update(observes, actions, advantages, advantages_var, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        var_func.fit(observes, disc_sum_delta_square, logger)  # update variance function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False

    scale, offset = scaler.get()
    data = {'SCALE': scale, 'OFFSET': offset}
    folder_save = os.path.join("saved_models", name_folder)
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    folder_save = os.path.join(folder_save, "Seed" + str(seed))
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    directory_to_store_data = folder_save
    file_name = directory_to_store_data + 'scale_and_offset.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

    logger.close()
    policy.close_sess()
    val_func.close_sess()
    var_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-s', '--seed', type=int, help='Seed Value',
                        default=1)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-p', '--psi', type=float, help='Psi Regularizer for Variance', default=0.0)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value, variance and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)

    args = parser.parse_args()
    main(**vars(args))
