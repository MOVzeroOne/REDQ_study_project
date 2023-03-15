import torch
import gymnasium as gym
from copy import deepcopy


def single_run_Q(env, state, policy, gamma, max_iter: int = -1):
    """
    Simulate the environment starting from the given current state using the given policy,
    and compute the total discounted reward obtained during the simulation.

    Args:
        env (gym.Env): the environment to simulate.
        state (numpy.ndarray): the current state of the simulation.
        policy (callable): a function that takes a PyTorch tensor as input and returns an action.
        gamma (float): the discount factor to use for computing the discounted reward.
        max_iter (int): the maximum number of steps to simulate the environment.
                        (if -1, simulate until termination) (default: -1)

    Returns:
        total_reward (float): the total discounted reward obtained during the simulation.
    """
    total_reward = 0.0
    # Simulate the environment until termination or until reaching max_iter.
    episode = 0
    done = False
    while not done and episode != max_iter:
        # Choose an action using the policy
        action = policy(torch.from_numpy(state).float())
        # Step the environment
        next_state, reward, done, *_ = env.step(action)
        # Update the total reward with the discounted reward
        total_reward += gamma * reward
        state = next_state
        # Update the counter.
        episode += 1
    return total_reward


def MC_Q(env, state, policy, num_samples, gamma):
    """
    Estimate the Q value at a given state by averaging the discounted rewards obtained
    from multiple simulations using the given policy.

    Args:
        env (gym.Env): the environment to simulate.
        state (ndarray): a given state to simulate from.
        policy (callable): a function that takes a state as input and returns an action.
        num_samples (int): the number of simulations to run.
        gamma (float): the discount factor.

    Returns:
        q_value (float): the estimated Q value
    """
    total_discounted_reward = 0.0
    # Run multiple simulations from the given state
    for i in range(num_samples):
        total_discounted_reward += single_run_Q(deepcopy(env), state, policy, gamma)
    q_value = total_discounted_reward / num_samples
    return q_value
