"""
 Solver.py

 Python function template to solve the stochastic
 shortest path problem.

 Dynamic Programming and Optimal Control
 Fall 2024
 Programming Exercise

 Contact: Antonio Terpin aterpin@ethz.ch

 Authors: Maximilian Stralz, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np
from utils import *


def initialize_policy(constants: Constants) -> np.array:
    """Initializes the policy.
    Args:
        Constants: The constants describing the problem instance.
    Returns:
        np.array: The initial policy.
    """
    # admissible_actions = compute_admissible_actions(Constants)
    # policy = []
    # # Random initial policy
    # for state in range(Constants.K):
    #     policy[state] = np.random.choice(list(admissible_actions[state]))
    return np.zeros(constants.K, dtype=int)


def solution(P, Q, Constants):
    """Computes the optimal cost and the optimal control input for each
    state of the state space solving the stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming;
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        Q  (np.array): A (K x L)-matrix containing the expected stage costs of all states
                       in the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP
        np.array: The optimal control policy for the stochastic SPP

    """
    # Constants
    policy = initialize_policy(Constants)
    value_func = np.zeros(Constants.K)

    epsilon = 1e-3
    last_policy = np.ones(Constants.K) * -1

    while not np.all(policy == last_policy):
        
        last_policy = policy.copy()
        # Update value function, policy is fixed
        old_value_func = value_func.copy()
        delta = np.inf

        while delta > epsilon:
            for state in range(Constants.K):
                action = policy[state]
                expected_stage_cost = Q[state][action]
                transition_probs = P[state, :, action]
                state_value = expected_stage_cost + np.dot(transition_probs, value_func)
                value_func[state] = state_value

            delta = np.max(np.abs(old_value_func - value_func))
            old_value_func = value_func.copy()

        # Update policy, value function is fixed
        old_policy = np.ones(Constants.K) * -1
        while not np.all(policy == old_policy):
            old_policy = policy.copy()
            for state in range(Constants.K):
                best_action = None
                best_value = -np.inf

                for action in range(Constants.L):
                    transition_probs = P[state, :, action]
                    current_value = Q[state][action] + np.dot(transition_probs, value_func)
                    if current_value > best_value:
                        best_value = current_value
                        best_action = action

                policy[state] = best_action

    # The policy converged, it is now optimal
    return value_func, policy
