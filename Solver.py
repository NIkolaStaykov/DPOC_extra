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

from scipy.optimize import linprog


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
    
    # find optimal cost using linear programming
    alpha = 0.99999       # discount factor
    value_func = linear_program(P, Q, Constants, alpha)
    
    # find the optimal policy
    P_transposed = np.transpose(P, (0, 2, 1))
    costs = Q + alpha * (P_transposed @ value_func)
    
    policy = np.argmin(costs, axis=1)
    
    row_mins = np.min(costs, axis=1)
    multiple_minima_rows = np.sum((costs == row_mins[:, None]), axis=1) > 1
    num_rows_with_multiple_minima = np.sum(multiple_minima_rows)
    print(f"Number of state with multiple optimal policies: {num_rows_with_multiple_minima}")

    # The policy converged, it is now optimal
    return value_func, policy


def linear_program(P: np.ndarray, Q: np.ndarray, const: Constants, alpha: float) -> np.ndarray:
    
    # re-order the dimensions of the transition probabiltiies matrix
    Pt = np.transpose(P, (2, 0, 1))
    
    # unfold P along its third dimension
    A = np.tile(np.eye(const.K), (const.L, 1)) - alpha * Pt.reshape((const.K * const.L, const.K))
    # A = csr_matrix(np.tile(np.eye(const.K), (const.L, 1)) - alpha * Pt.reshape((const.K * const.L, const.K)))
    
    # flatten the expected stage costs matrix
    b = Q.flatten(order="F")
    
    # define the cost vector to optimize
    c = np.ones(const.K) * -1
    
    # solve the linear program
    res = linprog(c, A_ub=A, b_ub=b, method="highs-ipm")
    
    
    return res.x
    