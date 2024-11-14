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
from scipy.sparse import csr_matrix


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
    
    ######################################################################
    # using linear programming
    value_func = linear_program(P, Q, Constants)

    # The policy converged, it is now optimal
    return value_func, policy



def linear_program(P: np.ndarray, Q: np.ndarray, const: Constants) -> np.ndarray:
    
    alpha = 0.99999
    
    Pt = np.transpose(P, (2, 0, 1))
    
    # unfold P along its third dimension
    A = np.tile(np.eye(const.K), (const.L, 1)) - alpha * Pt.reshape((const.K * const.L, const.K))
    
    b = Q.flatten(order="F")
    
    # define the cost vector
    c = np.ones(const.K) * -1
    
    # solve the linear program
    # res = linprog(c, A_ub=A, b_ub=b)
    # res = linprog(c, A_ub=A, b_ub=b, method="highs-ds")
    res = linprog(c, A_ub=A, b_ub=b, method="highs-ipm")
    
    return res.x
    