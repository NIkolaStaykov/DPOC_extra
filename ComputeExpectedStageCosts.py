"""
 ComputeExpectedStageCosts.py

 Python function template to compute the expected stage cost.

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

from utils import *

def compute_expected_stage_cost(Constants):
    """Computes the expected stage cost for the given problem.

    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L)
    """
    
    # Create a singleton class to store the instance
    values = SingletonClass()
    if values.value is None:
        P, Q = compute_probs_and_costs(Constants)
        values.set_value((P, Q))
        return Q
    else:
        Q = values.value[1]
        values.value = None
        return Q