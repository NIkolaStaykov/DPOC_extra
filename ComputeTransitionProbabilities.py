"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

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
from ComputeExpectedStageCosts import compute_probs_and_costs
import time
import os

####################################################################################
def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """
    # Create a singleton class to store the instance
    values = SingletonClass()
    if values.value is None:
        P, Q = compute_probs_and_costs(Constants)
        values.set_value((P, Q))
        return P
    else:
        P = values.value[0]
        values.value = None
        return P

####################################################################################
def get_transition_probs(state_idx: int, action_idx: int, respawn_idxs: np.ndarray,
                         M: int, N: int, K: int, goal_coords: np.ndarray, obs_coords: np.ndarray,
                         input_space: np.ndarray, current_prob: np.ndarray, swan_prob: float,
                         flow_field: np.ndarray) -> np.ndarray:
    """
    Computes the transition probabilities for all possible next states.

    Args:
        state (int): The current state index
        action (int): The action to be executed
    Returns:
        np.ndarray: The next states (idx) and their transition probabilities.
    """
    
    # get current state coordinates for drone and swan
    state_coords = idx2state(state_idx)
    drone_coords = state_coords[:2].astype(int)
    swan_coords = state_coords[2:].astype(int)
    
    # initialize the transition probabilities for all possible next states
    probs = np.zeros(K)
    
    # undefined states: same cell as swan, as static drone
    if np.all(drone_coords == swan_coords) or np.all(drone_coords == obs_coords, axis=1).any():
        return probs
    
    # if in goal (and swan not in goal), stay in goal (keep current state) with probability 1
    if np.all(drone_coords == goal_coords):
        return probs

    # 4 possibilities for disturbances: {}, {drone}, {swan}, {drone, swan}
    # for each, compute the next states, and check if new drone is needed
    # then, add onto the (running) correponsing transition probabilities
    
    # extract disturbance probabilities
    current_prob = current_prob[drone_coords[0], drone_coords[1]]
    
    # compute drone and swan disturbances
    drone_dist = flow_field[drone_coords[0], drone_coords[1]]
    swan_dist = compute_swan_movement(drone_coords, swan_coords)
    
    # case 1: no disturbances
    case1_prob = (1 - current_prob) * (1 - swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, np.array([0, 0]), np.array([0, 0]), case1_prob, K, M, N)            
    
    # case 2: drone disturbance, no swan movement
    case2_prob = (current_prob) * (1 - swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, drone_dist, np.array([0, 0]), 
                              case2_prob, K, M, N)
    
    # case 3: no drone disturbance, swan movement
    case3_prob = (1 - current_prob) * (swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, np.array([0, 0]), swan_dist, case3_prob, K, M, N)
    
    # case 4: drone disturbance, swan movement
    case4_prob = (current_prob) * (swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, drone_dist, swan_dist, 
                              case4_prob, K, M, N)
    
    
    return probs


####################################################################################
def get_prob_by_case(drone_coords: np.ndarray, input_space: np.ndarray, action_idx: int,
                     swan_coords: np.ndarray, obs_coords: np.ndarray, respawn_idxs: np.ndarray,
                     drone_dist: np.ndarray, swan_dist: np.ndarray, case_prob: float, 
                     K: int, M: int, N: int) -> np.ndarray:
    """
    Computes the transition probability for all possible next states given a disturbance case.
    
    Args:
        drone_coords (np.ndarray): The current drone coordinates.
        case (int): The disturbance case.
    Returns:
        np.ndarray: The transition probabilities vector (to be added to the running total).
    """
    # initilalize the transition probabilities for all possible next states
    case_probs = np.zeros(K)
    
    # compute next state coordinates for drone and swan
    next_drone_coords = get_drone_coords(drone_coords, input_space[action_idx], drone_dist)
    next_swan_coords = get_swan_coords(swan_coords, swan_dist)
    
    # check if respawn is needed
    if needs_respawn(drone_coords, next_drone_coords, next_swan_coords, obs_coords, M, N):
        # respawn needed, add probability to all possible respawn states
        # respawn states = (drone in start cell) and (swan not in start cell)
        case_probs[respawn_idxs] += case_prob / len(respawn_idxs)
    else:
        # respawn not needed, compute next state index
        next_state_idx = state2idx(np.concatenate((next_drone_coords, next_swan_coords)))
        # add onto transition probability
        case_probs[next_state_idx] += case_prob
        
    return case_probs
