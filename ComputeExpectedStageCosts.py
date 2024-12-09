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

import numpy as np
from utils import *
import os
import time

####################################################################################
def compute_probs_and_costs(Constants):
    P = np.zeros((Constants.K, Constants.K, Constants.L))
    Q = np.ones((Constants.K, Constants.L)) * np.inf
    M, N, K, L = Constants.M, Constants.N, Constants.K, Constants.L
    start_pos, goal_pos = Constants.START_POS, Constants.GOAL_POS
    static_drone_pos, input_space = Constants.DRONE_POS, Constants.INPUT_SPACE
    current_prob, swan_prob = Constants.CURRENT_PROB, Constants.SWAN_PROB
    flow_field = Constants.FLOW_FIELD
    time_cost, thruster_cost, drone_cost = Constants.TIME_COST, Constants.THRUSTER_COST, Constants.DRONE_COST
    

    # compute the indices of the respawn states
    swan_spawn_pos = np.ones((M, N))
    swan_spawn_pos[start_pos[0], start_pos[1]] = 0

    start_idx_grid = start_pos[0] + start_pos[1] * M
    respawn_idxs = np.arange(0, start_idx_grid)
    respawn_idxs = np.append(respawn_idxs, np.arange(start_idx_grid + 1, M * N))
    respawn_idxs = respawn_idxs*M*N + start_idx_grid
    
    # loop through all state indexes
    for state_idx in range(Constants.K):

        # loop through all admissble actions
        for action_idx in range(Constants.L):
            
            # compute the expected stage cost for the current state and action
            expected_stage_cost, next_state_probs = get_expected_stage_cost_and_transition_probs(state_idx, action_idx, respawn_idxs,
                                                    M, N, K, goal_pos, static_drone_pos, input_space,
                                                    current_prob, swan_prob, flow_field,
                                                    time_cost, thruster_cost, drone_cost)
            
            # write the expected stage cost to the matrix
            Q[state_idx, action_idx] = expected_stage_cost
            P[state_idx, :, action_idx] = next_state_probs

    return P, Q

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
    values = SingletonClass()
    if values.value is None:
        start = time.time()
        P, Q = compute_probs_and_costs(Constants)
        print("Time taken to compute P and Q: ", time.time() - start)
        values.set_value((P, Q))
        return P
    else:
        Q = values.value[1]
        values.value = None
        return Q
    
####################################################################################
def get_expected_stage_cost_and_transition_probs(state_idx: int, action_idx: int, respawn_idxs: np.ndarray,
                            M: int, N: int, K: int, goal_coords: np.ndarray, obs_coords: np.ndarray,
                            input_space: np.ndarray, current_prob: np.ndarray, swan_prob: float,
                            flow_field: np.ndarray, time_cost, thruster_cost, drone_cost) -> [float, np.ndarray]:
    """
    Computes the expected stage cost for a given state and action, across all possible next states.

    Args:
        state (int): The current state index
        action (int): The action (idx) to be executed
    Returns:
        float: The expected stage cost
    """
    
    # get current state coordinates for drone and swan
    state_coords = idx2state(state_idx)
    drone_coords = state_coords[:2].astype(int)
    swan_coords = state_coords[2:].astype(int)
    
    # initialize the expected stage cost to zero
    e_cost = 0.0
    transition_probs = np.zeros(K)
    
    # undefined states: same cell as swan, as static drone
    if np.all(drone_coords == swan_coords) or np.all(drone_coords == obs_coords, axis=1).any():
        return e_cost, transition_probs
    
    # if in goal (and swan not in goal), stay in goal (keep current state) with probability 1
    if np.all(drone_coords == goal_coords):
        return e_cost, transition_probs

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
    cost, probs = get_e_cost_and_probs_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, np.array([0, 0]), np.array([0, 0]), case1_prob,
                              time_cost, thruster_cost, drone_cost, K, M, N)            
    e_cost += cost
    transition_probs += probs

    # case 2: drone disturbance, no swan movement
    case2_prob = (current_prob) * (1 - swan_prob)
    cost, probs = get_e_cost_and_probs_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, drone_dist, np.array([0, 0]), 
                              case2_prob, time_cost, thruster_cost, drone_cost, K, M, N)
    e_cost += cost
    transition_probs += probs
    
    # case 3: no drone disturbance, swan movement
    case3_prob = (1 - current_prob) * (swan_prob)
    cost, probs = get_e_cost_and_probs_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, np.array([0, 0]), swan_dist, case3_prob,
                              time_cost, thruster_cost, drone_cost, K, M, N)
    e_cost += cost
    transition_probs += probs
    
    # case 4: drone disturbance, swan movement
    case4_prob = (current_prob) * (swan_prob)
    cost, probs = get_e_cost_and_probs_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, drone_dist, swan_dist, 
                              case4_prob, time_cost, thruster_cost, drone_cost, K, M, N)
    e_cost += cost
    transition_probs += probs
    
    return e_cost, transition_probs


####################################################################################
def get_e_cost_and_probs_by_case(drone_coords: np.ndarray, input_space: np.ndarray, action_idx: int,
                     swan_coords: np.ndarray, obs_coords: np.ndarray, respawn_idxs: np.ndarray,
                     drone_dist: np.ndarray, swan_dist: np.ndarray, case_prob: float,
                     time_cost, thruster_cost, drone_cost, K: int, M: int, N: int) -> [float, np.ndarray]:
    """
    Computes the expected stage cost over all possible next states given a disturbance case.
    
    Args:
        drone_coords (np.ndarray): The current drone coordinates.
        case (int): The disturbance case.
    Returns:
        float: The expected stage cost.
        np.ndarray: The transition probabilities vector (to be added to the running total).
    """
    # initilalize the transition probabilities and stage costs for all possible next states
    case_probs = np.zeros(K)
    case_costs = np.zeros(K)
    
    # compute next state coordinates for drone and swan
    next_drone_coords = get_drone_coords(drone_coords, input_space[action_idx], drone_dist)
    next_swan_coords = get_swan_coords(swan_coords, swan_dist)
    
    # check if respawn is needed
    if needs_respawn(drone_coords, next_drone_coords, next_swan_coords, obs_coords, M, N):
        # respawn needed, add probability to all possible respawn states
        case_probs[respawn_idxs] = case_prob / len(respawn_idxs)
        case_costs[respawn_idxs] = compute_stage_cost(time_cost, thruster_cost, drone_cost, action_idx, 1)
    else:
        # respawn not needed, compute next state index
        next_state_idx = state2idx(np.concatenate((next_drone_coords, next_swan_coords)))
        # add onto transition probability
        case_probs[next_state_idx] += case_prob
        case_costs[next_state_idx] = compute_stage_cost(time_cost, thruster_cost, drone_cost, action_idx, 0)
    
    # compute expected stage cost
    case_e_cost = np.dot(case_probs, case_costs)
    
    return case_e_cost, case_probs


####################################################################################
def compute_stage_cost(time_cost, thruster_cost, drone_cost, action_idx: int, new_drone: int) -> float:
    """
    Computes the stage cost for a given action and new drone.

    Args:
        time_cost (float): The cost of a time step.
        thruster_cost (float): The cost of using one thruster.
        drone_cost (float): The cost of sending a new drone.
        action_idx (int): The index of the action.
        new_drone (bool): Whether a new drone is needed.
    Returns:
        float: The stage cost.
    """
    # get action and its norm
    action = Constants.INPUT_SPACE[action_idx]
    action_norm = np.abs(action).sum()
    
    return time_cost + thruster_cost * action_norm + drone_cost * new_drone