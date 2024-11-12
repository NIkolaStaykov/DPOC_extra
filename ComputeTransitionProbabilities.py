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
    P = np.zeros((Constants.K, Constants.K, Constants.L))
    
    print_constants(Constants)
    
    # compute the indices of the respawn states
    M, N = Constants.M, Constants.N
    start_pos = Constants.START_POS
    respawn_idxs = np.array(
        [state2idx(np.array([start_pos[0], start_pos[1], k, l])) 
         for k in range(M) for l in range(N) 
         if (k, l) != (start_pos[0], start_pos[1])]
    )
    
    # loop through all state indexes
    for state_idx in range(Constants.K):

        # loop through all admissble actions
        for action_idx in range(Constants.L):
            
            # compute the list of probabilities for all possible next states
            next_state_probs = get_transition_probs(state_idx, action_idx, respawn_idxs)
            
            # write the probabilities to the transition matrix
            P[state_idx, :, action_idx] = next_state_probs
    
    return P





####################################################################################
def get_transition_probs(state_idx: int, action_idx: int, respawn_idxs: np.ndarray) -> np.ndarray:
    """
    Computes the transition probabilities for all possible next states.

    Args:
        state (int): The current state index
        action (int): The action to be executed
    Returns:
        np.ndarray: The next states (idx) and their transition probabilities.
    """
    
    # get start and goal coordinates (2x1)
    goal_coords = Constants.GOAL_POS
    
    # get static drone (obs) coordinates (numpy array of shape (n_obs, 2))
    obs_coords = Constants.DRONE_POS
    
    # get input space of drone
    input_space = Constants.INPUT_SPACE
    
    # get current state coordinates for drone and swan
    state_coords = idx2state(state_idx)
    drone_coords = state_coords[:2].astype(int)
    swan_coords = state_coords[2:].astype(int)
    
    # initialize the transition probabilities for all possible next states
    probs = np.zeros(Constants.K)
    
    # undefined states: same cell as swan, as static drone
    if np.all(drone_coords == swan_coords) or np.all(drone_coords == obs_coords, axis=1).any():
        # probs[respawn_idxs] = 1.0 / len(respawn_idxs)
        return probs
    
    # if in goal (and swan not in goal), stay in goal (keep current state) with probability 1
    if np.all(drone_coords == goal_coords):
        # probs[state_idx] = 0
        return probs

    # 4 possibilities for disturbances: {}, {drone}, {swan}, {drone, swan}
    # for each, compute the next states, and check if new drone is needed
    # then, add onto the (running) correponsing transition probabilities
    
    # extract disturbance probabilities
    current_prob = Constants.CURRENT_PROB[drone_coords[0], drone_coords[1]]
    swan_prob = Constants.SWAN_PROB
    
    # compute drone and swan disturbances
    drone_dist = Constants.FLOW_FIELD[drone_coords[0], drone_coords[1]]
    swan_dist = compute_swan_movement(drone_coords, swan_coords)
    
    # case 1: no disturbances
    case1_prob = (1 - current_prob) * (1 - swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, np.array([0, 0]), np.array([0, 0]), case1_prob)            
    
    # case 2: drone disturbance, no swan movement
    case2_prob = (current_prob) * (1 - swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, drone_dist, np.array([0, 0]), 
                              case2_prob)
    
    # case 3: no drone disturbance, swan movement
    case3_prob = (1 - current_prob) * (swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, np.array([0, 0]), swan_dist, case3_prob)
    
    # case 4: drone disturbance, swan movement
    case4_prob = (current_prob) * (swan_prob)
    probs += get_prob_by_case(drone_coords, input_space, action_idx, swan_coords, obs_coords,
                              respawn_idxs, drone_dist, swan_dist, 
                              case4_prob)
    
    
    return probs


####################################################################################
def get_prob_by_case(drone_coords: np.ndarray, input_space: np.ndarray, action_idx: int,
                     swan_coords: np.ndarray, obs_coords: np.ndarray, respawn_idxs: np.ndarray,
                     drone_dist: np.ndarray, swan_dist: np.ndarray, case_prob: float) -> np.ndarray:
    """
    Computes the transition probability for all possible next states given a disturbance case.
    
    Args:
        drone_coords (np.ndarray): The current drone coordinates.
        case (int): The disturbance case.
    Returns:
        np.ndarray: The transition probabilities vector (to be added to the running total).
    """
    # initilalize the transition probabilities for all possible next states
    case_probs = np.zeros(Constants.K)
    
    # compute next state coordinates for drone and swan
    next_drone_coords = get_drone_coords(drone_coords, input_space[action_idx], drone_dist)
    next_swan_coords = get_swan_coords(swan_coords, swan_dist)
    
    # check if respawn is needed
    if needs_respawn(drone_coords, next_drone_coords, next_swan_coords, obs_coords):
        # respawn needed, add probability to all possible respawn states
        # respawn states = (drone in start cell) and (swan not in start cell)
        case_probs[respawn_idxs] += case_prob / len(respawn_idxs)
    else:
        # respawn not needed, compute next state index
        next_state_idx = state2idx(np.concatenate((next_drone_coords, next_swan_coords)))
        # add onto transition probability
        case_probs[next_state_idx] += case_prob
        
    return case_probs


####################################################################################
def get_drone_coords(curr_drone_coords: np.ndarray, input: np.ndarray, 
                     disturbance: np.ndarray) -> np.ndarray:
    """
    Computes the next drone coordinates based on the current drone coordinates, the input and the disturbance.

    Args:
        curr_drone_coords (np.ndarray): The current drone coordinates.
        input (np.ndarray): The input to be executed
        disturbance (np.ndarray): The disturbance to be applied
    Returns:
        np.ndarray: The next drone coordinates.
    """
    # get current drone coordinates
    x, y = curr_drone_coords
    
    # get input
    ux, uy = input
    
    # get disturbance
    wx, wy = disturbance
    
    # compute next drone coordinates
    next_drone_coords = np.array([x + ux + wx, y + uy + wy])
    
    return next_drone_coords
    
    
####################################################################################
def get_swan_coords(curr_swan_coords: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
    """
    Computes the next swan coordinates based on the current swan coordinates and the disturbance.

    Args:
        curr_swan_coords (np.ndarray): The current swan coordinates.
        disturbance (np.ndarray): The disturbance to be applied
    Returns:
        np.ndarray: The next swan coordinates.
    """
    # get current swan coordinates
    x, y = curr_swan_coords
    
    # get disturbance
    wx, wy = disturbance
    
    # compute next swan coordinates
    next_swan_coords = np.array([x + wx, y + wy])
    
    return next_swan_coords


####################################################################################
def needs_respawn(curr_drone_coords: np.ndarray, next_drone_coords: np.ndarray, 
                  next_swan_coords: np.ndarray, obs_coords: np.ndarray) -> bool:
    """
    Checks if a new drone is needed for the given transition.

    Args:
        curr_drone_coords (np.ndarray): The current drone coordinates.
        next_drone_coords (np.ndarray): The next drone coordinates.
        next_swan_coords (np.ndarray): The next swan coordinates.
        obs_coords (np.ndarray): The coordinates of the static drones.
    Returns:
        bool: True if a new drone is needed, False otherwise.
    """
    
    # case 1: if drone collides with swan at next state, return True
    if np.all(next_drone_coords == next_swan_coords):
        return True
    
    # case 2: if drone goes out of bounds at next state, return True
    if next_drone_coords[0] < 0 or next_drone_coords[0] >= Constants.M \
        or next_drone_coords[1] < 0 or next_drone_coords[1] >= Constants.N:
        return True
    
    # case 3: if drone collides with static drone along path, return True
    drone_path_coords = bresenham(curr_drone_coords, next_drone_coords)   # list of tuples
    for coord in obs_coords:
        if (coord[0], coord[1]) in drone_path_coords:
            return True
    
    return False


####################################################################################
def compute_swan_movement(drone_coords: np.ndarray, swan_coords: np.ndarray) -> np.ndarray:
    """
    Computes the swan movement based on the drone and swan coordinates.

    Args:
        drone_coords (np.ndarray): The drone coordinates.
        swan_coords (np.ndarray): The swan coordinates.
    Returns:
        np.ndarray: The swan movement.
    """
    # get drone and swan coordinates
    dx, dy = drone_coords
    sx, sy = swan_coords
    
    # compute swan movement, 1 step in the direction of the drone
    heading_rad = np.arctan2(dy - sy, dx - sx)
    
    # find the corresponding bin of the heading and return the corresponding movement
    if -np.pi/8 <= heading_rad < np.pi/8:
        return np.array([1, 0])  # East (E)
    elif np.pi/8 <= heading_rad < 3*np.pi/8:
        return np.array([1, 1])  # North-East (NE)
    elif 3*np.pi/8 <= heading_rad < 5*np.pi/8:
        return np.array([0, 1])  # North (N)
    elif 5*np.pi/8 <= heading_rad < 7*np.pi/8:
        return np.array([-1, 1])  # North-West (NW)
    elif heading_rad >= 7*np.pi/8 or heading_rad < -7*np.pi/8:
        return np.array([-1, 0])  # West (W)
    elif -7*np.pi/8 <= heading_rad < -5*np.pi/8:
        return np.array([-1, -1])  # South-West (SW)
    elif -5*np.pi/8 <= heading_rad < -3*np.pi/8:
        return np.array([0, -1])  # South (S)
    elif -3*np.pi/8 <= heading_rad < -np.pi/8:
        return np.array([1, -1])  # South-East (SE)


def print_constants(c: Constants):
    print("="*100)
    print("M:", c.M)
    print("N:", c.N)
    print("N_DRONES:", c.N_DRONES)
    print("START_POS:", c.START_POS)
    print("GOAL_POS:", c.GOAL_POS)
    print("DRONE_POS:", c.DRONE_POS)
    print("STATE_SPACE:", c.STATE_SPACE)
    print("K:", c.K)
    print("INPUT_SPACE:", c.INPUT_SPACE)
    print("L:", c.L)
    # print("THRUSTER_COST:", c.THRUSTER_COST)
    # print("TIME_COST:", c.TIME_COST)
    # print("DRONE_COST:", c.DRONE_COST)
    # print("SWAN_PROB:", c.SWAN_PROB)
    # print("CURRENT_PROB:", c.CURRENT_PROB)
    # print("FLOW_FIELD:", c.FLOW_FIELD)
    print("="*100)
    print("")