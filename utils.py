"""
 utils.py

 Helper functions that are used in multiple files. Feel free to add more functions.

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
from Constants import Constants

from scipy.optimize import linprog

def bresenham(start, end):
    """
    Generates the coordinates of a line between two points using Bresenham's algorithm.

    Parameters:
        start (tuple or list): The starting point (x0, y0).
        end (tuple or list): The ending point (x1, y1).

    Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates.

    Example:
        >>> bresenham((2, 3), (10, 8))
        [(2, 3), (3, 4), (4, 4), (5, 5), (6, 6), (7, 6), (8, 7), (9, 7), (10, 8)]
    """
    x0, y0 = start
    x1, y1 = end

    points = []

    dx = x1 - x0
    dy = y1 - y0

    x_sign = 1 if dx > 0 else -1 if dx < 0 else 0
    y_sign = 1 if dy > 0 else -1 if dy < 0 else 0

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = x_sign, 0, 0, y_sign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, y_sign, x_sign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        px = x0 + x * xx + y * yx
        py = y0 + x * xy + y * yy
        points.append((px, py))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy

    return points

def idx2state(idx):
    """Converts a given index into the corresponding state.

    Args:
        idx (int): index of the entry whose state is required

    Returns:
        np.array: (x,y,x,y) state corresponding to the given index
    """
    state = np.empty(4)

    for i, j in enumerate(
        [
            Constants.M,
            Constants.N,
            Constants.M,
            Constants.N,
        ]
    ):
        state[i] = idx % j
        idx = idx // j
    return state


def state2idx(state):
    """Converts a given state into the corresponding index.

    Args:
        state (np.array): (x,y,x,y) entry in the state space

    Returns:
        int: index corresponding to the given state
    """
    idx = 0

    factor = 1
    for i, j in enumerate([Constants.M, Constants.N, Constants.M, Constants.N]):
        idx += state[i] * factor
        factor *= j

    return idx



##################################################################
##################################################################
##################################################################


####################################################################################
def linear_program(P: np.ndarray, Q: np.ndarray, const: Constants, alpha: float) -> np.ndarray:
    """
    Solves the linear program to find the optimal cost-to-go function for the stochastic shortest path problem.
    
    Args:
        P: The transition probabilities matrix.
        Q: The expected stage costs matrix.
        const: The constants describing the problem instance.
        alpha: The discount factor.
    
    Returns:
        np.ndarray: The optimal cost-to-go function.
    """
    
    # re-order the dimensions of the transition probabiltiies matrix
    Pt = np.transpose(P, (2, 0, 1))
    
    # unfold P along its third dimension
    A = np.tile(np.eye(const.K), (const.L, 1)) - alpha * Pt.reshape((const.K * const.L, const.K))
    
    # flatten the expected stage costs matrix
    b = Q.flatten(order="F")
    
    # define the cost vector to optimize
    c = np.ones(const.K) * -1
    
    # solve the linear program
    res = linprog(c, A_ub=A, b_ub=b, method="highs-ipm")
    
    return res.x


####################################################################################
def compute_probs_and_costs(const: Constants) -> [np.ndarray, np.ndarray]:
    """
    Computes the transition probabilities and expected stage costs matrices for all states and actions.

    Args:
        const (Constants): The constants of the problem.
    Returns:
        np.ndarray: The transition probabilities matrix.
        np.ndarray: The expected stage costs matrix.
    """
    
    P = np.zeros((const.K, const.K, const.L))
    # Q = np.ones((const.K, const.L)) * np.inf
    Q = np.zeros((const.K, const.L))
    
    start_pos = const.START_POS

    # Precompute the indices of the respawn states
    swan_spawn_pos = np.ones((const.M, const.N))
    swan_spawn_pos[start_pos[0], start_pos[1]] = 0

    start_idx_grid = start_pos[0] + start_pos[1] * const.M
    respawn_idxs = np.concatenate((np.arange(0, start_idx_grid), np.arange(start_idx_grid + 1, const.M * const.N))) * const.M * const.N + start_idx_grid

    # Loop through all state indexes
    for state_idx in range(const.K):
        # Loop through all admissible actions
        for action_idx in range(const.L):
            
            # get action
            action = const.INPUT_SPACE[action_idx]
            
            # Compute the expected stage cost for the current state and action
            expected_stage_cost, next_state_probs = get_expected_stage_cost_and_transition_probs(
                const, state_idx, action, respawn_idxs
            )

            # Write the expected stage cost to the matrix
            Q[state_idx, action_idx] = expected_stage_cost
            P[state_idx, :, action_idx] = next_state_probs

    return P, Q


####################################################################################
def get_expected_stage_cost_and_transition_probs(const: Constants, state_idx: int, action: np.ndarray,
                                                 respawn_idxs: np.ndarray) -> [float, np.ndarray]:
    """
    Computes the expected stage cost for a given state and action, across all possible next states.

    Args:
        const (Constants): The constants of the problem.
        state_idx (int): The current state index.
        action (np.ndarray): The action input of the drone.
        respawn_idxs (np.ndarray): The indices of the respawn states.
    Returns:
        float: The expected stage cost.
        np.ndarray: The transition probabilities vector to all possible next states.
    """
    
    # Get current state coordinates for drone and swan
    state_coords = idx2state(state_idx)
    drone_coords = state_coords[:2].astype(int)
    swan_coords = state_coords[2:].astype(int)
    
    # Initialize the expected stage cost to zero
    e_cost = 0.0
    transition_probs = np.zeros(const.K)
    
    # Undefined states: same cell as swan, as static drone
    if np.all(drone_coords == swan_coords) or np.all(drone_coords == const.DRONE_POS, axis=1).any():
        return e_cost, transition_probs
    
    # If in goal (and swan not in goal), stay in goal (keep current state) with probability 1
    if np.all(drone_coords == const.GOAL_POS):
        return e_cost, transition_probs

    # 4 possibilities for disturbances: {}, {drone}, {swan}, {drone, swan}
    # For each, compute the next states, and check if new drone is needed
    # Then, add onto the (running) corresponding transition probabilities
    
    # Extract disturbance probabilities
    current_prob = const.CURRENT_PROB[drone_coords[0], drone_coords[1]]
    
    # Compute drone and swan disturbances
    drone_dist = const.FLOW_FIELD[drone_coords[0], drone_coords[1]]
    swan_dist = compute_swan_movement(drone_coords, swan_coords)
    
    # Case 1: no disturbances
    case1_prob = (1 - current_prob) * (1 - const.SWAN_PROB)
    cost, probs = get_e_cost_and_probs_by_case(const, drone_coords, action, swan_coords,
                              respawn_idxs, np.array([0, 0]), np.array([0, 0]), case1_prob)            
    e_cost += cost
    transition_probs += probs

    # Case 2: drone disturbance, no swan movement
    case2_prob = (current_prob) * (1 - const.SWAN_PROB)
    cost, probs = get_e_cost_and_probs_by_case(const, drone_coords, action, swan_coords,
                              respawn_idxs, drone_dist, np.array([0, 0]), case2_prob)
    e_cost += cost
    transition_probs += probs
    
    # Case 3: no drone disturbance, swan movement
    case3_prob = (1 - current_prob) * (const.SWAN_PROB)
    cost, probs = get_e_cost_and_probs_by_case(const, drone_coords, action, swan_coords,
                              respawn_idxs, np.array([0, 0]), swan_dist, case3_prob)
    e_cost += cost
    transition_probs += probs
    
    # Case 4: drone disturbance, swan movement
    case4_prob = (current_prob) * (const.SWAN_PROB)
    cost, probs = get_e_cost_and_probs_by_case(const, drone_coords, action, swan_coords,
                              respawn_idxs, drone_dist, swan_dist, case4_prob)
    e_cost += cost
    transition_probs += probs
    
    return e_cost, transition_probs


####################################################################################
def get_e_cost_and_probs_by_case(const: Constants, drone_coords: np.ndarray, action: np.ndarray,
                     swan_coords: np.ndarray, respawn_idxs: np.ndarray, drone_dist: np.ndarray, 
                     swan_dist: np.ndarray, case_prob: float) -> [float, np.ndarray]:
    """
    Computes the expected stage cost over all possible next states given a disturbance case.
    
    Args:
        const (Constants): The constants of the problem.
        drone_coords (np.ndarray): The current drone coordinates.
        action (np.ndarray): The action input of the drone.
        swan_coords (np.ndarray): The current swan coordinates.
        respawn_idxs (np.ndarray): The indices of the respawn states.
        drone_dist (np.ndarray): The drone disturbance.
        swan_dist (np.ndarray): The swan disturbance.
        case_prob (float): The disturbance case probability.
    Returns:
        float: The expected stage cost.
        np.ndarray: The transition probabilities vector (to be added to the running total).
    """
    # Initialize the transition probabilities and stage costs for all possible next states
    case_probs = np.zeros(const.K)
    case_costs = np.zeros(const.K)
    
    # Compute next state coordinates for drone and swan
    next_drone_coords = np.array([drone_coords[0] + action[0] + drone_dist[0], 
                                  drone_coords[1] + action[1] + drone_dist[1]])
    next_swan_coords = np.array([swan_coords[0] + swan_dist[0], 
                                 swan_coords[1] + swan_dist[1]])
    
    # Check if respawn is needed
    if needs_respawn(const, drone_coords, next_drone_coords, next_swan_coords):
        # Respawn needed, add probability to all possible respawn states
        case_probs[respawn_idxs] = case_prob / len(respawn_idxs)
        case_costs[respawn_idxs] = const.TIME_COST + const.THRUSTER_COST * np.abs(action).sum() + const.DRONE_COST
    else:
        # Respawn not needed, compute next state index
        next_state_idx = state2idx(np.concatenate((next_drone_coords, next_swan_coords)))
        # Add onto transition probability
        case_probs[next_state_idx] += case_prob
        case_costs[next_state_idx] = const.TIME_COST + const.THRUSTER_COST * np.abs(action).sum()
    
    # Compute expected stage cost
    case_e_cost = np.dot(case_probs, case_costs)
    
    return case_e_cost, case_probs


####################################################################################
def needs_respawn(const: Constants, curr_drone_coords: np.ndarray, next_drone_coords: np.ndarray, 
                  next_swan_coords: np.ndarray) -> bool:
    """
    Checks if a new drone is needed for the given transition.

    Args:
        const (Constants): The constants of the problem.
        curr_drone_coords (np.ndarray): The current drone coordinates.
        next_drone_coords (np.ndarray): The next drone coordinates.
        next_swan_coords (np.ndarray): The next swan coordinates.
    Returns:
        bool: True if a new drone is needed, False otherwise.
    """
    
    # case 1: if drone collides with swan at next state, return True
    if np.all(next_drone_coords == next_swan_coords):
        return True
    
    # case 2: if drone goes out of bounds at next state, return True
    if not (0 <= next_drone_coords[0] < const.M and 0 <= next_drone_coords[1] < const.N):
        return True
    
    # case 3: if drone collides with static drone along path, return True        
    drone_path_coords = bresenham(curr_drone_coords, next_drone_coords)   # list of tuples
    drone_path_set = set(drone_path_coords)
    obs_coords_set = set(map(tuple, const.DRONE_POS))
    return len(drone_path_set.intersection(obs_coords_set)) > 0


####################################################################################
def compute_swan_movement(drone_coords: np.ndarray, swan_coords: np.ndarray) -> np.ndarray:
    """
    Computes the swan movement based on the drone and swan coordinates.

    Args:
        drone_coords (np.ndarray): The drone coordinates.
        swan_coords (np.ndarray): The swan coordinates.
    Returns:
        np.ndarray: The resulting new swan coordinates.
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


####################################################################################
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
    print("THRUSTER_COST:", c.THRUSTER_COST)
    print("TIME_COST:", c.TIME_COST)
    print("DRONE_COST:", c.DRONE_COST)
    print("SWAN_PROB:", c.SWAN_PROB)
    print("CURRENT_PROB:", c.CURRENT_PROB)
    print("FLOW_FIELD:", c.FLOW_FIELD)
    print("="*100)
    print("")


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class SingletonClass(metaclass=SingletonMeta):
    def __init__(self):
        self.value = None

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value
