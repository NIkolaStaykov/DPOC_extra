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
    return np.array([curr_drone_coords[0] + input[0] + disturbance[0],
                     curr_drone_coords[1] + input[1] + disturbance[1]])
    
    
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
    return np.array([curr_swan_coords[0] + disturbance[0],
                     curr_swan_coords[1] + disturbance[1]])


####################################################################################
def needs_respawn(curr_drone_coords: np.ndarray, next_drone_coords: np.ndarray, 
                  next_swan_coords: np.ndarray, obs_coords: np.ndarray, M: int, N: int) -> bool:
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
    if not (0 <= next_drone_coords[0] < M and 0 <= next_drone_coords[1] < N):
        return True
    
    # case 3: if drone collides with static drone along path, return True        
    drone_path_coords = bresenham(curr_drone_coords, next_drone_coords)   # list of tuples
    drone_path_set = set(drone_path_coords)
    obs_coords_set = set(map(tuple, obs_coords))
    return len(drone_path_set.intersection(obs_coords_set)) > 0


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
    # print("THRUSTER_COST:", c.THRUSTER_COST)
    # print("TIME_COST:", c.TIME_COST)
    # print("DRONE_COST:", c.DRONE_COST)
    # print("SWAN_PROB:", c.SWAN_PROB)
    # print("CURRENT_PROB:", c.CURRENT_PROB)
    # print("FLOW_FIELD:", c.FLOW_FIELD)
    print("="*100)
    print("")