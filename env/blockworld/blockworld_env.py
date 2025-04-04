

import open3d as o3d
from helchriss.envs.gripper_env import GripperSimulator
from helchriss.envs.recorder import SceneRecorder
import sys
import pybullet as p
import random
import numpy as np

blockworld_domain_str = """
(domain Blockworld)
(:type
    state - vector[float,3] ;; encoding of position and is holding
    position - vector[float,2]
)
(:predicate
    block_position ?x-state -> position
    on ?x-state ?y-state -> boolean
    clear ?x-state -> boolean
    holding ?x-state -> boolean
    hand-free -> boolean
)
(:action
    (
        name: pick
        parameters: ?o1
        precondition: (and (clear ?o1) (hand-free) )
        effect:
        (and-do
            (and-do
                (assign (holding ?o1) true)
                (assign (clear ?o1) false)
            )
            (assign (hand-free) false)
        )
    )
    (
        name: place
        parameters: ?o1 ?o2
        precondition:
            (and (holding ?o1) (clear ?o2))
        effect :
            (and-do
            (and-do
                        (assign (hand-free) true)
                (and-do
                        (assign (holding ?o1) false)
                    (and-do
                        (assign (clear ?o2) false)
                        (assign (clear ?o1) true)
                    )
                )
                
            )
                (assign (on ?x ?y) true)
            )
    )
)

"""

class Blockworld(GripperSimulator):
    def __init__(self, gui = True, robot_position = [0.0, 0.0, 0.6]):
        super().__init__(gui, robot_position = robot_position)
        table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
    
    def generate_blocks(self, blocks_list):
        """ it should be a list of dict representating blocks config
        a block have the config of 
            position : [0.1, 0.2, 0.0]
            size : [0.2, 0.2, 0.2]
            color : rgba or None
        """
        for block in blocks_list:
            self.add_box(block["position"], block["size"], block["color"])

def sample_points_in_rectangle_with_exclusion(outer_rect, inner_rect, k):
    """
    Uniformly sample k points in a large rectangle while avoiding a smaller inner rectangle.
    
    Parameters:
    -----------
    outer_rect : tuple
        (x_min, y_min, x_max, y_max) defining the outer rectangle
    inner_rect : tuple
        (x_min, y_min, x_max, y_max) defining the inner rectangle to exclude
    k : int
        Number of points to sample
    
    Returns:
    --------
    points : numpy.ndarray
        Array of shape (k, 2) containing the sampled points as (x, y) coordinates
    """
    # Extract rectangle coordinates
    outer_x_min, outer_y_min, outer_x_max, outer_y_max = outer_rect
    inner_x_min, inner_y_min, inner_x_max, inner_y_max = inner_rect
    
    # Calculate areas
    outer_area = (outer_x_max - outer_x_min) * (outer_y_max - outer_y_min)
    inner_area = (inner_x_max - inner_x_min) * (inner_y_max - inner_y_min)
    valid_area = outer_area - inner_area
    
    # Calculate how many points we need to sample to get k valid points
    # We'll oversample to ensure we have enough valid points
    oversample_factor = outer_area / valid_area
    n_samples = int(k * oversample_factor * 1.2)  # 20% extra to be safe
    
    points = []
    while len(points) < k:
        # Generate random points within the outer rectangle
        x = np.random.uniform(outer_x_min, outer_x_max, n_samples)
        y = np.random.uniform(outer_y_min, outer_y_max, n_samples)
        
        # Filter out points that fall within the inner rectangle
        valid_points = []
        for i in range(n_samples):
            if not (inner_x_min <= x[i] <= inner_x_max and 
                    inner_y_min <= y[i] <= inner_y_max):
                valid_points.append((x[i], y[i]))
                if len(points) + len(valid_points) >= k:
                    break
        
        # Add the valid points to our result list
        points.extend(valid_points[:k - len(points)])
    
    return np.array(points[:k])

def generate_block_config(num_blocks, constraints = None):
    outer_rectangle = (-0.45, -0.45, 0.45, 0.45)
    inner_rectangle = (-0.2, -0.2, 0.2, 0.2)

    constraints = [] if constraints is None else constraints
    block_configs = []
    for i in range(num_blocks):
        point = sample_points_in_rectangle_with_exclusion(outer_rectangle, inner_rectangle, 1)[0]
        block_configs.append(
            {"position" : [point[0], point[1], 0.65],
             "size" : [0.03, 0.03, 0.03],
             "color" : [random.random(), random.random(), random.random(), 1.0]
             }
        )
    return block_configs


if __name__ == "__main__":
    env = Blockworld(gui = True)
    for scene_id in range(2):
        block_config = generate_block_config(3)
        env.generate_blocks(block_config)
