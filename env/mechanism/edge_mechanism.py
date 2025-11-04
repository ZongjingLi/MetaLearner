

from helchriss.envs.gripper_env import GripperSimulator
from helchriss.envs.recorder import SceneRecorder
import sys
import pybullet as p

class GripEdge(GripperSimulator):
    def __init__(self, gui = True):
        super().__init__(gui, robot_position = [0.0, 0.0, 0.5])
        table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
