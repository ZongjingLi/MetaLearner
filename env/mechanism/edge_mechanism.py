

from rinarak.envs.gripper_env import GripperSimulator
from rinarak.envs.recorder import SceneRecorder
import sys
import pybullet as p

class GripEdge(GripperSimulator):
    def __init__(self, gui = True):
        super().__init__(gui, robot_position = [0.0, 0.0, 0.5])
        table_id = p.loadURDF("table/table.urdf", [0, 0, 0])

    def reset(self):
        for i in range(random.randint(1, 4)):
            pos_x = random.randint(45,45)/100.#(random.random()-0.5) * 0.8
            pos_y = random.randint(-45,45)/100.#(random.random()-0.5) * 0.8
            env.add_box([pos_x, pos_y, 0.7], size = [0.02,0.02,0.02])