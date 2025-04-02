
import random
from rinarak.envs.gripper_env import GripperSimulator
from rinarak.envs.recorder import SceneRecorder
from env.random_env import RandomizedEnv
import sys
import pybullet as p

class PickBlockEnv(GripperSimulator):
    def __init__(self, gui = True):
        super().__init__(gui, robot_position = [0.0, 0.0, 0.5])
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
        self.register_object("table", self.table_id)
        self.block_id = None

    def reset(self):
        pos_x = random.randint(45,45)/100.#(random.random()-0.5) * 0.8
        pos_y = random.randint(-45,45)/100.#(random.random()-0.5) * 0.8
        self.block_id = self.add_box([pos_x, pos_y, 0.7], size = [0.02,0.02,0.02])

        self.step(600) # step till the world is stablized
    
    def check_goal(self) -> bool:
        #self.build_contact_graph()
        #self.build_support_graph()
        self.build_contact_tensor()
        self.build_support_tensor()

        contact_tensor = self.contact_tensor > 0
        support_tensor = self.support_tensor > 0
        #print(self.robot)
        #print(self.table_id)
        #print(self.block_id)
        #print(contact_tensor)
        #print(support_tensor)
        return contact_tensor[self.robot_id, self.block_id]