import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import open3d as o3d
from rinarak.envs.gripper_env import GripperSimulator

class CabinetEnv(GripperSimulator):
    def __init__(self, gui = True, robot_position = [0., 0., 0.]):
        super().__init__(gui = gui, robot_position=robot_position)

        cabinet_path = "assets/single_door_cabinet.urdf"
        initial_position = [0, -1.0, 0]
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cabinet_id = p.loadURDF(cabinet_path, initial_position, initial_orientation, useFixedBase=1)
        self.cabinet_id = cabinet_id
        num_joints = p.getNumJoints(cabinet_id)

        self.door_joint_index = -1
        for i in range(num_joints):
            joint_info = p.getJointInfo(cabinet_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name == "door_joint":
                self.door_joint_index = i
                break

        door_slider = p.addUserDebugParameter("Door Angle", 0, 2.0, 0)
        small_cube = p.loadURDF("cube.urdf", [0, -1.0, 0.3], globalScaling=0.1)


    def pull_handle(self, target_angle, duration=1.0, steps=100):
        current_angle = p.getJointState(self.cabinet_id, self.door_joint_index)[0]
        angle_increment = (target_angle - current_angle) / steps
    
        for i in range(steps):
            new_angle = current_angle + angle_increment * (i + 1)
            p.setJointMotorControl2(self.cabinet_id, self.door_joint_index, 
                               p.POSITION_CONTROL, 
                               targetPosition=new_angle, 
                               force=5)
            p.stepSimulation()
            time.sleep(duration / steps)

if __name__ == "__main__":
    cabinet_env = CabinetEnv(1)

    cabinet_env.pull_handle(3.14, 3)