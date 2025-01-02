from typing import List, Tuple
import pybullet as p
import numpy as np

class BaseRobot:
    def __init__(self, config: dict):
        self.config = config
        self.robot_id = None
        self.joint_indices = None
        self.end_effector_index = None
        self._load_robot()
        
    def _load_robot(self):
        raise NotImplementedError
        
    def reset(self):
        for joint, pos in zip(self.joint_indices, self.config['default_joint_positions']):
            p.resetJointState(self.robot_id, joint, pos)
            
    def move_to_target(self, target_pos: List[float], target_orn: List[float]):
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn
        )
        
        for i, joint_pos in enumerate(joint_poses):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=self.joint_indices[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=self.config['max_force']
            )
            
    def get_end_effector_pose(self) -> Tuple[List[float], List[float]]:
        return p.getLinkState(self.robot_id, self.end_effector_index)[:2]