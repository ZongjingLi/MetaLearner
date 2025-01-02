from .base_robot import BaseRobot
import pybullet as p
import pybullet_data

class UR5(BaseRobot):
    def _load_robot(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF(
            "ur5/ur5.urdf",
            self.config['base_position'],
            useFixedBase=True
        )
        
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = range(self.num_joints)
        self.end_effector_index = self.num_joints - 1
        
        # Load gripper
        self.gripper_id = p.loadURDF(
            "gripper/wsg50_one_motor_gripper.urdf",
            [0.5, 0, 0.1],
            useFixedBase=True
        )
        
    def control_gripper(self, open_gripper: bool):
        target_pos = 0.05 if open_gripper else 0.0
        p.setJointMotorControl2(
            bodyIndex=self.gripper_id,
            jointIndex=0,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=50
        )