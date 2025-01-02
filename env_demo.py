import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Tuple, List

class RobotPickPlaceEnv:
    def __init__(self, render: bool = True):
        # Initialize PyBullet
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load UR5 robot
        self.robot_id = p.loadURDF("ur5/ur5.urdf", [0, 0, 0.02], useFixedBase=True)
        
        # Add gripper
        self.gripper_id = p.loadURDF("gripper/wsg50_one_motor_gripper.urdf", 
                                    [0.5, 0, 0.1], 
                                    useFixedBase=True)
        
        # Create table
        self.table_id = p.loadURDF("table/table.urdf", 
                                  [1.0, 0.0, 0.0],
                                  useFixedBase=True)
        
        # Load objects for manipulation
        self.object_ids = self._spawn_objects()
        
        # Initialize robot configuration
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = range(self.num_joints)
        
        # Set initial joint positions
        self.home_positions = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        self.reset_robot()
        
    def reset_robot(self):
        """Reset robot to home position"""
        for i, pos in enumerate(self.home_positions):
            p.resetJointState(self.robot_id, i, pos)
    
    def _spawn_objects(self) -> List[int]:
        """Spawn objects in the environment"""
        object_ids = []
        
        # Spawn a few cubes
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  # RGB colors
        positions = [[0.7, -0.2, 0.65], [0.7, 0, 0.65], [0.7, 0.2, 0.65]]
        
        for pos, color in zip(positions, colors):
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.02, 0.02, 0.02],
                rgbaColor=color
            )
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.02, 0.02, 0.02]
            )
            object_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pos
            )
            object_ids.append(object_id)
            
        return object_ids
    
    def get_object_position(self, object_id: int) -> Tuple[List[float], List[float]]:
        """Get position and orientation of an object"""
        pos, orn = p.getBasePositionAndOrientation(object_id)
        return pos, orn
    
    def move_arm_to_target(self, target_pos: List[float], target_orn: List[float]):
        """Move robot arm to target position and orientation"""
        # Calculate inverse kinematics
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.num_joints - 1,  # End effector link index
            target_pos,
            target_orn
        )
        
        # Set joint positions
        for i, pose in enumerate(joint_poses):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pose,
                force=500
            )
    
    def control_gripper(self, open_gripper: bool):
        """Control gripper state"""
        if open_gripper:
            target_pos = 0.05  # Open position
        else:
            target_pos = 0.0   # Closed position
            
        p.setJointMotorControl2(
            bodyIndex=self.gripper_id,
            jointIndex=0,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=50
        )
    
    def pick_and_place(self, object_id: int, target_pos: List[float]):
        """Execute pick and place operation"""
        # Get object position
        obj_pos, obj_orn = self.get_object_position(object_id)
        
        # Move above object
        above_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.2]
        self.move_arm_to_target(above_pos, [0, np.pi, 0])
        time.sleep(1)
        
        # Move to object
        self.move_arm_to_target(obj_pos, [0, np.pi, 0])
        time.sleep(1)
        
        # Close gripper
        self.control_gripper(False)
        time.sleep(1)
        
        # Lift object
        self.move_arm_to_target(above_pos, [0, np.pi, 0])
        time.sleep(1)
        
        # Move to target position
        above_target = [target_pos[0], target_pos[1], target_pos[2] + 0.2]
        self.move_arm_to_target(above_target, [0, np.pi, 0])
        time.sleep(1)
        
        # Lower object
        self.move_arm_to_target(target_pos, [0, np.pi, 0])
        time.sleep(1)
        
        # Open gripper
        self.control_gripper(True)
        time.sleep(1)
        
        # Move up
        self.move_arm_to_target(above_target, [0, np.pi, 0])
        time.sleep(1)
        
        # Return to home position
        self.reset_robot()
    
    def step_simulation(self):
        """Step the simulation"""
        p.stepSimulation()
    
    def close(self):
        """Close the PyBullet connection"""
        p.disconnect()

def main():
    # Create environment
    env = RobotPickPlaceEnv(render=True)
    
    # Let simulation stabilize
    for _ in range(100):
        env.step_simulation()
        time.sleep(1./240.)
    
    # Pick and place each object
    target_positions = [[0.7, -0.2, 0.75], [0.7, 0, 0.75], [0.7, 0.2, 0.75]]
    
    for obj_id, target_pos in zip(env.object_ids, target_positions):
        # Execute pick and place
        env.pick_and_place(obj_id, target_pos)
        
        # Step simulation for a while
        for _ in range(100):
            env.step_simulation()
            time.sleep(1./240.)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()