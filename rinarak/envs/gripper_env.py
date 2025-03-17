import pybullet as p
import pybullet_data
import numpy as np
import time
from datetime import datetime
from .base_env import BaseEnv

class GripperSimulator(BaseEnv):
    def __init__(self, gui=True, robot_position = [0.0, 0.0, 0.0]):
        super().__init__()
        self.gui = gui


        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        

        # Enable real-time simulation
        p.setRealTimeSimulation(0)
        
        self.objects = []
        self.robot = None
        
        # Panda robot constants
        self.PANDA_GRIPPER_INDEX = 9
        self.PANDA_EE_INDEX = 11
        self.PANDA_NUM_JOINTS = 12
        self.MAX_FORCE = 320

        # Add capture control parameters
        self.total_steps = 0         # Track total simulation steps

        self.add_ground()
        self.add_robot_arm(position = robot_position)


    def step(self, steps=1):
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/2560.0)
            
            self.total_steps += 1

    def close(self):p.disconnect()

    def add_ground(self):
        ground_id = p.loadURDF("plane.urdf")
        self.objects.append(ground_id)
        return ground_id

    def add_box(self, position, size=[0.02, 0.02, 0.06], rgba_color = None):
        rgba_color = [0, np.random.random(), np.random.random(), 1] if rgba_color is None else rgba_color
        col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=rgba_color)
        box_id = p.createMultiBody(0.1, col_box_id, vis_box_id, position)
        self.objects.append(box_id)
        return box_id

    def add_robot_arm(self, position=[0, 0, 0]):
        self.robot = p.loadURDF("franka_panda/panda.urdf", position, useFixedBase=True)
        
        # Set up joint damping
        for i in range(self.PANDA_NUM_JOINTS):
            p.changeDynamics(self.robot, i, linearDamping=0.1, angularDamping=0.1)
        
        # Reset all joints to a good starting position
        self.reset_arm()
        return self.robot
    
    def reset_arm(self):
        """Reset the robot to a default position"""
        default_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04, 0, 0, 0]
        for i in range(self.PANDA_NUM_JOINTS):
            p.resetJointState(self.robot, i, default_positions[i])
        self.step(100)  # Let the arm settle
    
    def control_gripper(self, open=True):
        target_pos = 0.04 if open else 0.01
        # Control both gripper fingers
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, target_pos, force=10)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, target_pos, force=10)
        self.step(200)
    
    def move_arm(self, target_position, target_orientation):
        """Move the arm using inverse kinematics with better control"""
        # Calculate inverse kinematics
        joint_positions = p.calculateInverseKinematics(
            self.robot,
            self.PANDA_EE_INDEX,
            target_position,
            target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Apply position control to the arm joints
        for i in range(7):  # Only control the arm joints (not gripper)
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=self.MAX_FORCE,
                maxVelocity=1.0
            )
        
        # Step simulation and wait for arm to reach target
        steps = 0
        max_steps = 2000
        threshold = 0.01
        
        while steps < max_steps:
            self.step(1)
            current_pos = p.getLinkState(self.robot, self.PANDA_EE_INDEX)[0]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_position))
            if distance < threshold:
                break
            steps += 1
    
    def pick_object(self, object_id):
        #if self.should_capture:self.start_capture()
        if self.robot is None:
            raise ValueError("No robot arm has been loaded.")

        # Get object position and calculate approach
        object_position, object_orientation = p.getBasePositionAndOrientation(object_id)
        gripper_orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Gripper facing down

        # Approach positions
        pre_pick_position = [object_position[0], object_position[1], object_position[2] + 0.2]
        pick_position = [object_position[0], object_position[1], object_position[2] - 0.0]
        
        # Execute pick sequence
        # 1. Move to position above object first
        self.move_arm(pre_pick_position, gripper_orientation)
    
        # 2. Open gripper while still above
        self.step(150)

        self.control_gripper(open=True)
        # 3. Move down to object
        self.move_arm(pick_position, gripper_orientation)
        

        # 4. Close gripper to grasp object
        self.control_gripper(open=False)
        self.step(50)  # Give time for grasp to settle
        
        # Lift the object
        lift_position = [pick_position[0], pick_position[1], pick_position[2] + 0.2]
        self.move_arm(lift_position, gripper_orientation)
        


if __name__ == "__main__":

    sim = GripperSimulator(gui=True)
    
    # Initialize simulation
    sim.add_box([0.5, 0.5, 0.2])

    gripper_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        
    # Let simulation settle
    sim.control_gripper(True)
    sim.move_arm([0.5, 0.5, 0.10], gripper_orientation)
    for _ in range(100):sim.step_simulation()

    sim.control_gripper(False)
    for _ in range(100):sim.step_simulation()
    
    sim.move_arm([0.5, 0.5, 0.5], gripper_orientation)
    for _ in range(100):sim.step_simulation()
    sim.control_gripper(True)

    sim.close()
