import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import os
from datetime import datetime

class PyBulletSimulator:
    def __init__(self, gui=True, record_video=False, video_path='outputs/output.mp4', output_dir='outputs'):
        self.gui = gui
        self.record_video = record_video
        self.video_path = video_path
        self.output_dir = output_dir
        self.cabinet_parts = {}  # Store cabinet components
        
        # Create output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rgb_dir = os.path.join(output_dir, self.timestamp, 'rgb')
        self.depth_dir = os.path.join(output_dir, self.timestamp, 'depth')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Camera settings
        self.width = 640
        self.height = 480
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1, 1, 1],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(self.width) / self.height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Enable real-time simulation
        p.setRealTimeSimulation(0)
        
        self.objects = []
        self.robot = None
        self.frame_count = 0
        self.video_frames = []
        
        # Panda robot constants
        self.PANDA_GRIPPER_INDEX = 9
        self.PANDA_EE_INDEX = 11
        self.PANDA_NUM_JOINTS = 12
        self.MAX_FORCE = 320

        # Add capture control parameters
        self.capture_interval = 20  # Capture every N simulation steps
        self.should_capture = False  # Flag to control when to capture
        self.last_capture_step = 0   # Track last capture step
        self.total_steps = 0         # Track total simulation steps
    
    def start_capture(self):
        """Start capturing frames"""
        self.should_capture = True
        
    def stop_capture(self):
        """Stop capturing frames"""
        self.should_capture = False

    def capture_frame(self):
        """Capture RGB and depth frames"""
        # Capture RGB and depth
        images = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(images[2], dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        depth_array = np.array(images[3])
        
        # Normalize depth for visualization
        depth_normalized = ((depth_array - depth_array.min()) * 255 / 
                          (depth_array.max() - depth_array.min())).astype(np.uint8)
        
        # Save images
        rgb_path = os.path.join(self.rgb_dir, f'frame_{self.frame_count:04d}.png')
        depth_path = os.path.join(self.depth_dir, f'frame_{self.frame_count:04d}.png')
        
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
        cv2.imwrite(depth_path, depth_normalized)
        
        if self.record_video:
            self.video_frames.append(rgb_array)
        
        self.frame_count += 1

    def save_video(self):
        """Save recorded frames as video"""
        if not self.video_frames:
            return
        
        video_path = os.path.join(self.output_dir, f'{self.timestamp}/simulation.mp4')
        height, width = self.video_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in self.video_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Video saved to: {video_path}")

    def step_simulation(self, steps=1):
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/2560.0)
            
            self.total_steps += 1
            
            # Capture frame if recording is enabled and it's time to capture
            if self.should_capture and (self.total_steps - self.last_capture_step >= self.capture_interval):
                self.capture_frame()
                self.last_capture_step = self.total_steps

    def close(self):
        if self.record_video:
            self.save_video()
        p.disconnect()

    def add_ground(self):
        ground_id = p.loadURDF("plane.urdf")
        self.objects.append(ground_id)
        return ground_id

    def add_box(self, position, size=[0.02, 0.02, 0.06]):
        col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0, np.random.random(), np.random.random(), 1])
        box_id = p.createMultiBody(0.1, col_box_id, vis_box_id, position)
        self.objects.append(box_id)
        return box_id

    def add_bowl(self, position, radius=0.1, height=0.05, thickness=0.005):
        """Create a bowl using multiple collision shapes"""
        # Create the main bowl body
        bowl_bottom_radius = radius - thickness
        bowl_height = height - thickness
        
        # Create bowl base (cylinder)
        base_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=thickness
        )
        base_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=thickness,
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        
        # Create bowl walls (hollow cylinder)
        wall_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=bowl_height,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        wall_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=bowl_height,
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        
        # Combine shapes into one multibody
        bowl_position = [position[0], position[1], position[2] + height/2]
        bowl_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=bowl_position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.objects.append(bowl_id)
        return bowl_id

    def add_cabinet(self, position, size=[0.4, 0.3, 0.5]):
        """Create a cabinet with an openable door"""
        cabinet_color = [0.6, 0.4, 0.2, 1]  # Brown color
        
        # Create cabinet body
        cabinet_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        cabinet_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2],
                                        rgbaColor=cabinet_color)
        
        # Create cabinet base
        cabinet_id = p.createMultiBody(
            baseMass=0,  # Static cabinet
            baseCollisionShapeIndex=cabinet_col,
            baseVisualShapeIndex=cabinet_vis,
            basePosition=position
        )
        
        # Create door
        door_size = [size[0]/2, 0.02, size[2]]
        door_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[door_size[0]/2, door_size[1]/2, door_size[2]/2])
        door_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[door_size[0]/2, door_size[1]/2, door_size[2]/2],
                                     rgbaColor=[0.5, 0.3, 0.1, 1])
        
        # Position door slightly in front of cabinet
        door_pos = [
            position[0] + size[0]/4,  # Centered on right half
            position[1] + size[1]/2,  # In front of cabinet
            position[2]
        ]
        
        door_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=door_col,
            baseVisualShapeIndex=door_vis,
            basePosition=door_pos
        )
        
        # Create hinge constraint for door
        hinge_pivot = [door_size[0]/2, 0, 0]
        hinge_axis = [0, 0, 1]  # Rotate around z-axis
        
        door_hinge = p.createConstraint(
            parentBodyUniqueId=cabinet_id,
            parentLinkIndex=-1,
            childBodyUniqueId=door_id,
            childLinkIndex=-1,
            jointType=p.JOINT_REVOLUTE,
            jointAxis=hinge_axis,
            parentFramePosition=[size[0]/4, 0, 0],
            childFramePosition=[-door_size[0]/2, 0, 0]
        )
        
        self.objects.extend([cabinet_id, door_id])
        return cabinet_id, door_id, door_hinge
    
    def open_cabinet_door(self, angle=1.57):  # 90 degrees in radians
        """Open the cabinet door to a specified angle"""
        if 'door' in self.cabinet_parts and 'hinge' in self.cabinet_parts:
            p.setJointMotorControl2(
                bodyUniqueId=self.cabinet_parts['door'],
                jointIndex=0,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=50
            )
            # Let the door move
            for _ in range(100):
                self.step_simulation()
    
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
        self.step_simulation(100)  # Let the arm settle
    
    def control_gripper(self, open=True):
        target_pos = 0.04 if open else 0.01
        # Control both gripper fingers
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, target_pos, force=10)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, target_pos, force=10)
        self.step_simulation(100)
    
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
        max_steps = 200
        threshold = 0.01
        
        while steps < max_steps:
            self.step_simulation(1)
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
        self.step_simulation(150)

        self.control_gripper(open=True)
        # 3. Move down to object
        self.move_arm(pick_position, gripper_orientation)
        
        # Create a constraint to keep the object in the gripper
        if 0:
            constraint = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.PANDA_EE_INDEX,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
            )

        # 4. Close gripper to grasp object
        self.control_gripper(open=False)
        self.step_simulation(50)  # Give time for grasp to settle
        
        # Lift the object
        lift_position = [pick_position[0], pick_position[1], pick_position[2] + 0.2]
        self.move_arm(lift_position, gripper_orientation)
        
        #return constraint
        #self.stop_capture()
    
    def place_object(self, target_position, constraint=None):
        #if self.should_capture:self.start_capture()
        if self.robot is None:
            raise ValueError("No robot arm has been loaded.")
        
        gripper_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        
        # Place sequence
        pre_place_position = [target_position[0], target_position[1], target_position[2] + 0.04]
        self.move_arm(pre_place_position, gripper_orientation)
        self.move_arm(target_position, gripper_orientation)
        
        # Remove constraint before opening gripper
        if constraint is not None:
            p.removeConstraint(constraint)
        
        self.control_gripper(open=True)
        self.move_arm(pre_place_position, gripper_orientation)
        #self.stop_capture()

if __name__ == "__main__":
    record = 0
    sim = PyBulletSimulator(gui=True, record_video=record)
    
    # Initialize simulation
    sim.add_ground()
    robot = sim.add_robot_arm()
    
    # Add cabinet and bowl
    #cabinet, door, hinge = sim.add_cabinet([0.8, 0, 0.25])  # Position the cabinet
    bowl = sim.add_bowl([0.5, 0.5, 0.0])  # Position the bowl
    
    # Add boxes for manipulation
    box1 = sim.add_box([0.5, 0.5, 0.11])
    box2 = sim.add_box([0.3, -0.4, 0.1])
    
    # Start capture for sequence
    if record: sim.start_capture()
    
    # Open cabinet door
    #sim.open_cabinet_door(angle=1.57)  # Open 90 degrees
    
    # Perform pick and place operations
    sim.pick_object(box1)    
    sim.place_object([0.8, 0, 0.2], None)  # Place inside cabinet
    
    sim.pick_object(box2)    
    sim.place_object([0.5, 0.5, 0.15], None)  # Place in bowl
    
    # Stop capture
    if record: sim.stop_capture()
    
    # Let simulation settle
    for _ in range(100):
        sim.step_simulation()
    
    sim.close()
