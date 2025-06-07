import pybullet as p
import pybullet_data
import numpy as np
import time
from datetime import datetime
from .base_env import BaseEnv
from .contact import ContactModel

class GripperSimulator(ContactModel, BaseEnv):
    def __init__(self, gui=True, robot_position=[0.0, 0.0, 0.0], auto_register=True):
        """
        Initialize the gripper simulator with integrated contact modeling.
        
        Args:
            gui: Whether to show the GUI
            robot_position: Initial position of the robot
            auto_register: Whether to automatically register objects for contact analysis
        """
        # Initialize the ContactModel parent class
        super().__init__(gravity_direction=(0, 0, -9.8))
        
        self.gui = gui
        self.auto_register = auto_register
        self.next_obj_id = 0  # For naming objects

        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Disable real-time simulation for better control
        p.setRealTimeSimulation(0)
        
        self.objects = []
        self.named_objects = {}  # New dictionary to store object names
        
        # Panda robot constants
        self.PANDA_GRIPPER_INDEX = 9
        self.PANDA_EE_INDEX = 11
        self.PANDA_NUM_JOINTS = 12
        self.MAX_FORCE = 320

        # Add capture control parameters
        self.total_steps = 0  # Track total simulation steps

        # Set up the environment
        self.ground_id = self.add_ground()
        self.robot = self.add_robot_arm(position=robot_position)
        self.reset_arm()
    
    @property
    def robot_id(self): return self.robot

    def step(self, steps=1, update_contacts=False):
        """
        Step the simulation forward.
        
        Args:
            steps: Number of steps to simulate
            update_contacts: Whether to update contact graphs after stepping
        """
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/2560.0)
            
            self.total_steps += 1
        
        # Update contact and support graphs if requested
        if update_contacts:
            self.update_contact_analysis()
            
        return self

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect()

    def add_ground(self, name="ground"):
        """Add a ground plane to the simulation."""
        ground_id = p.loadURDF("plane.urdf")
        self.objects.append(ground_id)
        
        # Register ground with contact model
        if self.auto_register:
            self.register_object(name, ground_id)
            self.named_objects[ground_id] = name
            
        return ground_id

    def add_box(self, position, size=[0.02, 0.02, 0.06], rgba_color=None, name=None):
        """
        Add a box to the simulation.
        
        Args:
            position: [x, y, z] position
            size: [width, depth, height] size as half-extents
            rgba_color: [r, g, b, a] color or None for random
            name: Optional name for the box (for contact analysis)
            
        Returns:
            box_id: The ID of the created box
        """
        # Generate a random color if none provided
        if rgba_color is None:
            rgba_color = [0, np.random.random(), np.random.random(), 1]
        
        # Create the box
        col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=rgba_color)
        box_id = p.createMultiBody(0.1, col_box_id, vis_box_id, position)
        self.objects.append(box_id)
        
        # Generate a name if none provided
        if name is None:
            name = f"box_{self.next_obj_id}"
            self.next_obj_id += 1
        
        # Register box with contact model
        if self.auto_register:
            self.register_object(name, box_id)
            self.named_objects[box_id] = name
            
        return box_id

    def add_sphere(self, position, radius=0.05, rgba_color=None, name=None):
        """
        Add a sphere to the simulation.
        
        Args:
            position: [x, y, z] position
            radius: Sphere radius
            rgba_color: [r, g, b, a] color or None for random
            name: Optional name for the sphere (for contact analysis)
            
        Returns:
            sphere_id: The ID of the created sphere
        """
        # Generate a random color if none provided
        if rgba_color is None:
            rgba_color = [np.random.random(), 0, np.random.random(), 1]
        
        # Create the sphere
        col_sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_sphere_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color)
        sphere_id = p.createMultiBody(0.1, col_sphere_id, vis_sphere_id, position)
        self.objects.append(sphere_id)
        
        # Generate a name if none provided
        if name is None:
            name = f"sphere_{self.next_obj_id}"
            self.next_obj_id += 1
        
        # Register sphere with contact model
        if self.auto_register:
            self.register_object(name, sphere_id)
            self.named_objects[sphere_id] = name
            
        return sphere_id

    def add_robot_arm(self, position=[0, 0, 0], name="robot"):
        """Add a Franka Panda robot arm to the simulation."""
        robot_id = p.loadURDF("franka_panda/panda.urdf", position, useFixedBase=True)
        
        # Set up joint damping
        for i in range(self.PANDA_NUM_JOINTS):
            p.changeDynamics(robot_id, i, linearDamping=0.1, angularDamping=0.1)
        
        # Reset all joints to a good starting position
        
        # Register robot with contact model
        if self.auto_register:
            self.register_object(name, robot_id)
            self.named_objects[robot_id] = name
            
        return robot_id
    
    def reset_arm(self):
        """Reset the robot to a default position"""
        default_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04, 0, 0, 0]
        for i in range(self.PANDA_NUM_JOINTS):
            #print(self.robot, i, default_positions[i])
            p.resetJointState(self.robot, i, default_positions[i])
        self.step(100)  # Let the arm settle
    
    def control_gripper(self, open=True):
        """
        Control the gripper (open or close).
        
        Args:
            open: True to open, False to close
        """
        target_pos = 0.04 if open else 0.01
        # Control both gripper fingers
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, target_pos, force=10)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, target_pos, force=10)
        self.step(200)
    
    def move_arm(self, target_position, target_orientation):
        """
        Move the arm using inverse kinematics.
        
        Args:
            target_position: [x, y, z] target position
            target_orientation: Quaternion target orientation
        """
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
        """
        Pick up an object with the robot arm.
        
        Args:
            object_id: The ID of the object to pick up
        """
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
        
        # Update contact analysis after picking
        if self.auto_register:
            self.update_contact_analysis()
    
    def update_contact_analysis(self, visualize=False):
        """
        Update the contact and support graph analysis.
        
        Args:
            visualize: Whether to show visualizations
        """
        self.get_contact_points()
        self.determine_support_relationships()
        self.build_contact_graph()
        self.build_support_graph()
        
        if visualize:
            self.print_contact_info()
            self.print_support_info()
            self.visualize_graph(self.contact_graph, "Contact Graph")
            self.visualize_graph(self.support_graph, "Support Graph")
        
        return self
    
    def get_object_name(self, object_id):
        """Get the name of an object from its ID."""
        return self.named_objects.get(object_id, f"object_{object_id}")
    
    def get_object_id(self, name):
        """Get the ID of an object from its name."""
        for obj_id, obj_name in self.named_objects.items():
            if obj_name == name:
                return obj_id
        return None
    
    def create_tower(self, base_position, num_blocks=3, block_size=None, spacing=0.0):
        """
        Create a tower of blocks.
        
        Args:
            base_position: [x, y, z] position of the base of the tower
            num_blocks: Number of blocks in the tower
            block_size: Size of each block or None for random sizes
            spacing: Extra vertical spacing between blocks
            
        Returns:
            List of block IDs in the tower (bottom to top)
        """
        block_ids = []
        current_height = base_position[2]
        
        for i in range(num_blocks):
            # Generate random or fixed size
            if block_size is None:
                size = [
                    0.03 + 0.02 * np.random.random(),
                    0.03 + 0.02 * np.random.random(),
                    0.03 + 0.02 * np.random.random()
                ]
            else:
                size = block_size.copy()
            
            # Calculate position
            pos = [base_position[0], base_position[1], current_height + size[2]]
            
            # Add block
            block_id = self.add_box(
                position=pos,
                size=size,
                name=f"tower_block_{i}"
            )
            
            block_ids.append(block_id)
            
            # Update height for next block
            current_height += 2 * size[2] + spacing
            
            # Let physics settle between blocks
            self.step(100)
        
        # Final settling
        self.step(500, update_contacts=True)
        
        return block_ids
    
    def remove_object(self, object_id=None, name=None):
        """
        Remove an object from the simulation by its ID or name.
        
        Args:
            object_id: The PyBullet ID of the object to remove
            name: The name of the object to remove
            
        Returns:
            True if object was removed, False otherwise
        """
        # If name is provided but not object_id, look up the ID
        if object_id is None and name is not None:
            object_id = self.get_object_id(name)
            if object_id is None:
                print(f"No object with name '{name}' found")
                return False
        
        # If we still don't have an object_id, we can't proceed
        if object_id is None:
            print("Must provide either object_id or name")
            return False
            
        # Don't remove the robot or ground
        if object_id == self.robot or object_id == self.ground_id:
            print("Cannot remove robot or ground")
            return False
        
        # Remove from PyBullet
        p.removeBody(object_id)
        
        # Remove from our object lists
        if object_id in self.objects:
            self.objects.remove(object_id)
        
        # Get the name before removing from named_objects
        name = self.named_objects.get(object_id, None)
        
        # Remove from named_objects
        if object_id in self.named_objects:
            del self.named_objects[object_id]
        
        # Unregister from contact model if name is available
        if name is not None and self.auto_register:
            try:
                self.unregister_object(name)
            except AttributeError as e:
                print(f"Warning: Cannot unregister object '{name}': {e}")
                # If the ContactModel parent class doesn't have unregister_object method,
                # We'll need to manually clean up contact model data
                if name in self.object_registry:
                    del self.object_registry[name]
                if name in self.object_attributes:
                    del self.object_attributes[name]
                if name in self.object_names:
                    self.object_names.remove(name)
        
        # Update the contact analysis
        if self.auto_register:
            try:
                self.update_contact_analysis()
            except Exception as e:
                print(f"Warning: Error updating contact analysis: {e}")
        
        return True

    def remove_last_object(self):
        """
        Remove the most recently added object.
        
        Returns:
            True if an object was removed, False otherwise
        """
        # Find the last added object that isn't the robot or ground
        for i in range(len(self.objects) - 1, -1, -1):
            object_id = self.objects[i]
            if object_id != self.robot and object_id != self.ground_id:
                return self.remove_object(object_id=object_id)
        
        print("No removable objects found")
        return False