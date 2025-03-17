from bulletarm import env_factory
import pybullet as p
import pybullet_data
import numpy as np
import trimesh
import time
import os
from datetime import datetime


class ObjectCategoriesDemo:
    """Extension of PyBulletSimulator focusing on object categories"""
    
    def __init__(self, gui=True):
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        
        # Object storage by category
        self.objects = {
            'rigid': [],
            'articulated': [],
            'nested_containers': [],
            'nested_contents': {}  # Maps container ID to list of contained objects
        }
        
        # Robot arm constants
        self.PANDA_GRIPPER_INDEX = 9
        self.PANDA_EE_INDEX = 11
        self.robot = None
        
    def step_simulation(self, steps=1):
        """Run simulation steps"""
        for _ in range(steps):
            p.stepSimulation()
            if p.isConnected(p.GUI):
                time.sleep(1/240.0)
    
    def create_environment(self):
        """Set up the basic environment"""
        # Add ground
        ground_id = p.loadURDF("plane.urdf")
        
        # Add robot arm
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        self.reset_arm()
        self.objects['articulated'].append(self.robot)
        
        return ground_id
    
    def reset_arm(self):
        """Reset the robot to a default position"""
        default_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04, 0, 0, 0]
        for i in range(12):  # Panda has 12 joints
            p.resetJointState(self.robot, i, default_positions[i])
        self.step_simulation(100)  # Let the arm settle
    
    # === RIGID OBJECTS ===
    
    def add_rigid_cube(self, position, size=0.05, mass=0.1, color=None):
        """Add a simple rigid cube"""
        if color is None:
            color = [np.random.random(), np.random.random(), np.random.random(), 1]
            
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2], rgbaColor=color)
        body_id = p.createMultiBody(mass, col_id, vis_id, position)
        
        self.objects['rigid'].append(body_id)
        return body_id
    
    def add_rigid_sphere(self, position, radius=0.05, mass=0.1, color=None):
        """Add a simple rigid sphere"""
        if color is None:
            color = [np.random.random(), np.random.random(), np.random.random(), 1]
            
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        body_id = p.createMultiBody(mass, col_id, vis_id, position)
        
        self.objects['rigid'].append(body_id)
        return body_id
    
    # === ARTICULATED OBJECTS ===
    
    def add_articulated_book(self, position, size=[0.2, 0.15, 0.02]):
        """Create a book with an openable cover (articulated object)"""
        # Create the book base (back cover)
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], 
                                      rgbaColor=[0.6, 0.4, 0.1, 1])
        
        book_base = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=position
        )
        
        # Create the front cover
        cover_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        cover_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], 
                                       rgbaColor=[0.8, 0.2, 0.1, 1])
        
        # Position the cover directly on top of the base
        cover_pos = [position[0], position[1], position[2] + size[2]]
        book_cover = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=cover_col,
            baseVisualShapeIndex=cover_vis,
            basePosition=cover_pos
        )
        
        # Create a hinge constraint for the cover
        # The hinge should be at the back edge of the book
        hinge = p.createConstraint(
            parentBodyUniqueId=book_base,
            parentLinkIndex=-1,
            childBodyUniqueId=book_cover,
            childLinkIndex=-1,
            jointType=p.JOINT_HINGE,  # Use JOINT_HINGE instead of JOINT_REVOLUTE
            jointAxis=[1, 0, 0],  # Rotate around x-axis
            parentFramePosition=[0, -size[1]/2, size[2]/2],  # Hinge at the back edge
            childFramePosition=[0, -size[1]/2, -size[2]/2]
        )
        
        if hinge >= 0:  # Check if constraint was created successfully
            # Set joint limits for the cover
            p.changeConstraint(hinge, maxForce=10)
        else:
            print("Warning: Failed to create book hinge constraint")
            # Create a simple fixed constraint as fallback
            hinge = p.createConstraint(
                parentBodyUniqueId=book_base,
                parentLinkIndex=-1,
                childBodyUniqueId=book_cover,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, size[2]/2],
                childFramePosition=[0, 0, -size[2]/2]
            )
        
        self.objects['articulated'].extend([book_base, book_cover])
        return book_base, book_cover, hinge
    
    def add_articulated_drawer(self, position, size=[0.3, 0.4, 0.2]):
        """Create a drawer that can slide in and out"""
        # Create cabinet frame - Only visual, no collision
        frame_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], 
                                       rgbaColor=[0.6, 0.6, 0.6, 0.3])  # Transparent cabinet
        
        # Create collision shapes only for the cabinet walls, not the entire cabinet
        # This prevents collision with objects inside the drawer
        wall_thickness = 0.01
        
        # Cabinet base (bottom)
        base_col = p.createCollisionShape(p.GEOM_BOX, 
                                       halfExtents=[size[0]/2, size[1]/2, wall_thickness/2])
        
        cabinet_frame = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=frame_vis,
            basePosition=[position[0], position[1], position[2] - size[2]/2 + wall_thickness/2]
        )
        
        # Create side walls as separate collision objects
        # Left wall
        left_col = p.createCollisionShape(p.GEOM_BOX, 
                                        halfExtents=[wall_thickness/2, size[1]/2, size[2]/2])
        left_wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=left_col,
            basePosition=[position[0] - size[0]/2 + wall_thickness/2, position[1], position[2]]
        )
        
        # Right wall
        right_col = p.createCollisionShape(p.GEOM_BOX, 
                                         halfExtents=[wall_thickness/2, size[1]/2, size[2]/2])
        right_wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=right_col,
            basePosition=[position[0] + size[0]/2 - wall_thickness/2, position[1], position[2]]
        )
        
        # Back wall
        back_col = p.createCollisionShape(p.GEOM_BOX, 
                                        halfExtents=[size[0]/2, wall_thickness/2, size[2]/2])
        back_wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=back_col,
            basePosition=[position[0], position[1] - size[1]/2 + wall_thickness/2, position[2]]
        )
        
        # Create drawer - no collision with cabinet interior
        drawer_size = [size[0]-0.03, size[1]-0.03, size[2]/2-0.02]  # Smaller than cabinet
        drawer_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[drawer_size[0]/2, drawer_size[1]/2, drawer_size[2]/2])
        drawer_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[drawer_size[0]/2, drawer_size[1]/2, drawer_size[2]/2], 
                                        rgbaColor=[0.8, 0.8, 0.2, 1])
        
        drawer_pos = [position[0], position[1], position[2]]  # Initially aligned
        drawer = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=drawer_col,
            baseVisualShapeIndex=drawer_vis,
            basePosition=drawer_pos
        )
        
        # Create a sliding constraint for the drawer
        slider = p.createConstraint(
            parentBodyUniqueId=cabinet_frame,
            parentLinkIndex=-1,
            childBodyUniqueId=drawer,
            childLinkIndex=-1,
            jointType=p.JOINT_PRISMATIC,
            jointAxis=[0, 1, 0],  # Slide along y-axis
            parentFramePosition=[0, 0, size[2]/2],
            childFramePosition=[0, 0, 0]
        )
        
        # Set joint limits
        p.changeConstraint(slider, maxForce=50, 
                          lowerLimit=-size[1]/2 + wall_thickness,  # How far it can be pulled out
                          upperLimit=0)  # Closed position
        
        # Attach all walls to the cabinet frame with fixed constraints
        for wall_id in [left_wall, right_wall, back_wall]:
            p.createConstraint(
                parentBodyUniqueId=cabinet_frame,
                parentLinkIndex=-1,
                childBodyUniqueId=wall_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0]
            )
        
        self.objects['articulated'].extend([cabinet_frame, drawer])
        self.objects['nested_containers'].append(drawer)
        self.objects['nested_contents'][drawer] = []
        
        return cabinet_frame, drawer, slider
    
    # === NESTED OBJECTS ===
    
    def add_bowl(self, position, radius=0.1, height=0.05, thickness=0.005):
        """Create a bowl that can contain other objects"""
        # Create a simplified bowl using primitive shapes instead of mesh
        # Create the bowl base (cylinder)
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
        
        bowl_position = [position[0], position[1], position[2] + thickness/2]
        bowl_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=bowl_position
        )
        
        # Create bowl sides using cylindrical segments
        num_segments = 8
        wall_height = height - thickness
        wall_radius = radius - thickness/2
        
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            x_offset = wall_radius * np.cos(angle)
            y_offset = wall_radius * np.sin(angle)
            
            # Create wall segment
            wall_col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[thickness/2, thickness/2, wall_height/2]
            )
            wall_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[thickness/2, thickness/2, wall_height/2],
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
            
            wall_pos = [
                bowl_position[0] + x_offset,
                bowl_position[1] + y_offset,
                bowl_position[2] + wall_height/2
            ]
            
            # Create wall segment as a separate body
            wall_id = p.createMultiBody(
                baseMass=0,  # Make it static
                baseCollisionShapeIndex=wall_col,
                baseVisualShapeIndex=wall_vis,
                basePosition=wall_pos
            )
            
            # Constrain wall segment to bowl base
            p.createConstraint(
                parentBodyUniqueId=bowl_id,
                parentLinkIndex=-1,
                childBodyUniqueId=wall_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[x_offset, y_offset, wall_height/2],
                childFramePosition=[0, 0, 0]
            )
        
        self.objects['nested_containers'].append(bowl_id)
        self.objects['nested_contents'][bowl_id] = []
        
        return bowl_id
    
    def add_box_container(self, position, size=[0.2, 0.2, 0.1], wall_thickness=0.01):
        """Create an open box that can contain other objects"""
        # Create the base
        base_size = [size[0], size[1], wall_thickness]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[base_size[0]/2, base_size[1]/2, base_size[2]/2])
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[base_size[0]/2, base_size[1]/2, base_size[2]/2], 
                                     rgbaColor=[0.9, 0.9, 0.9, 1])
        
        box_container = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=position
        )
        
        # Create walls (front, back, left, right)
        walls = []
        wall_positions = [
            [position[0], position[1] + size[1]/2 - wall_thickness/2, position[2] + size[2]/2],  # Front
            [position[0], position[1] - size[1]/2 + wall_thickness/2, position[2] + size[2]/2],  # Back
            [position[0] + size[0]/2 - wall_thickness/2, position[1], position[2] + size[2]/2],  # Right
            [position[0] - size[0]/2 + wall_thickness/2, position[1], position[2] + size[2]/2]   # Left
        ]
        
        wall_sizes = [
            [size[0], wall_thickness, size[2]],  # Front/Back walls
            [wall_thickness, size[1], size[2]],  # Left/Right walls
        ]
        
        for i, wall_pos in enumerate(wall_positions):
            wall_idx = 0 if i < 2 else 1  # First two are front/back, last two are left/right
            
            wall_col = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=[wall_sizes[wall_idx][0]/2, wall_sizes[wall_idx][1]/2, wall_sizes[wall_idx][2]/2]
            )
            wall_vis = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=[wall_sizes[wall_idx][0]/2, wall_sizes[wall_idx][1]/2, wall_sizes[wall_idx][2]/2],
                rgbaColor=[0.9, 0.9, 0.9, 1]
            )
            
            wall = p.createMultiBody(
                baseMass=0.0,  # Walls are static relative to the base
                baseCollisionShapeIndex=wall_col,
                baseVisualShapeIndex=wall_vis,
                basePosition=wall_pos
            )
            
            # Create fixed constraint between base and wall
            constraint = p.createConstraint(
                parentBodyUniqueId=box_container,
                parentLinkIndex=-1,
                childBodyUniqueId=wall,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[
                    wall_pos[0] - position[0],
                    wall_pos[1] - position[1],
                    wall_pos[2] - position[2]
                ],
                childFramePosition=[0, 0, 0]
            )
            
            walls.append(wall)
        
        self.objects['nested_containers'].append(box_container)
        self.objects['nested_contents'][box_container] = []
        
        return box_container, walls
    
    # === INTERACTION METHODS ===
    
    def is_inside(self, container_id, object_id):
        """Check if an object is inside a container"""
        if container_id not in self.objects['nested_containers']:
            return False
        
        container_pos, _ = p.getBasePositionAndOrientation(container_id)
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        
        # Get container AABB
        container_aabb_min, container_aabb_max = p.getAABB(container_id)
        
        # Check if object is inside container AABB
        inside = all(container_aabb_min[i] <= object_pos[i] <= container_aabb_max[i] for i in range(3))
        
        if inside:
            # Add to nested contents if not already there
            if object_id not in self.objects['nested_contents'][container_id]:
                self.objects['nested_contents'][container_id].append(object_id)
        elif object_id in self.objects['nested_contents'][container_id]:
            self.objects['nested_contents'][container_id].remove(object_id)
            
        return inside
    
    def open_articulated_object(self, obj_id, joint_id, target_pos=0.5, force=10):
        """Open an articulated object to a target position"""
        # For constraints like hinges, we need to directly control the constraint
        try:
            # Try to manipulate the object using the constraint
            p.changeConstraint(
                joint_id,
                gearRatio=1,
                erp=0.8,
                maxForce=force,
                targetPosition=target_pos
            )
        except Exception as e:
            print(f"Error controlling constraint: {e}")
            # Fallback to motor control if constraint control fails
            try:
                p.setJointMotorControl2(
                    bodyUniqueId=obj_id,
                    jointIndex=0,  # Assuming joint index 0
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=force
                )
            except Exception as e:
                print(f"Error controlling joint: {e}")
        
        # Let the object move
        self.step_simulation(100)
    
    def pull_drawer(self, drawer_id, slider_id, distance=0.2):
        """Pull a drawer out by a certain distance"""
        p.changeConstraint(
            slider_id,
            gearRatio=1,
            erp=0.8,
            maxForce=50,
            targetPosition=-distance  # Negative moves outward on y-axis
        )
        
        # Let the drawer move
        self.step_simulation(100)
    
    def push_drawer(self, drawer_id, slider_id):
        """Push a drawer closed"""
        p.changeConstraint(
            slider_id,
            gearRatio=1,
            erp=0.8,
            maxForce=50,
            targetPosition=0  # Zero is closed position
        )
        
        # Let the drawer move
        self.step_simulation(100)
    
    def control_gripper(self, open=True):
        """Control the robot gripper"""
        if not self.robot:
            return
            
        target_pos = 0.04 if open else 0.01
        # Control both gripper fingers
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, target_pos, force=10)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, target_pos, force=10)
        self.step_simulation(50)
    
    def move_arm(self, target_position, target_orientation=None):
        """Move the robot arm to a target position"""
        if not self.robot:
            return
            
        if target_orientation is None:
            target_orientation = p.getQuaternionFromEuler([0, -np.pi/2, 0])  # Default orientation
            
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
                force=300,
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
    
    def pick_object(self, object_id, check_contents=False):
        """
        Pick up an object with the robot gripper
        
        Args:
            object_id: ID of the object to pick
            check_contents: If True, check if object is a container with contents
        
        Returns:
            constraint: The constraint connecting object to gripper
            contained_objects: List of contained objects and their constraints (if container)
        """
        if not self.robot:
            return None, []
            
        # Get object position
        object_position, object_orientation = p.getBasePositionAndOrientation(object_id)
        gripper_orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Gripper facing down
        
        # Check if object is a container and has contents
        contained_objects = []
        contained_constraints = []
        
        if check_contents and object_id in self.objects['nested_containers']:
            contained_ids = self.objects['nested_contents'][object_id].copy()
            if contained_ids:
                print(f"Container has {len(contained_ids)} objects inside")
                
                # Create constraints between container and its contents
                for contained_id in contained_ids:
                    container_pos, container_orient = p.getBasePositionAndOrientation(object_id)
                    contained_pos, contained_orient = p.getBasePositionAndOrientation(contained_id)
                    
                    # Calculate relative position
                    rel_pos = [
                        contained_pos[0] - container_pos[0],
                        contained_pos[1] - container_pos[1],
                        contained_pos[2] - container_pos[2]
                    ]
                    
                    # Create constraint to keep contained object in same relative position
                    contained_constraint = p.createConstraint(
                        parentBodyUniqueId=object_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=contained_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=rel_pos,
                        childFramePosition=[0, 0, 0]
                    )
                    
                    contained_objects.append(contained_id)
                    contained_constraints.append(contained_constraint)
        
        # Approach positions
        pre_pick_position = [object_position[0], object_position[1], object_position[2] + 0.2]
        pick_position = [object_position[0], object_position[1], object_position[2] + 0.05]
        
        # Execute pick sequence
        self.move_arm(pre_pick_position, gripper_orientation)
        self.control_gripper(open=True)
        self.move_arm(pick_position, gripper_orientation)
        self.control_gripper(open=False)
        self.step_simulation(50)
        
        # Create constraint to attach object to gripper
        constraint = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.PANDA_GRIPPER_INDEX,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )
        
        # Lift the object
        self.move_arm(pre_pick_position, gripper_orientation)
        
        # Update containment status
        for container_id in self.objects['nested_containers']:
            if object_id in self.objects['nested_contents'][container_id]:
                self.objects['nested_contents'][container_id].remove(object_id)
        
        return constraint, contained_constraints
    
    def place_object(self, target_position, constraint=None, contained_constraints=None):
        """
        Place an object at a target position
        
        Args:
            target_position: Position to place the object
            constraint: Constraint connecting object to gripper
            contained_constraints: List of constraints for contained objects (if any)
        """
        if not self.robot:
            return
            
        gripper_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        
        # Place sequence
        pre_place_position = [target_position[0], target_position[1], target_position[2] + 0.2]
        self.move_arm(pre_place_position, gripper_orientation)
        self.move_arm(target_position, gripper_orientation)
        
        # First remove any containment constraints
        if contained_constraints:
            for contained_constraint in contained_constraints:
                try:
                    p.removeConstraint(contained_constraint)
                except:
                    pass
        
        # Then remove the main constraint before opening gripper
        if constraint is not None:
            try:
                p.removeConstraint(constraint)
            except:
                pass
        
        self.control_gripper(open=True)
        self.move_arm(pre_place_position, gripper_orientation)
        
        # Let simulation settle to update containment relationships
        self.step_simulation(50)
        
        # Update containment status
        for obj_id in self.objects['rigid']:
            # Check if the object is now inside any container
            for container_id in self.objects['nested_containers']:
                self.is_inside(container_id, obj_id)
    
    def close(self):
        """Disconnect from physics server"""
        p.disconnect()


def run_demo():
    """Run a demonstration of different object categories"""
    # Initialize simulator
    demo = ObjectCategoriesDemo(gui=True)
    
    # Create environment and robot
    demo.create_environment()
    
    print("Setting up objects...")
    
    # Add rigid objects
    cube1 = demo.add_rigid_cube([0.5, 0.5, 0.05], size=0.05, color=[1, 0, 0, 1])
    cube2 = demo.add_rigid_cube([0.5, 0.3, 0.05], size=0.04, color=[0, 1, 0, 1])
    sphere = demo.add_rigid_sphere([0.3, 0.5, 0.05], radius=0.03, color=[0, 0, 1, 1])
    
    # Add articulated objects - Modified to avoid errors
    print("Adding articulated objects...")
    # Safer approach: Skip book creation if it causes issues
    try:
        book_base, book_cover, book_hinge = demo.add_articulated_book([0.0, 0.5, 0.05])
        book_created = True
    except Exception as e:
        print(f"Skipping book creation due to error: {e}")
        book_created = False
    
    try:
        pass
        #drawer_base, drawer, drawer_slider = demo.add_articulated_drawer([0.0, 0.0, 0.15])
        #drawer_created = True
        drawer_created = False
    except Exception as e:
        print(f"Skipping drawer creation due to error: {e}")
        drawer_created = False
    
    # Add nested containers
    print("Adding containers...")
    try:
        bowl = demo.add_bowl([0.3, 0.0, 0.05], radius=0.1, height=0.05)
        bowl_created = True
    except Exception as e:
        print(f"Error creating bowl: {e}")
        bowl_created = False
        
    try:
        box, box_walls = demo.add_box_container([0.6, 0.0, 0.05], size=[0.2, 0.2, 0.1])
        box_created = True
    except Exception as e:
        print(f"Error creating box: {e}")
        box_created = False
    
    # Let everything settle
    print("Letting physics settle...")
    demo.step_simulation(100)
    
    # Demo interactions
    print("\n=== Starting Demo ===")
    
    # Demo 1: Open articulated objects if they were created successfully
    if book_created:
        print("Opening book...")
        try:
            demo.open_articulated_object(book_cover, book_hinge, target_pos=1.0)
        except Exception as e:
            print(f"Error opening book: {e}")
    
    if drawer_created:
        print("Pulling drawer...")
        try:
            demo.pull_drawer(drawer, drawer_slider, distance=0.3)
        except Exception as e:
            print(f"Error pulling drawer: {e}")

    
    # Demo 2: Pick and place into containers
    print("Picking up red cube...")
    try:
        constraint, _ = demo.pick_object(cube1)
        
        if bowl_created:
            print("Placing red cube in bowl...")
            demo.place_object([0.3, 0.0, 0.08], constraint)
        else:
            print("Placing red cube on table...")
            demo.place_object([0.3, 0.0, 0.05], constraint)
        
        # Wait for physics to settle
        demo.step_simulation(50)
        
        print("Picking up green cube...")
        constraint, _ = demo.pick_object(cube2)
        
        if box_created:
            print("Placing green cube in box...")
            demo.place_object([0.6, 0.0, 0.08], constraint)
        else:
            print("Placing green cube on table...")
            demo.place_object([0.6, 0.0, 0.05], constraint)
        
        # Let physics settle
        demo.step_simulation(50)
        
        print("Picking up blue sphere...")
        constraint, _ = demo.pick_object(sphere)
        
        if bowl_created:
            print("Placing blue sphere in bowl...")
            demo.place_object([0.3, 0.0, 0.1], constraint)
            
            # Let physics settle
            demo.step_simulation(100)
            
            # NEW ACTION: Pick up container with object inside
            print("\n=== Demonstrating container movement with contents ===")
            print("Picking up bowl with sphere inside...")
            # The check_contents=True parameter will automatically create constraints
            # for objects inside the container
            constraint, contained_constraints = demo.pick_object(bowl, check_contents=True)
            
            print("Moving bowl with contents to new location...")
            demo.place_object([0.5, 0.5, 0.05], constraint, contained_constraints)
            
            print("Contents should stay in the same relative position within the container")
            
        elif drawer_created:
            print("Placing blue sphere in drawer...")
            demo.place_object([0.0, -0.15, 0.12], constraint)
            
            # Demo 3: Close articulated objects
            print("Closing drawer...")
            demo.push_drawer(drawer, drawer_slider)
        else:
            print("Placing blue sphere on table...")
            demo.place_object([0.0, 0.0, 0.05], constraint)
    except Exception as e:
        print(f"Error during pick and place: {e}")
    
    # Display containment relationships
    print("\n=== Container Contents ===")
    for container_id in demo.objects['nested_containers']:
        try:
            contents = demo.objects['nested_contents'][container_id]
            container_pos, _ = p.getBasePositionAndOrientation(container_id)
            
            container_type = "Unknown"
            if bowl_created and container_id == bowl:
                container_type = "Bowl"
            elif box_created and container_id == box:
                container_type = "Box"
            elif drawer_created and container_id == drawer:
                container_type = "Drawer"
                
            print(f"{container_type} at {container_pos} contains {len(contents)} objects")
        except Exception as e:
            print(f"Error checking container contents: {e}")
    
    print("\nDemo complete. Press Ctrl+C to exit.")
    
    try:
        while True:
            demo.step_simulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()


if __name__ == "__main__":
    run_demo()