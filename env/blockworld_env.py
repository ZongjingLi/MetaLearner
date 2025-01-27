import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import os
import open3d as o3d

class PyBulletSimulator:
    def __init__(self, gui=True, record_video=False, video_path='output.mp4'):
        self.gui = gui
        self.record_video = record_video
        self.video_path = video_path
        
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -90.8)
        
        self.objects = []
        self.robot = None
        self.video_frames = []

    def add_ground(self):
        ground_id = p.loadURDF("plane.urdf")
        self.objects.append(ground_id)
        return ground_id

    def add_box(self, position, size=[0.1, 0.1, 0.1]):
        col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[1, 0, 0, 1])
        box_id = p.createMultiBody(0.1, col_box_id, vis_box_id, position)
        self.objects.append(box_id)
        return box_id
    
    def build_tower(self, base_position=[0, 0, 0.05], levels=5):
        for i in range(levels): self.add_box([base_position[0] + 0.5, base_position[1] + 0.5, base_position[2] + i * 0.3])
    
    def add_robot_arm(self, position=[0, 0, 0]):
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", position)
        return self.robot
    
    def pick_object(self, object_id):
        if self.robot is None:
            raise ValueError("No robot arm has been loaded.")

        # Define the end effector index (depends on the URDF)
        end_effector_index = 6  # Assuming last joint is the gripper or equivalent
        
        # Get the object position
        object_position, _ = p.getBasePositionAndOrientation(object_id)

        # Adjust pick position slightly above the object
        pre_pick_position = [object_position[0], object_position[1], object_position[2] + 0.2]
        pick_position = [object_position[0], object_position[1], object_position[2] + 0.05]

        # Move the arm above the object
        pre_pick_joint_positions = p.calculateInverseKinematics(self.robot, end_effector_index, pre_pick_position)
        for i, joint_pos in enumerate(pre_pick_joint_positions):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_pos)

        for _ in range(50):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/240.0)

        # Move the arm to the object's position
        pick_joint_positions = p.calculateInverseKinematics(self.robot, end_effector_index, pick_position)
        for i, joint_pos in enumerate(pick_joint_positions):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_pos)

        for _ in range(50):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/240.0)

        # Attach the object to the gripper using a constraint
        self.object_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=end_effector_index,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,  # -1 means base link
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],  # Adjust as needed
            childFramePosition=[0, 0, 0]
        )



    def place_object(self, target_position):
        if self.robot is None:
            raise ValueError("No robot arm has been loaded.")
        
        end_effector_index = 6  # Assuming last joint
        
        place_joint_positions = p.calculateInverseKinematics(self.robot, end_effector_index, target_position)
        for i, joint_pos in enumerate(place_joint_positions):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_pos)
        
        # Step the simulation to move the arm
        for _ in range(50):
            p.stepSimulation()
            if self.gui:time.sleep(1/240.0)

        # Remove the constraint to release the object
        if hasattr(self, 'object_constraint'):
            p.removeConstraint(self.object_constraint)
            del self.object_constraint
    
    def step_simulation(self, steps=1):
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/240.0)
    
    def capture_image(self, width=640, height=480):
        view_matrix = p.computeViewMatrix([1, 1, 1], [0, 0, 0.3], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 10)
        _, _, rgb_img, depth_img, seg_img = p.getCameraImage(width, height, view_matrix, proj_matrix)
        
        rgb_img = np.array(rgb_img, dtype=np.uint8)[:, :, :3]
        depth_img = np.array(depth_img, dtype=np.float32).reshape((height, width))
        seg_img = np.array(seg_img, dtype=np.int32).reshape((height, width))
        
        return rgb_img, depth_img, seg_img

    def save_images(self, path='outputs/images'):
        os.makedirs(path, exist_ok=True)
        rgb, depth, seg = self.capture_image()
        cv2.imwrite(os.path.join(path, 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
        cv2.imwrite(os.path.join(path, 'depth.png'), depth_normalized.astype(np.uint8))
        cv2.imwrite(os.path.join(path, 'segmentation.png'), seg.astype(np.uint8))

    def record_frame(self):
        rgb, _, _ = self.capture_image()
        self.video_frames.append(rgb)

    def save_video(self):
        if not self.video_frames:
            print("No frames recorded!")
            return
        
        h, w, _ = self.video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 30, (w, h))
        
        for frame in self.video_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Video saved to {self.video_path}")

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    sim = PyBulletSimulator(gui=True, record_video=True)
    
    sim.add_ground()
    sim.build_tower(levels=1)
    robot = sim.add_robot_arm()
    
    box = sim.add_box([-0.5, -0.5, 0.3])
    box = sim.objects[-2]
    sim.pick_object(box)    

    sim.place_object([-0.5, .7, 0.4])
    for _ in range(50):
        sim.step_simulation()
        sim.record_frame()
    
    sim.save_images()
    sim.save_video()
    sim.close()
