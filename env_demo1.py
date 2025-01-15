import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
import os
from PIL import Image

class BlockStackingEnv:
    def __init__(self, render=True, width=512, height=512):
        # Initialize PyBullet
        self.width = width
        self.height = height
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Camera setup
        self.camera_distance = 1.0
        self.camera_target = [0, 0, 0]
        self.camera_yaw = 45
        self.camera_pitch = -30
        
        # Block dimensions
        self.block_size = 0.05  # 5cm cubes
        
        # Initialize workspace
        self._setup_workspace()
        
    def _setup_workspace(self):
        """Setup the workspace with initial blocks"""
        self.blocks = []
        colors = [
            [1, 0, 0, 1],  # Red
            [1, 1, 0, 1],  # Yellow
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1]   # Blue
        ]
        
        # Create collision shape for blocks
        self.block_shape = p.createCollisionShape(p.GEOM_BOX, 
                                                halfExtents=[self.block_size/2]*3)
        
        # Create visual shape for colored blocks
        for i, color in enumerate(colors):
            visual_shape = p.createVisualShape(p.GEOM_BOX,
                                             halfExtents=[self.block_size/2]*3,
                                             rgbaColor=color)
            
            block_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=self.block_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, i*0.1 - 0.15, self.block_size/2]
            )
            self.blocks.append(block_id)
            
    def get_camera_image(self, mask=False):
        """Get RGB image and segmentation mask from camera"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.width)/self.height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image
        width, height, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(rgb)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        if mask:
            return rgb_array, np.array(seg)
        return rgb_array
        
    def save_images(self, output_dir="output", name_prefix="frame"):
        """Save RGB image and segmentation mask"""
        os.makedirs(output_dir, exist_ok=True)
        
        rgb, mask = self.get_camera_image(mask=True)
        
        # Save RGB image
        rgb_path = os.path.join(output_dir, f"{name_prefix}_rgb.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # Save segmentation mask
        mask_path = os.path.join(output_dir, f"{name_prefix}_mask.png")
        cv2.imwrite(mask_path, mask.astype(np.uint8))
        
    def stack_blocks(self):
        """Perform block stacking sequence"""
        for i, block in enumerate(self.blocks):
            if i > 0:
                target_pos = [0, 0, i * self.block_size + self.block_size/2]
                for _ in range(100):  # Interpolate movement
                    current_pos, _ = p.getBasePositionAndOrientation(block)
                    new_pos = [0.99 * c + 0.01 * t for c, t in zip(current_pos, target_pos)]
                    p.resetBasePositionAndOrientation(block, new_pos, [0, 0, 0, 1])
                    p.stepSimulation()
                    time.sleep(0.01)
                    
    def reset(self):
        """Reset the environment"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self._setup_workspace()
        
    def close(self):
        """Close the environment"""
        p.disconnect()

def main():
    # Create and initialize environment
    env = BlockStackingEnv(render=True)
    
    # Create output directory
    output_dir = "block_stacking_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture initial state
    env.save_images(output_dir=output_dir, name_prefix="initial")
    
    # Stack blocks and capture intermediate states
    env.stack_blocks()
    env.save_images(output_dir=output_dir, name_prefix="stacked")
    
    # Let physics stabilize
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Capture final state
    env.save_images(output_dir=output_dir, name_prefix="final")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()