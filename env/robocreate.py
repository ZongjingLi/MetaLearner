from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
import mujoco
import numpy as np
import cv2
import os
from mujoco_viewer import MujocoViewer

# Create the world and robot
mujoco_robot = Panda()
world = MujocoWorldBase()

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0.0, 0.0])
world.merge(mujoco_arena)

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

# Get model and create data
model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)

# Create renderer and viewer
renderer = mujoco.Renderer(model, height=480, width=640)
viewer = MujocoViewer(model, data)

# Create output directory for frames
os.makedirs("outputs", exist_ok=True)

def get_segmentation_mask():
    # Get segmentation mask
    segmentation = renderer.render(segmentid=True)
    # Convert to more usable format
    mask = np.zeros_like(segmentation)
    # Create mask for sphere (you may need to adjust the ID based on your scene)
    mask[segmentation == 3] = 1  # Sphere usually has ID 3
    return mask

def apply_control(data, target_joint_pos):
    """Apply position control to reach target joint positions"""
    kp = 100  # Position gain
    # Get current joint positions
    current_pos = data.qpos[0:7]  # First 7 DOF for Panda
    # Calculate position error
    pos_error = target_joint_pos - current_pos
    # Apply control
    data.ctrl[0:7] = kp * pos_error

# Example trajectory - moving joints to specific positions
target_positions = [
    np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]),  # Home position
    np.array([0.5, 0, 0, -2.0, 0, 1.0, 0]),  # Reaching position
]

frame_idx = 0
for t in range(1000):  # Simulation steps
    # Choose target position based on time
    target_idx = (t // 200) % len(target_positions)
    target_pos = target_positions[target_idx]
    
    # Apply control
    apply_control(data, target_pos)
    
    # Step simulation
    mujoco.mj_step(model, data)
    
    # Render every 10 steps
    if t % 10 == 0:
        # Render RGB image
        renderer.update_scene(data)
        rgb_img = renderer.render()
        
        # Get segmentation mask
        seg_mask = get_segmentation_mask()
        
        # Save renders
        cv2.imwrite(f"outputs/frame_{frame_idx:04d}.rgb.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"outputs/frame_{frame_idx:04d}.mask.png", (seg_mask * 255).astype(np.uint8))
        
        # Update viewer
        viewer.render()
        
        frame_idx += 1

print("Simulation complete. Renders saved in 'outputs' directory.")
viewer.close()  # Close the viewer window

# Optional: Create video from frames
def create_video():
    import cv2
    frame = cv2.imread("outputs/frame_0000.rgb.png")
    height, width, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('simulation.mp4', fourcc, 30.0, (width, height))
    
    frame_files = sorted([f for f in os.listdir("outputs") if f.endswith(".rgb.png")])
    for frame_file in frame_files:
        frame = cv2.imread(f"outputs/{frame_file}")
        out.write(frame)
    
    out.release()
    print("Video saved as 'simulation.mp4'")

create_video()