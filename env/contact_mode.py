import pybullet as p
import pybullet_data
import time
import numpy as np

# Initialize PyBullet
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Set up camera view to match the image
p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# Create plane with dark color
planeId = p.loadURDF("plane.urdf")
p.changeVisualShape(planeId, -1, rgbaColor=[0.2, 0.2, 0.2, 1])

# Add yellow triangle background
# We'll create this using a custom visual shape
vertices = [
    [0.5, 0, 0.5],
    [0, 0, 1],
    [-0.5, 0, 0.5]
]
indices = [0, 1, 2]


# Create small red cube
cube_size = 0.05
red_cube = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[cube_size/2, cube_size/2, cube_size/2],
    rgbaColor=[1, 0.4, 0.4, 1]
)
red_cube_collision = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[cube_size/2, cube_size/2, cube_size/2]
)
red_cube_body = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=red_cube_collision,
    baseVisualShapeIndex=red_cube,
    basePosition=[0.3, 0, cube_size/2]
)

# Load robotic arm (UR5 with gripper)
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
# Adjust robot position
p.resetBasePositionAndOrientation(
    robot_id,
    [0, 0, 0],
    p.getQuaternionFromEuler([0, 0, 0])
)

# Add a yellow tool/gripper to the end effector
tool_shape = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[0.02, 0.1, 0.01],
    rgbaColor=[1, 0.8, 0, 1]
)
tool_collision = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[0.02, 0.1, 0.01]
)
# Attach tool to the end effector (link 6)
end_effector_link = 6
tool_body = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=tool_collision,
    baseVisualShapeIndex=tool_shape,
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)
constraint = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=end_effector_link,
    childBodyUniqueId=tool_body,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)

# Set initial robot pose similar to the image
target_joints = [0, 0.3, 0, -1.5, 0, 0.5, 0]
for i in range(p.getNumJoints(robot_id)):
    if i < len(target_joints):
        p.resetJointState(robot_id, i, target_joints[i])

# Run simulation
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()