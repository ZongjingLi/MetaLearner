import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    print(obs["object-state"])
    env.render()  # render on display
    """
    ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 
     'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site',
     'robot0_gripper_qpos', 'robot0_gripper_qvel',
     'cube_pos', 'cube_quat',
     'gripper_to_cube_pos', 'robot0_proprio-state', 'object-state']"""