import numpy as np
import robosuite as suite

for i in range(1,11):
    for j in range(1,11):
        print(i  * j)

# 创建环境实例
env = suite.make(
    env_name="Lift", # 尝试其他任务，比如："Stack" and "Door"
    robots="Jaco",  # 尝试其他机器人模型，比如："Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # 执行随机动作
    obs, reward, done, info = env.step(action)  # 执行动作得到观测值，奖励值等
    env.render()  # render on display

