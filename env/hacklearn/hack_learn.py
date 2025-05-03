import gym
import minihack
from minihack import LevelGenerator
from minihack.envs import register
from nle.nethack import Command

# Create a level generator
lg = LevelGenerator(map="""
----------------
|.............|
|...........G.|
|.@............
|.............|
|.............S
----------------
""")

# Add some objects to the level
lg.add_object("gold piece", "%")
lg.add_object("apple", "%")
lg.add_object("dagger", "%")

# Add a monster
lg.add_monster("goblin", "random")

# Set the goal - reach the stairs (S)
lg.add_goal_pos((15, 5))

# Register the environment
register(
    id="MiniHack-Custom-v0",
    entry_point="minihack.envs.room:MiniHackRoom",
    kwargs={
        "des_file": lg.get_des(),
        "max_episode_steps": 100,
       
    },
)

# Example of how to use the environment
def main():
    env = gym.make("MiniHack-Custom-v0")
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # For this example, let's just take random actions
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        
        if done:
            print(f"Episode finished with reward {total_reward}")
            print(f"Goal reached: {info.get('goal_reached', False)}")
            break
    
    env.close()

if __name__ == "__main__":
    main()