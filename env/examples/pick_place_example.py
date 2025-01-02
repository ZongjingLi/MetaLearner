import yaml
from env.core.base_env import BaseEnv
from env.robots.ur5 import UR5
from env.scenes.pick_place import PickPlaceScene

def main():
    # Load config
    with open('env/configs/pick_place_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = BaseEnv(config)
    
    # Add robot and scene
    env.robot = UR5(config['robot'])
    env.scene = PickPlaceScene(config['scene'])
    
    # Run simulation
    env.reset()
    for _ in range(config['simulation']['max_steps']):
        env.step(None)  # Add your control logic here
        p.stepSimulation()
    
    env.close()

if __name__ == "__main__":
    main()