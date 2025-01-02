import pybullet as p
import yaml
from typing import Tuple, Dict, Any
import numpy as np

class BaseEnv:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_simulation()
        self.robot = None
        self.scene = None
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_simulation(self):
        if self.config['simulation']['render']:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setTimeStep(self.config['simulation']['timestep'])
        p.setGravity(*self.config['physics']['gravity'])
        
    def reset(self):
        if self.robot:
            self.robot.reset()
        if self.scene:
            self.scene.reset_scene()
            
    def step(self, action):
        raise NotImplementedError
        
    def close(self):
        p.disconnect()