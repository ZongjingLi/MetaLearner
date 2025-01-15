import pybullet as p
import yaml
from typing import Tuple, Dict, Any, Union
import numpy as np
import os
from copy import deepcopy

class BaseEnv:
    def __init__(self, config: Union[str,Dict]):
        base_config = self._load_base_config()
        task_config = config if isinstance(config, dict) else self._load_config(config)
        self.config = self._merge_configs(base_config, task_config)
        self._setup_simulation()
        self.robot = None
        self.scene = None
        
    def _load_base_config(self) -> Dict:
        base_path = os.path.join(os.path.dirname(__file__), "../configs/base_config.yaml")
        with open(base_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _merge_configs(self, base: Dict, task: Dict) -> Dict:
        merged = deepcopy(base)
        for key, value in task.items():
            if isinstance(value, dict) and key in merged:
                merged[key].update(value)
            else:
                merged[key] = value
        return merged
            
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