'''
 # @ Author: Zongjing Li
 # @ Create Time: 2025-01-03 00:53:41
 # @ Modified by: Zongjing Li
 # @ Modified time: 2025-01-03 00:53:45
 # @ Description: This file is distributed under the MIT license.
'''
from typing import List, Tuple
import pybullet as p

class BaseObject:
    def __init__(self, config: dict):
        self.config = config
        self.object_id = None
        self.initial_pos = config['position']
        self.initial_orn = config['orientation']
        self.spawn()
        
    def spawn(self):
        raise NotImplementedError
        
    def reset(self):
        p.resetBasePositionAndOrientation(
            self.object_id,
            self.initial_pos,
            self.initial_orn
        )
        
    def get_state(self) -> Tuple[List[float], List[float]]:
        return p.getBasePositionAndOrientation(self.object_id)
        
    def remove(self):
        if self.object_id is not None:
            p.removeBody(self.object_id)
            self.object_id = None