from typing import Dict, List
import pybullet as p
import pybullet_data

class BaseScene:
    def __init__(self, config: dict):
        self.config = config
        self.objects = {}
        self._load_scene()
        
    def _load_scene(self):
        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        
    def reset_scene(self):
        for obj in self.objects.values():
            obj.reset()
            
    def get_scene_state(self) -> Dict:
        return {
            name: obj.get_state() 
            for name, obj in self.objects.items()
        }
