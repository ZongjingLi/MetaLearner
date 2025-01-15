from .base_scene import BaseScene
from ..objects.simple_object import Box
import pybullet as p

class PickPlaceScene(BaseScene):
    def _load_scene(self):
        super()._load_scene()
        
        # Load table
        self.table_id = p.loadURDF(
            "table/table.urdf",
            self.config['table_position'],
            useFixedBase=True
        )
        
        # Load objects
        self.objects = {}
        for obj_config in self.config['objects']:
            obj = Box(obj_config)
            self.objects[obj_config['name']] = obj