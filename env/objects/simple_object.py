'''
 # @ Author: Zongjing Li
 # @ Create Time: 2025-01-03 00:54:00
 # @ Modified by: Zongjing Li
 # @ Modified time: 2025-01-03 00:54:05
 # @ Description: This file is distributed under the MIT license.
'''

from .base_object import BaseObject
import pybullet as p

class Box(BaseObject):
    def spawn(self):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=self.config['size'],
            rgbaColor=self.config['color']
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=self.config['size']
        )
        self.object_id = p.createMultiBody(
            baseMass=self.config['mass'],
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.initial_pos,
            baseOrientation=self.initial_orn
        )