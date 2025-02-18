# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-12 17:58:31
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-13 03:49:36
import ai2thor.controller
import numpy as np
import random
import time
from ai2thor.controller import Controller

class EmptyRoomEnv:
    def __init__(self, width=900, height=900):
        # Initialize controller with an empty room
        self.controller = Controller(
            scene="FloorPlan10",  # Using a simple training room
            width=width,
            height=height,
            renderDepthImage=True
        )
        self._clear_room()

        
    def _clear_room(self):
        """Remove all existing objects from the room"""
        # First get current state
        event = self.controller.step(action="MoveBack")
        
        # Collect all object IDs that need to be removed
        objects_to_remove = []
        for obj in event.metadata["objects"]:
            print(f"Found object: {obj['name']}, Pickupable: {obj['pickupable']}, Moveable: {obj['moveable']}")
            if obj["pickupable"] or obj["moveable"]:
                objects_to_remove.append(obj["objectId"])
        
        print(f"Found {len(objects_to_remove)} objects to remove")
        
        # Now remove all collected objects
        for obj_id in objects_to_remove:
        	removal_event = self.controller.step(
                action="RemoveFromScene",
                objectId=obj_id
            )
        	if removal_event.metadata["lastActionSuccess"]:
        		print(f"Successfully removed object {obj_id}")
        	else:
        		print(f"Failed to remove object {obj_id}")

        # Verify room is clear
        final_event = self.controller.step(action="Pass")
        remaining_objects = [obj["name"] for obj in final_event.metadata["objects"] 
                           if obj["pickupable"] or obj["moveable"]]
        if remaining_objects:
            print(f"Warning: These objects could not be removed: {remaining_objects}")

    def spawn_blocks(self, num_blocks=5):
        """Spawn blocks in random positions around the room"""
        # Get the room bounds to place blocks within valid areas
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        
        for _ in range(num_blocks):
            # Choose a random position from reachable positions
            pos = random.choice(reachable_positions)
            #print("Pos:",pos)
            
            # Add some randomness to the position
            pos_with_noise = {
                'x': pos['x'] + random.uniform(-0.2, 0.2),
                'y': 0.5,  # Fixed height off the ground
                'z': pos['z'] + random.uniform(-0.2, 0.2)
            }
            
            # Create a cube
            self.controller.step(
                action='CreateObject',
                objectType='Vase',
                position=pos_with_noise,
                rotation={'x': 0, 'y': random.uniform(0, 360), 'z': 0},
                forceAction=True
            )

    def get_observation(self):
        """Get the current observation"""
        event = self.controller.last_event
        return {
            'rgb': event.frame,
            'depth': event.depth_frame,
            'objects': event.metadata['objects']
        }

    def close(self):
        """Clean up resources"""
        self.controller.stop()

def main():
    # Create environment
    env = EmptyRoomEnv()
    
    try:
        # Spawn blocks
        env.spawn_blocks(num_blocks=5)
        
        # Move around to view the environment
        for _ in range(20):
            action = random.choice([
                'MoveAhead', 
                'MoveBack',
                'RotateRight', 
                'RotateLeft'
            ])
            
            event = env.controller.step(action=action)
            
            # Optional: print positions of all cubes
            for obj in event.metadata['objects']:
                if obj['objectType'] == 'Cube':
                    print(f"Cube position: {obj['position']}")
            
            time.sleep(0.1)
            
    finally:
        env.close()

if __name__ == "__main__":
    main()