# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-11 20:50:51
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-13 11:05:25

import ai2thor.controller
import numpy as np
import random
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

@dataclass
class EnvironmentConfig:
    """Configuration for AI2-THOR environment."""
    scene: str = "FloorPlan10"
    width: int = 900
    height: int = 900
    render_depth: bool = True
    render_instance_segmentation: bool = True
    grid_size: float = 0.25

class ThorObject:
    """Wrapper class for AI2-THOR objects."""
    def __init__(self, metadata: Dict[str, Any]):
        self.name = metadata["name"]
        self.position = metadata["position"]
        self.rotation = metadata["rotation"]
        self.object_id = metadata.get("objectId")
        self.pickupable = metadata.get("pickupable", False)
        self.moveable = metadata.get("moveable", False)
        self.properties = metadata

    def __str__(self):
        return f"{self.name} at position {self.position}"

class BaseEnvironment(ABC):
    """Abstract base class for AI2-THOR environments."""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self._setup_controller()
        self.objects: List[ThorObject] = []
        
    def _setup_controller(self):
        """Initialize AI2-THOR controller with given configuration."""
        # Clean up any existing FIFO pipes
        fifo_path = "/tmp/thor.fifo"
        if os.path.exists(fifo_path):
            os.remove(fifo_path)
            
        self.controller = ai2thor.controller.Controller(
            scene=self.config.scene,
            renderDepthImage=self.config.render_depth,
            renderInstanceSegmentation=self.config.render_instance_segmentation,
            width=self.config.width,
            height=self.config.height,
        )
        
    def reset(self):
        """Reset environment to initial state."""
        self.controller.reset(self.config.scene)
        self.objects.clear()
        return self._get_observation()
    
    def step(self, action: str, **action_args) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action in environment.
        
        Returns:
            Tuple containing (observation, reward, done, info)
        """
        event = self.controller.step(action=action, **action_args)
        observation = self._get_observation()
        reward = self._compute_reward(event)
        done = self._check_done(event)
        info = self._get_info(event)
        return observation, reward, done, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation of environment state."""
        event = self.controller.last_event
        return {
            'frame': event.frame,
            'depth': event.depth_frame if self.config.render_depth else None,
            'segmentation': event.instance_segmentation_frame if self.config.render_instance_segmentation else None,
            'objects': [ThorObject(obj) for obj in event.metadata['objects']]
        }
    
    @abstractmethod
    def _compute_reward(self, event) -> float:
        """Compute reward for current state."""
        pass
    
    @abstractmethod
    def _check_done(self, event) -> bool:
        """Check if episode is done."""
        pass
    
    def _get_info(self, event) -> Dict[str, Any]:
        """Get additional information about current state."""
        return {
            'last_action_success': event.metadata['lastActionSuccess'],
            'last_action': event.metadata['lastAction'],
            'agent': event.metadata['agent']
        }
    
    def close(self):
        """Clean up environment resources."""
        self.controller.stop()

class BlockWorld(BaseEnvironment):
    """Custom environment implementing a block world in AI2-THOR."""
    
    def __init__(self, config: EnvironmentConfig, num_blocks: int = 3):
        super().__init__(config)
        self.num_blocks = num_blocks
        self.block_ids = []
        
    def reset(self):
        """Reset block world environment and spawn blocks."""
        observation = super().reset()
        self._spawn_blocks()
        return self._get_observation()
    
    def _spawn_blocks(self):
        """Spawn blocks in random positions."""
        self.block_ids.clear()
        
        for _ in range(self.num_blocks):
            event = self.controller.step(
                action='CreateObject',
                objectType='Cube',  # Using Cube instead of Vase for block world
                position={
                    'x': random.uniform(-1, 1),
                    'y': 0.9,
                    'z': random.uniform(-1, 1)
                },
                rotation={
                    'x': 0,
                    'y': random.uniform(0, 360),
                    'z': 0
                }
            )
            
            if event.metadata['lastActionSuccess']:
                new_object = ThorObject(event.metadata['objects'][-1])
                self.objects.append(new_object)
                self.block_ids.append(new_object.object_id)
    
    def _compute_reward(self, event) -> float:
        """Compute reward based on block positions and agent interaction."""
        # Example reward function - customize based on your needs
        reward = 0
        if event.metadata['lastActionSuccess']:
            if event.metadata['lastAction'] == 'PickupObject':
                reward += 1
            elif event.metadata['lastAction'] == 'PutObject':
                # Check if block is placed on target location
                reward += 2
        return reward
    
    def _check_done(self, event) -> bool:
        """Check if episode is done based on task completion."""
        # Example done condition - customize based on your needs
        return False  # For continuous operation

def main():
    """Demo usage of the block world environment."""
    config = EnvironmentConfig(
        scene="FloorPlan10",
        width=900,
        height=900
    )
    
    env = BlockWorld(config, num_blocks=3)
    observation = env.reset()
    
    try:
        # Demo loop
        for _ in range(50):
            action = random.choice(['MoveAhead', 'RotateLeft', 'RotateRight', 'MoveBack'])
            obs, reward, done, info = env.step(action)
            
            # Display current frame
            #plt.imshow(obs['frame'])
            #plt.pause(0.01)
            time.sleep(0.1)
            #plt.clf()
            
            if done:
                break
                
    finally:
        env.close()

if __name__ == "__main__":
    pass
    #main()


def interactive_mode():
    """Interactive mode for controlling the AI2-THOR environment."""
    config = EnvironmentConfig(
        scene="FloorPlan10",
        width=900,
        height=900
    )

    env = BlockWorld(config, num_blocks=3)
    observation = env.reset()
    
    try:
        print("\nWelcome to the AI2-THOR Interactive Mode!")
        print("Type an action (e.g., MoveAhead, RotateLeft, PickupObject) or 'exit' to quit.")
        
        while True:
            # Show the current frame (optional)
            
            # Get user input
            action = input("\nEnter action: ").strip()

            # Exit condition
            if action.lower() in ["exit", "quit"]:
                print("Exiting interactive mode...")
                break

            # Execute the action
            obs, reward, done, info = env.step(action)
            observation = obs  # Update the observation
            
            # Print feedback
            print(f"Action: {action} | Reward: {reward} | Done: {done}")
            print(f"Objects in view: {[obj.name for obj in observation['objects']]}")
            print(f"Last action success: {info['last_action_success']}")

            if done:
                print("Episode finished. Resetting environment...")
                observation = env.reset()

    finally:
        env.close()

if __name__ == "__main__":
    interactive_mode()
