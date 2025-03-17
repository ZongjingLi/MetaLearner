import pybullet as p
import sys
from abc import abstractmethod

class BaseEnv(object):
    """a base simulator"""
    def __init__(self, save_path = None):
        self.save_path =  save_path

    @abstractmethod
    def reset(self):
        """
        Reset the simulation to its initial state.
        
        This method must be implemented by subclasses to define
        how the simulation environment is set up or reset.
        """
        pass

    @abstractmethod
    def step(self, action=None):
        """
        Step the simulation forward.
        
        Args:
            action: Action to take in the environment (if applicable)
            
        Returns:
            Observation after taking the step
        """
        pass