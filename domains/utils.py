from abc import abstractmethod

class BatchVisualizer:
    @abstractmethod
    def visualize(self, batched_data, save_file = None):
        return 
    
class NotGroundedException(Exception):

    def __init__(self, fn, ground_name, grounding):
        message = f"{fn} should be grounded on {ground_name} but not provided in {grounding.keys()}"
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message