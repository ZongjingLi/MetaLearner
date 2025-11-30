from abc import abstractmethod

class BatchVisualizer:
    @abstractmethod
    def visualize(self, batched_data, save_file = None):
        return 