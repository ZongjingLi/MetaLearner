
import torch
from torch.utils.data import Dataset, DataLoader


def collate_tensors_with_mask(tensor_list):
    """
    Collate a list of tensors with shape (n_i, ...) into a batch of shape (b, n_max, ...),
    along with a mask tensor of shape (b, n_max) indicating valid entries.

    Args:
        tensor_list (list of torch.Tensor): List of tensors with shape (n_i, ...).

    Returns:
        torch.Tensor: Collated tensor of shape (b, n_max, ...).
        torch.Tensor: Mask tensor of shape (b, n_max) with 1 for valid entries and 0 otherwise.
    """
    b = len(tensor_list)
    extra_dims = tensor_list[0].shape[1:]  # Assuming all tensors have the same extra dimensions
    n_max = max(tensor.shape[0] for tensor in tensor_list)

    # Initialize the batch tensor and the mask
    batch_shape = (b, n_max) + extra_dims
    batch_tensor = torch.zeros(batch_shape)
    mask = torch.zeros((b, n_max))

    # Fill in the batch tensor and mask
    for i, tensor in enumerate(tensor_list):
        n_i = tensor.shape[0]
        batch_tensor[i, :n_i, ...] = tensor
        mask[i, :n_i] = 1.0

    return batch_tensor, mask

# Example usage
tensor_list = [torch.randn(3, 4, 5), torch.randn(5, 4, 5), torch.randn(2, 4, 5)]
batch_tensor, mask = collate_tensors_with_mask(tensor_list)

#print("Batch Tensor:\n", batch_tensor)
#print("Mask:\n", mask)

from torch.utils.data import DataLoader
from helchriss.utils.data import ListDataset
from helchriss.utils.data import GroundBaseDataset
from helchriss.dsl.dsl_values import Value
from typing import List, Dict, Mapping, Union, Any


class SceneGroundingDataset(ListDataset):
    def __init__(self, queries : List[str], answers : List[Union[Value, Any]], groundings : None):
        query_size = len(queries)
        if groundings is None: groundings = [{} for _ in range(query_size)]
        data = [{"query":queries[i], "answer":answers[i], "grounding": groundings[i]} for i in range(query_size)]
        super().__init__(data)

    def shuffle(self):
        import random
        random.shuffle(self.data)



