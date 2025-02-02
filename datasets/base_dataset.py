
import torch
from torch.utils.data import Dataset, DataLoader

import torch

import torch

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

print("Batch Tensor:\n", batch_tensor)
print("Mask:\n", mask)


#tensor_list = [torch.randn(3, 4), torch.randn(5, 4), torch.randn(2, 4)]
#batch_tensor, mask = collate_tensors_with_mask(tensor_list)


class GroundTruthDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.perception_module_name = None

    def __len__(self):
        return []

    def __getitem__(self, idx):
        """ each item will be an a diction that contains, input, modal, ground truth, effective mask
        Returns:
            input: anything that an perception module can take as inoput
            modal: a object centric detection module that take
        """
        return idx
    
class QuestionAnsweringDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self): return 0

    def __getitem__(self, idx):
        return 