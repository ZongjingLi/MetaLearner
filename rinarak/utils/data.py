'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-24 18:42:10
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-24 18:42:21
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn
from typing import Dict, List

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_img(img):
    if len(img.shape) == 4:
        if not img.shape[1] in [1,3,4]: return img.permute(0,3,1,2)
    if len(img.shape) == 3:
        if not img.shape[0] in [1,3,4]: return img.permute(2,0,1)
def combine_dict_lists(*dicts: Dict, remove_duplicates: bool = False) -> List:
    """Flattens multiple dictionaries with list values into a single list.
    
    Args:
        *dicts: Variable number of dictionaries where values are lists
        remove_duplicates: If True, removes duplicate items from final list
        
    Returns:
        A single list containing all elements from the input dictionaries' lists
        
    Example:
        >>> d1 = {1: ['a', 'b'], 2: ['c']}
        >>> d2 = {3: ['d', 'e'], 4: ['f']}
        >>> combine_dict_lists(d1, d2)
        ['a', 'b', 'c', 'd', 'e', 'f']
    """
    combined = []
    
    # Flatten all dictionary values into a single list
    for d in dicts:
        for value_list in d.values():
            combined.extend(value_list)
    
    combineds = [str(e) for e in combined]
    
    # Remove duplicates if requested while preserving order
    if remove_duplicates:
        combined = list(dict.fromkeys(combined))
        
    return combined