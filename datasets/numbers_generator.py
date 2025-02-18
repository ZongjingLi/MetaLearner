# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-16 19:30:54
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-17 09:13:52
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NumbersDataset(Dataset):
	def __init__(self, nums = 12, split = "train", p = 0.4):
		super().__init__()
		self.portion = p
		self.nums = [i for i in range(nums)]
		self.mask = None #TODO: genetate a nxn binary tensor that have p percent element 1 others 0

	def __len__(self): return len(self.nums)

	def __getitem__(self, idx): 
		tokens = self.nums
		return {
            "input": tokens,  # List of (n_objects, 1024, 3) tensors
            "predicate": {
                "contact": contact_matrix,
                "end": end_score
            }
            "effective" : {
            	self.mask
            }
        }
