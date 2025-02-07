# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 01:14:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-07 05:01:23
#import open3d as o3d
import numpy as np
import random
import torch
import os
from torch.utils.data import Dataset, DataLoader

# ----- Directory Setup -----
BASE_DATA_DIR = "data"

# ----- Scene Dataset -----
class SceneDataset(Dataset):
    def __init__(self, name, split, points_per_object=1024):
        self.data_dir = os.path.join(BASE_DATA_DIR, name, split)
        self.points_per_object = points_per_object

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"No data found in '{self.data_dir}'. Run save_data() first.")
        
        self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".npz")]
        if not self.files:
            raise RuntimeError(f"No scene files found in '{self.data_dir}'. Run save_data() first.")

    def __len__(self):
        return len(self.files)

    def _normalize_pointcloud(self, points):
        """Ensure each object's point cloud has exactly `points_per_object` points."""
        n_points = points.shape[0]

        if n_points > self.points_per_object:
            # Randomly sample 1024 points
            idx = np.random.choice(n_points, self.points_per_object, replace=False)
            points = points[idx]
        elif n_points < self.points_per_object:
            # Repeat points to match 1024
            repeat_factor = (self.points_per_object // n_points) + 1
            points = np.tile(points, (repeat_factor, 1))[:self.points_per_object]

        return torch.tensor(points, dtype=torch.float32)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        # Process point clouds to be (n_objects, 1024, 3)
        pointclouds = [self._normalize_pointcloud(obj) for obj in data["input"]]

        contact_matrix = torch.tensor(data["contact"], dtype=torch.float32)
        end_score = torch.tensor(data["end"], dtype=torch.float32)
        
        return {
            "input": pointclouds,  # List of (n_objects, 1024, 3) tensors
            "predicate": {
                "contact": contact_matrix,
                "end": end_score
            }
        }

# ----- Custom Collate Function -----
def scene_collate(batch):
    inputs = [item['input'] for item in batch]
    predicates = [item['predicate'] for item in batch]

    collated_predicates = {key: [pred[key] for pred in predicates] for key in predicates[0]}

    for key in collated_predicates:
        if isinstance(collated_predicates[key][0], torch.Tensor):
            collated_predicates[key] = [collated_predicates[key]]
        elif isinstance(collated_predicates[key][0], (int, float)):
            collated_predicates[key] = torch.tensor(collated_predicates[key], dtype=torch.float32)

    return {
        "input": inputs,
        "predicate": collated_predicates
    }