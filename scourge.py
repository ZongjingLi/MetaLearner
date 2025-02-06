# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 05:45:31
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-06 06:03:45
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.scene_dataset import SceneDataset, scene_collate

from tqdm import tqdm
dataset = SceneDataset("contact_experiment", "train")
loader = DataLoader(dataset, batch_size = 5, collate_fn = scene_collate)

from rinarak.dklearn.nn.pnn import PointNetfeat
from core.encoders.pointcloud_encoder import PointCloudEncoder, PointCloudRelationEncoder
model = PointNetfeat()
#model = PointCloudRelationEncoder()

model(torch.randn([5,2024,3])) # -> [5,1024]

for sample in tqdm(loader):
	for scene in sample["input"]:
		scene = torch.stack(scene)
		print(scene.shape)
		state = model(scene)
		print(state.shape)