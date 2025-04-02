# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2024-10-14 13:23:40
# @Last Modified by:   Melkor
# @Last Modified time: 2024-10-16 06:20:10
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple

class Swissroll(Dataset):
   def __init__(self, tmin, tmax, N):
       t = tmin + torch.linspace(0, 1, N) * tmax
       self.vals = torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T
       self.vals = self.vals.unsqueeze(1)

   def __len__(self):
       return len(self.vals)
   
   def __getitem__(self, i):
       return {
           "data": self.vals[i].reshape([1,2]),
           "cond": {
               "edges": [(0, "online")]
           }
       }



def collate_graph_batch(batch):
    # Unpack batch efficiently
    batch_data = {
        "data": [],
        "cond": {"edges": []}
    }
    
    offset = 0
    for sample in batch:
        #(sample)
        state = sample["data"]
        edges = sample["cond"]["edges"]
        
        # Add state
        batch_data["data"].append(state)
        
        # Add edges with offset
        for e in edges:
            translated_edge = [e[k] + offset for k in range(len(e) - 1)]
            translated_edge.append(e[-1])
            batch_data["cond"]["edges"].append(translated_edge)
        offset += len(state)
    
    # Stack all states
    batch_data["data"] = torch.cat(batch_data["data"], dim=0)
    #print(batch_data["cond"]["edges"])
    
    return batch_data

@dataclass
class GeometricObject:
	"""the geometric object with point cloud as shape"""
	pointcloud : torch.FloatTensor
	pose : torch.FloatTensor

@dataclass
class GeometricGraph:
	"""set the geometric shape using point cloud and the constraint graph"""
	objects : List[GeometricObject]
	edges : List[Tuple[int, int, str]]

	def add_node(self, node : GeometricObject):
		self.objects.append(node)

	def get_transformed_pointclouds(self):
		pointclouds = []
		for i, obj in enumerate(self.objects):
			offset = obj.pose[:3]
			rot = obj.pose[3:6]
			scale = obj.pose[-1]
			_, n = obj.pointcloud.shape

			pts = obj.pointcloud* scale
			pts = pts + offset[..., None].repeat(1,n)
			

			pointclouds.append(pts)
		return pointclouds

	def get_pointcloud(self):
		return torch.cat([
			obj.pointcloud[None,...] for obj in self.objects
			], dim = 0)

	def get_control_variable(self):
		return torch.cat([
			obj.pose[None,...] for obj in self.objects
			], dim = 0)

	def update_control_variable(self, x_vars):
		n, _ = x_vars.shape
		for i in range(n):
			self.objects[i].pose = x_vars[i]

	def setup(self, flag = True):
		for obj in self.objects:
			obj.pose.requires_grad = flag


import plotly.graph_objs as go
import plotly.express as px

def plot_geom_graph(graph):
	r = 10.0
	pointclouds = graph.get_transformed_pointclouds()	
	traces = [go.Scatter3d(x=pts[0], y=pts[1], z=pts[2], mode='markers', marker=dict(size=3, colorscale='Viridis')) for pts in pointclouds]
	layout = go.Layout(
		scene=dict(
			xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
			xaxis = dict(nticks=4, range=[-r,r]),
			yaxis = dict(nticks=4, range=[-r,r]),
			zaxis = dict(nticks=4, range=[-r,r])  ))
	for edge in graph.edges:
		start_obj = graph.objects[edge[0]]
		end_obj = graph.objects[edge[1]]

		start_pos = start_obj.pose[:3]
		end_pos = end_obj.pose[:3]

		x = torch.linspace(start_pos[0], end_pos[0], 10)
		y = torch.linspace(start_pos[1], end_pos[1], 10)
		z = torch.linspace(start_pos[2], end_pos[2], 10)

		traces.append(go.Scatter3d(x=x, y=y,z=z, mode='lines', name = edge[2]))

	fig = go.Figure(data=traces, layout=layout)
	fig.show()


class SpatialCCSPDataset(Dataset):
	def __init__(self):
		super().__init__()
