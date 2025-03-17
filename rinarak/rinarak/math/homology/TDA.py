# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-11 05:13:46
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-12 14:49:39

import math
import torch
import numpy as np
import matplotlib.pyplot as plt

class PersistentHomology:
	def __init__(self, points, eps = 0.1):
		"""
		class of Persistent Homology
		"""
		self.eps = eps
		self.k = 3

		assert points.shape[1] in [2,3], print("spatial dim of TDA must be 2 or 3")
		self.points = points

		spatial_dim = points.shape[1]

	@staticmethod
	def persistent(samples = 4):
		return 0.0

	@staticmethod
	def homology(points, edges):
		return 0.0

	def fit(self, points):
		self.graph = buildGraph(points, self.eps)
		self.ripsComplex = ripsFiltration(self.graph,k = self.k)

	def transform(self, points):
		intervals = 0
		persistence = 0

		return persistence

	def fit_transform(self, points):
		self.fit(points)
		return self.transform(points)


def build_edges(points, threshold, soft = False):
	edges = []
	for i in range(len(points)):
		for j in range(len(points)):
			dist = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
			dist = math.sqrt(dist)
			if i != j and dist < threshold:edges.append([i,j])
	return edges

def homology_test_data(num_points = 100, dim = 2, noise_scale = 0.1):
	params = np.linspace(0, np.pi * 2, num_points)

	x = np.cos(params).reshape([-1,1])
	y = np.sin(params).reshape([-1,1])

	x += (np.random.random(x.shape)-0.5) * noise_scale
	y += (np.random.random(y.shape)-0.5) * noise_scale

	return np.concatenate([x.transpose(1,0),y.transpose(1,0)]).transpose(1,0)

if __name__ == "__main__":

	test_points = homology_test_data(num_points = 50)
	print(test_points.shape)

	tda = PersistentHomology(points = test_points)

	test_edges = build_edges(points = test_points, threshold = .3)

	import matplotlib.pyplot as plt
	count = 100
	eps_history = []
	homology_history = []

	for itr in range(count):
		plt.figure("Persistent Homology",figsize=[11,5])
		plt.subplot(1,2,1)
		plt.cla()
		plt.scatter(test_points[:,0], test_points[:,1])
		test_edges = build_edges(points = test_points, threshold = 1.0 * (itr+1)/count)
		plt.text(0.0,0.2,"eps:"+str((itr+1)/count))
		for ij in test_edges: 
			i, j = ij
			plt.plot([test_points[i][0], test_points[j][0]],[test_points[i][1], test_points[j][1]], color = "grey")
		homology_history.append(PersistentHomology.homology(test_points, test_edges))
		eps_history.append(1.0 * (itr+1)/count)

		plt.subplot(1,2,2)
		plt.cla()
		plt.plot(eps_history, homology_history)
		plt.pause(0.01)

	plt.show()