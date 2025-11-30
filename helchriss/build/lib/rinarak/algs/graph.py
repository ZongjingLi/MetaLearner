import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from .ds import *

class Graph:
    def __init__(self,nodes, edges, node_attrs = None, edge_attrs = None):
        self.nodes = nodes
        self.edges = edges
        
        # [Node and Edge Attributes]
        self.node_attributes = node_attrs
        self.edge_attributes = edge_attrs

        self.inf = 1e9

        # [Metric Space of Graph]
        self.metric = None
        self.dist   = None
    
    def metric_available(self):
        return self.metric is not None or self.dist is not None
    
    def render(self, color_map = None):
        if color_map is not None:
            pass
        return 

    def generate_layout(self):
        if self.metric is not None:
            return 
        return     

    def bellman_ford(self, start, to = None, get_path = True):
        inf = self.inf
        distances = {}
        for node in self.nodes: distances[node] = inf
        distances[start] = 0.0
        predecor = {}
        for r in range(len(self.nodes) - 1):
            for edge in self.edges:
                u,v = edge
                #if v==start:print(v,distances[v])
                if self.metric is not None:
                    #print(self.node_attributes,u,v)
                    attr_u = self.node_attributes[u]
                    attr_v = self.node_attributes[v]
                    dist_uv = self.metric(attr_u,attr_v)
                    if distances[u]+dist_uv< distances[v]:predecor[v] = u
                distances[v] = min(distances[u]+dist_uv, distances[v])
        if to is not None:
            path = []
            curr_node = to
            while True:
                if curr_node in predecor:
                    pred_node = predecor[curr_node]
                    path.append([pred_node, curr_node])
                    curr_node = pred_node
                else: break  
            path = list(reversed(path))
            return distances, path
        return distances
    
    def has_negative_cycle(self):
        return 

    def astar_search(self,goal,heurstics):
        return 
    
def manhattan_disance(x,y):
    x = x["pos"]
    y = y["pos"]
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

class GridGraph(Graph):
    def __init__(self, width, height, \
                dtype = "manhattan",):
        self.width = width
        self.height = height

        nodes = [(i,j) for i in range(width) for j in range(height)]
        if dtype == "manhattan":
            dirs =[(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]
        else:
            assert False, print("Unknown Distance Method")
        edges = [
            [(i,j),(i+dir[0], j+dir[1])] \
            for dir in dirs for j in range(1,height-1)for i in range(1,width-1)]
        for i in range(1,width-1):
            for j in range(1,height-1):
                for dir in dirs:edges.append([(i+dir[0], j+dir[1]), (i,j)])

        
        node_attrs = {}
        for key in nodes: node_attrs[key] = {"pos":key}
        edge_attrs = edges

        # [Node and Edge Attributes]
        super().__init__(nodes, edges, node_attrs, edge_attrs)

        if dtype == "manhattan":
            self.metric = manhattan_disance

    def render(self, sequence = None, cmap = None):
        color_map = np.zeros([self.width, self.height])
        print(self.node_attributes[(0,0)])
        if cmap is not None or "height" in self.node_attributes[(0,0)]:
            for name in self.node_attributes:
                node_attr = self.node_attributes[name]
                pos = node_attr["pos"]
                if color_map is not None:
                    color_map[pos[0]][pos[1]] = cmap(node_attr)
                else: color_map[pos[0]][pos[1]] = node_attr["height"]
        for edge in sequence:
            u,v = edge
            plt.plot([v[1], u[1]], [v[0],u[0]], c="red")
            #color_map[u[0]][u[1]] = 0.5
            #color_map[v[0]][v[1]] = 0.5
        return color_map