'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-11-29 00:36:17
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-11-29 00:36:24
 # @ Description: This file is distributed under the MIT license.
 '''

import numpy as np

from typing import Optional, Tuple, List

__all__ = ["RRTNode", "RRTTree", "smooth_path"]

class RRTNode(object):
    def __init__(self, config, parent = None):
        self.config = config
        self.parent = parent
        self.children = list()
    
    def add_to_children(self, other):
        self.children.append(other)
        return self
    
    def attach_to(self, other):
        self.parent = other
        self.parent.add_to_children(self)
        return self
    
    def backtrace(self, config = True):
        path = list()

        def dfs(x):
            if x.parent is not None:
                dfs(x.parent)
            path.append(x.config if config else x)
        
        try:
            dfs(self)
            return path
        finally:
            del dfs

    @classmethod
    def from_states(cls, states):
        if isinstance(states, list):
            return [cls(s) for s in states]
        else:
            return cls(states)
    
    def __repr__(self):
        return f"RRTNode(config = {self.config}, parent={self.parent})"

def traverse_rrt_bfs(nodes):
    queue = nodes.copy()
    results = list()

    while (len(queue) > 0):
        x = queue[0]
        queue = queue[1:]
        results.append(x)
        for y in x.children:queue.append(y)
    return results

class RRTTree(object):
    """The RRT Tree"""

    def __init__(self, pspace, roots):
        if isinstance(roots, RRTNode):
            roots = [roots]
        else: assert isinstance(roots, list)
    
        self.pspace = pspace
        self.roots = roots
        self.size = len(roots)
    
    def extend(self, parent, child_config):
        child = RRTNode(child_config).attach_to(parent)
        self.size += 1
        return child

    def nearest(self, config, pspace = None):
        if pspace is None:
            pspace = self.pspace
        best_node, best_value = None, np.inf
        for node in traverse_rrt_bfs(self.roots):
            distance = pspace.distance(node.config, config)
            if distance < best_value:
                best_value = distance
                best_node = node
        return best_node

def smooth_path(pspace, path, num_attempts = 100, use_fine_path: bool = False):
    if use_fine_path:
        path = get_smooth_path(pspace, path)

    for i in range(num_attempts):
        if len(path) <= 2:
            break;
        # use the uniform pair sampling method
        a, b = np.random.randint(0, len(path) - 1), np.random.randint(0, len(path) - 1)
        if a > b:
            a, b = b, a + 1
        else:
            b = b + 1
        if use_fine_path:
            success, _, subpath = pspace.try_extend_path(path[a], path[b])
            if success:
                path = path[:a + 1] + subpath[1:] + path[b:]
        else:
            if pspace.validate_path(path[a], path[b]):
                path = path[:a + 1] + path[b:]        
    return path

def get_smooth_path(pspace, path):
    cpath = [path[0]]
    for config in path[1:]:
        cpath.extend(pspace.cspace.gen_path(cpath[-1], config)[1][1:])
    path = cpath
    return path