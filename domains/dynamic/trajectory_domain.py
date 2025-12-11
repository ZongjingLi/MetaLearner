'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-12-10 14:36:31
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-12-10 16:26:24

'''

'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-12-10 14:38:16
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-12-10 14:38:27
'''
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

integer_domain_str = """
(domain :: Trajectory)
(def type
    point - Embedding[point2d, 2]
    line - Embedding[segment, 4] ;; directed segment
    circle - Embedding[circle, 3] ;; circle encoded by (x, y, r)
)
(def function
    start (x : line) : point := by pass
    end   (x : line) : point := by pass
    on_line (x : point, y : line) : boolean := by pass
    length (x : line) : float := by pass

    connect_segment (x y : point) : line := by pass

    on_radius (x : point, y : circle) : boolean := by pass
    inside    (x : point, y : circle) : boolean := by pass
    outside   (x : point, y : circle) : boolean := by pass

    contain (x : circle, y : circle) : boolean := by pass
)

"""

trajectory_domain = load_domain_string(integer_domain_str)

class TrajectoryExecutor(CentralExecutor):

    def start(self, x): return x[:2]

    def end(self, x): return x[2:]

euclid_executor = TrajectoryExecutor(trajectory_domain)
