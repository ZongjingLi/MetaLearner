# @Author: Yiqi Sun
# @Create Time: 2025-12-10 14:38:16
# @Modified by: Yiqi Sun
# @Modified time: 2025-12-10 14:38:27

import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

integer_domain_str = """
(domain :: Euclid)
(def type
    point - Embedding[point2d, 2]
    line - Embedding[segment, 4] ;; directed segment
    circle - Embedding[circle, 3] ;; circle encoded by (x, y, r)
)
(def function
    start (x : line) : point := by pass
    end   (x : line) : point := by pass
    on_line (x : point) (y : line) : boolean := by pass
    length (x : line) : float := by pass
    intersect (x : line) (y : line) : boolean := by pass 
    parallel (x : line) (y : line) : boolean := by pass
    perpendicular (x : line) (y : line) : boolean := by pass

    connect_segment (x y : point) : line := by pass

    ;; Core circle operations
    center (x : circle) : point := by pass                ;; Extract center of circle
    radius (x : circle) : float := by pass                ;; Extract radius of circle
    on_radius (x : point) (y : circle) : boolean := by pass ;; Is point on circle's perimeter
    inside (x : point) (y : circle) : boolean := by pass   ;; Is point inside circle
    outside (x : point) (y : circle) : boolean := by pass  ;; Is point outside circle
    contain (x : circle) (y : circle) : boolean := by pass ;; Does circle x fully contain circle y

    ;; Line-circle interactions
    intersect_line_circle (x : line) (y : circle) : boolean := by pass ;; Does line intersect circle
    tangent (x : line) (y : circle) : boolean := by pass           ;; Is line tangent to circle
    contained_line (x : line) (y : circle) : boolean := by pass     ;; Is entire line inside circle
)
"""

euclid_domain = load_domain_string(integer_domain_str)

class EuclidExecutor(CentralExecutor):
    def __init__(self, domain):
        super().__init__(domain)
        self.epsilon = torch.tensor(1e-6)  # Differentiable tolerance

    # Core line operations
    def start(self, x: torch.Tensor) -> torch.Tensor:
        return x[:2]

    def end(self, x: torch.Tensor) -> torch.Tensor:
        return x[2:]

    def length(self, x: torch.Tensor) -> torch.Tensor:
        start_p = self.start(x)
        end_p = self.end(x)
        return torch.norm(end_p - start_p, p=2)

    def on_line(self, point: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
        px, py = point
        x1, y1, x2, y2 = line
        
        # Collinearity check (cross product)
        cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        collinear = torch.isclose(cross, torch.tensor(0.0), atol=self.epsilon)
        
        # Bounding box check
        in_x = torch.logical_and(px >= torch.min(x1, x2), px <= torch.max(x1, x2))
        in_y = torch.logical_and(py >= torch.min(y1, y2), py <= torch.max(y1, y2))
        in_bounds = torch.logical_and(in_x, in_y)
        
        return torch.logical_and(collinear, in_bounds)

    def intersect(self, line1: torch.Tensor, line2: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # CCW helper (differentiable)
        def ccw(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        
        a = torch.tensor([x1, y1])
        b = torch.tensor([x2, y2])
        c = torch.tensor([x3, y3])
        d = torch.tensor([x4, y4])
        
        ccw1 = ccw(a, c, d)
        ccw2 = ccw(b, c, d)
        ccw3 = ccw(a, b, c)
        ccw4 = ccw(a, b, d)
        
        return torch.logical_and(
            torch.not_equal(ccw1, ccw2),
            torch.not_equal(ccw3, ccw4)
        )

    def parallel(self, line1: torch.Tensor, line2: torch.Tensor) -> torch.Tensor:
        # Direction vectors
        dir1 = self.end(line1) - self.start(line1)
        dir2 = self.end(line2) - self.start(line2)
        
        # Cross product = 0 → parallel
        cross = dir1[0] * dir2[1] - dir1[1] * dir2[0]
        return torch.isclose(cross, torch.tensor(0.0), atol=self.epsilon)

    def perpendicular(self, line1: torch.Tensor, line2: torch.Tensor) -> torch.Tensor:
        # Direction vectors
        dir1 = self.end(line1) - self.start(line1)
        dir2 = self.end(line2) - self.start(line2)
        
        # Dot product = 0 → perpendicular
        dot = torch.dot(dir1, dir2)
        return torch.isclose(dot, torch.tensor(0.0), atol=self.epsilon)

    def connect_segment(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y])  # (x1,y1,x2,y2) from two points

    # Core circle operations
    def center(self, x: torch.Tensor) -> torch.Tensor:
        return x[:2]

    def radius(self, x: torch.Tensor) -> torch.Tensor:
        return x[2]

    def on_radius(self, point: torch.Tensor, circle: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(point - self.center(circle), p=2)
        return torch.isclose(dist, self.radius(circle), atol=self.epsilon)

    def inside(self, point: torch.Tensor, circle: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(point - self.center(circle), p=2)
        return dist < (self.radius(circle) - self.epsilon)

    def outside(self, point: torch.Tensor, circle: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(point - self.center(circle), p=2)
        return dist > (self.radius(circle) + self.epsilon)

    def contain(self, circle1: torch.Tensor, circle2: torch.Tensor) -> torch.Tensor:
        center_dist = torch.norm(self.center(circle1) - self.center(circle2), p=2)
        return (center_dist + self.radius(circle2)) < (self.radius(circle1) - self.epsilon)

    # Line-circle interactions
    def intersect_line_circle(self, line: torch.Tensor, circle: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = line
        cx, cy, r = circle
        
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy
        
        a = dx*dx + dy*dy
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r
        
        discriminant = b*b - 4*a*c
        no_intersect = discriminant < 0
        
        # Avoid sqrt of negative (differentiable)
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        in_range = torch.logical_or(
            torch.logical_and(t1 >= 0, t1 <= 1),
            torch.logical_and(t2 >= 0, t2 <= 1)
        )
        return torch.logical_and(torch.logical_not(no_intersect), in_range)

    def tangent(self, line: torch.Tensor, circle: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = line
        cx, cy, r = circle
        
        # Distance from center to line
        numerator = torch.abs((y2 - y1)*cx - (x2 - x1)*cy + x2*y1 - y2*x1)
        denominator = torch.norm(torch.tensor([x2-x1, y2-y1]), p=2)
        denominator = torch.clamp(denominator, min=1e-8)  # Avoid division by zero
        dist = numerator / denominator
        
        # Check distance == radius (tangent)
        is_tangent_dist = torch.isclose(dist, r, atol=self.epsilon)
        
        # Check tangent point is on segment
        t = ((cx - x1)*(x2 - x1) + (cy - y1)*(y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
        t_clamped = torch.clamp(t, 0.0, 1.0)
        px = x1 + t_clamped*(x2 - x1)
        py = y1 + t_clamped*(y2 - y1)
        
        tangent_point_dist = torch.norm(torch.tensor([px, py]) - self.center(circle), p=2)
        is_tangent_point = torch.isclose(tangent_point_dist, r, atol=self.epsilon)
        
        return torch.logical_and(is_tangent_dist, is_tangent_point)

    def contained_line(self, line: torch.Tensor, circle: torch.Tensor) -> torch.Tensor:

        start_inside = self.inside(self.start(line), circle)
        end_inside = self.inside(self.end(line), circle)
        
        mid_x = (self.start(line)[0] + self.end(line)[0]) / 2
        mid_y = (self.start(line)[1] + self.end(line)[1]) / 2
        mid_point = torch.tensor([mid_x, mid_y])
        mid_inside = self.inside(mid_point, circle)
        
        return torch.logical_and(torch.logical_and(start_inside, end_inside), mid_inside)


euclid_executor = EuclidExecutor(euclid_domain)

# Minimal test (verify differentiability)
if __name__ == "__main__":
    # Test tensors (requires_grad=True for differentiability)
    line = torch.tensor([1.0, 1.0, 5.0, 5.0], requires_grad=True)
    circle = torch.tensor([3.0, 3.0, 2.0], requires_grad=True)
    point = torch.tensor([3.0, 3.0], requires_grad=True)
    
    # Test core methods
    print("Line start:", euclid_executor.start(line))
    print("Circle center:", euclid_executor.center(circle))
    print("Point inside circle:", euclid_executor.inside(point, circle))
    print("Line intersects circle:", euclid_executor.intersect_line_circle(line, circle))
    
    # Verify gradient flow (differentiable check)
    loss = euclid_executor.length(line)
    loss.backward()
    print("Line gradient (length):", line.grad)  # Non-null → differentiable