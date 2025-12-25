import torch
import random
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple

# --------------------------
# Configuration Constants
# --------------------------
BASE_COLORS = {
    "red": torch.tensor([1.0, 0.0, 0.0]),
    "green": torch.tensor([0.0, 1.0, 0.0]),
    "blue": torch.tensor([0.0, 0.0, 1.0]),
    "yellow": torch.tensor([1.0, 1.0, 0.0])
}
COLOR_VARIATION = 0.3
CANVAS_SIZE = 128
MIN_LINE_LENGTH = 30
MAX_LINE_LENGTH = 102
MIN_CIRCLE_RADIUS = 10
MAX_CIRCLE_RADIUS = 40
EDGE_BUFFER = 7
LINE_WIDTH = 2

# Constraint weights (balance different constraint priorities)
CONSTRAINT_WEIGHTS = {
    "parallel": 3000.0,
    "perpendicular": 3000.0,
    "tangent": 1000.0,
    "contain_circle": 1000.0,
    "point_inside": 500.0,
    "point_outside": 500.0,
    "point_on_line": 1000.0,
    "point_on_radius": 1000.0,
    "line_length": 100.0,
    "edge_buffer": 1500.0,
    "circle_radius_bounds": 100.0,
    "line_intersect": 800.0,
    "line_circle_intersect": 800.0,
    "contained_line": 800.0,
    "start_point": 1000.0,
    "end_point": 1000.0,
    "circle_overlap": 1000.0 
}

# --------------------------
# Extended Smooth Loss Functions
# --------------------------
def smooth_parallel_loss(line1: np.ndarray, line2: np.ndarray) -> float:
    """Smooth loss for parallelism between two lines (x1,y1,x2,y2). Loss=0 when perfectly parallel."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    dir1_x = x2 - x1
    dir1_y = y2 - y1
    dir2_x = x4 - x3
    dir2_y = y4 - y3
    
    cross_product = dir1_x * dir2_y - dir1_y * dir2_x
    dir1_mag = np.sqrt(dir1_x**2 + dir1_y**2) + 1e-8
    dir2_mag = np.sqrt(dir2_x**2 + dir2_y**2) + 1e-8
    normalized_cross = cross_product / (dir1_mag * dir2_mag)
    
    return normalized_cross ** 2

def smooth_perpendicular_loss(line1: np.ndarray, line2: np.ndarray) -> float:
    """Smooth loss for perpendicularity between two lines (x1,y1,x2,y2). Loss=0 when perfectly perpendicular."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    dir1_x = x2 - x1
    dir1_y = y2 - y1
    dir2_x = x4 - x3
    dir2_y = y4 - y3
    
    dot_product = dir1_x * dir2_x + dir1_y * dir2_y
    dir1_mag = np.sqrt(dir1_x**2 + dir1_y**2) + 1e-8
    dir2_mag = np.sqrt(dir2_x**2 + dir2_y**2) + 1e-8
    normalized_dot = dot_product / (dir1_mag * dir2_mag)
    
    return normalized_dot ** 2

def smooth_line_length_loss(line: np.ndarray, min_len: float, max_len: float) -> float:
    """Smooth loss for line length bounds. Loss=0 when length is within [min_len, max_len]."""
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    loss_below = np.log1p(np.exp(-length + min_len))
    loss_above = np.log1p(np.exp(length - max_len))
    
    return loss_below + loss_above

def smooth_circle_containment_loss(outer_circle: np.ndarray, inner_circle: np.ndarray) -> float:
    """Smooth loss for circle containment. Loss=0 when outer circle fully contains inner circle."""
    cx1, cy1, r1 = outer_circle
    cx2, cy2, r2 = inner_circle
    
    center_dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    deviation = center_dist + r2 - r1
    
    return np.log1p(np.exp(deviation))

def smooth_line_circle_tangency_loss(line: np.ndarray, circle: np.ndarray) -> float:
    """Smooth loss for line-circle tangency. Loss=0 when line is perfectly tangent to circle."""
    x1, y1, x2, y2 = line
    cx, cy, r = circle
    
    numerator = np.abs((y2 - y1)*cx - (x2 - x1)*cy + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) + 1e-8
    dist_center_to_line = numerator / denominator
    
    deviation = np.abs(dist_center_to_line - r)
    return deviation ** 2

def smooth_point_in_circle_loss(point: np.ndarray, circle: np.ndarray) -> float:
    """Smooth loss for point inside circle. Loss=0 when point is inside circle."""
    x, y = point
    cx, cy, r = circle
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    deviation = dist - r
    
    return np.log1p(np.exp(deviation))

def smooth_point_outside_circle_loss(point: np.ndarray, circle: np.ndarray) -> float:
    """Smooth loss for point outside circle. Loss=0 when point is outside circle."""
    x, y = point
    cx, cy, r = circle
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    deviation = r - dist  # 与inside相反，当dist>r时deviation<0，损失为0
    
    return np.log1p(np.exp(deviation))

def smooth_point_on_line_loss(point: np.ndarray, line: np.ndarray) -> float:
    """Smooth loss for point lying on line. Loss=0 when point is perfectly on line."""
    x, y = point
    x1, y1, x2, y2 = line
    
    # 点到直线的距离公式
    numerator = np.abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) + 1e-8
    dist_point_to_line = numerator / denominator
    
    # 额外约束：点在线段的包围盒内（避免点在直线延长线上）
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    x_deviation = np.log1p(np.exp(-(x - x_min))) + np.log1p(np.exp(x - x_max))
    y_deviation = np.log1p(np.exp(-(y - y_min))) + np.log1p(np.exp(y - y_max))
    
    return (dist_point_to_line ** 2) + 0.5 * (x_deviation + y_deviation)

def smooth_point_on_circle_radius_loss(point: np.ndarray, circle: np.ndarray) -> float:
    """Smooth loss for point lying on circle's perimeter (on_radius). Loss=0 when point is perfectly on perimeter."""
    x, y = point
    cx, cy, r = circle
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    deviation = np.abs(dist - r)  # 距离与半径的绝对差，为0时在圆上
    
    return deviation ** 2

def smooth_edge_buffer_loss(point: np.ndarray, canvas_size: int, edge_buffer: int) -> float:
    """Smooth loss for point staying within canvas edge buffer. Loss=0 when point is inside buffer."""
    x, y = point
    min_bound = edge_buffer
    max_bound = canvas_size - edge_buffer
    
    loss_x_left = np.log1p(np.exp(min_bound - x))
    loss_x_right = np.log1p(np.exp(x - max_bound))
    loss_y_top = np.log1p(np.exp(min_bound - y))
    loss_y_bottom = np.log1p(np.exp(y - max_bound))
    
    return loss_x_left + loss_x_right + loss_y_top + loss_y_bottom

def smooth_line_intersection_loss(segment1: np.ndarray, segment2: np.ndarray) -> float:
    """Smooth loss for line-line intersection. Loss=0 when lines intersect (within segment bounds)."""
    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2
    
    # 计算方向向量叉积（判断是否平行）
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # 若两条线段近似平行（无交点），返回大损失
    if np.abs(denom) < 1e-8:
        return 1e6 
    
    # 计算线段参数t（对应segment1）和u（对应segment2），仅在[0,1]内表示交点在线段上
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # 平滑约束t在[0,1]范围内（softplus构建的平滑损失，替代硬阈值）
    t_deviation = np.log1p(np.exp(-t)) + np.log1p(np.exp(t - 1))
    # 平滑约束u在[0,1]范围内（约束segment2的线段范围）
    u_deviation = np.log1p(np.exp(-u)) + np.log1p(np.exp(u - 1))
    
    # 总损失为两个线段的约束损失之和
    return float(t_deviation + u_deviation)

def smooth_line_circle_intersection_loss(line: np.ndarray, circle: np.ndarray) -> float:
    """Smooth loss for line-circle intersection. Loss=0 when line intersects circle."""
    x1, y1, x2, y2 = line
    cx, cy, r = circle
    
    # 圆心到直线的距离
    numerator = np.abs((y2 - y1)*cx - (x2 - x1)*cy + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) + 1e-8
    dist_center_to_line = numerator / denominator
    
    # 当距离≤半径时，直线与圆相交
    deviation = dist_center_to_line - r
    # 同时确保交点在线段上（简单包围盒约束）
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    cx_deviation = np.log1p(np.exp(-(cx - x_min))) + np.log1p(np.exp(cx - x_max))
    cy_deviation = np.log1p(np.exp(-(cy - y_min))) + np.log1p(np.exp(cy - y_max))
    
    return np.log1p(np.exp(-deviation)) + 0.3 * (cx_deviation + cy_deviation)

def smooth_contained_line_loss(line: np.ndarray, circle: np.ndarray) -> float:
    """Smooth loss for entire line being contained in circle. Loss=0 when line is fully inside circle."""
    x1, y1, x2, y2 = line
    cx, cy, r = circle
    
    # 约束线段的两个端点都在圆内
    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    loss_p1 = smooth_point_in_circle_loss(point1, circle)
    loss_p2 = smooth_point_in_circle_loss(point2, circle)
    
    # 额外约束：线段中点也在圆内（增强包含效果）
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    mid_point = np.array([mid_x, mid_y])
    loss_mid = smooth_point_in_circle_loss(mid_point, circle)
    
    return loss_p1 + loss_p2 + 0.5 * loss_mid

def smooth_line_start_point_loss(point: np.ndarray, line: np.ndarray) -> float:
    """Smooth loss for point being the start point of line. Loss=0 when point is (x1,y1) of line."""
    px, py = point
    x1, y1, x2, y2 = line
    
    deviation_x = np.abs(px - x1)
    deviation_y = np.abs(py - y1)
    
    return (deviation_x ** 2) + (deviation_y ** 2)

def smooth_line_end_point_loss(point: np.ndarray, line: np.ndarray) -> float:
    """Smooth loss for point being the end point of line. Loss=0 when point is (x2,y2) of line."""
    px, py = point
    x1, y1, x2, y2 = line
    
    deviation_x = np.abs(px - x2)
    deviation_y = np.abs(py - y2)
    
    return (deviation_x ** 2) + (deviation_y ** 2)

def smooth_circle_overlap_loss(circle1: np.ndarray, circle2: np.ndarray) -> float:
    """
    新增：Smooth loss for circle-circle overlap (相交/重叠). 
    Loss=0 when two circles overlap (存在公共区域)，Loss>0 when they are fully separated or one is fully contained in the other (无公共区域).
    """
    cx1, cy1, r1 = circle1
    cx2, cy2, r2 = circle2
    
    # 计算两圆心距离
    center_dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    # 两圆半径之和与半径之差的绝对值
    radius_sum = r1 + r2
    radius_diff = np.abs(r1 - r2)
    
    # 重叠条件：center_dist ≤ radius_sum 且 center_dist ≥ radius_diff
    # 损失1：center_dist > radius_sum（分离），产生损失
    loss_separate = np.log1p(np.exp(center_dist - radius_sum))
    # 损失2：center_dist < radius_diff（内含，无重叠），产生损失
    loss_contain_no_overlap = np.log1p(np.exp(radius_diff - center_dist))
    
    # 总损失：分离损失 + 内含无重叠损失
    return loss_separate + loss_contain_no_overlap

# --------------------------
# Extended DSL Generators for Geometric Shapes
# --------------------------
def colored_triangle(colors=["red", "green", "blue"]):
    program = f"l1:line(p1, p2)[color({colors[0]})];l2:line(p2, p3)[color({colors[1]})];l3:line(p3, p1)[color({colors[2]})];"
    return program

def colored_square(colors=["red", "green", "blue", "yellow"]):
    program = f"""
l1:line(p1, p2)[color({colors[0]})];
p2:point(p2)[on_line(p2, l1)];
l2:line(p2, p3)[color({colors[1]}), perpendicular(l2, l1)];
p3:point(p3)[on_line(p3, l2)];
l3:line(p3, p4)[color({colors[2]}), perpendicular(l3, l2), parallel(l3, l1)];
p4:point(p4)[on_line(p4, l3)];
l4:line(p4, p1)[color({colors[3]}), perpendicular(l4, l3), parallel(l4, l2)];
p1:point(p1)[on_line(p1, l4)];
"""
    return program.strip()

def colored_circle_with_perimeter_point(circle_color="blue", point_color="red"):
    program = f"""
c1:circle(p1, p2)[color({circle_color})];
p3:point(p3)[color({point_color}), on_radius(p3, c1)];
"""
    return program.strip()

def colored_rectangle(colors=["red", "green", "blue", "yellow"]):
    program = f"""
l1:line(p1, p2)[color({colors[0]})];
p2:point(p2)[on_line(p2, l1)];
l2:line(p2, p3)[color({colors[1]}), perpendicular(l2, l1)];
p3:point(p3)[on_line(p3, l2)];
l3:line(p3, p4)[color({colors[2]}), parallel(l3, l1)];
p4:point(p4)[on_line(p4, l3)];
l4:line(p4, p1)[color({colors[3]}), parallel(l4, l2)];
p1:point(p1)[on_line(p1, l4)];
"""
    return program.strip()

def tangent_line_and_circle(line_color="red", circle_color="blue"):
    program = f"""
c1:circle(p1, p2)[color({circle_color})];
l1:line(p3, p4)[color({line_color}), tangent(l1, c1)];
p3:point(p3)[on_line(p3, l1)];
p4:point(p4)[on_line(p4, l1)];
"""
    return program.strip()

def non_intersecting_lines(line1_color="red", line2_color="green"):
    """示例：使用!intersect约束生成不相交的两条直线"""
    program = f"""
l1:line(p1, p2)[color({line1_color})];
l2:line(p3, p4)[color({line2_color}), !intersect(l2, l1)];
p1:point(p1)[start(p1, l1)];
p2:point(p2)[end(p2, l1)];
p3:point(p3)[start(p3, l2)];
p4:point(p4)[end(p4, l2)];
"""
    return program.strip()

def line_outside_circle(line_color="red", circle_color="blue"):
    """示例：使用!contained_line和!inside约束生成圆外的直线"""
    program = f"""
c1:circle(p1, p2)[color({circle_color})];
l1:line(p3, p4)[color({line_color}), !contained_line(l1, c1)];
p3:point(p3)[!inside(p3, c1)];
p4:point(p4)[!inside(p4, c1)];
"""
    return program.strip()

def overlapping_circles(circle1_color="red", circle2_color="blue"):
    """新增示例：使用overlap约束生成两个重叠的圆"""
    program = f"""
c1:circle(p1, p2)[color({circle1_color})];
c2:circle(p3, p4)[color({circle2_color}), overlap(c2, c1)];
"""
    return program.strip()

def non_overlapping_circles(circle1_color="red", circle2_color="blue"):
    """新增示例：使用!overlap约束生成两个完全不重叠的圆"""
    program = f"""
c1:circle(p1, p2)[color({circle1_color})];
c2:circle(p3, p4)[color({circle2_color}), !overlap(c2, c1)];
"""
    return program.strip()

# --------------------------
# Extended CCSP Geometric Solver
# --------------------------
class CCSPGeometricSolver:
    def __init__(self):
        self.canvas_size = CANVAS_SIZE
        self.edge_buffer = EDGE_BUFFER
        self.min_line_length = MIN_LINE_LENGTH
        self.max_line_length = MAX_LINE_LENGTH
        self.min_circle_radius = MIN_CIRCLE_RADIUS
        self.max_circle_radius = MAX_CIRCLE_RADIUS
        self.constraint_weights = CONSTRAINT_WEIGHTS
        
        # Object registry
        self.objects = {}
        self.points = {}
        self.constraints = []
        self.var_count = 0
        # 记录点的约束类型（自由点/线上点/圆上点）
        self.point_constraint_types = {}

    def parse_dsl(self, dsl_program: str) -> List[Dict]:
        """Parse DSL program into structured object specs and constraints (支持!否定约束和overlap约束)."""
        parsed_objects = []
        statements = [s.strip() for s in dsl_program.split(";") if s.strip()]
    
        for stmt in statements:
            if "[" not in stmt or "]" not in stmt:
                raise ValueError(f"Invalid DSL statement (missing constraints bracket): {stmt}")
        
            obj_part, constraint_part = stmt.split("[", 1)
            constraint_part = constraint_part.rsplit("]", 1)[0]
        
            if ":" not in obj_part:
                raise ValueError(f"Invalid object part (missing colon): {obj_part}")
            obj_name, type_params = obj_part.split(":", 1)
        
            if "(" not in type_params or ")" not in type_params:
                raise ValueError(f"Invalid type params (missing parentheses): {type_params}")
            obj_type, params_str = type_params.split("(", 1)
            params_str = params_str.rsplit(")", 1)[0].strip()
            param_list = [p.strip() for p in params_str.split(",")] if params_str else []
        
            # Parse constraints with parenthesis handling
            constraints = []
            current_constraint = ""
            parenthesis_count = 0
        
            for char in constraint_part:
                if char == "(":
                    parenthesis_count += 1
                    current_constraint += char
                elif char == ")":
                    parenthesis_count -= 1
                    current_constraint += char
                elif char == "," and parenthesis_count == 0:
                    if current_constraint.strip():
                        constraints.append(current_constraint.strip())
                    current_constraint = ""
                else:
                    current_constraint += char
        
            if current_constraint.strip():
                constraints.append(current_constraint.strip())
        
            # Separate color and geometric constraints
            color = None
            geo_constraints = []
            for const in constraints:
                if const.startswith("color("):
                    color_val = const.replace("color(", "").replace(")", "").strip()
                    color = color_val if color_val in BASE_COLORS.keys() else None
                else:
                    geo_constraints.append(const)
                    is_negative = False
                    const_clean = const
                    if const.startswith("!"):
                        is_negative = True
                        const_clean = const[1:]  # 去掉!符号
                    
                    const_type = const_clean.split("(")[0]
                    const_args = [arg.strip() for arg in const_clean.split("(")[1].rstrip(")").split(",")]
                    self.constraints.append({
                        "type": const_type,
                        "args": const_args,
                        "obj_name": obj_name.strip(),
                        "is_negative": is_negative  # 添加否定标识
                    })
                    
                    if obj_type.strip() == "point":
                        if const_type == "on_line":
                            self.point_constraint_types[obj_name.strip()] = {"type": "on_line", "ref_obj": const_args[1]}
                        elif const_type == "on_radius":
                            self.point_constraint_types[obj_name.strip()] = {"type": "on_radius", "ref_obj": const_args[1]}
                        elif const_type == "start":
                            self.point_constraint_types[obj_name.strip()] = {"type": "start", "ref_obj": const_args[1]}
                        elif const_type == "end":
                            self.point_constraint_types[obj_name.strip()] = {"type": "end", "ref_obj": const_args[1]}
        
            parsed_obj = {
                "name": obj_name.strip(),
                "type": obj_type.strip(),
                "params": param_list,
                "color": color,
                "geo_constraints": geo_constraints
            }
            parsed_objects.append(parsed_obj)
            
            self.objects[obj_name.strip()] = {
                "type": obj_type.strip(),
                "params": param_list,
                "parsed_obj": parsed_obj
            }
        
        return parsed_objects

    def initialize_variables(self) -> np.ndarray:
        """Initialize optimization variables (points + circle radii) as numpy array (extended for constrained points)."""
        variables = []
        self.points = {}
        self.var_count = 0
        self.point_constraint_types = {}
        
        # First, initialize free points (p1, p2, ...) - no constraints
        for obj in self.objects.values():
            for param in obj["params"]:
                if param not in self.points and param.startswith("p"):

                    is_free_point = True
                    for const in self.constraints:
                        if const["obj_name"] == param and const["type"] in ["on_line", "on_radius", "start", "end"]:
                            is_free_point = False
                            break
                    
                    if is_free_point:
                        init_x = random.uniform(self.edge_buffer, self.canvas_size - self.edge_buffer)
                        init_y = random.uniform(self.edge_buffer, self.canvas_size - self.edge_buffer)

                        self.points[param] = {
                            "x": init_x,
                            "y": init_y,
                            "var_idx": self.var_count,
                            "is_free": True
                        }
                        variables.extend([init_x, init_y])
                        self.var_count += 2
        
        # Then, initialize constrained points (on line/circle)
        for obj_name, obj in self.objects.items():
            if obj["type"] == "point" and obj_name.startswith("p") and obj_name not in self.points:
                const_info = self.point_constraint_types.get(obj_name, None)
                init_x, init_y = 0.0, 0.0
                
                if const_info:
                    ref_obj = self.objects.get(const_info["ref_obj"])
                    if ref_obj:
                        if const_info["type"] == "on_line":
                            line_params = ref_obj["params"]
                            if len(line_params) >= 2 and line_params[0] in self.points and line_params[1] in self.points:
                                p1 = self.points[line_params[0]]
                                p2 = self.points[line_params[1]]
                                t = random.uniform(0, 1)
                                init_x = p1["x"] + t * (p2["x"] - p1["x"])
                                init_y = p1["y"] + t * (p2["y"] - p1["y"])
                        elif const_info["type"] == "on_radius":
                            circle_params = ref_obj["params"]
                            if len(circle_params) >= 1 and circle_params[0] in self.points:
                                center = self.points[circle_params[0]]
                                angle = random.uniform(0, 2 * np.pi)
                                temp_r = (self.min_circle_radius + self.max_circle_radius) / 2
                                init_x = center["x"] + temp_r * np.cos(angle)
                                init_y = center["y"] + temp_r * np.sin(angle)
                        elif const_info["type"] in ["start", "end"]:
                            line_params = ref_obj["params"]
                            if len(line_params) >= 2 and line_params[0] in self.points and line_params[1] in self.points:
                                p_start = self.points[line_params[0]]
                                p_end = self.points[line_params[1]]
                                if const_info["type"] == "start":
                                    init_x, init_y = p_start["x"], p_start["y"]
                                else:
                                    init_x, init_y = p_end["x"], p_end["y"]
                
                if init_x < self.edge_buffer or init_x > self.canvas_size - self.edge_buffer:
                    init_x = random.uniform(self.edge_buffer, self.canvas_size - self.edge_buffer)
                if init_y < self.edge_buffer or init_y > self.canvas_size - self.edge_buffer:
                    init_y = random.uniform(self.edge_buffer, self.canvas_size - self.edge_buffer)
                
                self.points[obj_name] = {
                    "x": init_x,
                    "y": init_y,
                    "var_idx": self.var_count,
                    "is_free": False
                }
                variables.extend([init_x, init_y])
                self.var_count += 2
        
        # Initialize circle radii
        for obj_name, obj in self.objects.items():
            if obj["type"] == "circle" and obj_name not in [p for p in self.points.keys()]:
                init_r = random.uniform(self.min_circle_radius, self.max_circle_radius)
                self.objects[obj_name]["radius_var_idx"] = self.var_count
                variables.append(init_r)
                self.var_count += 1
        
        return np.array(variables, dtype=np.float64)

    def _update_state_from_variables(self, x: np.ndarray):
        """Update internal points/radii state from optimization variables."""
        # Update points
        for point_name, point_data in self.points.items():
            var_idx = point_data["var_idx"]
            point_data["x"] = x[var_idx]
            point_data["y"] = x[var_idx + 1]
        
        # Update circle radii
        for obj_name, obj_data in self.objects.items():
            if obj_data["type"] == "circle" and "radius_var_idx" in obj_data:
                var_idx = obj_data["radius_var_idx"]
                obj_data["radius"] = max(self.min_circle_radius, min(self.max_circle_radius, x[var_idx]))

    def constraint_loss(self, x: np.ndarray) -> float:
        """Total smooth loss function for geometric constraint optimization (支持!否定约束和overlap约束)."""
        total_loss = 0.0
        self._update_state_from_variables(x)
        
        # 1. Edge Buffer Loss
        for point_name, point_data in self.points.items():
            point = np.array([point_data["x"], point_data["y"]])
            edge_loss = smooth_edge_buffer_loss(
                point=point,
                canvas_size=self.canvas_size,
                edge_buffer=self.edge_buffer
            )
            total_loss += self.constraint_weights["edge_buffer"] * edge_loss
        
        # 2. Line Length Loss
        for obj_name, obj_data in self.objects.items():
            if obj_data["type"] == "line":
                p1_data = self.points[obj_data["params"][0]]
                p2_data = self.points[obj_data["params"][1]]
                line = np.array([
                    p1_data["x"], p1_data["y"],
                    p2_data["x"], p2_data["y"]
                ])
                length_loss = smooth_line_length_loss(
                    line=line,
                    min_len=self.min_line_length,
                    max_len=self.max_line_length
                )
                total_loss += self.constraint_weights["line_length"] * length_loss
        
        # 3. Circle Radius Bounds Loss
        for obj_name, obj_data in self.objects.items():
            if obj_data["type"] == "circle":
                radius = obj_data.get("radius", self.min_circle_radius)
                loss_min_r = np.log1p(np.exp(self.min_circle_radius - radius))
                loss_max_r = np.log1p(np.exp(radius - self.max_circle_radius))
                radius_loss = loss_min_r + loss_max_r
                total_loss += self.constraint_weights["circle_radius_bounds"] * radius_loss
        
        # 4. Extended Geometric Constraints (支持否定约束和overlap约束)
        for constraint in self.constraints:
            const_type = constraint["type"]
            const_args = constraint["args"]
            obj_name = constraint["obj_name"]
            is_negative = constraint["is_negative"]  # 获取否定标识
            obj_data = self.objects.get(obj_name)
            if not obj_data:
                continue
            
            # Get main object
            main_obj = None
            if obj_data["type"] == "line":
                p1 = self.points[obj_data["params"][0]]
                p2 = self.points[obj_data["params"][1]]
                main_obj = np.array([p1["x"], p1["y"], p2["x"], p2["y"]])
            elif obj_data["type"] == "circle":
                center = self.points[obj_data["params"][0]]
                radius = obj_data.get("radius", self.min_circle_radius)
                main_obj = np.array([center["x"], center["y"], radius])
            elif obj_data["type"] == "point":
                point_data = self.points[obj_data["params"][0]]
                main_obj = np.array([point_data["x"], point_data["y"]])
            
            # Get reference object
            ref_obj = None
            ref_obj_data = None
            if len(const_args) > 1:
                ref_obj_name = const_args[1]
                ref_obj_data = self.objects.get(ref_obj_name)
                if ref_obj_data:
                    if ref_obj_data["type"] == "line":
                        p1_ref = self.points[ref_obj_data["params"][0]]
                        p2_ref = self.points[ref_obj_data["params"][1]]
                        ref_obj = np.array([p1_ref["x"], p1_ref["y"], p2_ref["x"], p2_ref["y"]])
                    elif ref_obj_data["type"] == "circle":
                        center_ref = self.points[ref_obj_data["params"][0]]
                        radius_ref = ref_obj_data.get("radius", self.min_circle_radius)
                        ref_obj = np.array([center_ref["x"], center_ref["y"], radius_ref])
                    elif ref_obj_data["type"] == "point":
                        point_ref_data = self.points[ref_obj_data["params"][0]]
                        ref_obj = np.array([point_ref_data["x"], point_ref_data["y"]])
            
            # Apply extended constraint losses (含否定处理，新增overlap约束)
            current_loss = 0.0
            if const_type == "parallel" and ref_obj is not None and obj_data["type"] == "line" and ref_obj_data["type"] == "line":
                current_loss = smooth_parallel_loss(main_obj, ref_obj)
                # 否定：希望两条直线不平行，即让平行损失尽可能大（取1 - 损失，确保非负）
                if is_negative:
                    current_loss = np.clip(1 - current_loss, 0, 1) ** 2  # 平方增强惩罚
                total_loss += self.constraint_weights["parallel"] * current_loss
            
            elif const_type == "perpendicular" and ref_obj is not None and obj_data["type"] == "line" and ref_obj_data["type"] == "line":
                current_loss = smooth_perpendicular_loss(main_obj, ref_obj)
                # 否定：希望两条直线不垂直
                if is_negative:
                    current_loss = np.clip(1 - current_loss, 0, 1) ** 2
                total_loss += self.constraint_weights["perpendicular"] * current_loss
            
            elif const_type == "tangent" and ref_obj is not None and obj_data["type"] == "line" and ref_obj_data["type"] == "circle":
                current_loss = smooth_line_circle_tangency_loss(main_obj, ref_obj)
                # 否定：希望直线与圆不相切（距离与半径的差尽可能大）
                if is_negative:
                    current_loss = 1 - np.exp(-current_loss)  # 平滑反转
                total_loss += self.constraint_weights["tangent"] * current_loss
            
            elif const_type == "contain" and ref_obj is not None and obj_data["type"] == "circle" and ref_obj_data["type"] == "circle":
                current_loss = smooth_circle_containment_loss(main_obj, ref_obj)
                # 否定：希望外圆不包含内圆（deviation尽可能大）
                if is_negative:
                    # 原损失是log1p(exp(deviation))，否定则希望-deviation大，即log1p(exp(-deviation))
                    cx1, cy1, r1 = main_obj
                    cx2, cy2, r2 = ref_obj
                    center_dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                    deviation = center_dist + r2 - r1
                    current_loss = np.log1p(np.exp(-deviation))
                total_loss += self.constraint_weights["contain_circle"] * current_loss
            
            elif const_type == "inside" and ref_obj is not None and obj_data["type"] == "point" and ref_obj_data["type"] == "circle":
                current_loss = smooth_point_in_circle_loss(main_obj, ref_obj)
                # 否定：希望点在圆外（即原outside损失）
                if is_negative:
                    current_loss = smooth_point_outside_circle_loss(main_obj, ref_obj)
                total_loss += self.constraint_weights["point_inside"] * current_loss
            
            elif const_type == "outside" and ref_obj is not None and obj_data["type"] == "point" and ref_obj_data["type"] == "circle":
                current_loss = smooth_point_outside_circle_loss(main_obj, ref_obj)
                # 否定：希望点在圆内（即原inside损失）
                if is_negative:
                    current_loss = smooth_point_in_circle_loss(main_obj, ref_obj)
                total_loss += self.constraint_weights["point_outside"] * current_loss
            
            elif const_type == "on_line" and ref_obj is not None and obj_data["type"] == "point" and ref_obj_data["type"] == "line":
                current_loss = smooth_point_on_line_loss(main_obj, ref_obj)
                # 否定：希望点不在直线上（距离尽可能大）
                if is_negative:
                    current_loss = 1 - np.exp(-current_loss)
                total_loss += self.constraint_weights["point_on_line"] * current_loss
            
            elif const_type == "on_radius" and ref_obj is not None and obj_data["type"] == "point" and ref_obj_data["type"] == "circle":
                current_loss = smooth_point_on_circle_radius_loss(main_obj, ref_obj)
                # 否定：希望点不在圆的周长上（距离与半径的差尽可能大）
                if is_negative:
                    current_loss = 1 - np.exp(-current_loss)
                total_loss += self.constraint_weights["point_on_radius"] * current_loss
            
            elif const_type == "intersect" and ref_obj is not None and obj_data["type"] == "line" and ref_obj_data["type"] == "line":
                current_loss = smooth_line_intersection_loss(main_obj, ref_obj)
                # 否定：希望两条直线不相交（原损失越大越好，直接使用原损失的相反数惩罚）
                if is_negative:
                    # 原损失小表示相交，否定则希望损失大，使用np.log1p(np.exp(-current_loss))增强不相交的惩罚
                    current_loss = np.log1p(np.exp(-current_loss))
                total_loss += self.constraint_weights["line_intersect"] * current_loss
            
            elif const_type == "intersect_line_circle" and ref_obj is not None and obj_data["type"] == "line" and ref_obj_data["type"] == "circle":
                current_loss = smooth_line_circle_intersection_loss(main_obj, ref_obj)
                # 否定：希望直线与圆不相交
                if is_negative:
                    x1, y1, x2, y2 = main_obj
                    cx, cy, r = ref_obj
                    numerator = np.abs((y2 - y1)*cx - (x2 - x1)*cy + x2*y1 - y2*x1)
                    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) + 1e-8
                    dist_center_to_line = numerator / denominator
                    deviation = dist_center_to_line - r
                    current_loss = np.log1p(np.exp(deviation))  # 反转deviation符号
                total_loss += self.constraint_weights["line_circle_intersect"] * current_loss
            
            elif const_type == "contained_line" and ref_obj is not None and obj_data["type"] == "line" and ref_obj_data["type"] == "circle":
                current_loss = smooth_contained_line_loss(main_obj, ref_obj)
                # 否定：希望直线不完全在圆内（端点至少有一个在圆外）
                if is_negative:
                    # 反转每个点的损失：使用outside损失
                    x1, y1, x2, y2 = main_obj
                    cx, cy, r = ref_obj
                    point1 = np.array([x1, y1])
                    point2 = np.array([x2, y2])
                    loss_p1 = smooth_point_outside_circle_loss(point1, ref_obj)
                    loss_p2 = smooth_point_outside_circle_loss(point2, ref_obj)
                    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                    mid_point = np.array([mid_x, mid_y])
                    loss_mid = smooth_point_outside_circle_loss(mid_point, ref_obj)
                    current_loss = loss_p1 + loss_p2 + 0.5 * loss_mid
                total_loss += self.constraint_weights["contained_line"] * current_loss
            
            elif const_type == "start" and ref_obj is not None and obj_data["type"] == "point" and ref_obj_data["type"] == "line":
                current_loss = smooth_line_start_point_loss(main_obj, ref_obj)
                # 否定：希望点不是直线的起点
                if is_negative:
                    current_loss = 1 - np.exp(-current_loss)
                total_loss += self.constraint_weights["start_point"] * current_loss
            
            elif const_type == "end" and ref_obj is not None and obj_data["type"] == "point" and ref_obj_data["type"] == "line":
                current_loss = smooth_line_end_point_loss(main_obj, ref_obj)
                # 否定：希望点不是直线的终点
                if is_negative:
                    current_loss = 1 - np.exp(-current_loss)
                total_loss += self.constraint_weights["end_point"] * current_loss
            
            # 新增：overlap约束（圆-圆重叠）
            elif const_type == "overlap" and ref_obj is not None and obj_data["type"] == "circle" and ref_obj_data["type"] == "circle":
                current_loss = smooth_circle_overlap_loss(main_obj, ref_obj)
                # 否定：!overlap表示两圆完全不重叠（分离或内含无公共区域）
                if is_negative:
                    # 反转损失逻辑：希望原重叠损失的两个部分都尽可能小（即分离或内含）
                    cx1, cy1, r1 = main_obj
                    cx2, cy2, r2 = ref_obj
                    center_dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                    radius_sum = r1 + r2
                    radius_diff = np.abs(r1 - r2)
                    # 否定损失：希望center_dist > radius_sum 或 center_dist < radius_diff（与原损失相反）
                    loss_overlap = np.log1p(np.exp(radius_sum - center_dist)) + np.log1p(np.exp(center_dist - radius_diff))
                    current_loss = loss_overlap
                total_loss += self.constraint_weights["circle_overlap"] * current_loss

        # L2 Regularization
        total_loss += 1e-6 * np.sum(x ** 2)
        return total_loss

    def _get_variable_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for optimization variables."""
        bounds = []
        
        # Bounds for points (x, y)
        for _ in self.points.values():
            bounds.append((self.edge_buffer, self.canvas_size - self.edge_buffer))
            bounds.append((self.edge_buffer, self.canvas_size - self.edge_buffer))
        
        # Bounds for circle radii
        for obj_data in self.objects.values():
            if obj_data["type"] == "circle":
                bounds.append((self.min_circle_radius, self.max_circle_radius))
        
        return bounds

    def solve(self, dsl_program: str, max_restarts: int = 3) -> Tuple[List[Dict], Dict]:
        """Main CCSP solve function: optimize to satisfy constraints with smooth loss (extended)."""
        # Reset state
        self.objects = {}
        self.points = {}
        self.constraints = []
        self.var_count = 0
        self.point_constraint_types = {}
        
        # Parse DSL
        parsed_objects = self.parse_dsl(dsl_program)
        
        # Initialize variables
        best_result = None
        best_loss = float("inf")
        
        for restart in range(max_restarts):
            x0 = self.initialize_variables()
            if restart > 0:
                x0 += np.random.normal(0, 1e-4, size=x0.shape)
                for i, (lb, ub) in enumerate(self._get_variable_bounds()):
                    x0[i] = np.clip(x0[i], lb, ub)
            
            bounds = self._get_variable_bounds()
            
            # Run optimization
            result = minimize(
                fun=self.constraint_loss,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 30000,
                    "maxfun": 100000,
                    "gtol": 1e-9,
                    "ftol": 1e-8,
                    "eps": 1e-9,
                    "disp": False,
                    "maxls": 50
                }
            )
            
            # Track best result
            current_loss = self.constraint_loss(result.x)
            if current_loss < best_loss:
                best_loss = current_loss
                best_result = result
            #print(result["success"])
        
        # Update state with best result
        self._update_state_from_variables(best_result.x)
        
        # Compile results
        scene_metadata = []
        solved_objects = {}
        
        for parsed_obj in parsed_objects:
            obj_name = parsed_obj["name"]
            obj_type = parsed_obj["type"]
            color_name = parsed_obj["color"]
            
            # Get object geometry
            if obj_type == "line":
                p1_data = self.points[parsed_obj["params"][0]]
                p2_data = self.points[parsed_obj["params"][1]]
                geometry = torch.tensor([
                    p1_data["x"], p1_data["y"],
                    p2_data["x"], p2_data["y"]
                ], dtype=torch.float32)
                solved_objects[parsed_obj["params"][0]] = torch.tensor([p1_data["x"], p1_data["y"]])
                solved_objects[parsed_obj["params"][1]] = torch.tensor([p2_data["x"], p2_data["y"]])
            elif obj_type == "circle":
                center_data = self.points[parsed_obj["params"][0]]
                radius = self.objects[obj_name].get("radius", self.min_circle_radius)
                geometry = torch.tensor([
                    center_data["x"], center_data["y"], radius
                ], dtype=torch.float32)
                solved_objects[parsed_obj["params"][0]] = torch.tensor([center_data["x"], center_data["y"]])
                solved_objects[parsed_obj["params"][1]] = torch.tensor([center_data["x"] + radius, center_data["y"]])
            elif obj_type == "point":
                point_data = self.points[parsed_obj["params"][0]]
                geometry = torch.tensor([point_data["x"], point_data["y"]], dtype=torch.float32)
                solved_objects[parsed_obj["params"][0]] = geometry
            
            # Get color
            color_rgb = get_varied_color(color_name) if color_name else BASE_COLORS["red"]
            
            solved_objects[obj_name] = geometry
            scene_metadata.append({
                "name": obj_name,
                "type": obj_type,
                "geometry": geometry,
                "color_name": color_name,
                "color_rgb": color_rgb
            })
        
        return scene_metadata, solved_objects

# --------------------------
# Helper Functions
# --------------------------
def get_varied_color(base_color_name: str) -> torch.Tensor:
    """Generate slightly varied color within base category."""
    base = BASE_COLORS[base_color_name].clone()
    if base_color_name == "red":
        base[1] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
        base[2] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
    elif base_color_name == "green":
        base[0] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
        base[2] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
    elif base_color_name == "blue":
        base[0] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
        base[1] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
    elif base_color_name == "yellow":
        base[0] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
        base[1] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
    return torch.clamp(base, 0.0, 1.0)

def _rasterize_triangle(xx: torch.Tensor, yy: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Rasterize a triangle defined by 3 vertices into a binary mask."""
    v0, v1, v2 = vertices
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = torch.stack([xx - v0[0], yy - v0[1]], dim=-1)
    
    dot00 = torch.sum(v0v1 * v0v1)
    dot01 = torch.sum(v0v1 * v0v2)
    dot02 = torch.sum(v0v1 * v0p, dim=-1)
    dot11 = torch.sum(v0v2 * v0v2)
    dot12 = torch.sum(v0v2 * v0p, dim=-1)
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    mask = torch.logical_and(
        torch.logical_and(u >= 0.0, v >= 0.0),
        (u + v) <= 1.0
    ).float()
    
    return mask

# --------------------------
# Rendering Functions (Extended for Point Rendering & Multi-Channel Masks)
# --------------------------
def render_filled_circle(
    canvas: torch.Tensor, 
    object_masks: torch.Tensor,  # 改为多通道掩码
    circle: torch.Tensor, 
    color: torch.Tensor,
    object_channel: int,  # 改为物体对应的通道索引
    line_width: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render circle to canvas and update multi-channel object masks with object's binary mask."""
    cx, cy, r = circle
    y_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    x_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
    circle_mask = (dist <= r).float()
    
    # 更新画布（原有逻辑不变）
    canvas = canvas * (1 - circle_mask.unsqueeze(0)) + color.unsqueeze(1).unsqueeze(2) * circle_mask.unsqueeze(0)
    # 更新对应通道的物体掩码（仅当前物体通道设为1）
    object_masks[:, :, object_channel] = circle_mask
    
    return canvas, object_masks

def render_directed_line(
    canvas: torch.Tensor, 
    object_masks: torch.Tensor,  # 改为多通道掩码
    line: torch.Tensor, 
    color: torch.Tensor,
    object_channel: int,  # 改为物体对应的通道索引
    line_width: int = LINE_WIDTH
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render directed line (with small arrow) to canvas and update multi-channel object masks."""
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    line_length = torch.sqrt(dx**2 + dy**2)
    
    if line_length < 1e-6:
        return canvas, object_masks
    
    arrow_size = 2.0 if line_length > 4 else 1.0
    
    y_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    x_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    numerator = torch.abs((dy) * xx - (dx) * yy + x2*y1 - y2*x1)
    denominator = torch.sqrt(dx**2 + dy**2) + 1e-8
    dist_to_line = numerator / denominator
    line_mask = (dist_to_line <= line_width).float()
    
    x_min, x_max = torch.min(x1, x2) - line_width, torch.max(x1, x2) + line_width
    y_min, y_max = torch.min(y1, y2) - line_width, torch.max(y1, y2) + line_width
    bbox_mask = torch.logical_and(
        torch.logical_and(xx >= x_min, xx <= x_max),
        torch.logical_and(yy >= y_min, yy <= y_max)
    ).float()
    line_mask = line_mask * bbox_mask
    
    # Arrow mask
    dir_x = dx / line_length
    dir_y = dy / line_length
    perp_x = -dir_y
    perp_y = dir_x
    
    p1 = torch.tensor([x2, y2])
    p2 = torch.tensor([x2 - arrow_size*dir_x - arrow_size*perp_x, y2 - arrow_size*dir_y - arrow_size*perp_y])
    p3 = torch.tensor([x2 - arrow_size*dir_x + arrow_size*perp_x, y2 - arrow_size*dir_y + arrow_size*perp_y])
    arrow_vertices = torch.stack([p1, p2, p3])
    arrow_mask = _rasterize_triangle(xx, yy, arrow_vertices)
    
    total_mask = (line_mask + arrow_mask) > 0
    total_mask = total_mask.float()
    
    # 更新画布（原有逻辑不变）
    canvas = canvas * (1 - total_mask.unsqueeze(0)) + color.unsqueeze(1).unsqueeze(2) * total_mask.unsqueeze(0)
    # 更新对应通道的物体掩码
    object_masks[:, :, object_channel] = total_mask
    
    return canvas, object_masks

def render_point(
    canvas: torch.Tensor,
    object_masks: torch.Tensor,  # 改为多通道掩码
    point: torch.Tensor,
    color: torch.Tensor,
    object_channel: int,  # 改为物体对应的通道索引
    point_radius: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a point as a small circle to canvas and update multi-channel object masks."""
    x, y = point
    y_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    x_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    dist = torch.sqrt((xx - x)**2 + (yy - y)**2)
    point_mask = (dist <= point_radius).float()
    
    # 更新画布（原有逻辑不变）
    canvas = canvas * (1 - point_mask.unsqueeze(0)) + color.unsqueeze(1).unsqueeze(2) * point_mask.unsqueeze(0)
    # 更新对应通道的物体掩码
    object_masks[:, :, object_channel] = point_mask
    
    return canvas, object_masks

def render_scene(scene_metadata: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Render scene and multi-channel object masks (instead of combined segment map)."""
    canvas = torch.zeros((3, CANVAS_SIZE, CANVAS_SIZE), dtype=torch.float32)
    num_objects = len(scene_metadata)
    # 初始化多通道物体掩码：[64,64,num_objects]，初始全0
    object_masks = torch.zeros((CANVAS_SIZE, CANVAS_SIZE, num_objects), dtype=torch.float32)
    
    # Render order: circles (bottom) → lines → points (top) for proper occlusion
    # 遍历每个物体，对应一个通道
    for channel_idx, obj_meta in enumerate(scene_metadata):
        obj_type = obj_meta["type"]
        if obj_type == "circle":
            canvas, object_masks = render_filled_circle(
                canvas, object_masks, obj_meta["geometry"], obj_meta["color_rgb"], channel_idx
            )
        elif obj_type == "line":
            canvas, object_masks = render_directed_line(
                canvas, object_masks, obj_meta["geometry"], obj_meta["color_rgb"], channel_idx
            )
        elif obj_type == "point":
            canvas, object_masks = render_point(
                canvas, object_masks, obj_meta["geometry"], obj_meta["color_rgb"], channel_idx
            )
    
    # Metadata
    metadata = {
        "num_objects": num_objects,
        "mask_info": {
            "background_value": 0,
            "object_channel_mapping": {obj["name"]: idx for idx, obj in enumerate(scene_metadata)},
            "mask_shape": object_masks.shape
        },
        "canvas_size": CANVAS_SIZE,
        "objects": scene_metadata
    }
    
    return canvas, object_masks, metadata


def generate_constrained_scene(dsl_program):
    solver = CCSPGeometricSolver()
    scene_metadata, solved_objects = solver.solve(dsl_program)
    
    # Render scene (返回多通道掩码)
    scene, object_masks, metadata = render_scene(scene_metadata)
    return scene, object_masks, metadata


# --------------------------
# Main Execution (Extended with Overlap Constraint Scenes)
# --------------------------
if __name__ == "__main__":
    # 保留原有DSL选择逻辑
    # dsl_program = colored_triangle()
    # dsl_program = colored_square()
    # dsl_program = colored_circle_with_perimeter_point()
    # dsl_program = colored_rectangle()
    # dsl_program = tangent_line_and_circle()
    # dsl_program = intersecting_lines()
    # dsl_program = line_inside_circle()
    # dsl_program = non_intersecting_lines()
    # dsl_program = line_outside_circle()
    dsl_program = overlapping_circles()  # 测试overlap约束（重叠圆）
    # dsl_program = non_overlapping_circles()  # 测试!overlap约束（不重叠圆）
    
    print(f"Using DSL Program:\n{dsl_program}")
    print("="*50)
    
    # Initialize solver and solve
    solver = CCSPGeometricSolver()
    scene_metadata, solved_objects = solver.solve(dsl_program)
    
    # Render scene（现在返回多通道掩码）
    scene, object_masks, metadata = render_scene(scene_metadata)
    
    # Print results（调整输出信息，适配多通道掩码）
    print("Scene Generation Results:")
    print(f"Number of objects: {metadata['num_objects']}")
    print(f"Object channel mapping: {metadata['mask_info']['object_channel_mapping']}")
    print(f"Scene tensor shape: {scene.shape}")
    print(f"Multi-channel object masks shape: {object_masks.shape}")  # 输出多通道掩码形状
    print(f"Each channel is a binary mask (1=object, 0=background)")
    
    # Optional: Save outputs（掩码改为保存多通道格式）
    # torch.save(scene, "geometric_scene.pt")
    # torch.save(object_masks, "multi_channel_object_masks.pt")
    print("\nScene generation completed successfully!")