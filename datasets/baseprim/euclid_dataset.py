import torch
import random
from typing import Dict, List, Tuple
from domains.math.euclid_domain import EuclidExecutor

BASE_COLORS = {
    "red": torch.tensor([1.0, 0.0, 0.0]),
    "green": torch.tensor([0.0, 1.0, 0.0]),
    "blue": torch.tensor([0.0, 0.0, 1.0])
}
COLOR_VARIATION = 0.9
CANVAS_SIZE = 256
MIN_LINE_LENGTH = 20
MAX_LINE_LENGTH = 80
MIN_CIRCLE_RADIUS = 15
MAX_CIRCLE_RADIUS = 40
EDGE_BUFFER = 30


RELATION_PROBABILITY = 0.7  # 35% of scenes have intentional geometric relations
RELATION_TYPES = ["parallel", "perpendicular", "tangent", "contained_line", "contain_circle"]



def get_varied_color(base_color_name: str) -> torch.Tensor:
    """Generate slightly varied color within base category."""
    base = BASE_COLORS[base_color_name].clone()
    #print(base_color_name)
    if base_color_name == "red":
        base[1] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
        base[2] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
    elif base_color_name == "green":
        base[0] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
        base[2] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
    else:
        base[0] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
        base[1] += random.uniform(-COLOR_VARIATION, COLOR_VARIATION)
    return torch.clamp(base, 0.0, 1.0)

# Predefined DSL templates (covers all core relations)
DSL_TEMPLATES = [
    # Tangent line-circle
    "c1:circle(p1, p2)[color({color_c})]; l1:line(p3, p4)[tangent(l1, c1), color({color_l})]",
    # Parallel lines
    "l1:line(p1, p2)[color({color1})]; l2:line(p3, p4)[parallel(l2, l1), color({color2})]",
    # Perpendicular lines
    "l1:line(p1, p2)[color({color1})]; l2:line(p3, p4)[perpendicular(l2, l1), color({color2})]",
    # Contained line in circle
    "c1:circle(p1, p2)[color({color_c})]; l1:line(p3, p4)[contained_line(l1, c1), color({color_l})]",
    # Contained circles
    "c1:circle(p1, p2)[color({color1})]; c2:circle(p3, p4)[contain(c1, c2), color({color2})]",
    # Line-line intersect + circle tangent
    "l1:line(p1, p2)[color({color1})]; l2:line(p3, p4)[intersect(l2, l1), color({color2})]; c1:circle(p5, p6)[tangent(c1, l1), color({color3})]",

    # Tree Like
]

DSL_TEMPLATES =[
    #"l1:line(p1,p2)[color({color1})];l2:line(p2,p3)[color({color2})];l3:line(p2,p4)[color({color3})]",
    #"l1:line(p1,p2)[color({color1})];l2:line(p2,p3)[color({color2})];c1:circle(p1,p3)[color({color3})]",
    #"l1:line(p1,p2)[color({color1})];l2:line(p2,p3)[color({color2})];c1:circle(p3,p4)[color({color3})];l3:line(p3,p5)[perpendicular(l3, l2), color({color2})]"
    "l1:line(p1,p2)[color({color1})];l2:line(p2,p3)[perpendicular(l2, l1),color({color2})];l3:line(p3,p4)[perpendicular(l3, l2),color({color3})] ;l4:line(p4,p1)[color({color_c})]"
]

def generate_dsl_program() -> str:
    """Generate random valid EuclidConstraint DSL program (fixed color mapping)."""
    template = random.choice(DSL_TEMPLATES)
    colors = ["red", "green", "blue"]
    
    # Step 1: Assign base colors sequentially (no circular reference)
    color_c = random.choice(colors)
    color_l = random.choice([c for c in colors if c != color_c])  # Different from color_c
    color1 = random.choice(colors)
    color2 = random.choice([c for c in colors if c != color1])   # Different from color1
    available_colors3 = [c for c in colors if c not in [color1, color2]]
    color3 = random.choice(available_colors3) if available_colors3 else random.choice(colors)
    
    # Step 2: Map to color placeholders (matches template variables)
    color_map = {
        "color_c": color_c,
        "color_l": color_l,
        "color1": color1,
        "color2": color2,
        "color3": color3
    }
    
    # Step 3: Replace placeholders in template
    for key, val in color_map.items():
        if key in template:
            template = template.replace(f"{{{key}}}", val)
    #(template)
    return template


class DSLConstraintSolver:
    def __init__(self, executor: EuclidExecutor):
        self.executor = executor
        self.canvas_size = 256
        self.edge_buffer = 30
        self.min_line_length = 20
        self.max_line_length = 80
        self.min_circle_radius = 15
        self.max_circle_radius = 40


    def parse_dsl(self, dsl_program: str) -> List[Dict]:
        """Parse DSL program into structured object specs (type, params, constraints) — fixed constraint splitting."""
        objects = []
        statements = [s.strip() for s in dsl_program.split(";") if s.strip()]
    
        for stmt in statements:
            # Split obj_name:type(params)[constraints] (handle edge case where [] is at end)
            if "[" not in stmt or "]" not in stmt:
                raise ValueError(f"Invalid DSL statement (missing constraints bracket): {stmt}")
        
            obj_part, constraint_part = stmt.split("[", 1)  # Split only on first "["
            constraint_part = constraint_part.rsplit("]", 1)[0]  # Remove only last "]"
        
            # Parse obj_name:type(params)
            if ":" not in obj_part:
                raise ValueError(f"Invalid object part (missing colon): {obj_part}")
            obj_name, type_params = obj_part.split(":", 1)
        
            if "(" not in type_params or ")" not in type_params:
                raise ValueError(f"Invalid type params (missing parentheses): {type_params}")
            obj_type, params_str = type_params.split("(", 1)
            params_str = params_str.rsplit(")", 1)[0].strip()  # Extract params inside ()
        
            # Parse params (free points or object refs)
            param_list = [p.strip() for p in params_str.split(",")] if params_str else []
        
            # ------------------------------
            # Fixed constraint parsing (handles commas inside parentheses)
            # ------------------------------
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
                    # Split only when comma is outside parentheses
                    if current_constraint.strip():
                        constraints.append(current_constraint.strip())
                    current_constraint = ""
                else:
                    current_constraint += char
        
            # Add the last constraint after loop ends
            if current_constraint.strip():
                constraints.append(current_constraint.strip())
        
            # Separate color constraint and geometric constraints
            color = None
            geo_constraints = []
            for const in constraints:
                if const.startswith("color("):
                    # Extract color(red) → red (handle optional whitespace)
                    color_val = const.replace("color(", "").replace(")", "").strip()
                    color = color_val if color_val in ["red", "green", "blue"] else None
                else:
                    geo_constraints.append(const)  # e.g., "perpendicular(l2, l1)"
        
            objects.append({
            "name": obj_name.strip(),
            "type": obj_type.strip(),
            "params": param_list,
            "color": color,
            "geo_constraints": geo_constraints
        })
    
        return objects
    def solve_free_point(self) -> torch.Tensor:
        """Solve free point (p1, p2) → random valid (x,y) coordinate."""
        x = random.uniform(self.edge_buffer, self.canvas_size - self.edge_buffer)
        y = random.uniform(self.edge_buffer, self.canvas_size - self.edge_buffer)
        return torch.tensor([x, y], dtype=torch.float32)

    def solve_circle(self, circle_spec: Dict, solved_objects: Dict) -> torch.Tensor:
        """Solve circle config (x,y,r) from specs (params + constraints)."""
        params = circle_spec["params"]
        if len(params) == 2:
            # Params: center point + edge point (e.g., circle(p1, p2))
            p1 = solved_objects[params[0]] if params[0] in solved_objects else self.solve_free_point()
            p2 = solved_objects[params[1]] if params[1] in solved_objects else self.solve_free_point()
            center = p1
            radius = torch.norm(p2 - p1, p = 2 )
            # Clamp radius to valid range
            #radius = torch.clamp(radius, self.min_circle_radius, self.max_circle_radius)
        else:
            # Default: random center + radius (fallback)
            center = self.solve_free_point()
            radius = torch.tensor(random.uniform(self.min_circle_radius, self.max_circle_radius), dtype=torch.float32)
        
        # Apply constraints (e.g., contain(c1, c2))
        for const in circle_spec["geo_constraints"]:
            if const.startswith("contain("):
                # contain(c1, c2) → c1 must contain c2 (c1 is already solved)
                ref_obj_name = const.split("(")[1].split(",")[0].strip()
                ref_obj = solved_objects[ref_obj_name]  # ref_obj is circle (x,y,r)
                # Adjust current circle (c2) to be inside ref_obj (c1)
                ref_center = ref_obj[:2]
                ref_radius = ref_obj[2]
                # Random offset inside ref circle
                offset_angle = torch.tensor(random.uniform(0, 2*torch.pi), dtype=torch.float32)
                offset_dist = torch.tensor(random.uniform(0, ref_radius - self.min_circle_radius - 5), dtype=torch.float32)
                center = ref_center + torch.tensor([
                    offset_dist * torch.cos(offset_angle),
                    offset_dist * torch.sin(offset_angle)
                ])
                radius = torch.tensor(random.uniform(self.min_circle_radius, ref_radius - offset_dist - 5), dtype=torch.float32)
        
        return torch.cat([center, radius[None,...]])

    def solve_line(self, line_spec: Dict, solved_objects: Dict) -> torch.Tensor:
        """Solve line config (x1,y1,x2,y2) from specs (params + constraints)."""
        params = line_spec["params"]
        if len(params) == 2:
            #print(solved_objects.keys())
            # Params: start + end points (e.g., line(p1, p2))

            p1 = solved_objects[params[0]] if params[0] in solved_objects else self.solve_free_point()
            p2 = solved_objects[params[1]] if params[1] in solved_objects else self.solve_free_point()
            line = torch.cat([p1, p2])
            # Clamp to valid length
            #print(p1, solved_objects[params[0]] if params[0] in solved_objects else "not")
            length = self.executor.length(line)
            if length < self.min_line_length:
                # Extend line to min length
                dir_vec = (p2 - p1) / (length + 1e-8)
                p2 = p1 + dir_vec * self.min_line_length
                line = torch.cat([p1, p2])
            #print(line)

        else:

            # Default: random line (fallback)
            start = self.solve_free_point()
            angle = torch.tensor(random.uniform(0, 2*torch.pi), dtype=torch.float32)
            length = torch.tensor(random.uniform(self.min_line_length, self.max_line_length), dtype=torch.float32)
            end = start + torch.tensor([length * torch.cos(angle), length * torch.sin(angle)])
            end = torch.clamp(end, self.edge_buffer, self.canvas_size - self.edge_buffer)
            line = torch.cat([start, end])

        for const in line_spec["geo_constraints"]:
            #print("const:",const, solved_objects.keys())
            if const.startswith("parallel("):
                # parallel(l2, l1) → l2 parallel to l1 (l1 is solved)
                ref_line_name = const.split("(")[1].split(",")[1].strip()[:-1]
                ref_line = solved_objects[ref_line_name]
                # Keep start point, adjust end point to match ref line direction
                start = line[:2]
                ref_dir = ref_line[2:] - ref_line[:2]
                ref_dir = ref_dir / (torch.norm(ref_dir) + 1e-8)
                length = self.executor.length(line)
                end = start + ref_dir * length
                line = torch.cat([start, end])
            
            elif const.startswith("perpendicular("):
                # perpendicular(l2, l1) → l2 perpendicular to l1
                #print(const)
                ref_line_name = const.split("(")[1].split(",")[1].strip()[:-1]
                ref_line = solved_objects[ref_line_name]
                start = line[:2]
                ref_dir = ref_line[2:] - ref_line[:2]
                perp_dir = torch.tensor([-ref_dir[1], ref_dir[0]])  # Perpendicular direction
                perp_dir = perp_dir / (torch.norm(perp_dir) + 1e-8)
                length = self.executor.length(line)
                end = start + perp_dir * length
                line = torch.cat([start, end])
            
            elif const.startswith("tangent("):
                # tangent(l1, c1) → line tangent to circle (c1 is solved)
                circle_name = const.split("(")[1].split(",")[1].strip()[:-1]
                circle = solved_objects[circle_name]
                circle_center = circle[:2]
                circle_radius = circle[2]
                # Solve tangent line (as in your structured generator)
                tangent_angle = torch.tensor(random.uniform(0, 2*torch.pi), dtype=torch.float32)
                tangent_point = circle_center + torch.tensor([
                    circle_radius * torch.cos(tangent_angle),
                    circle_radius * torch.sin(tangent_angle)
                ])
                line_angle = tangent_angle + torch.pi/2
                length = self.executor.length(line)
                start = tangent_point - torch.tensor([
                    (length/2) * torch.cos(line_angle),
                    (length/2) * torch.sin(line_angle)
                ])
                end = tangent_point + torch.tensor([
                    (length/2) * torch.cos(line_angle),
                    (length/2) * torch.sin(line_angle)
                ])
                start = torch.clamp(start, self.edge_buffer, self.canvas_size - self.edge_buffer)
                end = torch.clamp(end, self.edge_buffer, self.canvas_size - self.edge_buffer)
                line = torch.cat([start, end])
            
            elif const.startswith("contained_line("):
                # contained_line(l1, c1) → line inside circle (c1 is solved)
                #print(const)
                circle_name = const.split("(")[1].split(",")[1].strip()[:-1]
                circle = solved_objects[circle_name]
                circle_center = circle[:2]
                circle_radius = circle[2]
                # Solve line inside circle (as in your structured generator)
                length = torch.tensor(random.uniform(self.min_line_length, circle_radius * 1.5), dtype=torch.float32)
                line_angle = torch.tensor(random.uniform(0, 2*torch.pi), dtype=torch.float32)
                start_offset = torch.tensor(random.uniform(0, circle_radius - length/2 - 5), dtype=torch.float32)
                start = circle_center + torch.tensor([
                    start_offset * torch.cos(line_angle),
                    start_offset * torch.sin(line_angle)
                ])
                end = start + torch.tensor([
                    length * torch.cos(line_angle + torch.pi/4),
                    length * torch.sin(line_angle + torch.pi/4)
                ])
                line = torch.cat([start, end])
        
        return line

    def solve(self, dsl_program: str) -> Tuple[List[Dict], Dict]:
        """Main solve function: parse DSL → compute all object configs."""
        parsed_objects = self.parse_dsl(dsl_program)
        solved_objects = {}  # Key: obj_name, Value: geometry tensor (line:4D, circle:3D)
        scene_metadata = []
        
        for obj in parsed_objects:
            obj_name = obj["name"]
            obj_type = obj["type"]
            color = obj["color"]


            
            # Solve geometry based on type
            if obj_type == "circle":
                geometry = self.solve_circle(obj, solved_objects)
            elif obj_type == "line":
                geometry = self.solve_line(obj, solved_objects)
                #print(obj["name"], obj["params"], geometry)
                for i,param in enumerate(obj["params"]):
                    solved_objects[param] = geometry[2*i:2*i+2]


            else:
                raise ValueError(f"Unknown object type: {obj_type}")

            # Store solved geometry and metadata
            solved_objects[obj_name] = geometry
            scene_metadata.append({
                "name": obj_name,
                "type": obj_type,
                "geometry": geometry,
                "color_name": color,
                "color_rgb": get_varied_color(color)  # Reuse your color variation function
            })
        
        return scene_metadata, solved_objects
    

def render_filled_circle(canvas: torch.Tensor, circle: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    cx, cy, r = circle
    y_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    x_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
    circle_mask = (dist <= r).float()
    canvas = canvas * (1 - circle_mask.unsqueeze(0)) + color.unsqueeze(1).unsqueeze(2) * circle_mask.unsqueeze(0)
    return canvas


def render_directed_line(canvas: torch.Tensor, line: torch.Tensor, color: torch.Tensor, line_width: int = 2) -> torch.Tensor:
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    line_length = torch.sqrt(dx**2 + dy**2)
    if line_length < 1e-6:
        return canvas
    
    y_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    x_coords = torch.linspace(0, CANVAS_SIZE-1, CANVAS_SIZE, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    numerator = torch.abs(dy * xx - dx * yy + x2*y1 - y2*x1)
    denominator = torch.sqrt(dy**2 + dx**2)
    dist_to_line = numerator / denominator
    line_mask = (dist_to_line <= line_width).float()
    
    x_min, x_max = torch.min(x1, x2), torch.max(x1, x2)
    y_min, y_max = torch.min(y1, y2), torch.max(y1, y2)
    bbox_mask = torch.logical_and(
        torch.logical_and(xx >= x_min - line_width, xx <= x_max + line_width),
        torch.logical_and(yy >= y_min - line_width, yy <= y_max + line_width)
    ).float()
    line_mask = line_mask * bbox_mask
    
    canvas = canvas * (1 - line_mask.unsqueeze(0)) + color.unsqueeze(1).unsqueeze(2) * line_mask.unsqueeze(0)
    
    # Arrowhead
    arrow_size = 8
    perp_dx = -dy / line_length * arrow_size
    perp_dy = dx / line_length * arrow_size
    p1 = torch.tensor([x2, y2])
    p2 = p1 - torch.tensor([dx/line_length * arrow_size, dy/line_length * arrow_size]) + torch.tensor([perp_dx, perp_dy])
    p3 = p1 - torch.tensor([dx/line_length * arrow_size, dy/line_length * arrow_size]) - torch.tensor([perp_dx, perp_dy])
    arrow_vertices = torch.stack([p1, p2, p3])
    arrow_mask = _rasterize_triangle(xx, yy, arrow_vertices)
    canvas = canvas * (1 - arrow_mask.unsqueeze(0)) + color.unsqueeze(1).unsqueeze(2) * arrow_mask.unsqueeze(0)
    
    return canvas


def _rasterize_triangle(xx: torch.Tensor, yy: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = vertices
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = torch.stack([xx - v0[0], yy - v0[1]], dim=-1)
    dot00 = torch.sum(v0v1 * v0v1)
    dot01 = torch.sum(v0v1 * v0v2)
    dot02 = torch.sum(v0v1 * v0p)
    dot11 = torch.sum(v0v2 * v0v2)
    dot12 = torch.sum(v0v2 * v0p)
    
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    mask = torch.logical_and(torch.logical_and(u >= 0, v >= 0), (u + v) <= 1).float()
    return mask


def generate_constrained_scene(executor: EuclidExecutor) -> Tuple[torch.Tensor, Dict]:
    """End-to-end pipeline: DSL → Solve → Render."""
    # Step 1: Generate random DSL constraint program
    dsl_program = generate_dsl_program()
    print(f"Generated DSL Program: {dsl_program}")
    
    # Step 2: Solve constraints for object configs
    solver = DSLConstraintSolver(executor)

    scene_metadata, solved_objects = solver.solve(dsl_program)
    
    # Step 3: Render scene (reuse your existing renderers)
    canvas = torch.zeros((3, CANVAS_SIZE, CANVAS_SIZE), dtype=torch.float32)
    
    # Render circles first, then lines (avoid occlusion)
    for obj_meta in scene_metadata:
        if obj_meta["type"] == "circle":
            canvas = render_filled_circle(canvas, obj_meta["geometry"], obj_meta["color_rgb"])
    
    for obj_meta in scene_metadata:
        if obj_meta["type"] == "line":
            canvas = render_directed_line(canvas, obj_meta["geometry"], obj_meta["color_rgb"])
    
    # Add DSL program to metadata for traceability
    full_metadata = {
        "dsl_program": dsl_program,
        "objects": scene_metadata,
        "num_objects": len(scene_metadata)
    }
    
    return canvas, full_metadata
