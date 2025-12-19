import pygame
import pymunk
import random
import cv2
import numpy as np
import os
import time
import json
import torch
from torch.utils.data import DataLoader

# --------------------------
# Core Configuration (Explicit Object Properties)
# --------------------------
# Fixed color palette (explicit list)
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (255, 0, 255),
    "cyan": (0, 255, 255)
}
COLOR_LIST = list(COLORS.keys())

# Fixed texture types (explicit list)
TEXTURES = ["solid", "striped", "checkered"]

# Spatial relations (allowed)
RELATIONS = ["left", "right", "above", "below", "on"]

# Simulation parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30
VIDEO_DURATION = 5  # Seconds per video
VIDEO_FPS = 30
OBJECT_SIZE_RANGE = (40, 60)  # Min/max diameter of circles
TOWER_HEIGHT_RANGE = (2, 5)  # Min/max number of objects in tower
STABILITY_THRESHOLD = 0.1  # Max velocity for stable objects
FALLING_THRESHOLD = 5.0  # Min velocity to consider object falling

# --------------------------
# Pygame/Pymunk Initialization
# --------------------------
def init_simulation():
    """Initialize physics and rendering environment"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    mask_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Physics space
    space = pymunk.Space()
    space.gravity = (0, 980)  # Downward gravity
    space.damping = 0.9
    
    # Add ground
    ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_shape = pymunk.Segment(ground_body, (0, SCREEN_HEIGHT-50), (SCREEN_WIDTH, SCREEN_HEIGHT-50), 5)
    ground_shape.friction = 0.8
    space.add(ground_body, ground_shape)
    
    return screen, mask_surface, clock, space, ground_body, ground_shape

# --------------------------
# Object/Texture Generation
# --------------------------
def generate_unique_mask_color(object_id):
    """Generate unique RGB color for object segmentation mask"""
    offset = 30
    r = (object_id * offset) % 256
    g = ((object_id + 8) * offset) % 256
    b = ((object_id + 16) * offset) % 256
    return (r, g, b)

def create_texture(width, height, texture_type, color_rgb):
    """Generate texture for circle objects"""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill(color_rgb)
    
    if texture_type == "striped":
        # Horizontal stripes (50% darker)
        stripe_color = tuple(c // 2 for c in color_rgb)
        stripe_height = height // 8
        for i in range(0, height, stripe_height * 2):
            pygame.draw.rect(surf, stripe_color, (0, i, width, stripe_height))
    
    elif texture_type == "checkered":
        # Checkerboard pattern (50% darker)
        check_color = tuple(c // 2 for c in color_rgb)
        checker_size = width // 8
        for x in range(0, width, checker_size * 2):
            for y in range(0, height, checker_size * 2):
                pygame.draw.rect(surf, check_color, (x, y, checker_size, checker_size))
                pygame.draw.rect(surf, check_color, (x+checker_size, y+checker_size, checker_size, checker_size))
    
    return surf

def create_circle_object(x, y, size, color_name, texture_type, object_id):
    """Create circle object with physics + visual properties"""
    color_rgb = COLORS[color_name]
    mask_color = generate_unique_mask_color(object_id)
    
    # Physics body (mass proportional to size)
    mass = (size / 50) ** 2
    moment = pymunk.moment_for_circle(mass, 0, size/2)
    body = pymunk.Body(mass, moment)
    body.position = (x, y)
    body.angular_damping = 0.1  # Circles roll naturally
    
    # Collision shape (circle)
    shape = pymunk.Circle(body, size/2)
    shape.friction = 0.6
    shape.elasticity = 0.05
    
    # Visual texture
    texture = create_texture(size, size, texture_type, color_rgb)
    
    return {
        "id": object_id,
        "body": body,
        "shape": shape,
        "size": size,
        "color": color_name,
        "texture": texture_type,
        "mask_color": mask_color,
        "color_rgb": color_rgb,
        "position": (x, y),
        "falling": False  # Will be updated during simulation
    }

def build_tower(space):
    """Build tower of circle objects (random properties)"""
    tower_objects = []
    tower_height = random.randint(*TOWER_HEIGHT_RANGE)
    base_x = SCREEN_WIDTH // 2
    
    for layer in range(tower_height):
        # Random properties
        offset_x = random.randint(-20, 20) if layer > 0 else 0
        size = random.randint(*OBJECT_SIZE_RANGE)
        color_name = random.choice(COLOR_LIST)
        texture_type = random.choice(TEXTURES)
        object_id = layer + 1
        
        # Position above previous layer
        y_pos = SCREEN_HEIGHT - 100 - (layer * (size + 10))
        
        # Create object and add to physics space
        obj = create_circle_object(base_x + offset_x, y_pos, size, color_name, texture_type, object_id)
        space.add(obj["body"], obj["shape"])
        tower_objects.append(obj)
    
    return tower_objects

# --------------------------
# Simulation & Frame Capture
# --------------------------
def render_rgb_frame(screen, tower_objects, ground_body):
    """Render RGB frame of simulation"""
    screen.fill((240, 240, 240))  # Light gray background
    
    # Draw ground
    pygame.draw.line(screen, (50, 50, 50), (0, SCREEN_HEIGHT-50), (SCREEN_WIDTH, SCREEN_HEIGHT-50), 5)
    
    # Draw objects
    for obj in tower_objects:
        body = obj["body"]
        texture = obj["texture_surf"]
        size = obj["size"]
        
        # Rotate texture to match physics rotation
        rotated_texture = pygame.transform.rotate(texture, -body.angle * 180 / np.pi)
        rect = rotated_texture.get_rect(center=(body.position.x, body.position.y))
        screen.blit(rotated_texture, rect)
    
    pygame.display.flip()
    return screen.copy()

def render_mask_frame(mask_surface, tower_objects):
    """Render segmentation mask frame (unique color per object)"""
    mask_surface.fill((0, 0, 0))  # Black background
    
    # Draw ground (gray)
    pygame.draw.line(mask_surface, (128, 128, 128), (0, SCREEN_HEIGHT-50), (SCREEN_WIDTH, SCREEN_HEIGHT-50), 5)
    
    # Draw objects with unique mask colors
    for obj in tower_objects:
        body = obj["body"]
        size = obj["size"]
        mask_color = obj["mask_color"]
        
        # Create solid color mask surface
        mask_obj_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        mask_obj_surf.fill(mask_color)
        
        # Rotate and position
        rotated_mask = pygame.transform.rotate(mask_obj_surf, -body.angle * 180 / np.pi)
        rect = rotated_mask.get_rect(center=(body.position.x, body.position.y))
        mask_surface.blit(rotated_mask, rect)
    
    return mask_surface.copy()

def run_simulation(space, screen, mask_surface, clock, tower_objects):
    """Run simulation and capture frames + object state"""
    rgb_frames = []
    mask_frames = []
    start_time = time.time()
    running = True
    
    # Pre-generate texture surfaces (optimization)
    for obj in tower_objects:
        obj["texture_surf"] = create_texture(obj["size"], obj["size"], obj["texture"], COLORS[obj["color"]])
    
    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # Step physics
        space.step(1/FPS)
        
        # Update falling state for each object
        for obj in tower_objects:
            vel = obj["body"].velocity
            speed = (vel.x**2 + vel.y**2) ** 0.5
            obj["falling"] = speed > FALLING_THRESHOLD
        
        # Capture frames
        rgb_frame = render_rgb_frame(screen, tower_objects, space)
        mask_frame = render_mask_frame(mask_surface, tower_objects)
        rgb_frames.append(rgb_frame)
        mask_frames.append(mask_frame)
        
        # Stop after video duration
        if time.time() - start_time >= VIDEO_DURATION:
            running = False
        
        clock.tick(FPS)
    
    # Convert frames to OpenCV format (BGR)
    cv_rgb_frames = []
    cv_mask_frames = []
    for frame in rgb_frames:
        arr = pygame.surfarray.array3d(frame)
        arr = np.transpose(arr, (1, 0, 2))  # Pygame (x,y) → OpenCV (y,x)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv_rgb_frames.append(arr)
    
    for frame in mask_frames:
        arr = pygame.surfarray.array3d(frame)
        arr = np.transpose(arr, (1, 0, 2))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv_mask_frames.append(arr)
    
    return cv_rgb_frames, cv_mask_frames, tower_objects

# --------------------------
# VQA Question/Answer Generation
# --------------------------
def get_object_relations(tower_objects):
    """Calculate spatial relations between objects"""
    relations = {}
    
    # Sort objects by initial layer (bottom to top)
    objects_sorted = sorted(tower_objects, key=lambda x: x["id"])
    
    for i, obj1 in enumerate(objects_sorted):
        obj1_pos = obj1["body"].position
        relations[obj1["id"]] = {}
        
        for j, obj2 in enumerate(objects_sorted):
            if i == j:
                continue
            
            obj2_pos = obj2["body"].position
            
            # Calculate spatial relations
            relations[obj1["id"]][obj2["id"]] = {
                "left": obj1_pos.x < obj2_pos.x - 5,  # 5px tolerance
                "right": obj1_pos.x > obj2_pos.x + 5,
                "above": obj1_pos.y < obj2_pos.y - 5,
                "below": obj1_pos.y > obj2_pos.y + 5,
                "on": (abs(obj1_pos.x - obj2_pos.x) < obj1["size"]/2) and (obj1_pos.y < obj2_pos.y - 5)
            }
    
    return relations

def gen_boolean_question(tower_objects, relations):
    """Generate boolean VQA question (yes/no)"""
    # Helper to get valid object IDs for a color
    def get_obj_ids_by_color(color):
        return [obj["id"] for obj in tower_objects if obj["color"] == color]
    
    # Question templates (FIXED scope + robust object selection)
    templates = [
        # Color-based
        ("Is there a {color} object?", 
         lambda obj, params: obj["color"] == params["color"], 
         lambda: {"color": random.choice(COLOR_LIST)}),
        
        # Texture-based
        ("Is there a {texture} object?", 
         lambda obj, params: obj["texture"] == params["texture"], 
         lambda: {"texture": random.choice(TEXTURES)}),
        
        # Falling-based
        ("Is there a falling object?", 
         lambda obj, params: obj["falling"] == True, 
         lambda: {}),
        
        # Color + falling
        ("Is the {color} object falling?", 
         lambda obj, params: obj["color"] == params["color"] and obj["falling"] == True, 
         lambda: {"color": random.choice(COLOR_LIST)}),
        
        # Spatial relation (binary) - FULLY FIXED
        ("Is the {color1} object {relation} the {color2} object?", 
         lambda obj, params: obj["id"] == params["obj1_id"] and relations[params["obj1_id"]][params["obj2_id"]][params["relation"]], 
         lambda: {
             "color1": random.choice(COLOR_LIST),
             "color2": random.choice([c for c in COLOR_LIST if c != random.choice(COLOR_LIST)]),
             "relation": random.choice(RELATIONS)
         })
    ]
    
    # Select random template
    template, condition, param_generator = random.choice(templates)
    
    # Generate params (call generator to avoid scope errors)
    params = param_generator()
    
    # Resolve valid object IDs for spatial relations
    if "color1" in params and "color2" in params:
        # Get valid objects for color1 (fallback to any object if none)
        color1_objs = get_obj_ids_by_color(params["color1"])
        if not color1_objs:
            color1_objs = [obj["id"] for obj in tower_objects]
        params["obj1_id"] = random.choice(color1_objs)
        
        # Get valid objects for color2 (different from obj1)
        color2_objs = [obj["id"] for obj in tower_objects if obj["color"] == params["color2"] and obj["id"] != params["obj1_id"]]
        if not color2_objs:
            # If no color2 objects, pick any object except obj1
            color2_objs = [obj["id"] for obj in tower_objects if obj["id"] != params["obj1_id"]]
            # Update color2 to match the fallback object (for accurate question text)
            fallback_obj = next(obj for obj in tower_objects if obj["id"] == color2_objs[0])
            params["color2"] = fallback_obj["color"]
        params["obj2_id"] = random.choice(color2_objs)
    
    # Generate question text (now uses valid params)
    question = template.format(**params)
    
    # Calculate answer (check if condition is true for any object)
    answer = any(condition(obj, params) for obj in tower_objects)
    
    # Generate logical program (simplified)
    program = f"exists:Logic(filter:Logic(scene:Objects(), { {k:v for k,v in params.items() if k in ['color1','color2','relation']} }))"
    
    return question, program, answer

def gen_numeric_question(tower_objects):
    """Generate numeric VQA question (count/sum/etc.)"""
    # Question templates
    templates = [
        # Count by color
        ("How many {color} objects are there?", 
         lambda params: len([obj for obj in tower_objects if obj["color"] == params["color"]]),
         {"color": random.choice(COLOR_LIST)}),
        
        # Count by texture
        ("How many {texture} objects are there?", 
         lambda params: len([obj for obj in tower_objects if obj["texture"] == params["texture"]]),
         {"texture": random.choice(TEXTURES)}),
        
        # Count falling objects
        ("How many objects are falling?", 
         lambda params: len([obj for obj in tower_objects if obj["falling"] == True]),
         {}),
        
        # Count total objects
        ("How many objects are in the tower?", 
         lambda params: len(tower_objects),
         {}),
        
        # Max size by color
        ("What is the size of the largest {color} object?", 
         lambda params: max([obj["size"] for obj in tower_objects if obj["color"] == params["color"]], default=0),
         {"color": random.choice(COLOR_LIST)})
    ]
    
    # Select random template
    template, compute_answer, params = random.choice(templates)
    
    # Generate question text
    question = template.format(**params)
    
    # Calculate answer
    answer = compute_answer(params)
    
    # Generate logical program
    program = f"count:Logic(filter:Logic(scene:Objects(), {params}))" if "many" in question else f"max:Integer(filter:Logic(scene:Objects(), {params}))"
    
    return question, program, answer

def gen_vqa_question(tower_objects, relations, question_type=None):
    """Generate random VQA question (boolean or numeric)"""
    if question_type is None:
        question_type = random.choice(["boolean", "numeric"])
    
    if question_type == "boolean":
        return gen_boolean_question(tower_objects, relations)
    else:
        return gen_numeric_question(tower_objects)

# --------------------------
# Dataset Class (Filterable, PyTorch Compatible)
# --------------------------
class TowerVQADatasetUnwrapped:
    def __init__(self, dataset_size, output_dir="tower_vqa_dataset"):
        self.dataset_size = dataset_size
        self.output_dir = output_dir
        self.data = self._generate_dataset()
        
    def _generate_dataset(self):
        """Generate full VQA dataset"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        
        dataset = {
            "videos": [],
            "masks": [],
            "questions": [],
            "programs": [],
            "answers": [],
            "object_info": [],
            "relations": []
        }
        
        # Initialize simulation
        screen, mask_surface, clock, space, ground_body, ground_shape = init_simulation()
        
        for idx in range(self.dataset_size):
            print(f"Generating sample {idx+1}/{self.dataset_size}...")
            
            # Reset physics space
            space.remove(*space.bodies, *space.shapes)
            space.add(ground_body, ground_shape)
            
            # Build tower and run simulation
            tower_objects = build_tower(space)
            rgb_frames, mask_frames, tower_objects = run_simulation(space, screen, mask_surface, clock, tower_objects)
            
            # Calculate object relations
            relations = get_object_relations(tower_objects)
            
            # Generate VQA question
            question, program, answer = gen_vqa_question(tower_objects, relations)
            
            # Save video and mask
            video_path = os.path.join(self.output_dir, f"videos/tower_{idx:04d}.mp4")
            mask_path = os.path.join(self.output_dir, f"masks/mask_{idx:04d}.mp4")
            
            # Write video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))
            mask_writer = cv2.VideoWriter(mask_path, fourcc, VIDEO_FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            for frame in rgb_frames:
                video_writer.write(frame)
            for frame in mask_frames:
                mask_writer.write(frame)
            
            video_writer.release()
            mask_writer.release()
            
            # Collect object info (serializable)
            object_info = []
            for obj in tower_objects:
                object_info.append({
                    "id": obj["id"],
                    "color": obj["color"],
                    "texture": obj["texture"],
                    "size": obj["size"],
                    "falling": obj["falling"],
                    "position": (float(obj["body"].position.x), float(obj["body"].position.y))
                })
            
            # Add to dataset
            dataset["videos"].append(video_path)
            dataset["masks"].append(mask_path)
            dataset["questions"].append(question)
            dataset["programs"].append(program)
            dataset["answers"].append(answer)
            dataset["object_info"].append(object_info)
            dataset["relations"].append(relations)
            
            # Cleanup
            for obj in tower_objects:
                space.remove(obj["body"], obj["shape"])
        
        # Save metadata
        with open(os.path.join(self.output_dir, "dataset_metadata.json"), "w") as f:
            json.dump(dataset, f, indent=2)
        
        pygame.quit()
        return dataset
    
    def __getitem__(self, index):
        """Get single sample (PyTorch compatible)"""
        # Load first frame of video as image tensor
        cap = cv2.VideoCapture(self.data["videos"][index])
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert to RGB, CHW, normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.transpose(2, 0, 1)
            frame = frame / 255.0
            frame = torch.tensor(frame, dtype=torch.float32)
        else:
            frame = torch.zeros((3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=torch.float32)
        
        return {
            "image": frame,
            "video_path": self.data["videos"][index],
            "mask_path": self.data["masks"][index],
            "query": self.data["questions"][index],
            "program": self.data["programs"][index],
            "answer": self.data["answers"][index],
            "object_info": self.data["object_info"][index],
            "relations": self.data["relations"][index]
        }
    
    def __len__(self):
        return self.dataset_size

class TowerVQADatasetFilterableView:
    def __init__(self, unwrapped_dataset):
        self._dataset = unwrapped_dataset
        self.indices = list(range(len(unwrapped_dataset)))
    
    def filter_by_question_type(self, question_type):
        """Filter by boolean/numeric questions"""
        filtered_indices = []
        for idx in self.indices:
            answer = self._dataset.data["answers"][idx]
            q_type = "boolean" if isinstance(answer, bool) else "numeric"
            if q_type == question_type:
                filtered_indices.append(idx)
        
        self.indices = filtered_indices
        return self
    
    def filter_by_color(self, color):
        """Filter samples with specific color objects"""
        filtered_indices = []
        for idx in self.indices:
            objects = self._dataset.data["object_info"][idx]
            if any(obj["color"] == color for obj in objects):
                filtered_indices.append(idx)
        
        self.indices = filtered_indices
        return self
    
    def filter_by_falling(self, falling=True):
        """Filter samples with falling objects"""
        filtered_indices = []
        for idx in self.indices:
            objects = self._dataset.data["object_info"][idx]
            if any(obj["falling"] == falling for obj in objects):
                filtered_indices.append(idx)
        
        self.indices = filtered_indices
        return self
    
    def make_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        """Create PyTorch DataLoader"""
        # Create subset dataset
        class SubsetDataset:
            def __init__(self, parent, indices):
                self.parent = parent
                self.indices = indices
            
            def __getitem__(self, idx):
                return self.parent[self.indices[idx]]
            
            def __len__(self):
                return len(self.indices)
        
        subset = SubsetDataset(self._dataset, self.indices)
        
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def __getitem__(self, idx):
        return self._dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)

def TowerVQADataset(dataset_size, output_dir="tower_vqa_dataset"):
    """Create filterable VQA dataset for tower simulations"""
    unwrapped = TowerVQADatasetUnwrapped(dataset_size, output_dir)
    return TowerVQADatasetFilterableView(unwrapped)

# --------------------------
# Save Dataset (Human-Readable Format)
# --------------------------
def save_tower_vqa_dataset(dataset_size, save_root="tower_vqa_dataset"):
    """Save dataset to structured folder with human-readable metadata"""
    # Generate dataset
    dataset = TowerVQADatasetUnwrapped(dataset_size, save_root)
    
    # Create metadata files
    metadata = {
        "dataset_size": dataset_size,
        "video_fps": VIDEO_FPS,
        "video_duration": VIDEO_DURATION,
        "color_options": COLOR_LIST,
        "texture_options": TEXTURES,
        "relations": RELATIONS,
        "samples": []
    }
    
    # Per-sample metadata
    for idx in range(dataset_size):
        metadata["samples"].append({
            "index": idx,
            "video_path": dataset.data["videos"][idx],
            "mask_path": dataset.data["masks"][idx],
            "question": dataset.data["questions"][idx],
            "program": dataset.data["programs"][idx],
            "answer": dataset.data["answers"][idx],
            "num_objects": len(dataset.data["object_info"][idx]),
            "has_falling_objects": any(obj["falling"] for obj in dataset.data["object_info"][idx])
        })
    
    # Save metadata
    with open(os.path.join(save_root, "full_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save questions/answers separately
    with open(os.path.join(save_root, "questions.txt"), "w") as f:
        f.write("\n".join(dataset.data["questions"]))
    
    with open(os.path.join(save_root, "answers.json"), "w") as f:
        json.dump(dataset.data["answers"], f, indent=2)
    
    print(f"Dataset saved to {save_root}")
    print(f"- {dataset_size} video samples")
    print(f"- {len(COLOR_LIST)} color options: {COLOR_LIST}")
    print(f"- {len(TEXTURES)} texture options: {TEXTURES}")
    print(f"- Supported relations: {RELATIONS}")

# --------------------------
# Usage Example
# --------------------------
if __name__ == "__main__":
    # Generate and save dataset (100 samples)
    save_tower_vqa_dataset(
        dataset_size=100,
        save_root="tower_vqa_dataset"
    )
    
    # Create filterable dataset
    dataset = TowerVQADataset(100)
    
    # Example filters
    boolean_questions = dataset.filter_by_question_type("boolean")
    red_objects = dataset.filter_by_color("red")
    falling_objects = dataset.filter_by_falling(True)
    
    # Create dataloader
    dataloader = dataset.make_dataloader(batch_size=8, shuffle=True)
    
    # Iterate through data
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Question: {batch['query'][0]}")
        print(f"Answer: {batch['answer'][0]}")
        print(f"Image shape: {batch['image'].shape}")
        break