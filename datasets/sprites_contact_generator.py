import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import shutil
from sklearn.model_selection import train_test_split

class SpritesDatasetGenerator:
    def __init__(self, 
                 output_dir="data/sprites_contact", 
                 img_size=64, 
                 min_shapes=2, 
                 max_shapes=5,
                 level=3,
                 test_size=0.2,
                 random_seed=42):
        """
        Initialize the sprite dataset generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the generated dataset
        img_size : int
            Size of the square images to generate (img_size x img_size)
        min_shapes : int
            Minimum number of shapes per image
        max_shapes : int
            Maximum number of shapes per image
        level : int
            Dataset complexity level:
            1 - Only circles
            2 - Regular shapes (circles and rectangles)
            3 - All shapes (circles, rectangles, triangles)
        test_size : float
            Proportion of the dataset to include in the test split (0.0 to 1.0)
        random_seed : int
            Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.img_size = img_size
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.level = level
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Available shape types based on level
        if level == 1:
            self.shape_types = ["circle"]
        elif level == 2:
            self.shape_types = ["circle", "rectangle"]
        else:  # level 3
            self.shape_types = ["circle", "rectangle", "triangle"]
        
        # Create output directories if they don't exist
        self.train_images_dir = os.path.join(output_dir, "train", "images")
        self.train_metadata_dir = os.path.join(output_dir, "train", "metadata")
        self.test_images_dir = os.path.join(output_dir, "test", "images")
        self.test_metadata_dir = os.path.join(output_dir, "test", "metadata")
        
        # Create temp directory for initial generation
        self.temp_images_dir = os.path.join(output_dir, "temp", "images")
        self.temp_metadata_dir = os.path.join(output_dir, "temp", "metadata")
        
        # Create all directories
        for dir_path in [self.train_images_dir, self.train_metadata_dir, 
                         self.test_images_dir, self.test_metadata_dir,
                         self.temp_images_dir, self.temp_metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_random_color(self):
        """
        Generate a random RGB color from a predefined set of colors.
        Returns both the RGB tuple and the color name.
        """
        # Define a set of distinct colors with their RGB values and names
        colors = [
            ((255, 0, 0), "red"),
            ((0, 255, 0), "green"),
            ((0, 0, 255), "blue"),
            ((255, 255, 0), "yellow"),
            ((255, 0, 255), "magenta"),
            ((0, 255, 255), "cyan"),
            ((255, 165, 0), "orange"),
            ((128, 0, 128), "purple"),
            ((165, 42, 42), "brown"),
            ((0, 128, 0), "dark_green")
        ]
        
        # Randomly select a color
        rgb, name = colors[np.random.randint(0, len(colors))]
        return rgb, name
    
    def generate_random_shape_params(self):
        """Generate random parameters for a shape."""
        shape_type = np.random.choice(self.shape_types)
        
        # Size should be significant but not too large
        min_size = self.img_size // 8
        max_size = self.img_size // 3
        
        # Generate random color
        color_rgb, color_name = self.generate_random_color()
        
        if shape_type == "circle":
            # For circle: (x, y, radius)
            radius = np.random.randint(min_size, max_size) // 2
            x = np.random.randint(radius, self.img_size - radius)
            y = np.random.randint(radius, self.img_size - radius)
            params = {
                "type": shape_type,
                "center": (x, y),
                "radius": radius,
                "color": color_rgb,
                "color_name": color_name,
                "bbox": (x-radius, y-radius, x+radius, y+radius)
            }
        
        elif shape_type == "rectangle":
            # For rectangle: (x1, y1, x2, y2) - top-left and bottom-right corners
            width = np.random.randint(min_size, max_size)
            height = np.random.randint(min_size, max_size)
            x = np.random.randint(0, self.img_size - width)
            y = np.random.randint(0, self.img_size - height)
            params = {
                "type": shape_type,
                "bbox": (x, y, x + width, y + height),
                "color": color_rgb,
                "color_name": color_name
            }
        
        elif shape_type == "triangle":
            # For triangle: create a simpler, more predictable triangle
            # This makes it easier to ensure consistency between image and mask
            
            # Decide triangle size (height and base width)
            base_width = np.random.randint(min_size, max_size)
            height = np.random.randint(min_size, max_size)
            
            # Position the triangle (top-left corner of bounding box)
            x = np.random.randint(0, self.img_size - base_width)
            y = np.random.randint(0, self.img_size - height)
            
            # Create a simple triangle (pointing up by default)
            # Three points: bottom-left, top-middle, bottom-right
            p1 = (x, y + height)                 # Bottom left
            p2 = (x + base_width // 2, y)        # Top middle
            p3 = (x + base_width, y + height)    # Bottom right
            
            # Store as a list for drawing
            points = [p1, p2, p3]
            
            # Calculate bounding box from the points (should be the same as constructed above)
            min_x = x
            min_y = y
            max_x = x + base_width
            max_y = y + height
            
            params = {
                "type": shape_type,
                "points": points,
                "color": color_rgb,
                "color_name": color_name,
                "bbox": (min_x, min_y, max_x, max_y)
            }
        
        return params
    
    def draw_shape(self, draw, shape_params):
        """Draw a shape on the image using the provided parameters."""
        if shape_params["type"] == "circle":
            draw.ellipse(
                (
                    shape_params["center"][0] - shape_params["radius"],
                    shape_params["center"][1] - shape_params["radius"],
                    shape_params["center"][0] + shape_params["radius"],
                    shape_params["center"][1] + shape_params["radius"]
                ),
                fill=shape_params["color"]
            )
        
        elif shape_params["type"] == "rectangle":
            draw.rectangle(
                shape_params["bbox"],
                fill=shape_params["color"]
            )
        
        elif shape_params["type"] == "triangle":
            draw.polygon(
                shape_params["points"],
                fill=shape_params["color"]
            )
    
    def check_overlap(self, shape1, shape2):
        """Check if two shapes overlap (have contact)."""
        # Get bounding boxes
        bbox1 = shape1["bbox"]
        bbox2 = shape2["bbox"]
        
        # Check if bounding boxes don't overlap
        if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
            bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
            return False
        
        # For more precise overlap detection, we could implement
        # shape-specific checks, but bounding box overlap works well
        # for a basic implementation
        return True
    
    def check_subset(self, shape1, shape2):
        """Check if shape1 is a subset of shape2 (shape1 is completely inside shape2)."""
        # Get bounding boxes
        bbox1 = shape1["bbox"]
        bbox2 = shape2["bbox"]
        
        # Check if shape1's bbox is completely inside shape2's bbox
        if (bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and
            bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]):
            # For a more precise check, we could implement shape-specific checks
            # but this bounding box check is a reasonable approximation
            return True
        
        return False
    
    def get_shape_center(self, shape):
        """Get the center coordinates of a shape."""
        x1, y1, x2, y2 = shape["bbox"]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_distance(self, shape1, shape2):
        """Calculate the Euclidean distance between the centers of two shapes."""
        center1 = self.get_shape_center(shape1)
        center2 = self.get_shape_center(shape2)
        
        # Calculate distance between centers
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        return np.sqrt(dx**2 + dy**2)
    
    def get_direction(self, shape1, shape2):
        """Determine directional relationships between shape1 and shape2."""
        # Get centers of shapes
        center1 = self.get_shape_center(shape1)
        center2 = self.get_shape_center(shape2)
        
        # Calculate direction vector
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]  # Note: in images, y increases downward
        
        # Calculate angle in degrees (0 is to the right, increases clockwise)
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Initialize directional relationships
        directions = {
            "north": False,
            "northeast": False,
            "east": False, 
            "southeast": False,
            "south": False,
            "southwest": False,
            "west": False,
            "northwest": False
        }
        
        # Set the appropriate direction based on angle
        # North is up (-90 degrees), increases clockwise
        if -112.5 <= angle < -67.5:
            directions["north"] = True
        elif -67.5 <= angle < -22.5:
            directions["northeast"] = True
        elif -22.5 <= angle < 22.5:
            directions["east"] = True
        elif 22.5 <= angle < 67.5:
            directions["southeast"] = True
        elif 67.5 <= angle < 112.5:
            directions["south"] = True
        elif 112.5 <= angle < 157.5:
            directions["southwest"] = True
        elif angle >= 157.5 or angle < -157.5:
            directions["west"] = True
        elif -157.5 <= angle < -112.5:
            directions["northwest"] = True
        
        return directions

    def determine_size_category(self, shape):
        """
        Determine the size category of a shape (small, medium, large).
        
        Size is determined based on the area of the shape's bounding box
        relative to the image size.
        """
        x1, y1, x2, y2 = shape["bbox"]
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Total image area
        img_area = self.img_size * self.img_size
        
        # Determine size category based on percentage of image area
        if area < img_area * 0.05:  # Less than 5% of image area
            return "small"
        elif area < img_area * 0.15:  # Less than 15% of image area
            return "medium"
        else:  # 15% or more of image area
            return "large"
    
    def create_one_hot_features(self, shapes):
        """
        Create one-hot encoded features for each shape.
        
        Returns:
            Dict of tensors for color, shape type, and size features
        """
        n = len(shapes)
        
        # Get all possible colors, shapes, and sizes
        all_colors = ["red", "green", "blue", "yellow", "magenta", 
                     "cyan", "orange", "purple", "brown", "dark_green"]
        all_shapes = ["circle", "rectangle", "triangle"]
        all_sizes = ["small", "medium", "large"]
        
        # Initialize feature matrices
        color_features = np.zeros((n, len(all_colors)), dtype=int)
        shape_features = np.zeros((n, len(all_shapes)), dtype=int)
        size_features = np.zeros((n, len(all_sizes)), dtype=int)
        
        # Fill in features for each shape
        for i, shape in enumerate(shapes):
            # Color features
            color_name = shape["color_name"]
            color_idx = all_colors.index(color_name)
            color_features[i, color_idx] = 1
            
            # Shape type features
            shape_type = shape["type"]
            shape_idx = all_shapes.index(shape_type)
            shape_features[i, shape_idx] = 1
            
            # Size features
            size = self.determine_size_category(shape)
            size_idx = all_sizes.index(size)
            size_features[i, size_idx] = 1
        
        return {
            "color": color_features,
            "shape": shape_features,
            "size": size_features,
            "color_names": all_colors,
            "shape_names": all_shapes,
            "size_names": all_sizes
        }
    
    def generate_sample(self, idx, output_images_dir, output_metadata_dir):
        """Generate a single sample with random shapes and compute relation matrices."""
        # Create a blank white image
        img = Image.new('RGB', (self.img_size, self.img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Determine number of shapes for this image
        num_shapes = np.random.randint(self.min_shapes, self.max_shapes + 1)
        
        # Generate shapes with retries to avoid too much overlap
        shapes = []
        max_attempts = 100  # Avoid infinite loops
        
        for _ in range(num_shapes):
            shape_params = self.generate_random_shape_params()
            
            # Try to avoid shapes that overlap too much with existing shapes
            attempts = 0
            while attempts < max_attempts:
                # Check overlap with existing shapes
                overlap_count = sum(1 for s in shapes if self.check_overlap(shape_params, s))
                
                # If overlaps with few shapes or reached max attempts, accept this shape
                if overlap_count <= 1 or attempts >= max_attempts - 1:
                    break
                
                # Try a new shape
                shape_params = self.generate_random_shape_params()
                attempts += 1
            
            shapes.append(shape_params)
        
        # Draw all shapes
        for shape_params in shapes:
            self.draw_shape(draw, shape_params)
        
        # Calculate all relationship matrices
        n = len(shapes)
        
        # Initialize matrices
        contact_matrix = np.zeros((n, n), dtype=int)
        subset_matrix = np.zeros((n, n), dtype=int)
        near_matrix = np.zeros((n, n), dtype=int)
        far_matrix = np.zeros((n, n), dtype=int)
        distance_matrix = np.zeros((n, n), dtype=float)
        
        # Direction matrices
        direction_matrices = {
            "north": np.zeros((n, n), dtype=int),
            "northeast": np.zeros((n, n), dtype=int),
            "east": np.zeros((n, n), dtype=int),
            "southeast": np.zeros((n, n), dtype=int),
            "south": np.zeros((n, n), dtype=int),
            "southwest": np.zeros((n, n), dtype=int),
            "west": np.zeros((n, n), dtype=int),
            "northwest": np.zeros((n, n), dtype=int)
        }
        
        # Set thresholds for near and far relationships
        near_threshold = self.img_size / 3
        far_threshold = self.img_size * 2 / 3
        
        # Calculate relationships between all shape pairs
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Skip self-relationships
                    continue
                
                # Calculate distance
                distance = self.get_distance(shapes[i], shapes[j])
                distance_matrix[i, j] = distance
                
                # Check contact/overlap
                if self.check_overlap(shapes[i], shapes[j]):
                    contact_matrix[i, j] = 1
                
                # Check subset relationship
                if self.check_subset(shapes[i], shapes[j]):
                    subset_matrix[i, j] = 1
                
                # Check near/far
                if distance < near_threshold and not contact_matrix[i, j]:
                    near_matrix[i, j] = 1
                if distance >= far_threshold:
                    far_matrix[i, j] = 1
                
                # Check directional relationships
                directions = self.get_direction(shapes[i], shapes[j])
                for direction, is_direction in directions.items():
                    if is_direction:
                        direction_matrices[direction][i, j] = 1
        
        # Create one-hot encoded features for unary properties (color, shape, size)
        unary_features = self.create_one_hot_features(shapes)
        
        # Save image
        img_path = os.path.join(output_images_dir, f"sprite_{idx:05d}.png")
        img.save(img_path)
        
        # Prepare shapes metadata with all necessary information
        shapes_metadata = []
        for i, s in enumerate(shapes):
            shape_info = {
                "id": i,
                "type": s["type"],
                "color": s["color"],
                "color_name": s["color_name"],
                "bbox": s["bbox"],
                "size_category": self.determine_size_category(s)
            }

            
            # Add shape-specific details for accurate mask generation
            if s["type"] == "circle":
                shape_info["center"] = s["center"]
                shape_info["radius"] = s["radius"]
            elif s["type"] == "triangle":
                shape_info["points"] = s["points"]
            
            shapes_metadata.append(shape_info)
        
        # Create metadata with all relation matrices and unary features
        metadata = {
            "image_id": idx,
            "num_shapes": len(shapes),
            "shapes": shapes_metadata,
            "relations": {
                "contact": contact_matrix.tolist(),
                "subset": subset_matrix.tolist(),
                "near": near_matrix.tolist(),
                "far": far_matrix.tolist(),
                "directions": {
                    direction: matrix.tolist()
                    for direction, matrix in direction_matrices.items()
                },
                "distance": distance_matrix.tolist()
            },
            "unary_features": {
                "color": unary_features["color"].tolist(),
                "shape": unary_features["shape"].tolist(),
                "size": unary_features["size"].tolist(),
                "color_names": unary_features["color_names"],
                "shape_names": unary_features["shape_names"],
                "size_names": unary_features["size_names"]
            }
        }
        
        metadata_path = os.path.join(output_metadata_dir, f"sprite_{idx:05d}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return img, shapes, metadata["relations"]
    
    def split_dataset(self, num_samples):
        """Split dataset into train and test sets."""
        # Get all sample indices
        indices = list(range(num_samples))
        
        # Split indices into train and test sets
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=self.test_size,
            random_state=self.random_seed
        )
        
        # Move files from temp to train/test directories
        for idx in train_indices:
            img_src = os.path.join(self.temp_images_dir, f"sprite_{idx:05d}.png")
            meta_src = os.path.join(self.temp_metadata_dir, f"sprite_{idx:05d}.json")
            
            img_dst = os.path.join(self.train_images_dir, f"sprite_{idx:05d}.png")
            meta_dst = os.path.join(self.train_metadata_dir, f"sprite_{idx:05d}.json")
            
            shutil.copy(img_src, img_dst)
            shutil.copy(meta_src, meta_dst)
        
        for idx in test_indices:
            img_src = os.path.join(self.temp_images_dir, f"sprite_{idx:05d}.png")
            meta_src = os.path.join(self.temp_metadata_dir, f"sprite_{idx:05d}.json")
            
            img_dst = os.path.join(self.test_images_dir, f"sprite_{idx:05d}.png")
            meta_dst = os.path.join(self.test_metadata_dir, f"sprite_{idx:05d}.json")
            
            shutil.copy(img_src, img_dst)
            shutil.copy(meta_src, meta_dst)
        
        # Clean up temp directory
        shutil.rmtree(os.path.join(self.output_dir, "temp"))
        
        return len(train_indices), len(test_indices)
    
    def visualize_sample(self, idx, split="train"):
        """Visualize a generated sample with its relation matrices."""
        # Determine which directory to use
        if split == "train":
            images_dir = self.train_images_dir
            metadata_dir = self.train_metadata_dir
        else:  # test
            images_dir = self.test_images_dir
            metadata_dir = self.test_metadata_dir
        
        # Load metadata
        metadata_path = os.path.join(metadata_dir, f"sprite_{idx:05d}.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load image
        img_path = os.path.join(images_dir, f"sprite_{idx:05d}.png")
        img = Image.open(img_path)
        
        # Check if we have the new relations format
        if "relations" in metadata:
            # New format with all relation types
            relation_matrix = np.array(metadata["relations"]["contact"])
        else:
            # Old format with just contact matrix
            relation_matrix = np.array(metadata["relation_matrix"])
        
        # Create figure with basic visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot image
        ax1.imshow(img)
        ax1.set_title(f"Sprite Image {idx} ({split})")
        ax1.axis('off')
        
        # Plot contact relation matrix as a heatmap
        ax2.imshow(relation_matrix, cmap='Blues')
        ax2.set_title("Contact Relation Matrix")
        
        # Add labels to relation matrix
        num_shapes = len(relation_matrix)
        ax2.set_xticks(range(num_shapes))
        ax2.set_yticks(range(num_shapes))
        ax2.set_xticklabels([f"{i}" for i in range(num_shapes)])
        ax2.set_yticklabels([f"{i}" for i in range(num_shapes)])
        
        for i in range(num_shapes):
            for j in range(num_shapes):
                text = ax2.text(j, i, relation_matrix[i, j],
                               ha="center", va="center", 
                               color="black" if relation_matrix[i, j] == 0 else "white")
        
        plt.tight_layout()
        plt.show()
        
        # If we have the new relations format, also visualize other relationships
        if "relations" in metadata:
            self.visualize_all_relations(metadata, img)
        
        return fig
    
    def visualize_all_relations(self, metadata, img):
        """Visualize all relationship matrices for a sample."""
        # Get all relation matrices
        relations = metadata["relations"]
        contact_matrix = np.array(relations["contact"])
        subset_matrix = np.array(relations["subset"])
        near_matrix = np.array(relations["near"])
        far_matrix = np.array(relations["far"])
        
        # Get direction matrices
        direction_matrices = {
            direction: np.array(matrix) 
            for direction, matrix in relations["directions"].items()
        }
        
        # Get distance matrix
        distance_matrix = np.array(relations["distance"])
        
        # Number of shapes
        num_shapes = len(contact_matrix)
        
        # Create figure with subplots for all relations
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Helper function to plot a matrix
        def plot_matrix(ax, matrix, title):
            im = ax.imshow(matrix, cmap='Blues')
            ax.set_title(title)
            # Add text annotations
            for i in range(num_shapes):
                for j in range(num_shapes):
                    text = ax.text(j, i, matrix[i, j],
                                  ha="center", va="center", 
                                  color="black" if matrix[i, j] < 0.5 else "white",
                                  fontsize=8)
        
        # Plot general relation matrices
        plot_matrix(axes[1], contact_matrix, 'Contact')
        plot_matrix(axes[2], subset_matrix, 'Subset')
        plot_matrix(axes[3], near_matrix, 'Near')
        plot_matrix(axes[4], far_matrix, 'Far')
        
        # Plot directional relation matrices
        directions = ['north', 'northeast', 'east', 'southeast', 
                     'south', 'southwest', 'west', 'northwest']
        
        for i, direction in enumerate(directions):
            plot_matrix(axes[i+5], direction_matrices[direction], direction.capitalize())
        
        # Plot distance matrix
        # Normalize distances for visualization
        if np.max(distance_matrix) > 0:
            normalized_distance = distance_matrix / np.max(distance_matrix)
        else:
            normalized_distance = distance_matrix
            
        im = axes[13].imshow(normalized_distance, cmap='viridis')
        axes[13].set_title('Distance (normalized)')
        plt.colorbar(im, ax=axes[13])
        
        # Adjust layout and spacing
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()
        
        return fig
    
    def generate_dataset(self, num_samples=300):
        """Generate a dataset with the specified number of samples."""
        print(f"Generating {num_samples} level-{self.level} samples to {self.output_dir}...")
        
        # Generate all samples to temp directory first
        for idx in range(num_samples):
            self.generate_sample(idx, self.temp_images_dir, self.temp_metadata_dir)
            if (idx + 1) % 10 == 0:
                print(f"Generated {idx + 1}/{num_samples} samples")
        
        # Split dataset into train and test sets
        print("Splitting dataset into train and test sets...")
        train_size, test_size = self.split_dataset(num_samples)
        
        print(f"Dataset split complete: {train_size} training samples, {test_size} test samples")
        print(f"Train images: {self.train_images_dir}")
        print(f"Test images: {self.test_images_dir}")
        
        # Write a dataset info file
        info = {
            "dataset_name": f"sprites_contact_level{self.level}",
            "description": f"Random sprites with contact relation matrices (Level {self.level})",
            "level": self.level,
            "shape_types": self.shape_types,
            "total_samples": num_samples,
            "train_samples": train_size,
            "test_samples": test_size,
            "test_split_ratio": self.test_size,
            "image_size": self.img_size,
            "creation_date": str(np.datetime64('now')),
            "random_seed": self.random_seed
        }
        
        with open(os.path.join(self.output_dir, "dataset_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        return self.output_dir


def generate_all_levels(base_dir="data/sprites_contact", num_samples=100, test_size=0.2, random_seed=45):
    """Generate datasets for all three levels."""
    datasets = []
    
    for level in [1, 2, 3]:
        level_dir = os.path.join(base_dir, f"level{level}")
        
        # Create generator for this level
        generator = SpritesDatasetGenerator(
            output_dir=level_dir,
            img_size=64,
            min_shapes=2,
            max_shapes=4,
            level=level,
            test_size=test_size,
            random_seed=44
        )
        
        # Generate the dataset
        generator.generate_dataset(num_samples=num_samples)
        
        # Add to list of generated datasets
        datasets.append({
            "level": level,
            "path": level_dir,
            "generator": generator
        })
    
    return datasets


def main():
    # Generate datasets for all three levels
    datasets = generate_all_levels(
        base_dir="data/sprites_contact",
        num_samples=300,
        test_size=0.2,
        random_seed=45
    )
    
    # Visualize examples from each level
    for dataset in datasets:
        level = dataset["level"]
        generator = dataset["generator"]
        
        print(f"\nLevel {level} Dataset Examples:")
        
        # Visualize one train and one test example
        generator.visualize_sample(0, split="train")
        generator.visualize_sample(0, split="test")


if __name__ == "__main__":
    main()