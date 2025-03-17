import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import random
import json
from tqdm import tqdm

class SpriteDatasetGenerator:
    def __init__(self, 
                 output_dir="data/sprite_dataset",
                 img_height=32,
                 img_width=32*3,
                 n_samples=1000,
                 seed=42):
        """
        Initialize the Sprite Dataset Generator.
        
        Args:
            output_dir: Directory to save dataset
            img_height: Height of the image (default: 32)
            img_width: Width of the image (default: 96)
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.img_height = img_height
        self.img_width = img_width
        self.n_samples = n_samples
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        
        # Define shapes and colors
        self.shapes = ["circle", "cube", "triangle"]
        self.colors = ["red", "green", "blue"]
        
        # RGB values for colors
        self.color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        }
        
    def draw_shape(self, shape, color, size=32):
        """
        Draw a shape with specified color on a white background.
        
        Args:
            shape: One of ["circle", "cube", "triangle"]
            color: One of ["red", "green", "blue"]
            size: Size of the image (default: 32)
            
        Returns:
            PIL Image and mask
        """
        # Create a white background
        img = Image.new('RGB', (size, size), color=(255, 255, 255))
        mask = Image.new('L', (size, size), color=0)
        
        draw = ImageDraw.Draw(img)
        mask_draw = ImageDraw.Draw(mask)
        
        # Get RGB color
        rgb_color = self.color_map[color]
        
        # Draw shape
        padding = 4
        if shape == "circle":
            draw.ellipse([padding, padding, size-padding, size-padding], fill=rgb_color)
            mask_draw.ellipse([padding, padding, size-padding, size-padding], fill=255)
        elif shape == "cube":
            draw.rectangle([padding, padding, size-padding, size-padding], fill=rgb_color)
            mask_draw.rectangle([padding, padding, size-padding, size-padding], fill=255)
        elif shape == "triangle":
            points = [(size//2, padding), (padding, size-padding), (size-padding, size-padding)]
            draw.polygon(points, fill=rgb_color)
            mask_draw.polygon(points, fill=255)
            
        return img, mask
    
    def generate_sample(self):
        """
        Generate a single sample with 3 sprites.
        
        Returns:
            Dictionary containing the sample data
        """
        # Select random shapes and colors for each position
        selected_shapes = random.choices(self.shapes, k=3)
        selected_colors = random.choices(self.colors, k=3)
        
        # Create full image (32x96)
        full_img = Image.new('RGB', (self.img_width, self.img_height), color=(255, 255, 255))
        
        # Create masks for each object
        object_masks = []
        
        # Draw each shape
        for i in range(3):
            # Generate shape and mask
            img, mask = self.draw_shape(selected_shapes[i], selected_colors[i])
            
            # Paste onto full image
            full_img.paste(img, (i*32, 0))
            object_masks.append(mask)
        
        # Create binary tensors for attributes
        shape_tensors = {shape: np.zeros((3, 1)) for shape in self.shapes}
        color_tensors = {color: np.zeros((3, 1)) for color in self.colors}
        
        for i in range(3):
            shape_tensors[selected_shapes[i]][i, 0] = 1
            color_tensors[selected_colors[i]][i, 0] = 1
        
        # Create relation matrix (left/right)
        relation_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i < j:  # i is to the left of j
                    relation_matrix[i, j] = 1
                elif i > j:  # i is to the right of j
                    relation_matrix[i, j] = -1
                # i == j: stays 0 (same position)
        
        # Create sample dictionary
        sample = {
            "image": np.array(full_img),
            "masks": [np.array(mask) for mask in object_masks],
            "relation_matrix": relation_matrix.tolist(),
            "shapes": selected_shapes,
            "colors": selected_colors
        }
        
        # Add binary tensors
        for shape in self.shapes:
            sample[shape] = shape_tensors[shape].tolist()
        
        for color in self.colors:
            sample[color] = color_tensors[color].tolist()
        
        return sample
    
    def generate_dataset(self):
        """
        Generate the full dataset and save it.
        
        Returns:
            List of samples
        """
        dataset = []
        
        for i in tqdm(range(self.n_samples)):
            sample = self.generate_sample()
            
            # Save image and masks
            img = Image.fromarray(np.uint8(sample["image"]))
            img.save(os.path.join(self.output_dir, "images", f"img_{i:05d}.png"))
            
            # Save masks
            for j, mask in enumerate(sample["masks"]):
                mask_img = Image.fromarray(np.uint8(mask))
                mask_img.save(os.path.join(self.output_dir, "masks", f"img_{i:05d}_obj_{j}.png"))
            
            # Convert numpy arrays to lists for JSON serialization
            sample_json = {k: v for k, v in sample.items() if k not in ["image", "masks"]}
            
            dataset.append(sample_json)
        
        # Save dataset metadata
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump({
                "n_samples": self.n_samples,
                "img_height": self.img_height,
                "img_width": self.img_width,
                "shapes": self.shapes,
                "colors": self.colors
            }, f)
        
        # Save dataset
        with open(os.path.join(self.output_dir, "dataset.json"), "w") as f:
            json.dump(dataset, f)
            
        return dataset
    
    def visualize_sample(self, sample_idx=None):
        """
        Visualize a sample from the dataset.
        
        Args:
            sample_idx: Index of the sample to visualize, if None, generates a new sample
        
        Returns:
            None (displays the sample)
        """
        if sample_idx is not None:
            # Load from saved dataset
            img_path = os.path.join(self.output_dir, "images", f"img_{sample_idx:05d}.png")
            img = np.array(Image.open(img_path))
            
            mask_paths = [
                os.path.join(self.output_dir, "masks", f"img_{sample_idx:05d}_obj_{j}.png")
                for j in range(3)
            ]
            masks = [np.array(Image.open(path)) for path in mask_paths]
            
            # Load metadata
            with open(os.path.join(self.output_dir, "dataset.json"), "r") as f:
                dataset = json.load(f)
                sample_data = dataset[sample_idx]
        else:
            # Generate new sample
            sample = self.generate_sample()
            img = sample["image"]
            masks = sample["masks"]
            sample_data = {k: v for k, v in sample.items() if k not in ["image", "masks"]}
        
        # Plot
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # Plot full image
        axes[0].imshow(img)
        axes[0].set_title("Scene Image")
        axes[0].axis('off')
        
        # Plot masks
        for i in range(3):
            axes[i+1].imshow(masks[i], cmap='gray')
            axes[i+1].set_title(f"Mask {i+1}: {sample_data['shapes'][i]}, {sample_data['colors'][i]}")
            axes[i+1].axis('off')
        
        # Plot relation matrix
        axes[4].matshow(np.array(sample_data["relation_matrix"]))
        axes[4].set_title("Relation Matrix")
        axes[4].set_xlabel("Object Index")
        axes[4].set_ylabel("Object Index")
        
        for i in range(3):
            for j in range(3):
                axes[4].text(j, i, f"{sample_data['relation_matrix'][i][j]}", 
                             ha="center", va="center", color="w")
        
        plt.tight_layout()
        plt.show()
        
        # Print attributes
        print("Attributes:")
        print(f"Shapes: {sample_data['shapes']}")
        print(f"Colors: {sample_data['colors']}")
        print("\nBinary Tensors:")
        for shape in self.shapes:
            print(f"{shape}: {sample_data[shape]}")
        for color in self.colors:
            print(f"{color}: {sample_data[color]}")


if __name__ == "__main__":
    # Create generator
    generator = SpriteDatasetGenerator(n_samples=64)
    
    # Generate dataset
    dataset = generator.generate_dataset()
    
    # Visualize a sample
    generator.visualize_sample(sample_idx=0)