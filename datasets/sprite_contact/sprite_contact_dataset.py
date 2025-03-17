import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms


class SpritesContactDataset(Dataset):
    """
    PyTorch Dataset for loading sprite images with masks and contact matrices.
    
    Attributes:
        root_dir (str): Root directory of the dataset
        level (int): Dataset complexity level (1, 2, or 3)
        split (str): 'train' or 'test' split
        transform (callable, optional): Optional transform to be applied to images
        preload (bool): Whether to preload all data into memory
    """
    
    def __init__(self, root_dir="data/sprites_contact", level=3, split="train", 
                 transform=None, preload=False, return_masks=True):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the dataset
            level (int): Dataset complexity level (1, 2, or 3)
            split (str): 'train' or 'test' split
            transform (callable, optional): Optional transform to be applied to images
            preload (bool): Whether to preload all data into memory
            return_masks (bool): Whether to return individual masks for each sprite
        """
        self.root_dir = os.path.join(root_dir, f"level{level}")
        self.level = level
        self.split = split
        self.transform = transform
        self.preload = preload
        self.return_masks = return_masks
        
        # Define directories
        self.images_dir = os.path.join(self.root_dir, split, "images")
        self.metadata_dir = os.path.join(self.root_dir, split, "metadata")
        
        # Get dataset info
        with open(os.path.join(self.root_dir, "dataset_info.json"), 'r') as f:
            self.dataset_info = json.load(f)
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                  if f.endswith('.png')])
        
        # Preload data if requested
        self.cached_data = {}
        if preload:
            print(f"Preloading {len(self.image_files)} {split} samples into memory...")
            for idx in range(len(self.image_files)):
                self.cached_data[idx] = self._load_item(idx)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_files)
    
    def _load_item(self, idx):
        """
        Load a single item (image, masks, contact matrix) from disk.
        
        Args:
            idx (int): Index of the item to load
            
        Returns:
            tuple: (image, masks, contact_matrix, metadata) and other relation matrices and features
        """
        # Get file paths
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        metadata_path = os.path.join(self.metadata_dir, 
                                     self.image_files[idx].replace('.png', '.json'))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get relation matrices
        relations = metadata.get('relations', {})
        
        # Check for old format vs new format
        if 'relation_matrix' in metadata:
            # Old format
            contact_matrix = torch.tensor(metadata['relation_matrix'], dtype=torch.float32)
            subset_matrix = torch.zeros_like(contact_matrix)
            near_matrix = torch.zeros_like(contact_matrix)
            far_matrix = torch.zeros_like(contact_matrix)
            direction_matrices = {
                "north": torch.zeros_like(contact_matrix),
                "northeast": torch.zeros_like(contact_matrix),
                "east": torch.zeros_like(contact_matrix),
                "southeast": torch.zeros_like(contact_matrix),
                "south": torch.zeros_like(contact_matrix),
                "southwest": torch.zeros_like(contact_matrix),
                "west": torch.zeros_like(contact_matrix),
                "northwest": torch.zeros_like(contact_matrix)
            }
            distance_matrix = torch.zeros_like(contact_matrix)
        else:
            # New format with all relation types
            contact_matrix = torch.tensor(relations['contact'], dtype=torch.float32)
            subset_matrix = torch.tensor(relations['subset'], dtype=torch.float32)
            near_matrix = torch.tensor(relations['near'], dtype=torch.float32)
            far_matrix = torch.tensor(relations['far'], dtype=torch.float32)
            
            direction_matrices = {
                direction: torch.tensor(matrix, dtype=torch.float32)
                for direction, matrix in relations['directions'].items()
            }
            
            distance_matrix = torch.tensor(relations['distance'], dtype=torch.float32)
        
        # Create individual masks for each sprite
        if self.return_masks:
            masks = self._generate_masks(metadata, image.size)
        else:
            masks = None
        
        # Extract unary features
        unary_features = metadata.get('unary_features', {})
        color_features = torch.tensor(unary_features.get('color', []), dtype=torch.float32)
        shape_features = torch.tensor(unary_features.get('shape', []), dtype=torch.float32)
        size_features = torch.tensor(unary_features.get('size', []), dtype=torch.float32)
        
        # Get feature names
        color_names = unary_features.get('color_names', [])
        shape_names = unary_features.get('shape_names', [])
        size_names = unary_features.get('size_names', [])
        
        # Extract sprite information
        sprite_colors = []
        sprite_types = []
        sprite_sizes = []
        
        for shape in metadata['shapes']:

            sprite_colors.append(shape['color_name'])
            sprite_types.append(shape['type'])
            sprite_sizes.append(shape['size_category'])
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to convert to tensor
            image = transforms.ToTensor()(image)
        
        return (image, masks, contact_matrix, subset_matrix, near_matrix, far_matrix, 
                direction_matrices, distance_matrix, color_features, shape_features, 
                size_features, sprite_colors, sprite_types, sprite_sizes,
                color_names, shape_names, size_names, metadata)
    
    def _generate_masks(self, metadata, img_size):
        """
        Generate binary masks for each sprite in the image.
        
        Args:
            metadata (dict): Image metadata containing sprite information
            img_size (tuple): Image size (width, height)
            
        Returns:
            torch.Tensor: Binary masks for each sprite [num_sprites, H, W]
        """
        shapes = metadata['shapes']
        masks = []
        
        for shape in shapes:
            # Create an empty mask for this shape
            mask = Image.new('L', img_size, 0)
            draw = ImageDraw.Draw(mask)
            
            if shape['type'] == 'circle':
                # Extract bbox
                x1, y1, x2, y2 = shape['bbox']
                # Calculate radius and center
                radius = (x2 - x1) // 2
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw the circle
                draw.ellipse(
                    (center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius),
                    fill=255
                )
            
            elif shape['type'] == 'rectangle':
                # Draw the rectangle
                draw.rectangle(shape['bbox'], fill=255)
            
            elif shape['type'] == 'triangle':
                # For triangles, we must recreate the EXACT same triangle
                if 'points' in shape:
                    # The points are stored directly in the metadata as a list of coordinate tuples
                    # We need to convert them from the JSON format back to Python tuples
                    points = []
                    for point in shape['points']:
                        # Handle both list format and tuple format for compatibility
                        if isinstance(point, list):
                            points.append(tuple(point))
                        else:
                            points.append(point)
                    
                    # Draw using exactly the same points with polygon
                    draw.polygon(points, fill=255)
                else:
                    # Fallback to the simplest triangle approximation if points not found
                    # (this should not happen with updated generator)
                    print(f"Warning: No points data found for triangle. Using bbox approximation.")
                    x1, y1, x2, y2 = shape['bbox']
                    
                    # Recreate simple triangle (pointing up) from bbox
                    points = [
                        (x1, y2),                  # Bottom left
                        ((x1 + x2) // 2, y1),      # Top middle
                        (x2, y2)                   # Bottom right
                    ]
                    draw.polygon(points, fill=255)
            
            # Convert to tensor and add to list
            mask_tensor = transforms.ToTensor()(mask)
            masks.append(mask_tensor)
        
        # Stack all masks into a single tensor [num_sprites, H, W]
        if masks:
            masks_tensor = torch.stack(masks)
        else:
            # If no sprites, return an empty tensor
            masks_tensor = torch.zeros((0, img_size[1], img_size[0]), dtype=torch.float32)
        
        return masks_tensor
    
    def __getitem__(self, idx):
        """
        Get a dataset item by index.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            dict: All item attributes
        """
        if self.preload and idx in self.cached_data:
            image, masks, contact_matrix, subset_matrix, near_matrix, far_matrix, \
            direction_matrices, distance_matrix, color_features, shape_features, \
            size_features, sprite_colors, sprite_types, sprite_sizes, \
            color_names, shape_names, size_names, metadata = self.cached_data[idx]
        else:
            image, masks, contact_matrix, subset_matrix, near_matrix, far_matrix, \
            direction_matrices, distance_matrix, color_features, shape_features, \
            size_features, sprite_colors, sprite_types, sprite_sizes, \
            color_names, shape_names, size_names, metadata = self._load_item(idx)
    
        # Create return dictionary with all elements
        item = {
            'image': image,
            'contact_matrix': contact_matrix,
            'subset_matrix': subset_matrix,
            'near_matrix': near_matrix,
            'far_matrix': far_matrix,
            'direction_matrices': direction_matrices,
            'distance_matrix': distance_matrix,
            'color_features': color_features,
            'shape_features': shape_features,
            'size_features': size_features,
            'sprite_colors': sprite_colors,
            'sprite_types': sprite_types,
            'sprite_sizes': sprite_sizes,
            'color_names': color_names,
            'shape_names': shape_names,
            'size_names': size_names,
            'num_sprites': len(metadata['shapes'])
        }
        
        if self.return_masks:
            item['masks'] = masks
            item['num_sprites'] = masks.shape[0]
    
        return item
    
    def visualize_sample(self, idx):
        """
        Visualize a sample with its masks and contact matrix.
        Also overlay masks on the original image to verify alignment.
        
        Args:
            idx (int): Index of the sample to visualize
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get the sample
        sample = self[idx]
        image = sample['image']
        masks = sample['masks']
        contact_matrix = sample['contact_matrix']
        num_sprites = sample['num_sprites']
        
        # Convert image from tensor to numpy for visualization
        img_np = image.permute(1, 2, 0).numpy()
        
        # Create figure with subplots for better visualization
        num_cols = min(3, num_sprites + 1)  # At most 3 columns
        num_rows = 2 + (num_sprites + num_cols - 1) // num_cols  # Rows for masks and overlays
        
        # Create figure
        fig = plt.figure(figsize=(4 * num_cols, 3 * num_rows))
        
        # Plot original image in the first subplot
        plt.subplot(num_rows, num_cols, 1)
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot contact matrix in the second subplot
        plt.subplot(num_rows, num_cols, 2)
        plt.imshow(contact_matrix.numpy(), cmap='Blues')
        plt.title('Contact Matrix')
        for i in range(num_sprites):
            for j in range(num_sprites):
                plt.text(j, i, contact_matrix[i, j].item(),
                         ha="center", va="center", 
                         color="black" if contact_matrix[i, j].item() == 0 else "white")
        
        # Plot masks for each sprite and overlay on image
        for i in range(num_sprites):
            # Get mask for this sprite
            mask_np = masks[i, 0].numpy()
            
            # Plot individual mask
            plt.subplot(num_rows, num_cols, i + 3)  # Start from the third position
            plt.imshow(mask_np, cmap='gray')
            plt.title(f'Mask {i}: {sample["sprite_types"][i]}')
            plt.axis('off')
            
            # Create overlay - show mask boundary on the original image
            if i < num_sprites:
                overlay_idx = i + 3 + num_sprites
                if overlay_idx <= num_rows * num_cols:  # Make sure we don't exceed subplot grid
                    plt.subplot(num_rows, num_cols, overlay_idx)
                    plt.imshow(img_np)
                    
                    # Create colored overlay of the mask (semi-transparent)
                    colored_mask = np.zeros_like(img_np)
                    # Use different colors for different shape types
                    if sample["sprite_types"][i] == "circle":
                        colored_mask[:, :, 0] = mask_np * 1.0  # Red for circles
                    elif sample["sprite_types"][i] == "rectangle":
                        colored_mask[:, :, 1] = mask_np * 1.0  # Green for rectangles
                    else:  # Triangle or other
                        colored_mask[:, :, 2] = mask_np * 1.0  # Blue for triangles
                    
                    # Overlay mask on image
                    plt.imshow(colored_mask, alpha=0.5)
                    plt.title(f'Overlay: {sample["sprite_types"][i]}')
                    plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
        
    def visualize_relations(self, idx):
        """
        Visualize all relation matrices for a sample.
        
        Args:
            idx (int): Index of the sample to visualize
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get the sample
        sample = self[idx]
        image = sample['image']
        contact_matrix = sample['contact_matrix']
        subset_matrix = sample['subset_matrix']
        near_matrix = sample['near_matrix']
        far_matrix = sample['far_matrix']
        direction_matrices = sample['direction_matrices']
        distance_matrix = sample['distance_matrix']
        num_sprites = sample['num_sprites']
        
        # Convert image from tensor to numpy for visualization
        img_np = image.permute(1, 2, 0).numpy()
        
        # Create figure with subplots for all relations
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        # Plot original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Helper function to plot a matrix
        def plot_matrix(ax, matrix, title):
            matrix_np = matrix.numpy()
            im = ax.imshow(matrix_np, cmap='Blues')
            ax.set_title(title)
            # Add text annotations
            for i in range(num_sprites):
                for j in range(num_sprites):
                    text = ax.text(j, i, matrix_np[i, j],
                                  ha="center", va="center", 
                                  color="black" if matrix_np[i, j] < 0.5 else "white",
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
        distance_np = distance_matrix.numpy()
        if np.max(distance_np) > 0:
            normalized_distance = distance_np / np.max(distance_np)
        else:
            normalized_distance = distance_np
            
        im = axes[13].imshow(normalized_distance, cmap='viridis')
        axes[13].set_title('Distance (normalized)')
        plt.colorbar(im, ax=axes[13])
        
        # Adjust layout and spacing
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()
        
        return fig


def get_dataloader(root_dir="data/sprites_contact", level=3, split="train", 
                   batch_size=16, shuffle=True, num_workers=4, **kwargs):
    """
    Create a DataLoader for the sprites dataset.
    
    Args:
        root_dir (str): Root directory of the dataset
        level (int): Dataset complexity level (1, 2, or 3)
        split (str): 'train' or 'test' split
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        **kwargs: Additional arguments to pass to the SpritesContactDataset
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified dataset
    """
    dataset = SpritesContactDataset(root_dir=root_dir, level=level, split=split, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_sprites  # Use custom collate function for variable sprites
    )


def collate_variable_sprites(batch):
    """
    Custom collate function to handle variable number of sprites per image.
    
    Args:
        batch (list): Batch of samples from the dataset
        
    Returns:
        dict: Collated batch
    """
    # Extract all items from the batch
    images = [item['image'] for item in batch]
    contact_matrices = [item['contact_matrix'] for item in batch]
    subset_matrices = [item['subset_matrix'] for item in batch]
    near_matrices = [item['near_matrix'] for item in batch]
    far_matrices = [item['far_matrix'] for item in batch]
    direction_matrices_dict = {k: [] for k in batch[0]['direction_matrices'].keys()}
    distance_matrices = [item['distance_matrix'] for item in batch]
    num_sprites = [item['num_sprites'] for item in batch]
    
    # Get direction matrices for each direction
    for item in batch:
        for dir_key in direction_matrices_dict.keys():
            direction_matrices_dict[dir_key].append(item['direction_matrices'][dir_key])
    
    # Stack images
    images = torch.stack(images)
    
    # For masks, we can't simply stack because each image may have a different number of sprites
    if 'masks' in batch[0]:
        # Get max number of sprites in this batch
        max_sprites = max(num_sprites)
        
        # Pad masks to have the same number of sprites
        padded_masks = []
        for item in batch:
            masks = item['masks']
            curr_sprites = masks.shape[0]
            
            if curr_sprites < max_sprites:
                # Pad with empty masks
                padding = torch.zeros(
                    (max_sprites - curr_sprites, 1, masks.shape[2], masks.shape[3]),
                    dtype=masks.dtype
                )
                padded_mask = torch.cat([masks, padding], dim=0)
            else:
                padded_mask = masks
            
            padded_masks.append(padded_mask)
        
        # Stack padded masks
        masks = torch.stack(padded_masks)
    else:
        masks = None
    
    # For relation matrices, pad to max_sprites x max_sprites
    max_sprites = max(num_sprites)
    
    # Helper function to pad matrices
    def pad_and_stack_matrices(matrices):
        padded_matrices = []
        for mat, n in zip(matrices, num_sprites):
            # Create a max_sprites x max_sprites matrix filled with zeros
            padded = torch.zeros((max_sprites, max_sprites), dtype=mat.dtype)
            # Copy the original matrix into the top-left corner
            padded[:n, :n] = mat
            padded_matrices.append(padded)
        return torch.stack(padded_matrices)
    
    # Pad and stack all relation matrices
    contact_matrices = pad_and_stack_matrices(contact_matrices)
    subset_matrices = pad_and_stack_matrices(subset_matrices)
    near_matrices = pad_and_stack_matrices(near_matrices)
    far_matrices = pad_and_stack_matrices(far_matrices)
    distance_matrices = pad_and_stack_matrices(distance_matrices)
    
    # Pad and stack direction matrices
    direction_matrices = {}
    for dir_key, matrices in direction_matrices_dict.items():
        direction_matrices[dir_key] = pad_and_stack_matrices(matrices)
    
    # Create masks for valid sprite positions (1 for valid, 0 for padding)
    sprite_masks = torch.zeros((len(batch), max_sprites), dtype=torch.bool)
    for i, n in enumerate(num_sprites):
        sprite_masks[i, :n] = 1
    
    # Construct the batch dictionary
    batch_dict = {
        'images': images,
        'contact_matrices': contact_matrices,
        'subset_matrices': subset_matrices, 
        'near_matrices': near_matrices,
        'far_matrices': far_matrices,
        'direction_matrices': direction_matrices,
        'distance_matrices': distance_matrices,
        'num_sprites': torch.tensor(num_sprites),
        'sprite_masks': sprite_masks,  # Indicates which sprites are valid (not padding)
    }
    
    if masks is not None:
        batch_dict['masks'] = masks
    
    return batch_dict


def demo():
    """Demonstrate the use of the SpritesContactDataset."""
    # Create a dataset instance
    dataset = SpritesContactDataset(
        root_dir="data/sprites_contact",
        level=3,
        split="train",
        transform=None,
        preload=False
    )
    
    # Print dataset info
    print(f"Dataset level: {dataset.level}")
    print(f"Dataset split: {dataset.split}")
    print(f"Number of samples: {len(dataset)}")
    
    # Get and visualize a sample
    sample_idx = 0
    dataset.visualize_sample(sample_idx)
    
    # Visualize all relation matrices for the sample
    dataset.visualize_relations(sample_idx)
    
    # Create a DataLoader
    dataloader = get_dataloader(
        root_dir="data/sprites_contact",
        level=3,
        split="train",
        batch_size=4,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Print batch info
    print("\nBatch info:")
    print(f"Batch size: {batch['images'].shape[0]}")
    print(f"Images shape: {batch['images'].shape}")
    print(f"Masks shape: {batch['masks'].shape}")
    print(f"Contact matrices shape: {batch['contact_matrices'].shape}")
    print(f"Subset matrices shape: {batch['subset_matrices'].shape}")
    print(f"Direction matrices shape: {list(batch['direction_matrices'].keys())}")
    print(f"Number of sprites per image: {batch['num_sprites']}")
    
    # Visualize the first image in the batch with its masks
    b_idx = 0
    
    plt.figure(figsize=(15, 10))
    
    # Plot the image
    plt.subplot(2, 4, 1)
    img = batch['images'][b_idx].permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(f"Image {b_idx}")
    plt.axis('off')
    
    # Plot some relation matrices
    plt.subplot(2, 4, 2)
    cm = batch['contact_matrices'][b_idx].numpy()
    valid_sprites = batch['num_sprites'][b_idx].item()
    plt.imshow(cm[:valid_sprites, :valid_sprites], cmap='Blues')
    plt.title(f"Contact Matrix")
    
    plt.subplot(2, 4, 3)
    sm = batch['subset_matrices'][b_idx].numpy()
    plt.imshow(sm[:valid_sprites, :valid_sprites], cmap='Greens')
    plt.title(f"Subset Matrix")
    
    plt.subplot(2, 4, 4)
    nm = batch['near_matrices'][b_idx].numpy()
    plt.imshow(nm[:valid_sprites, :valid_sprites], cmap='Oranges')
    plt.title(f"Near Matrix")
    
    # Plot directional matrices
    plt.subplot(2, 4, 5)
    dm_north = batch['direction_matrices']['north'][b_idx].numpy()
    plt.imshow(dm_north[:valid_sprites, :valid_sprites], cmap='Purples')
    plt.title(f"North Direction")
    
    plt.subplot(2, 4, 6)
    dm_east = batch['direction_matrices']['east'][b_idx].numpy()
    plt.imshow(dm_east[:valid_sprites, :valid_sprites], cmap='Purples')
    plt.title(f"East Direction")
    
    plt.subplot(2, 4, 7)
    dm_south = batch['direction_matrices']['south'][b_idx].numpy()
    plt.imshow(dm_south[:valid_sprites, :valid_sprites], cmap='Purples')
    plt.title(f"South Direction")
    
    plt.subplot(2, 4, 8)
    dm_west = batch['direction_matrices']['west'][b_idx].numpy()
    plt.imshow(dm_west[:valid_sprites, :valid_sprites], cmap='Purples')
    plt.title(f"West Direction")
    
    plt.tight_layout()
    plt.show()
    
    # Now visualize some masks
    plt.figure(figsize=(12, 4))
    
    # Plot masks for the first 4 sprites
    for i in range(min(4, valid_sprites)):
        plt.subplot(1, 4, i + 1)
        mask = batch['masks'][b_idx, i, 0].numpy()
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()