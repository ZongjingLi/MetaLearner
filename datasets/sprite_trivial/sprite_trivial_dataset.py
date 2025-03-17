import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class SpriteDataset(Dataset):
    """
    PyTorch Dataset for the Sprite dataset.
    
    Args:
        dataset_dir: Directory containing the dataset
        transform: Optional transform to apply to the images
        target_transform: Optional transform to apply to the targets
        load_in_memory: Whether to load all images into memory (faster but uses more RAM)
    """
    def __init__(self, dataset_dir="data/sprite_dataset", transform=None, target_transform=None, load_in_memory=False):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.load_in_memory = load_in_memory
        
        # Load metadata
        with open(os.path.join(dataset_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        # Load dataset information
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
            self.data = json.load(f)
        
        # Pre-load images if requested
        self.images = None
        self.masks = None
        
        if load_in_memory:
            self.images = []
            self.masks = []
            
            for i in range(len(self.data)):
                # Load image
                img_path = os.path.join(self.dataset_dir, "images", f"img_{i:05d}.png")
                img = np.array(Image.open(img_path))
                self.images.append(img)
                
                # Load masks
                masks = []
                for j in range(3):
                    mask_path = os.path.join(self.dataset_dir, "masks", f"img_{i:05d}_obj_{j}.png")
                    mask = np.array(Image.open(mask_path))
                    masks.append(mask)
                self.masks.append(masks)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image
        if self.load_in_memory:
            image = self.images[idx]
            masks = self.masks[idx]
        else:
            # Load image
            img_path = os.path.join(self.dataset_dir, "images", f"img_{idx:05d}.png")
            image = np.array(Image.open(img_path))
            
            # Load masks
            masks = []
            for j in range(3):
                mask_path = os.path.join(self.dataset_dir, "masks", f"img_{idx:05d}_obj_{j}.png")
                mask = np.array(Image.open(mask_path))
                masks.append(mask)
        
        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Convert to CHW format
        masks_tensor = torch.tensor(np.array(masks), dtype=torch.float32) / 255.0  # [3, H, W]
        
        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Get sample data
        sample_data = self.data[idx]
        
        # Create sample dictionary
        sample = {
            "image": image_tensor,
            "masks": masks_tensor,
            "relation_matrix": torch.tensor(sample_data["relation_matrix"], dtype=torch.float32),
            "shapes": sample_data["shapes"],
            "colors": sample_data["colors"],
        }
        
        # Add binary attribute tensors
        for shape in self.metadata["shapes"]:
            sample[shape] = torch.tensor(sample_data[shape], dtype=torch.float32)
        
        for color in self.metadata["colors"]:
            sample[color] = torch.tensor(sample_data[color], dtype=torch.float32)
        
        # Apply target transform if provided
        if self.target_transform:
            for key in sample:
                if key not in ["image", "masks", "shapes", "colors"]:
                    sample[key] = self.target_transform(sample[key])
        
        return sample

    def visualize_sample(self, idx):
        """
        Visualize a sample from the dataset.
        
        Args:
            idx: Index of the sample to visualize
        """
        sample = self[idx]
        
        # Convert tensors to numpy for visualization
        image = sample["image"].permute(1, 2, 0).numpy() * 255
        masks = sample["masks"].numpy() * 255
        relation_matrix = sample["relation_matrix"].numpy()
        
        # Plot
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # Plot full image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title("Scene Image")
        axes[0].axis('off')
        
        # Plot masks
        for i in range(3):
            axes[i+1].imshow(masks[i], cmap='gray')
            axes[i+1].set_title(f"Mask {i+1}: {sample['shapes'][i]}, {sample['colors'][i]}")
            axes[i+1].axis('off')
        
        # Plot relation matrix
        axes[4].matshow(relation_matrix)
        axes[4].set_title("Relation Matrix")
        axes[4].set_xlabel("Object Index")
        axes[4].set_ylabel("Object Index")
        
        for i in range(3):
            for j in range(3):
                axes[4].text(j, i, f"{relation_matrix[i, j]:.0f}", 
                            ha="center", va="center", color="w")
        
        plt.tight_layout()
        plt.show()
        
        # Print attributes
        print("Attributes:")
        print(f"Shapes: {sample['shapes']}")
        print(f"Colors: {sample['colors']}")
        print("\nBinary Tensors:")
        for shape in self.metadata["shapes"]:
            print(f"{shape}: {sample[shape].numpy()}")
        for color in self.metadata["colors"]:
            print(f"{color}: {sample[color].numpy()}")


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = SpriteDataset(dataset_dir="data/sprite_dataset")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Visualize a sample
    dataset.visualize_sample(0)
    
    # Iterate through dataloader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Masks shape: {batch['masks'].shape}")
        print(f"Relation matrix shape: {batch['relation_matrix'].shape}")
        
        # Process only first batch as example
        break
    
    # Example showing how to use the dataset for a model
    def simple_model_usage():
        # Create a simple model (example only)
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(16 * 16 * 48, 9)  # For relation matrix prediction
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv(x)))
                x = self.pool(torch.relu(self.conv(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                x = x.view(-1, 3, 3)  # Reshape to relation matrix
                return x
        
        # Create model, loss, optimizer
        model = SimpleModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop (example only)
        for epoch in range(2):  # Just 2 epochs for example
            for batch in dataloader:
                # Get data
                images = batch["image"]
                targets = batch["relation_matrix"]
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        print("Training finished!")
    
    # Uncomment to test model usage
    # simple_model_usage()