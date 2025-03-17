# Assuming you already have defined SceneDataset and collate_variable_components
from datasets.scene_dataset import SceneDataset, collate_variable_components
from rinarak.dklearn.nn.pnn import PointNetfeat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm


class PointCloudDecoder(nn.Module):
    def __init__(self, feature_dim=1024, output_points=1024, output_dim=3):
        super().__init__()
        
        # MLP to generate points from the latent features
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, output_points * output_dim)
        )
        
        self.output_points = output_points
        self.output_dim = output_dim
    
    def forward(self, x):
        # x is the feature vector from PointNetfeat [batch_size, feature_dim]
        batch_size = x.size(0)

        x = self.mlp(x)
        
        # Reshape to [batch_size, output_points, output_dim]
        x = x.view(batch_size, self.output_points, self.output_dim)
        
        return x

class PointAutoEncoder(nn.Module):
    def __init__(self, input_channels=3, decode_points=1024, feature_dim=1024):
        super().__init__()
        self.encoder = PointNetfeat(global_feat=True, feature_transform=True, channel=input_channels)
        self.decoder = PointCloudDecoder(feature_dim=feature_dim, output_points=decode_points, output_dim=input_channels)
        self.decode_points = decode_points
        self.scale = 2.0
    
    def forward(self, x):
        features = self.encoder(x)

        reconstructed = self.decoder(features)
        return reconstructed, features
    
    def encode(self, x): return self.encoder(x)
    
    def decode(self, x): return self.decoder(x) * self.scale

# Chamfer Distance loss for point cloud reconstruction
def chamfer_distance(x, y):
    """
    Compute chamfer distance between two point clouds
    x: [batch_size, num_points, dim]
    y: [batch_size, num_points, dim]
    """
    # For each point in x, find the nearest point in y
    x_expanded = x.unsqueeze(2)  # [batch_size, num_points, 1, dim]
    y_expanded = y.unsqueeze(1)  # [batch_size, 1, num_points, dim]
    
    # Compute distances
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)  # [batch_size, num_points, num_points]
    
    # Get minimum distances in both directions
    minx = torch.min(dist, dim=2)[0]  # [batch_size, num_points]
    miny = torch.min(dist, dim=1)[0]  # [batch_size, num_points]
    
    # Sum over all points
    chamfer_loss = torch.mean(minx) + torch.mean(miny)
    
    return chamfer_loss

def train(model, train_loader, epochs, device, save_dir="outputs/models"):
    """Train the autoencoder model"""
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Process each component in each scene
            batch_loss = 0
            batch_count = 0
            
            for components in batch["components"]:
                for comp in components:
                    # Get point positions (and colors if available)
                    if "colors" in comp and comp["colors"] is not None:
                        xyzs, rgbs = torch.tensor(comp["points"]), torch.tensor(comp["colors"])
                        input_features = torch.cat([xyzs, rgbs], dim=-1).float().unsqueeze(0)
                        input_channels = 6  # xyz + rgb
                    else:
                        xyzs = torch.tensor(comp["points"])
                        input_features = xyzs.float().unsqueeze(0)
                        input_channels = 3  # xyz only

                    # Skip if too few points
                    if input_features.shape[1] < 10:
                        continue
                    
                    input_features = input_features.to(device)
                    
                    # Forward pass
                    reconstructed, _ = model(input_features)
                    
                    # Compute loss

                    num_pts = input_features.shape[1]
                    reconstructed = reconstructed[:,:num_pts, :]

                    loss = chamfer_distance(input_features[:,:,:3], reconstructed[:,:,:3])
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss += loss.item()
                    batch_count += 1
            
            if batch_count > 0:
                avg_batch_loss = batch_loss / batch_count
                epoch_loss += avg_batch_loss
                num_batches += 1
                pbar.set_postfix({"loss": avg_batch_loss})
        
        scheduler.step()
        
        # Calculate average loss for the epoch
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
            
            # Save model if it's the best so far
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_autoencoder.pth"))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, os.path.join(save_dir, f"autoencoder_checkpoint_epoch_{epoch+1}.pth"))
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_autoencoder.pth"))
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    
    return train_losses

def visualize_reconstruction(model, dataset, device, num_samples=5, output_dir="outputs/results"):
    """Visualize original and reconstructed point clouds"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        count = 0
        for idx in range(min(len(dataset), 20)):  # Check first 20 items to find enough samples
            if count >= num_samples:
                break
                
            data = dataset[idx]
            
            for comp_idx, comp in enumerate(data["components"]):
                if count >= num_samples:
                    break
                    
                # Skip if too few points
                if comp["points"].shape[0] < 10:
                    continue
                
                # Get point positions (and colors if available)
                if "colors" in comp and comp["colors"] is not None:
                    xyzs, rgbs = torch.tensor(comp["points"]), torch.tensor(comp["colors"])
                    input_features = torch.cat([xyzs, rgbs], dim=-1).float().unsqueeze(0)
                else:
                    xyzs = torch.tensor(comp["points"])
                    input_features = xyzs.float().unsqueeze(0)
                
                input_features = input_features.to(device)
                
                # Get reconstruction
                reconstructed, _ = model(input_features)
                
                
                # Move tensors to CPU for visualization
                input_points = input_features.cpu().numpy()[0]

                reconstructed_points = reconstructed.cpu().numpy()[0]
                num_pts = input_points.shape[0]
                reconstructed_points = reconstructed_points[:num_pts, :]
                
                # Create directory for this sample
                sample_dir = os.path.join(output_dir, f"sample_{count+1}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save as PLY files
                save_point_cloud(input_points[:, :3], os.path.join(sample_dir, "original.ply"))
                save_point_cloud(reconstructed_points[:, :3], os.path.join(sample_dir, "reconstructed.ply"))
                
                # Visualize
                fig = plt.figure(figsize=(12, 6))
                
                # Original
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], s=2)
                ax1.set_title('Original Point Cloud')
                set_axes_equal(ax1)
                
                # Reconstructed
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], s=2)
                ax2.set_title('Reconstructed Point Cloud')
                set_axes_equal(ax2)
                
                plt.savefig(os.path.join(sample_dir, "comparison.png"))
                plt.close()
                
                count += 1

def save_point_cloud(points, filename):
    import open3d as o3d
    """Save points as a PLY file"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale"""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

if __name__ == "__main__":
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Example data directory path
    data_dir = "data"
    
    # Create the dataset
    dataset = SceneDataset(
        root_dir=data_dir,
        n_points=1024,  # Sample 1024 points per point cloud
        load_views=False  # Don't load view directories
    )
    
    print(f"Found {len(dataset)} scenes")
    
    # Create a data loader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_variable_components
    )
    
    # Check if we have colors in our point clouds
    has_colors = False
    input_channels = 3
    for data in dataset:
        for comp in data["components"]:
            if "colors" in comp and comp["colors"] is not None:
                has_colors = True
                input_channels = 6
                break
        if has_colors:
            break

    
    print(f"Input channels: {input_channels} ({'with colors' if has_colors else 'positions only'})")
    
    # Create the autoencoder model
    model = PointAutoEncoder(input_channels=input_channels, decode_points=2048)
    
    # Train the model
    epochs = 350
    train_losses = train(model, dataloader, epochs, device)
    
    # Visualize reconstructions
    visualize_reconstruction(model, dataset, device)
    
    print("Training and visualization complete!")