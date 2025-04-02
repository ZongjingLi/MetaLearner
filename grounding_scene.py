# Assuming you already have defined SceneDataset and collate_variable_components
from datasets.scene_dataset import SceneDataset, collate_variable_components

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class PointCloudUpsampler:
    @staticmethod
    def upsample_density_aware(points, target_count, radius=0.1, density_weight=1.0):
        """
        Density-aware upsampling that adds more points in sparse regions.
        
        Args:
            points (np.ndarray or torch.Tensor): Input point cloud of shape (N, D)
            target_count (int): Desired number of points after upsampling
            radius (float): Radius for density estimation
            density_weight (float): Weight for density influence (higher = more uniform)
        
        Returns:
            np.ndarray or torch.Tensor: Upsampled point cloud of shape (target_count, D)
        """
        # If we already have enough points, just return the original or subsample
        if points.shape[0] >= target_count:
            if isinstance(points, torch.Tensor):
                indices = torch.randperm(points.shape[0])[:target_count]
                return points[indices]
            else:
                indices = np.random.choice(points.shape[0], target_count, replace=False)
                return points[indices]
        
        # Convert to numpy if tensor, ensuring we use float32
        is_tensor = isinstance(points, torch.Tensor)
        device = points.device if is_tensor else None
        
        if is_tensor:
            points_np = points.detach().cpu().numpy().astype(np.float32)
        else:
            points_np = points.astype(np.float32)
        
        # Build KD-tree for neighbor search
        nn_model = NearestNeighbors(n_neighbors=min(10, points_np.shape[0]), algorithm='auto', n_jobs=-1)
        nn_model.fit(points_np)
        
        # Estimate local density for each point
        distances, _ = nn_model.kneighbors(points_np)
        # Use average distance to k nearest neighbors as density indicator
        densities = np.mean(distances, axis=1)
        
        # Normalize and invert densities (lower density = higher probability for new points)
        if np.max(densities) > np.min(densities):
            normalized_densities = 1.0 - (densities - np.min(densities)) / (np.max(densities) - np.min(densities))
        else:
            normalized_densities = np.ones_like(densities)
        
        # Apply density weight (higher weight = more uniform sampling)
        sampling_weights = np.power(normalized_densities, density_weight)
        sampling_weights = sampling_weights / np.sum(sampling_weights)
        
        # Generate new points based on estimated density
        num_points_to_add = target_count - points_np.shape[0]
        new_points = np.zeros((num_points_to_add, points_np.shape[1]))
        
        for i in range(num_points_to_add):
            # Sample base point with probability inversely proportional to density
            base_idx = np.random.choice(len(points_np), p=sampling_weights)
            base_point = points_np[base_idx]
            
            # Find nearest neighbors
            distances, indices = nn_model.kneighbors([base_point], n_neighbors=min(3, points_np.shape[0]))
            
            if len(indices[0]) < 2:  # If not enough neighbors, use random offset
                offset = np.random.normal(0, 0.02, points_np.shape[1])
            else:
                # Get vector to neighbor
                neighbor_idx = indices[0][1]  # First neighbor (excluding itself)
                neighbor = points_np[neighbor_idx]
                
                # Create offset vector (with random scaling and rotation)
                direction = neighbor - base_point
                
                # Scale the offset
                scale = np.random.random() * 0.7 + 0.3  # 0.3 to 1.0
                offset = direction * scale * 0.5  # Half the distance to neighbor
                
                # Add small random noise for diversity
                offset += np.random.normal(0, 0.01, points_np.shape[1])
            
            # Create new point
            new_points[i] = base_point + offset
        
        # Combine original and new points
        upsampled_points = np.vstack([points_np, new_points])
        
        # Convert back to tensor if necessary, always using float32 for MPS compatibility
        if is_tensor:
            return torch.tensor(upsampled_points, dtype=torch.float32, device=device)
        else:
            return upsampled_points.astype(np.float32)


# Define the Point Cloud Autoencoder (using the simpler model provided)
class PointCloudAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128, output_points=1024):
        super(PointCloudAutoencoder, self).__init__()
        self.input_channels = input_channels
        self.output_points = output_points
        
        # Encoder layers
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        # Max pooling to handle variable number of points
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # FC layers for latent space
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, output_points * input_channels)
        
    def encode(self, x):
        # x shape: batch_size x input_channels x num_points
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Max pooling across points
        x = self.max_pool(x)  # shape: batch_size x 512 x 1
        x = x.view(-1, 512)   # shape: batch_size x 512
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)       # shape: batch_size x latent_dim
        
        return x
        
    def decode(self, z):
        # z shape: batch_size x latent_dim
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)       # shape: batch_size x (output_points*input_channels)
        
        # Reshape to point cloud
        x = x.view(-1, self.output_points, self.input_channels)
        
        return x
        
    def forward(self, x):
        # x shape: batch_size x num_points x input_channels
        # Transpose to match encoder input shape
        x_t = x.transpose(1, 2)  # shape: batch_size x input_channels x num_points
        
        # Encode and decode
        z = self.encode(x_t)
        y = self.decode(z)
        
        return y, z


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
                    
                    # Match number of points for loss calculation
                    num_pts = min(input_features.shape[1], reconstructed.shape[1])
                    input_subset = input_features[:, :num_pts, :]
                    reconstructed_subset = reconstructed[:, :num_pts, :]
                    
                    # Compute loss (only on XYZ coordinates if we have colors)
                    if input_channels > 3:
                        loss = chamfer_distance(input_subset[:, :, :3], reconstructed_subset[:, :, :3])
                    else:
                        loss = chamfer_distance(input_subset, reconstructed_subset)
                    
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
    """
    Visualize original and reconstructed point clouds at both component and scene levels
    
    Args:
        model: The trained autoencoder model
        dataset: The dataset containing scenes with components
        device: The device to run inference on
        num_samples: Number of scenes to visualize
        output_dir: Directory to save visualizations
    """
    import open3d as o3d
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        # Process a limited number of scenes
        for scene_idx in range(min(len(dataset), num_samples)):
            scene_data = dataset[scene_idx]
            
            # Create directory for this scene
            scene_dir = os.path.join(output_dir, f"scene_{scene_idx+1}")
            os.makedirs(scene_dir, exist_ok=True)
            
            # Lists to collect all original and reconstructed points for the entire scene
            all_original_points = []
            all_reconstructed_points = []
            
            # Process each component in the scene
            valid_components = 0
            for comp_idx, comp in enumerate(scene_data["components"]):
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
                
                # Match number of points
                num_pts = min(input_features.shape[1], reconstructed.shape[1])
                
                # Move tensors to CPU for visualization
                input_points = input_features.cpu().numpy()[0][:num_pts]
                reconstructed_points = reconstructed.cpu().numpy()[0][:num_pts]
                
                # Add to scene collection (using only XYZ coordinates)
                all_original_points.append(input_points[:, :3])
                all_reconstructed_points.append(reconstructed_points[:, :3])
                
                # Create directory for this component
                comp_dir = os.path.join(scene_dir, f"component_{comp_idx+1}")
                os.makedirs(comp_dir, exist_ok=True)
                
                # Save as PLY files
                save_point_cloud(input_points[:, :3], os.path.join(comp_dir, "original.ply"))
                save_point_cloud(reconstructed_points[:, :3], os.path.join(comp_dir, "reconstructed.ply"))
                
                # Visualize component
                fig = plt.figure(figsize=(12, 6))
                
                # Original
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], s=2)
                ax1.set_title(f'Original Component {comp_idx+1}')
                set_axes_equal(ax1)
                
                # Reconstructed
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], s=2)
                ax2.set_title(f'Reconstructed Component {comp_idx+1}')
                set_axes_equal(ax2)
                
                plt.savefig(os.path.join(comp_dir, "comparison.png"))
                plt.close()
                
                valid_components += 1
            
            # If we have valid components, visualize the entire scene
            if valid_components > 0:
                # Concatenate all points for scene-level visualization
                scene_original = np.vstack(all_original_points)
                scene_reconstructed = np.vstack(all_reconstructed_points)
                
                # Save scene PLY files
                save_point_cloud(scene_original, os.path.join(scene_dir, "original_scene.ply"))
                save_point_cloud(scene_reconstructed, os.path.join(scene_dir, "reconstructed_scene.ply"))
                
                # Visualize complete scene
                fig = plt.figure(figsize=(15, 7))
                
                # Original scene
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(scene_original[:, 0], scene_original[:, 1], scene_original[:, 2], 
                           s=1, alpha=0.7, c=scene_original[:, 2], cmap='viridis')
                ax1.set_title(f'Original Scene {scene_idx+1} ({len(scene_original)} points)')
                set_axes_equal(ax1)
                
                # Reconstructed scene
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(scene_reconstructed[:, 0], scene_reconstructed[:, 1], scene_reconstructed[:, 2], 
                           s=1, alpha=0.7, c=scene_reconstructed[:, 2], cmap='viridis')
                ax2.set_title(f'Reconstructed Scene {scene_idx+1} ({len(scene_reconstructed)} points)')
                set_axes_equal(ax2)
                
                plt.savefig(os.path.join(scene_dir, "scene_comparison.png"), dpi=300)
                plt.close()
                
                # Create open3d visualization for interactive viewing
                create_open3d_comparison(scene_original, scene_reconstructed, 
                                        os.path.join(scene_dir, "scene_comparison_o3d.png"))
                
                print(f"Visualized scene {scene_idx+1} with {valid_components} components")


def create_open3d_comparison(original_points, reconstructed_points, output_path):
    """
    Create an Open3D visualization of original vs reconstructed point clouds
    
    Args:
        original_points: Original point cloud (N, 3) numpy array
        reconstructed_points: Reconstructed point cloud (M, 3) numpy array
        output_path: Path to save the comparison image
    """
    import open3d as o3d
    
    # Create Open3D point cloud objects
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    
    reconstructed_pcd = o3d.geometry.PointCloud()
    reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed_points)
    
    # Paint original red and reconstructed blue
    original_pcd.paint_uniform_color([1, 0.3, 0.3])  # Light red
    reconstructed_pcd.paint_uniform_color([0.3, 0.3, 1])  # Light blue
    
    # Create a visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=900)
    
    # Add the geometries
    vis.add_geometry(original_pcd)
    vis.add_geometry(reconstructed_pcd)
    
    # Set up view
    view_control = vis.get_view_control()
    view_control.set_zoom(0.7)
    
    # Capture image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path)
    vis.destroy_window()


def analyze_reconstruction_quality(model, dataset, device, num_samples=10):
    """
    Analyze the reconstruction quality metrics for multiple scenes
    
    Args:
        model: The trained autoencoder model
        dataset: The dataset containing scenes with components
        device: The device to run inference on
        num_samples: Number of scenes to analyze
        
    Returns:
        dict: Dictionary containing quality metrics
    """
    model.eval()
    
    metrics = {
        'chamfer_distances': [],
        'component_counts': [],
        'point_counts': [],
        'scene_ids': []
    }
    
    with torch.no_grad():
        for scene_idx in range(min(len(dataset), num_samples)):
            scene_data = dataset[scene_idx]
            scene_cd = 0.0
            scene_components = 0
            scene_points = 0
            
            for comp in scene_data["components"]:
                if comp["points"].shape[0] < 10:
                    continue
                
                # Process point cloud
                if "colors" in comp and comp["colors"] is not None:
                    xyzs, rgbs = torch.tensor(comp["points"]), torch.tensor(comp["colors"])
                    input_features = torch.cat([xyzs, rgbs], dim=-1).float().unsqueeze(0)
                else:
                    xyzs = torch.tensor(comp["points"])
                    input_features = xyzs.float().unsqueeze(0)
                
                input_features = input_features.to(device)
                
                # Get reconstruction
                reconstructed, _ = model(input_features)
                
                # Match number of points
                num_pts = min(input_features.shape[1], reconstructed.shape[1])
                input_subset = input_features[:, :num_pts, :3]  # Only XYZ coords
                reconstructed_subset = reconstructed[:, :num_pts, :3]  # Only XYZ coords
                
                # Calculate Chamfer distance
                cd = chamfer_distance(input_subset, reconstructed_subset).item()
                
                scene_cd += cd
                scene_components += 1
                scene_points += num_pts
            
            if scene_components > 0:
                metrics['chamfer_distances'].append(scene_cd / scene_components)
                metrics['component_counts'].append(scene_components)
                metrics['point_counts'].append(scene_points)
                metrics['scene_ids'].append(scene_idx)
    
    # Calculate overall statistics
    metrics['mean_chamfer_distance'] = np.mean(metrics['chamfer_distances'])
    metrics['std_chamfer_distance'] = np.std(metrics['chamfer_distances'])
    
    return metrics


def save_point_cloud(points, filename):
    """Save points as a PLY file"""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def visualize_all_reconstructions(model, dataset, device, output_dir="outputs/results"):
    """Comprehensive visualization and analysis of reconstructions"""
    # Create output directory structure
    scene_output_dir = os.path.join(output_dir, "scenes")
    stats_output_dir = os.path.join(output_dir, "statistics")
    os.makedirs(scene_output_dir, exist_ok=True)
    os.makedirs(stats_output_dir, exist_ok=True)
    
    # Basic visualization
    visualize_reconstruction(model, dataset, device, num_samples=5, output_dir=scene_output_dir)
    
    # Calculate quality metrics
    metrics = analyze_reconstruction_quality(model, dataset, device, num_samples=min(10, len(dataset)))
    
    # Plot chamfer distance distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(metrics['chamfer_distances'])), metrics['chamfer_distances'])
    plt.xlabel('Scene Index')
    plt.ylabel('Average Chamfer Distance')
    plt.title(f'Reconstruction Quality (Mean CD: {metrics["mean_chamfer_distance"]:.4f})')
    plt.savefig(os.path.join(stats_output_dir, 'chamfer_distances.png'))
    plt.close()
    
    # Plot relationship between component count and reconstruction quality
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['component_counts'], metrics['chamfer_distances'])
    plt.xlabel('Number of Components')
    plt.ylabel('Average Chamfer Distance')
    plt.title('Relationship Between Scene Complexity and Reconstruction Quality')
    plt.savefig(os.path.join(stats_output_dir, 'complexity_vs_quality.png'))
    plt.close()
    
    # Save metrics to CSV
    import pandas as pd
    df = pd.DataFrame({
        'scene_id': metrics['scene_ids'],
        'components': metrics['component_counts'],
        'points': metrics['point_counts'],
        'chamfer_distance': metrics['chamfer_distances']
    })
    df.to_csv(os.path.join(stats_output_dir, 'reconstruction_metrics.csv'), index=False)
    
    print(f"Visualization and analysis complete. Results saved to {output_dir}")
    return metrics


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
        n_points=3000,  # Sample points per point cloud
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
    model = PointCloudAutoencoder(
        input_channels=input_channels,
        latent_dim=128,
        output_points=3000  # Use same number as dataset sampling
    ).to(device)
    
    # Train the model
    epochs = 200
    train_losses = train(model, dataloader, epochs, device)
    
    # Visualize reconstructions with both component-level and scene-level comparisons
    visualize_all_reconstructions(model, dataset, device)
    
    print("Training and visualization complete!")