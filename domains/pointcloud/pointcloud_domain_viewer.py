import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

def visualize_reconstruction(model, data, device, num_samples=5):
    model.eval()
    fig = plt.figure(figsize=(20, 8))
    
    with torch.no_grad():
        for idx in range(num_samples):
            # Original pointcloud
            ax1 = fig.add_subplot(2, num_samples, idx + 1, projection='3d')
            points = data[idx].cpu().numpy()
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c = "white")
            ax1.set_title(f'Original {idx+1}')
            ax1.axis('off')
            
            # Reconstructed pointcloud
            ax2 = fig.add_subplot(2, num_samples, idx + num_samples + 1, projection='3d')
            input_data = data[idx:idx+1].to(device)
            recon_batch, _, _ = model(input_data)
            recon_points = recon_batch[0].cpu().numpy()
            ax2.scatter(recon_points[:, 0], recon_points[:, 1], recon_points[:, 2], s=1, c = "white")
            ax2.set_title(f'Reconstructed {idx+1}')
            ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("outputs/point_samples.png",transparent=True)
    plt.show()

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_pointcloud_states(model, z=None, num_samples=5, device='cuda'):
    model.eval()
    fig = plt.figure(figsize=(8, 8))
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = z if z is not None else torch.randn(num_samples, 128).to(device)
        # Generate samples
        samples = model.decoder(z)
        samples = samples.cpu().numpy()
        
        for idx in range(z.shape[0]):
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            points = samples[idx]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5.0)
            
            # Set axis limits to be equal
            max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                                  points[:, 1].max() - points[:, 1].min(),
                                  points[:, 2].max() - points[:, 2].min()]).max() / 1.5
            
            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_title(f'Sample {idx+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("outputs/pointcloud_states")



def visualize_latent_space(model, test_loader, device):
    model.eval()
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            mu, _ = model.encoder(data)
            latent_vectors.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Reduce dimensionality to 2D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

def interpolate_shapes(model, data, device, num_steps=10):
    model.eval()
    with torch.no_grad():
        # Get two random shapes
        shape1 = data[0:1].to(device)
        shape2 = data[1:2].to(device)
        
        # Get their latent representations
        mu1, _ = model.encoder(shape1)
        mu2, _ = model.encoder(shape2)
        
        # Create interpolation steps
        alphas = np.linspace(0, 1, num_steps)
        interpolated_shapes = []
        
        for alpha in alphas:
            # Interpolate in latent space
            z = alpha * mu1 + (1 - alpha) * mu2
            # Decode
            shape = model.decoder(z)
            interpolated_shapes.append(shape.cpu().numpy()[0])
        
        # Visualize
        fig = plt.figure(figsize=(20, 4))
        for idx, shape in enumerate(interpolated_shapes):
            ax = fig.add_subplot(1, num_steps, idx + 1, projection='3d')
            ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2], s=1)
            ax.set_title(f'Step {idx+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Generate dataset
    from pointcloud_repr import *
    from dataset_generator import *
    #generator = GeometricDataset()
    #points, labels = generator.generate_dataset(n_samples_per_class=100)
    
    # Create dataset and dataloader
    dataset = GeometricDataset()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Load trained model (assuming you have already trained it)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointCloudVAE(num_points=2048, latent_dim=128)
    model = model.to(device)

    #print(model.decoder(torch.randn([2,128])))
    
    # Load your trained model weights here
    model.load_state_dict(torch.load('domains/pointcloud/pointcloud_vae_state.pth', map_location = device))
    
    # Get some test data
    test_data = next(iter(dataloader))[0]
    
    # Visualize reconstructions
    print("Visualizing reconstructions...")
    visualize_reconstruction(model, test_data, device)
    
    # Visualize random samples
    print("Visualizing random samples...")
    visualize_pointcloud_states(model, device=device)
    
    # Visualize latent space
    print("Visualizing latent space...")
    visualize_latent_space(model, dataloader, device)
    
    # Visualize shape interpolation
    print("Visualizing shape interpolation...")
    interpolate_shapes(model, test_data, device)