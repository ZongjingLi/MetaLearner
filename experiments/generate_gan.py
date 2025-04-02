import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
batch_size = 64
n_steps = 10  # Number of points in the path
latent_dim = 32  # Dimension of noise vector for GAN
device = torch.device("cuda" if torch.cuda.is_available() else "mps:0")
print(f"Using device: {device}")

# Generate synthetic data: smooth paths between two points (same as for diffusion model)
class PathDataset(Dataset):
    def __init__(self, num_samples=10000, n_steps=50):
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.paths = []
        self.start_points = []
        self.end_points = []
        
        for _ in range(num_samples):
            # Generate random start and end points
            start_point = np.random.uniform(-5, 5, size=(2,))
            end_point = np.random.uniform(-5, 5, size=(2,))
            
            # Generate a smooth path (simple linear interpolation with some noise)
            t = np.linspace(0, 1, n_steps).reshape(-1, 1)
            path = start_point * (1 - t) + end_point * t
            
            # Add some random noise to make the path interesting
            noise_scale = np.linalg.norm(end_point - start_point) * 0.1
            noise = np.random.normal(0, noise_scale, size=(n_steps, 2))
            
            # Ensure the noise doesn't affect start and end points
            smoothing = np.sin(np.pi * t)
            noise = noise * smoothing
            
            path = path + noise
            
            self.paths.append(path.astype(np.float32))
            self.start_points.append(start_point.astype(np.float32))
            self.end_points.append(end_point.astype(np.float32))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'path': self.paths[idx],
            'start': self.start_points[idx],
            'end': self.end_points[idx]
        }

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=32, n_steps=50, condition_dim=4, hidden_dim=256):
        super().__init__()
        
        self.n_steps = n_steps
        
        # Condition embedding (start and end points)
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Latent embedding
        self.latent_embedding = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Combined processing
        self.main = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_steps * 2)  # Output is path with n_steps points, each with x,y
        )
        
    def forward(self, z, condition):
        # Embed condition (start and end points)
        c_emb = self.condition_embedding(condition)
        
        # Embed latent vector
        z_emb = self.latent_embedding(z)
        
        # Combine and generate path
        combined = torch.cat([z_emb, c_emb], dim=1)
        output = self.main(combined)
        
        # Reshape to path format
        path = output.view(-1, self.n_steps, 2)
        
        # Force the start and end points to match the condition
        # Extract start and end points from the condition
        start_points = condition[:, :2]
        end_points = condition[:, 2:]
        
        # Replace the generated start and end points
        path[:, 0, :] = start_points
        path[:, -1, :] = end_points
        
        return path

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, n_steps=50, condition_dim=4, hidden_dim=256):
        super().__init__()
        
        # Path embedding
        self.path_embedding = nn.Sequential(
            nn.Linear(n_steps * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Condition embedding (start and end points)
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Combined processing
        self.main = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, path, condition):
        # Flatten the path
        batch_size = path.size(0)
        path_flat = path.view(batch_size, -1)
        
        # Embed path
        path_emb = self.path_embedding(path_flat)
        
        # Embed condition
        cond_emb = self.condition_embedding(condition)
        
        # Combine and discriminate
        combined = torch.cat([path_emb, cond_emb], dim=1)
        output = self.main(combined)
        
        return output

# Function to train the GAN
def train_gan(generator, discriminator, dataloader, num_epochs=100, device=device):
    # Initialize optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Move models to device
    generator.to(device)
    discriminator.to(device)
    
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    
    for epoch in range(num_epochs):
        g_epoch_loss = 0
        d_epoch_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get real data
            real_paths = batch['path'].to(device)
            start_points = batch['start'].to(device)
            end_points = batch['end'].to(device)
            
            # Combine start and end points as condition
            condition = torch.cat([start_points, end_points], dim=1)
            
            batch_size = real_paths.size(0)
            
            # Create labels
            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()
            
            # Train with real batch
            real_output = discriminator(real_paths, condition)
            d_loss_real = criterion(real_output, real_label)
            
            # Train with fake batch
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_paths = generator(z, condition)
            fake_output = discriminator(fake_paths.detach(), condition)
            d_loss_fake = criterion(fake_output, fake_label)
            
            # Combined discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            g_optimizer.zero_grad()
            
            # Generate another fake batch
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_paths = generator(z, condition)
            fake_output = discriminator(fake_paths, condition)
            
            # Calculate generator loss
            g_loss = criterion(fake_output, real_label)
            
            # Add path smoothness loss (penalize sharp turns)
            smoothness_loss = path_smoothness_loss(fake_paths)
            
            # Add start-end constraint loss (ensure path connects start and end)
            # (Already enforced in the generator's forward pass)
            
            # Combined generator loss
            combined_g_loss = g_loss + 0.1 * smoothness_loss
            combined_g_loss.backward()
            g_optimizer.step()
            
            # Track losses
            g_epoch_loss += combined_g_loss.item()
            d_epoch_loss += d_loss.item()
        
        avg_g_loss = g_epoch_loss / len(dataloader)
        avg_d_loss = d_epoch_loss / len(dataloader)
        
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}")
        
        # Visualize samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_gan_samples(generator, device, epoch + 1)
    
    return generator, discriminator, G_losses, D_losses

# Path smoothness loss - penalize sharp turns in the path
def path_smoothness_loss(paths):
    # Compute the difference between consecutive points
    diffs = paths[:, 1:, :] - paths[:, :-1, :]
    
    # Compute the difference in direction (penalize sharp turns)
    dir_change = diffs[:, 1:, :] - diffs[:, :-1, :]
    
    # Return the mean squared magnitude of direction changes
    return torch.mean(torch.sum(dir_change**2, dim=2))

# Generate samples with the trained generator
@torch.no_grad()
def generate_path(generator, start_point, end_point, device=device):
    generator.eval()
    
    # Convert points to tensors
    start = torch.tensor(start_point, dtype=torch.float32).to(device)
    end = torch.tensor(end_point, dtype=torch.float32).to(device)
    
    # Combine as condition
    condition = torch.cat([start, end]).unsqueeze(0)
    
    # Generate random latent vector
    z = torch.randn(1, latent_dim, device=device)
    
    # Generate path
    generated_path = generator(z, condition)
    
    return generated_path.cpu().numpy()[0]

# Visualize generated paths
def visualize_gan_samples(generator, device, epoch=None):
    generator.eval()
    plt.figure(figsize=(15, 5))
    
    # Generate 3 samples with different start and end points
    for i in range(3):
        # Generate random start and end points
        start_point = np.random.uniform(-5, 5, size=(2,))
        end_point = np.random.uniform(-5, 5, size=(2,))
        
        # Generate a path
        generated_path = generate_path(generator, start_point, end_point, device)
        
        # Plot the path
        plt.subplot(1, 3, i+1)
        plt.plot(generated_path[:, 0], generated_path[:, 1], 'b-')
        plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
        plt.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
        plt.grid(True)
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title(f"Generated Path {i+1}")
        if i == 0:
            plt.legend()
    
    if epoch:
        plt.suptitle(f"GAN Generated Paths After Epoch {epoch}")
        plt.savefig(f"outputs/gan_path_samples_epoch_{epoch}.png")
    else:
        plt.suptitle("Final GAN Generated Paths")
        plt.savefig("outputs/final_gan_path_samples.png")
    
    plt.close()

# Plot training losses
def plot_gan_losses(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator')
    plt.plot(D_losses, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.savefig('gan_training_losses.png')
    plt.close()

def main():
    # Create dataset and dataloader
    dataset = PathDataset(num_samples=10000, n_steps=n_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    generator = Generator(latent_dim=latent_dim, n_steps=n_steps)
    discriminator = Discriminator(n_steps=n_steps)
    
    # Train the GAN
    generator, discriminator, G_losses, D_losses = train_gan(generator, discriminator, dataloader, num_epochs=1000, device=device)
    
    # Save models
    torch.save(generator.state_dict(), "outputs/gan_path_generator.pt")
    torch.save(discriminator.state_dict(), "outputs/gan_path_discriminator.pt")
    
    # Plot losses
    plot_gan_losses(G_losses, D_losses)
    
    # Generate final visualizations
    visualize_gan_samples(generator, device)
    
    print("Training complete! Models saved.")

if __name__ == "__main__":
    main()