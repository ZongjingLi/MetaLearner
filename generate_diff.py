import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
batch_size = 64
n_steps = 5  # Number of points in the path
n_timesteps = 3200  # Number of diffusion steps
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# Beta schedule for the diffusion process
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Calculate alphas and other constants
def get_diffusion_parameters(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }

# Generate synthetic data: smooth paths between two points
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

# UNet-like architecture for the diffusion model
class ConditionedUNet(nn.Module):
    def __init__(self, n_steps=50, time_emb_dim=128, condition_dim=4):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Condition embedding (start and end points)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Downsampling path
        self.down1 = nn.Linear(n_steps * 2, 512)
        self.down2 = nn.Linear(512, 256)
        self.down3 = nn.Linear(256, 128)
        
        # Middle layers with time and condition embedding
        self.mid = nn.Sequential(
            nn.Linear(128 + time_emb_dim + time_emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU()
        )
        
        # Upsampling path
        self.up1 = nn.Linear(128 + 128, 256)
        self.up2 = nn.Linear(256 + 256, 512)
        self.up3 = nn.Linear(512 + 512, n_steps * 2)
        
    def forward(self, x, timestep, condition):
        # Embed time
        temb = self.time_embed(timestep.unsqueeze(1).float())
        
        # Embed condition (start and end points)
        cemb = self.condition_embed(condition)
        
        # Initial shape: [batch, n_steps, 2]
        batch_size = x.shape[0]
        
        # Flatten the path
        x_flat = x.reshape(batch_size, -1)
        
        # Downsampling
        d1 = F.silu(self.down1(x_flat))
        d2 = F.silu(self.down2(d1))
        d3 = F.silu(self.down3(d2))
        
        # Middle with time and condition embedding
        mid_input = torch.cat([d3, temb, cemb], dim=1)
        mid = self.mid(mid_input)
        
        # Upsampling with skip connections
        u1 = F.silu(self.up1(torch.cat([mid, d3], dim=1)))
        u2 = F.silu(self.up2(torch.cat([u1, d2], dim=1)))
        u3 = self.up3(torch.cat([u2, d1], dim=1))
        
        # Reshape back to path
        return u3.reshape(batch_size, n_steps, 2)

# Forward diffusion process
def q_sample(params, x_start, t, noise=None):
    """
    Forward diffusion process: add noise to the data
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = params["sqrt_alphas_cumprod"][t].reshape(-1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = params["sqrt_one_minus_alphas_cumprod"][t].reshape(-1, 1, 1)
    
    # mean + variance
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Loss function
def p_losses(model, params, x_start, t, condition, noise=None):
    """
    Training loss for the diffusion model
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Forward diffusion to get noisy x and the added noise
    x_noisy, target = q_sample(params, x_start, t, noise)
    
    # Predict the noise using the model
    predicted = model(x_noisy, t, condition)
    
    # Calculate loss
    loss = F.mse_loss(predicted, target)
    
    return loss

# Sampling function (reverse diffusion)
@torch.no_grad()
def p_sample(model, params, x, t, t_index, condition):
    """
    Sample from the model at timestep t
    """
    betas_t = params["betas"][t].reshape(-1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = params["sqrt_one_minus_alphas_cumprod"][t].reshape(-1, 1, 1)
    sqrt_recip_alphas_t = params["sqrt_recip_alphas"][t].reshape(-1, 1, 1)
    
    # Predict the noise
    predicted_noise = model(x, t, condition)
    
    # No noise at step 0
    if t_index == 0:
        return (1 / sqrt_recip_alphas_t) * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    else:
        # Add noise scaled by the posterior variance
        posterior_variance_t = params["posterior_variance"][t].reshape(-1, 1, 1)
        noise = torch.randn_like(x)
        return (1 / sqrt_recip_alphas_t) * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t) + torch.sqrt(posterior_variance_t) * noise

# Full sampling loop
@torch.no_grad()
def p_sample_loop(model, params, shape, n_timesteps, condition):
    """
    Generate samples by running the reverse diffusion process
    """
    device = next(model.parameters()).device
    b = shape[0]
    
    # Start with random noise
    img = torch.randn(shape, device=device)
    
    # Iteratively denoise
    for i in reversed(range(n_timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, params, img, t, i, condition)
    
    return img

# Training function
def train(model, dataloader, optimizer, params, n_epochs=100, device=device):
    model.to(device)
    
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Get data and move to device
            x = batch['path'].to(device)
            start_points = batch['start'].to(device)
            end_points = batch['end'].to(device)
            
            # Combine start and end points as condition
            condition = torch.cat([start_points, end_points], dim=1)
            
            # Sample random timesteps
            batch_size = x.shape[0]
            t = torch.randint(0, n_timesteps, (batch_size,), device=device).long()
            
            # Calculate loss
            loss = p_losses(model, params, x, t, condition)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        
        # Save a visualization every 10 epochs
        if (epoch + 1) % 200 == 0:
            visualize_samples(model, params, device, epoch + 1)
    
    return model

# Function to visualize generated paths
def visualize_samples(model, params, device, epoch=None):
    model.eval()
    plt.figure(figsize=(15, 5))
    
    # Generate 3 samples with different start and end points
    for i in range(3):
        # Generate random start and end points
        start_point = torch.tensor(np.random.uniform(-5, 5, size=(2,)), device=device, dtype=torch.float32)
        end_point = torch.tensor(np.random.uniform(-5, 5, size=(2,)), device=device, dtype=torch.float32)
        
        # Condition for the model (start and end points)
        condition = torch.cat([start_point, end_point]).unsqueeze(0)
        
        # Generate a path
        sample_shape = (1, n_steps, 2)
        generated_path = p_sample_loop(model, params, sample_shape, n_timesteps, condition)
        
        # Plot the path
        plt.subplot(1, 3, i+1)
        generated_path = generated_path.cpu().numpy()[0]
        plt.plot(generated_path[:, 0], generated_path[:, 1], 'b-')
        plt.plot(start_point.cpu().numpy()[0], start_point.cpu().numpy()[1], 'go', markersize=10, label='Start')
        plt.plot(end_point.cpu().numpy()[0], end_point.cpu().numpy()[1], 'ro', markersize=10, label='End')
        plt.grid(True)
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title(f"Generated Path {i+1}")
        if i == 0:
            plt.legend()
    
    if epoch:
        plt.suptitle(f"Generated Paths After Epoch {epoch}")
        plt.savefig(f"outputs/path_samples_epoch_{epoch}.png")
    else:
        plt.suptitle("Final Generated Paths")
        plt.savefig("outputs/final_path_samples.png")
    
    plt.close()

def main():
    # Create dataset and dataloader
    dataset = PathDataset(num_samples=10000, n_steps=n_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize diffusion parameters
    betas = linear_beta_schedule(n_timesteps).to(device)
    diffusion_params = get_diffusion_parameters(betas)
    
    # Initialize model
    model = ConditionedUNet(n_steps=n_steps)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    model = train(model, dataloader, optimizer, diffusion_params, n_epochs=1000, device=device)
    
    # Save the model
    torch.save(model.state_dict(), "outputs/path_diffusion_model.pt")
    
    print("Training complete! Model saved as 'path_diffusion_model.pt'")
    
    # Generate final visualizations
    visualize_samples(model, diffusion_params, device)

if __name__ == "__main__":
    main()