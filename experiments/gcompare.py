import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Import models from the previous code files
from generate_diff import ConditionedUNet, get_diffusion_parameters, p_sample_loop, linear_beta_schedule, n_steps, n_timesteps
from generate_gan import Generator, latent_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    # Load diffusion model
    diffusion_model = ConditionedUNet(n_steps=n_steps)
    diffusion_model.load_state_dict(torch.load("outputs/path_diffusion_model.pt", map_location=device))
    diffusion_model.to(device)
    diffusion_model.eval()
    
    # Load GAN generator
    gan_generator = Generator(latent_dim=latent_dim, n_steps=n_steps)
    gan_generator.load_state_dict(torch.load("outputs/gan_path_generator.pt", map_location=device))
    gan_generator.to(device)
    gan_generator.eval()
    
    return diffusion_model, gan_generator

# Sampling functions
@torch.no_grad()
def sample_from_diffusion(model, start_point, end_point, params, device=device):
    # Convert points to tensors
    start = torch.tensor(start_point, dtype=torch.float32).to(device)
    end = torch.tensor(end_point, dtype=torch.float32).to(device)
    
    # Combine as condition
    condition = torch.cat([start, end]).unsqueeze(0)
    
    # Sample from the model
    sample_shape = (1, n_steps, 2)
    generated_path = p_sample_loop(model, params, sample_shape, n_timesteps, condition)
    
    return generated_path.cpu().numpy()[0]

@torch.no_grad()
def sample_from_gan(model, start_point, end_point, device=device):
    # Convert points to tensors
    start = torch.tensor(start_point, dtype=torch.float32).to(device)
    end = torch.tensor(end_point, dtype=torch.float32).to(device)
    
    # Combine as condition
    condition = torch.cat([start, end]).unsqueeze(0)
    
    # Generate random latent vector
    z = torch.randn(1, latent_dim, device=device)
    
    # Generate path
    generated_path = model(z, condition)
    
    return generated_path.cpu().numpy()[0]

def generate_diffusion_process_visualization(model, params, start_point, end_point, save_dir="outputs/specific_examples"):
    """Visualize the diffusion denoising process step by step"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert points to tensors
    start = torch.tensor(start_point, dtype=torch.float32).to(device)
    end = torch.tensor(end_point, dtype=torch.float32).to(device)
    
    # Combine as condition
    condition = torch.cat([start, end]).unsqueeze(0)
    
    # Sample shape
    sample_shape = (1, n_steps, 2)
    
    # Start with random noise
    x = torch.randn(sample_shape, device=device)
    
    # Save the initial noise
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x[0, :, 0].cpu().numpy(), x[0, :, 1].cpu().numpy(), 'b-', alpha=0.5)
    ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    ax.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
    ax.grid(True)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title(f"Initial Noise (Timestep {n_timesteps})")
    ax.legend()
    plt.savefig(f"{save_dir}/diffusion_process_step_0.png")
    plt.close()
    
    # Visualize intermediate steps
    steps_to_visualize = [900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10, 0]
    
    for i, step in enumerate(steps_to_visualize):
        if step < n_timesteps:
            # Create a copy of the current x
            current_x = x.clone()
            
            # Denoise until the current step
            for t in range(n_timesteps - 1, step - 1, -1):
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                with torch.no_grad():
                    # Use the model to predict the noise
                    predicted_noise = model(current_x, t_tensor, condition)
                    
                    # Update x using the model's prediction
                    betas_t = params["betas"][t_tensor].reshape(-1, 1, 1)
                    sqrt_one_minus_alphas_cumprod_t = params["sqrt_one_minus_alphas_cumprod"][t_tensor].reshape(-1, 1, 1)
                    sqrt_recip_alphas_t = params["sqrt_recip_alphas"][t_tensor].reshape(-1, 1, 1)
                    
                    if t > 0:
                        noise = torch.randn_like(current_x)
                        posterior_variance_t = params["posterior_variance"][t_tensor].reshape(-1, 1, 1)
                        current_x = (1 / sqrt_recip_alphas_t) * (current_x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t) + torch.sqrt(posterior_variance_t) * noise
                    else:
                        current_x = (1 / sqrt_recip_alphas_t) * (current_x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            
            # Plot the current state
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(current_x[0, :, 0].cpu().numpy(), current_x[0, :, 1].cpu().numpy(), 'b-')
            ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
            ax.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
            ax.grid(True)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_title(f"Diffusion Process (Timestep {step})")
            ax.legend()
            plt.savefig(f"{save_dir}/diffusion_process_step_{i+1}.png")
            plt.close()

def visualize_gan_latent_space(model, start_point, end_point, num_samples=10, save_dir="outputs/specific_examples"):
    """Visualize how different latent vectors affect the GAN output"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert points to tensors
    start = torch.tensor(start_point, dtype=torch.float32).to(device)
    end = torch.tensor(end_point, dtype=torch.float32).to(device)
    
    # Combine as condition
    condition = torch.cat([start, end]).unsqueeze(0)
    
    # Generate multiple paths with different latent vectors
    plt.figure(figsize=(10, 8))
    
    # Generate paths with different random seeds
    for i in range(num_samples):
        # Use a specific seed for reproducibility
        torch.manual_seed(i)
        z = torch.randn(1, latent_dim, device=device)
        
        # Generate path
        with torch.no_grad():
            generated_path = model(z, condition)
        
        # Plot the path
        path = generated_path.cpu().numpy()[0]
        plt.plot(path[:, 0], path[:, 1], alpha=0.6, label=f"Latent {i+1}")
    
    plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    plt.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
    plt.grid(True)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title("GAN Outputs with Different Latent Vectors")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gan_latent_space.png")
    plt.close()

def compare_difficult_cases(diffusion_model, gan_model, params, num_cases=3, save_dir="outputs/specific_examples"):
    """Compare models on difficult cases (e.g., distant points, obstacles)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Define difficult cases
    difficult_cases = [
        # Case 1: Very distant points
        {
            "start": np.array([-5.0, -5.0]),
            "end": np.array([5.0, 5.0]),
            "description": "Distant Points"
        },
        # Case 2: Sharp turn required
        {
            "start": np.array([0.0, -4.0]),
            "end": np.array([0.0, 4.0]),
            "obstacle": np.array([0.0, 0.0]),
            "description": "Path with Obstacle"
        },
        # Case 3: Complex setup with multiple constraints
        {
            "start": np.array([-4.0, 0.0]),
            "end": np.array([4.0, 0.0]),
            "obstacles": [
                np.array([-2.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([2.0, 0.0])
            ],
            "description": "Multiple Obstacles"
        }
    ]
    
    for i, case in enumerate(difficult_cases):
        start_point = case["start"]
        end_point = case["end"]
        
        # Generate multiple paths with each model
        diffusion_paths = []
        gan_paths = []
        
        for _ in range(5):  # Generate 5 paths for each model
            diffusion_paths.append(sample_from_diffusion(diffusion_model, start_point, end_point, params))
            gan_paths.append(sample_from_gan(gan_model, start_point, end_point))
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot diffusion model paths
        for path in diffusion_paths:
            ax1.plot(path[:, 0], path[:, 1], 'b-', alpha=0.4)
        ax1.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
        ax1.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
        
        # Plot GAN model paths
        for path in gan_paths:
            ax2.plot(path[:, 0], path[:, 1], 'r-', alpha=0.4)
        ax2.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
        ax2.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
        
        # Add obstacles if present
        if "obstacle" in case:
            obstacle = case["obstacle"]
            obs_radius = 0.5
            circle1 = plt.Circle((obstacle[0], obstacle[1]), obs_radius, color='gray', alpha=0.5)
            circle2 = plt.Circle((obstacle[0], obstacle[1]), obs_radius, color='gray', alpha=0.5)
            ax1.add_patch(circle1)
            ax2.add_patch(circle2)
            
        if "obstacles" in case:
            obs_radius = 0.5
            for obstacle in case["obstacles"]:
                circle1 = plt.Circle((obstacle[0], obstacle[1]), obs_radius, color='gray', alpha=0.5)
                circle2 = plt.Circle((obstacle[0], obstacle[1]), obs_radius, color='gray', alpha=0.5)
                ax1.add_patch(circle1)
                ax2.add_patch(circle2)
        
        # Set titles and labels
        ax1.set_title(f"Diffusion Model: {case['description']}")
        ax2.set_title(f"GAN Model: {case['description']}")
        
        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.legend()
        
        plt.suptitle(f"Comparison on {case['description']}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/difficult_case_{i+1}.png")
        plt.close()

def compare_interpolation_behavior(diffusion_model, gan_model, params, save_dir="outputs/specific_examples"):
    """Compare how models interpolate between multiple points"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Define points for interpolation
    points = np.array([
        [-4.0, -3.0],  # Point 1
        [-2.0, 3.0],   # Point 2
        [2.0, -2.0],   # Point 3
        [4.0, 3.0]     # Point 4
    ])
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='black', s=100)
    for i, point in enumerate(points):
        plt.annotate(f"Point {i+1}", (point[0], point[1]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=12)
    
    # Generate paths for consecutive point pairs
    colors = ['blue', 'green', 'red']
    
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i+1]
        
        # Generate path with diffusion model
        diffusion_path = sample_from_diffusion(diffusion_model, start_point, end_point, params)
        
        # Generate path with GAN model
        gan_path = sample_from_gan(gan_model, start_point, end_point)
        
        # Plot paths
        plt.plot(diffusion_path[:, 0], diffusion_path[:, 1], color=colors[i], linestyle='-', 
                 linewidth=2, alpha=0.7, label=f"Diffusion {i+1}->{i+2}")
        plt.plot(gan_path[:, 0], gan_path[:, 1], color=colors[i], linestyle='--', 
                 linewidth=2, alpha=0.7, label=f"GAN {i+1}->{i+2}")
    
    plt.grid(True)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title("Path Interpolation Comparison", fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/interpolation_comparison.png")
    plt.close()

def plot_model_uncertainty(diffusion_model, gan_model, params, num_samples=50, save_dir="outputs/specific_examples"):
    """Visualize the uncertainty in model predictions by generating many paths"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Define a specific start and end point
    start_point = np.array([-3.0, 0.0])
    end_point = np.array([3.0, 0.0])
    
    # Generate many paths with each model
    diffusion_paths = []
    gan_paths = []
    
    for _ in range(num_samples):
        diffusion_paths.append(sample_from_diffusion(diffusion_model, start_point, end_point, params))
        gan_paths.append(sample_from_gan(gan_model, start_point, end_point))
    
    # Convert to numpy arrays
    diffusion_array = np.array(diffusion_paths)
    gan_array = np.array(gan_paths)
    
    # Calculate mean and standard deviation at each point along the path
    diffusion_mean = np.mean(diffusion_array, axis=0)
    diffusion_std = np.std(diffusion_array, axis=0)
    
    gan_mean = np.mean(gan_array, axis=0)
    gan_std = np.std(gan_array, axis=0)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot diffusion model uncertainty
    for path in diffusion_paths:
        ax1.plot(path[:, 0], path[:, 1], 'b-', alpha=0.1)
    
    # Plot the mean path
    ax1.plot(diffusion_mean[:, 0], diffusion_mean[:, 1], 'b-', linewidth=2, label='Mean Path')
    
    # Plot uncertainty bounds
    ax1.fill_between(
        diffusion_mean[:, 0],
        diffusion_mean[:, 1] - diffusion_std[:, 1],
        diffusion_mean[:, 1] + diffusion_std[:, 1],
        color='blue', alpha=0.2, label='±1 Std Dev'
    )
    
    ax1.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    ax1.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
    ax1.set_title("Diffusion Model Path Uncertainty", fontsize=14)
    ax1.grid(True)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-3, 3)
    ax1.legend()
    
    # Plot GAN model uncertainty
    for path in gan_paths:
        ax2.plot(path[:, 0], path[:, 1], 'r-', alpha=0.1)
    
    # Plot the mean path
    ax2.plot(gan_mean[:, 0], gan_mean[:, 1], 'r-', linewidth=2, label='Mean Path')
    
    # Plot uncertainty bounds
    ax2.fill_between(
        gan_mean[:, 0],
        gan_mean[:, 1] - gan_std[:, 1],
        gan_mean[:, 1] + gan_std[:, 1],
        color='red', alpha=0.2, label='±1 Std Dev'
    )
    
    ax2.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    ax2.plot(end_point[0], end_point[1], 'ro', markersize=10, label='End')
    ax2.set_title("GAN Model Path Uncertainty", fontsize=14)
    ax2.grid(True)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-3, 3)
    ax2.legend()
    
    plt.suptitle("Model Uncertainty Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/uncertainty_comparison.png")
    plt.close()

def main():
    print("Running specific example comparisons...")
    
    # Load models
    diffusion_model, gan_generator = load_models()
    
    # Get diffusion parameters
    betas = linear_beta_schedule(n_timesteps).to(device)
    diffusion_params = get_diffusion_parameters(betas)
    
    # Define a specific start and end point for visualization
    start_point = np.array([-3.0, 2.0])
    end_point = np.array([3.0, -2.0])
    
    # Generate diffusion process visualization
    print("Generating diffusion process visualization...")
    generate_diffusion_process_visualization(diffusion_model, diffusion_params, start_point, end_point)
    
    # Visualize GAN latent space
    print("Visualizing GAN latent space...")
    visualize_gan_latent_space(gan_generator, start_point, end_point)
    
    # Compare difficult cases
    print("Comparing models on difficult cases...")
    compare_difficult_cases(diffusion_model, gan_generator, diffusion_params)
    
    # Compare interpolation behavior
    print("Comparing interpolation behavior...")
    compare_interpolation_behavior(diffusion_model, gan_generator, diffusion_params)
    
    # Visualize model uncertainty
    print("Visualizing model uncertainty...")
    plot_model_uncertainty(diffusion_model, gan_generator, diffusion_params)
    
    print("All visualizations complete!")

if __name__ == "__main__":
    main()