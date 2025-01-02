import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
from matplotlib.gridspec import GridSpec
import os
import imageio
import cv2
from io import BytesIO
from tqdm import tqdm

class GeometricVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        # Set the style for transparent background
        plt.style.use('dark_background')
        
    def _setup_transparent_figure(self):
        """Helper method to setup transparent figure settings."""
        fig = plt.gcf()
        ax = plt.gca()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    
    def _setup_clean_figure(self, square_size: float = 1.2):
        """Helper method to setup clean figure settings."""
        fig = plt.gcf()
        ax = plt.gca()
        
        # Make background transparent
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set square aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-square_size, square_size)
        ax.set_ylim(-square_size, square_size)

    def plot_single_shape(self, points: np.ndarray, title: str = "",
                         show_points: bool = True, save_path: Optional[str] = None) -> None:
        """Plot a single shape from its points."""
        plt.figure(figsize=(6, 6))
        self._setup_clean_figure()
        
        if show_points:
            plt.scatter(points[:, 0], points[:, 1], c='white', alpha=0.6, s=20)
        plt.plot(points[:, 0], points[:, 1], 'white', alpha=0.3)
        
        if save_path:
            plt.savefig(f'outputs/{save_path}', transparent=True, bbox_inches='tight', 
                       pad_inches=0, dpi=300)
        plt.show()
        plt.close()

    def plot_shape_gallery(self, shapes: Dict[str, np.ndarray],
                          max_shapes: int = 25,
                          save_path: Optional[str] = None) -> None:
        """Plot a gallery of shapes."""
        num_shapes = min(len(shapes), max_shapes)
        rows = int(np.ceil(np.sqrt(num_shapes)))
        cols = rows

        plt.figure(figsize=(18, 18))
        for idx, (name, points) in enumerate(list(shapes.items())[:max_shapes]):
            plt.subplot(rows, cols, idx + 1)
            self._setup_clean_figure()
            plt.scatter(points[:, 0], points[:, 1], c='white', alpha=0.6, s=10)
            plt.plot(points[:, 0], points[:, 1], 'white', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f'outputs/{save_path}', transparent=True, bbox_inches='tight', 
                       pad_inches=0, dpi=300)
        plt.show()
        plt.close()

    def plot_reconstruction_comparison(self, original: torch.Tensor,
                                    reconstruction: torch.Tensor,
                                    num_samples: int = 5,
                                    save_path: Optional[str] = None) -> None:
        """Plot original shapes alongside their reconstructions."""
        fig = plt.figure(figsize=self.figsize)
        for idx in range(num_samples):
            # Original
            plt.subplot(2, num_samples, idx + 1)
            self._setup_clean_figure()
            points = original[idx].detach().cpu().numpy()
            plt.scatter(points[:, 0], points[:, 1], c='white', alpha=0.6, s=10)
            plt.plot(points[:, 0], points[:, 1], 'white', alpha=0.3)

            # Reconstruction
            plt.subplot(2, num_samples, num_samples + idx + 1)
            self._setup_clean_figure()
            points = reconstruction[idx].detach().cpu().numpy()
            plt.scatter(points[:, 0], points[:, 1], c='white', alpha=0.6, s=10)
            plt.plot(points[:, 0], points[:, 1], 'white', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f'outputs/{save_path}', transparent=True, bbox_inches='tight', 
                       pad_inches=0, dpi=300)
        plt.show()
        plt.close()

    def plot_latent_space_samples(self, model: 'PointCloudVAE',
                                num_samples: int = 25,
                                save_path: Optional[str] = None,
                                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """Generate and plot shapes from random latent space samples."""
        model.eval()
        model.to(device)
        rows = int(np.sqrt(num_samples))
        cols = rows
        z = torch.randn(num_samples, model.decoder.fc1.in_features).to(device)

        with torch.no_grad():
            samples = model.decoder(z.to(device)).cpu().numpy()

        plt.figure(figsize=self.figsize)
        for idx in range(num_samples):
            plt.subplot(rows, rows, idx + 1)
            self._setup_clean_figure()
            points = samples[idx]
            plt.scatter(points[:, 0], points[:, 1], c='white', alpha=0.6, s=10)
            plt.plot(points[:, 0], points[:, 1], 'white', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f'outputs/{save_path}', transparent=True, bbox_inches='tight', 
                       pad_inches=0, dpi=300)
        plt.show()
        plt.close()

    def plot_latent_space_interpolation(self, model: 'PointCloudVAE',
                                      num_steps: int = 10,
                                      save_path: Optional[str] = None,
                                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """Generate shapes by interpolating between two random points in latent space."""
        model.eval()
        z1 = torch.randn(1, model.decoder.fc1.in_features)
        z2 = torch.randn(1, model.decoder.fc1.in_features)

        alphas = np.linspace(0, 1, num_steps)
        interpolated_z = torch.stack([
            z1 * (1-alpha) + z2 * alpha for alpha in alphas
        ]).squeeze(1).to(device)

        with torch.no_grad():
            samples = model.decoder(interpolated_z).cpu().numpy()

        plt.figure(figsize=(15, 3))
        for idx in range(num_steps):
            plt.subplot(1, num_steps, idx + 1)
            self._setup_clean_figure()
            points = samples[idx]
            plt.scatter(points[:, 0], points[:, 1], c='white', alpha=0.6, s=10)
            plt.plot(points[:, 0], points[:, 1], 'white', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f'outputs/{save_path}', transparent=True, bbox_inches='tight', 
                       pad_inches=0, dpi=300)
        plt.show()
        plt.close()

    def plot_training_progress(self, losses: List[float],
                             save_path: Optional[str] = None) -> None:
        """Plot training loss over epochs."""
        plt.figure(figsize=(10, 10))
        self._setup_clean_figure(square_size=max(losses))
        plt.plot(losses, 'white', linewidth=2)
        
        if save_path:
            plt.savefig(f'outputs/{save_path}', transparent=True, bbox_inches='tight', 
                       pad_inches=0, dpi=300)
        plt.show()
        plt.close()

    def create_animation_data(self, model: 'PointCloudVAE',
                            num_frames: int = 300,
                            device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> np.ndarray:
        """Generate data for creating animations of shape morphing."""
        model.eval()
        model = model.to(device)
        # Create a circular path in 2D latent space
        theta = np.linspace(0, 2*np.pi, num_frames)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)

        # Extend to full latent dimension
        z = torch.zeros((num_frames, model.decoder.fc1.in_features))
        for i in range(10):
            z[:, i*2] = torch.from_numpy(circle_x)
            z[:, i*2+1] = torch.from_numpy(circle_y)

        with torch.no_grad():
            frames = model.decoder(z.to(device)).cpu().numpy()
        return frames

    def save_animation_frames(self, model: 'PointCloudVAE',
                         output_dir: str = 'outputs/frames',
                         num_frames: int = 300,
                         resolution: Tuple[int, int] = (1920, 1920),
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Create and save animation frames as PNG files with transparent background.
        
        Args:
            model: The VAE model
            output_dir: Directory to save the PNG frames
            num_frames: Number of frames to generate
            resolution: Output resolution (width, height)
            device: Device to run the model on
        """
        frames_data = self.create_animation_data(model, num_frames, device)
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating frames...")
        for idx, points in enumerate(tqdm(frames_data)):
            # Create figure with transparent background
            dpi = 100
            figsize = (resolution[0]/dpi, resolution[1]/dpi)
            fig = plt.figure(figsize=figsize, dpi=dpi)
            
            # Setup clean figure
            ax = plt.gca()
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            
            # Remove all spines, ticks, and labels
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set square aspect ratio and limits
            ax.set_aspect('equal')
            square_size = 1.2  # Adjust this to control the view window
            ax.set_xlim(-square_size, square_size)
            ax.set_ylim(-square_size, square_size)
            
            # Plot the shape with white points and lines
            plt.scatter(points[:, 0], points[:, 1], c='white', alpha=1.0, s=50)
            plt.plot(points[:, 0], points[:, 1], 'white', alpha=1.0)
            
            # Save frame
            frame_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
            plt.savefig(frame_path,
                       transparent=True,
                       bbox_inches='tight',
                       pad_inches=0,
                       dpi=dpi,
                       facecolor='none',
                       edgecolor='none')
            plt.close()

        print(f"Frames saved to {output_dir}")
        print(f"Total frames generated: {num_frames}")

def create_shape_morphing_video(model_path: str, 
                              output_path: str = 'outputs/animation.mp4',
                              resolution: Tuple[int, int] = (1920, 1080),
                              fps: int = 10,
                              num_frames: int = 500):
    """
    Convenience function to create a shape morphing video from a saved model.
    
    Args:
        model_path: Path to the saved model state dict
        output_path: Path where the video should be saved
        resolution: Output video resolution (width, height)
        fps: Frames per second
        num_frames: Number of frames in the animation
    """
    # Load the model
    model = PointCloudVAE(latent_dim=64, num_points=320)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Create visualizer and save video
    viz = GeometricVisualizer()
    viz.save_animation_video(
        model,
        output_path=output_path,
        num_frames=num_frames,
        fps=fps,
        resolution=resolution
    )

from PIL import Image
import glob
from typing import Optional

def create_gif_from_pngs(input_dir, output_path, duration=50):
    """
    Create a GIF animation from PNG frames in the specified directory.
    Preserves transparency and handles frames independently.
    
    Args:
        input_dir (str): Directory containing PNG frame files
        output_path (str): Path where the output GIF will be saved
        duration (int): Duration for each frame in milliseconds (default: 50ms = 20fps)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get list of PNG files in the input directory
        frames = []
        png_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith('.png')]
        
        if not png_files:
            print("No PNG files found in the input directory")
            return False
        
        # Load and convert each frame
        for filename in png_files:
            filepath = os.path.join(input_dir, filename)
            # Open the image and convert to RGBA to ensure proper transparency handling
            with Image.open(filepath) as img:
                # Convert to RGBA if not already
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Create a copy of the frame to ensure independence
                frame_copy = Image.new('RGBA', img.size, (0, 0, 0, 0))
                frame_copy.paste(img, (0, 0), img)
                
                # Convert to P mode (palette) with transparency
                alpha = frame_copy.split()[3]
                frame_p = frame_copy.convert('RGB').convert('P', palette=Image.Palette.ADAPTIVE, colors=255)
                mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                frame_p.paste(255, mask)  # 255 is the transparent color index
                frames.append(frame_p)
        
        # Save the frames as an animated GIF
        frames[0].save(
            output_path,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,  # 0 means loop forever
            optimize=False,  # Disable optimization to prevent frame bleeding
            disposal=2,  # Clear the frame before rendering the next one
            transparency=255,  # Set transparent color index
        )
        
        print(f"GIF animation created successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating GIF: {str(e)}")
        return False


# Example usage:
if __name__ == "__main__":
    # Create generator and dataset

    from curve_repr import *
    from dataset_generator import *
    generator = GeometricShapeGenerator(num_points=100)
    shapes = generator.generate_shapes()
    model = PointCloudVAE(latent_dim=64, num_points=320)
    model.load_state_dict(torch.load("domains/curve/state_curve.pth", map_location="cpu"))
    viz = GeometricVisualizer()
    viz.save_animation_frames(model, output_dir='outputs/frames', num_frames=500, resolution=(1920, 1920))


    # Basic usage
    create_gif_from_pngs(
        input_dir="outputs/frames",
        output_path="outputs/animation.gif",
        duration=50  # 50ms per frame = 20fps
    )
    print("done")
    
    """
    # Initialize visualizer
    model = PointCloudVAE(latent_dim=64, num_points=320)
    model.load_state_dict(torch.load("domains/curve/state_curve.pth", map_location="cpu"))
    model.to("cpu")
    viz = GeometricVisualizer()

    # Plot gallery of original shapes
    viz.plot_shape_gallery(shapes, save_path="curve_gallery")

    # Create and train VAE (assuming model is trained)
    #model = PointCloudVAE(latent_dim=32, num_points=100)
    # ... (training code here)
    dataset = GeometricDataset(num_samples=320, num_points=320)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Plot reconstructions
    batch = next(iter(train_loader))
    with torch.no_grad():
        recon, _, _ = model(batch[0])
    viz.plot_reconstruction_comparison(batch[0], recon,save_path = "curve_recons")

    # Plot latent space samples
    viz.plot_latent_space_samples(model, save_path="curve_samples")

    # Plot interpolation
    viz.plot_latent_space_interpolation(model, save_path = "curve_interpolate")

    # Animation data (can be used with external animation libraries)
    frames = viz.create_animation_data(model)
    """