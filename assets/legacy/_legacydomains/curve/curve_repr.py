import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from scipy.integrate import quad
import math


class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, num_points: int = 100):
        super().__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_points * 2)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.num_points, 2)
        return x

class PointCloudVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, num_points: int = 100):
        super().__init__()
        self.encoder = PointCloudEncoder(latent_dim)
        self.decoder = PointCloudDecoder(latent_dim, num_points)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

def vae_loss(reconstruction: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss * 1.0

# Training function
def train_vae(model: PointCloudVAE,
              train_loader: DataLoader,
              num_epochs: int = 100,
              learning_rate: float = 2e-4,
              device: str = 'cuda' if torch.cuda.is_available() else 'mps') -> List[float]:

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    losses = []


    from tqdm import tqdm

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device).float()
            #print(data.shape)

            optimizer.zero_grad()

            reconstruction, mu, log_var = model(data)
            loss = vae_loss(reconstruction, data, mu, log_var)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), "curve_vae_state.pth")
    return losses

import torch
from torch.utils.data import Dataset
import numpy as np

class GeometricDataset(Dataset):
    """
    A dataset of point clouds representing 2D curves, including:
    - Open paths with start and end points (linear, Bezier, sine wave)
    - Closed loops (circles, ellipses, spirals)
    """

    def __init__(self, num_samples: int = 1000, num_points: int = 320):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            num_points (int): Number of points per curve.
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.data = self._generate_dataset()

    def _generate_dataset(self):
        """Generates a dataset of open and closed curves in 2D."""
        dataset = []

        for _ in range(self.num_samples):
            curve_type = np.random.choice(["line", "bezier", "sine", "spiral", "circle", "ellipse"])

            if curve_type == "line":
                curve = self._generate_linear_path(self.num_points)
            elif curve_type == "bezier":
                curve = self._generate_bezier_curve(self.num_points)
            elif curve_type == "sine":
                curve = self._generate_sine_wave(self.num_points)
            elif curve_type == "spiral":
                curve = self._generate_spiral(self.num_points)
            elif curve_type == "circle":
                curve = self._generate_closed_curve(self.num_points, shape="circle")
            elif curve_type == "ellipse":
                curve = self._generate_closed_curve(self.num_points, shape="ellipse")
            else:
                continue  # Should never happen

            dataset.append(curve)

        return torch.tensor(np.array(dataset), dtype=torch.float32)

    def _generate_linear_path(self, num_points):
        """Generates a linear path between two random points."""
        start = np.random.uniform(-1, 1, size=2)
        end = np.random.uniform(-1, 1, size=2)

        t = np.linspace(0, 1, num_points)
        x = start[0] * (1 - t) + end[0] * t
        y = start[1] * (1 - t) + end[1] * t

        return np.stack([x, y], axis=1)

    def _generate_bezier_curve(self, num_points):
        """Generates a smooth Bezier curve with random control points."""
        start = np.random.uniform(-1, 1, size=2)
        end = np.random.uniform(-1, 1, size=2)
        ctrl1 = np.random.uniform(-1, 1, size=2)
        ctrl2 = np.random.uniform(-1, 1, size=2)

        t = np.linspace(0, 1, num_points)
        x = (1 - t) ** 3 * start[0] + 3 * (1 - t) ** 2 * t * ctrl1[0] + 3 * (1 - t) * t ** 2 * ctrl2[0] + t ** 3 * end[0]
        y = (1 - t) ** 3 * start[1] + 3 * (1 - t) ** 2 * t * ctrl1[1] + 3 * (1 - t) * t ** 2 * ctrl2[1] + t ** 3 * end[1]

        return np.stack([x, y], axis=1)

    def _generate_sine_wave(self, num_points):
        """Generates a sine wave path."""
        t = np.linspace(0, 2 * np.pi, num_points)
        x = np.linspace(-1, 1, num_points)
        y = np.sin(4 * t) * np.random.uniform(0.2, 0.5)

        return np.stack([x, y], axis=1)

    def _generate_spiral(self, num_points):
        """Generates an Archimedean spiral."""
        theta = np.linspace(0, 4 * np.pi, num_points)
        r = np.linspace(0, 1, num_points)  # Radius grows outward

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.stack([x, y], axis=1)

    def _generate_closed_curve(self, num_points, shape="circle"):
        """Generates a closed curve (circle or ellipse)."""
        theta = np.linspace(0, 2 * np.pi, num_points)

        if shape == "circle":
            r = np.random.uniform(0.7, 1.2)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        else:  # Ellipse
            a, b = np.random.uniform(0.7, 1.2), np.random.uniform(0.5, 1.0)
            x = a * np.cos(theta)
            y = b * np.sin(theta)

        return np.stack([x, y], axis=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0  # Return data and a dummy label (needed for DataLoader)




# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = GeometricDataset(num_samples=320, num_points=320)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train VAE
    model = PointCloudVAE(latent_dim=64, num_points=320)
    losses = train_vae(model, train_loader, num_epochs=30000)