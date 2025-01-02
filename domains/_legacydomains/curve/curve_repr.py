import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from scipy.integrate import quad
import math
from .dataset_generator import GeometricShapeGenerator

class GeometricDataset(Dataset):
    def __init__(self, num_samples: int = 1000, num_points: int = 100):
        super().__init__()
        self.generator = GeometricShapeGenerator(num_points)#EnhancedShapeGenerator()#
        self.shapes = self.generator.generate_shapes()
        self.data = []
        self.labels = []

        shapes_list = list(self.shapes.keys())
        for i in range(num_samples):

            shape_name = np.random.choice(shapes_list)
            shape_points = self.shapes[shape_name]
            self.data.append(shape_points)
            self.labels.append(shapes_list.index(shape_name))

        self.data = torch.FloatTensor(np.array(self.data))
        self.labels = torch.LongTensor(self.labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
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
              device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[float]:

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

    return losses

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = GeometricDataset(num_samples=320, num_points=320)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train VAE
    model = PointCloudVAE(latent_dim=64, num_points=320)
    losses = train_vae(model, train_loader, num_epochs=30000)