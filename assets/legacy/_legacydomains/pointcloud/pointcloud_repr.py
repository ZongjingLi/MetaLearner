import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, points, labels):
        self.points = torch.FloatTensor(points)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]

class PointNetEncoder(nn.Module):
    def __init__(self, num_points=1024, latent_dim=128):
        super(PointNetEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc1 = nn.Linear(256, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        
        x = F.relu(self.fc_bn1(self.fc1(x)))
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var

class PointNetDecoder(nn.Module):
    def __init__(self, num_points=1024, latent_dim=128):
        super(PointNetDecoder, self).__init__()
        
        self.num_points = num_points
        
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, num_points * 3)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        b, d = x.shape
        if b == 1:
            x = torch.cat([x, torch.rand_like(x)])
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        if b == 1:
            x = x[0:1]

        x = x.view(-1, self.num_points, 3)
        return x

class PointCloudVAE(nn.Module):
    def __init__(self, num_points=1024, latent_dim=128):
        super(PointCloudVAE, self).__init__()
        
        self.encoder = PointNetEncoder(num_points, latent_dim)
        self.decoder = PointNetDecoder(num_points, latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        #print(z.shape)
        reconstruction = self.decoder(z)
        #print(reconstruction.shape)
        return reconstruction, mu, log_var

def chamfer_distance(x, y):
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    
    dist = torch.sum((x - y) ** 2, dim=-1)
    
    min_dist_x = torch.min(dist, dim=2)[0]
    min_dist_y = torch.min(dist, dim=1)[0]
    
    chamfer_dist = torch.mean(min_dist_x) + torch.mean(min_dist_y)
    return chamfer_dist


def train_vae(model, train_loader, num_epochs=100, learning_rate=2e-4, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            
            # Reconstruction loss (Chamfer distance)
            recon_loss = chamfer_distance(recon_batch, data)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon_loss + 0.01 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')


def train_vae_with_saving(model, train_loader, num_epochs=100, learning_rate=2e-4, device='cuda', ctl = 0.000001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    val_loader = train_loader
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            recon_loss = chamfer_distance(recon_batch, data)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + ctl * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                recon_loss = chamfer_distance(recon_batch, data)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + ctl*kl_loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_vae.pth')      

if __name__ == "__main__":
    # Generate dataset
    from datasets.geometric_generator import *

    generator = GeometryGenerator(n_points=2048)
    points, labels = generator.generate_dataset(n_samples_per_class=100)
    
    # Create dataset and dataloader
    dataset = PointCloudDataset(points, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = PointCloudVAE(num_points=2048, latent_dim=128)
    vae.load_state_dict(torch.load("pointcloud_vae_state.pth"))
    train_vae_with_saving(vae, dataloader, num_epochs=1000, device=device)
    torch.save(vae.state_dict(), "state.pth")