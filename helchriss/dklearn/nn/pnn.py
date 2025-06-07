# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-11-10 00:16:27
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-03-16 21:19:56
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np

class STN3d(nn.Module):
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.device = "cuda" if torch.cuda.is_available() else "mps"


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)


        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        

        if x.is_cuda:
            iden = iden.cuda()

        iden = iden.to(x.device)
        x = x + iden

        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(channel=3)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.channel = channel
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[1]
        batch_size = x.shape[0]
        if batch_size == 1: x = x.repeat(2, 1, 1)
        
        # Transpose to (batch_size, channels, n_points)
        x = x.permute(0, 2, 1)
        
        # For colored point clouds, we only apply the spatial transform (STN3d) to the xyz coordinates
        if self.channel > 3:
            # Split into coordinates and features (colors)
            coords = x[:, :3, :]
            features = x[:, 3:, :]
            
            # Apply STN only to coordinates
            trans = self.stn(coords)
            
            # Transform coordinates
            coords = coords.transpose(2, 1)
            coords = torch.bmm(coords, trans)
            coords = coords.transpose(2, 1)
            
            # Recombine coordinates with features
            x = torch.cat([coords, features], dim=1)
        else:
            # Original implementation for xyz only
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            
        # Continue with the standard PointNet pipeline
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        if self.global_feat:
            return x[:batch_size, :]
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointAutoEncoder(nn.Module):
    def __init__(self, decode_points=1024, channel=3):
        super().__init__()
        # Encoder
        self.encoder = PointNetfeat(global_feat=True, feature_transform=True, channel=channel)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, decode_points * channel)
        )
        
        self.decode_points = decode_points
        self.channel = channel
    
    def forward(self, x):
        # x shape: [batch_size, num_points, channel]
        # Encode
        features = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(features)
        reconstructed = reconstructed.view(-1, self.decode_points, self.channel)
        
        return reconstructed, features
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        reconstructed = self.decoder(x)
        return reconstructed.view(-1, self.decode_points, self.channel)

# Example of how to use the model with colored point clouds
if __name__ == "__main__":
    # Example with XYZ only (3 channels)
    xyz_point_cloud = torch.randn(8, 1024, 3)  # [batch_size, num_points, 3]
    
    # Example with XYZ+RGB (6 channels)
    colored_point_cloud = torch.randn(8, 1024, 6)  # [batch_size, num_points, 6]
    
    # Create models
    xyz_autoencoder = PointAutoEncoder(decode_points=1024, channel=3)
    colored_autoencoder = PointAutoEncoder(decode_points=1024, channel=6)
    
    # Forward pass
    xyz_reconstructed, xyz_features = xyz_autoencoder(xyz_point_cloud)
    colored_reconstructed, colored_features = colored_autoencoder(colored_point_cloud)
    
    print(f"XYZ input shape: {xyz_point_cloud.shape}")
    print(f"XYZ features shape: {xyz_features.shape}")
    print(f"XYZ reconstructed shape: {xyz_reconstructed.shape}")
    
    print(f"Colored input shape: {colored_point_cloud.shape}")
    print(f"Colored features shape: {colored_features.shape}")
    print(f"Colored reconstructed shape: {colored_reconstructed.shape}")