import torch
import torch.nn as nn

class ConvolutionUnits(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim = 128, num_convs = 3):
        super().__init__()

        self.pre_conv = nn.Conv2d(input_dim, latent_dim, 5, 1, 2)
        self.conv_modules = nn.ModuleList([
            nn.Conv2d(latent_dim, latent_dim, 5, 1, 2) for _ in range(num_convs)
        ])
        self.final_conv = nn.Conv2d(latent_dim, output_dim, 5, 1, 2)
    
    def forward(self, x):
        x = self.pre_conv(x)
        for conv_module in self.conv_modules:
            x = conv_module(x)
        x = self.final_conv(x)
        return x

class ConvolutionBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim = 128):
        super().__init__()
    
    def forward(self, x):
        return x

class GridConvolution(nn.Module):
    def __init__(self,input_dim, output_dim, latent_dim):
        super().__init___()

    def forward(self, x):
        return x