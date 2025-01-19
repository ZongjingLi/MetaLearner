import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ImageEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        input_channels: int = 3,
        base_channels: int = 64,
        num_conv_blocks: int = 4,
        mask_channels: int = 1
    ):
        """
        Initialize the image encoder with mask support.
        
        Args:
            embedding_dim: Dimension of the output embeddings
            input_channels: Number of input image channels (3 for RGB)
            base_channels: Base number of convolutional channels
            num_conv_blocks: Number of convolutional blocks
            mask_channels: Number of mask channels (typically 1 for binary mask)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        
        # Image encoder backbone
        self.conv_blocks = nn.ModuleList()
        curr_channels = input_channels + mask_channels  # Concatenate mask with image
        
        for i in range(num_conv_blocks):
            out_channels = base_channels * (2 ** i)
            self.conv_blocks.append(
                ConvBlock(
                    curr_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2 if i < num_conv_blocks-1 else 1
                )
            )
            curr_channels = out_channels
        
        # Global pooling and embedding projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_proj = nn.Sequential(
            nn.Linear(curr_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Standard image transforms
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Mask transform
        self.mask_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def preprocess_image(
        self, 
        image: Union[str, Image.Image, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Preprocess image to tensor."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            image = self.transforms(image)
        elif isinstance(image, torch.Tensor) and image.dim() == 3:
            image = self.transforms(image)
            
        return image

    def preprocess_mask(
        self, 
        mask: Union[str, Image.Image, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Preprocess mask to tensor."""
        if isinstance(mask, str):
            mask = Image.open(mask).convert('L')
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
            
        if isinstance(mask, Image.Image):
            mask = self.mask_transform(mask)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        return mask

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            image: Image tensor of shape (B, C, H, W)
            mask: Mask tensor of shape (B, 1, H, W)
            
        Returns:
            Embedding tensor of shape (B, embedding_dim)
        """
        # Concatenate image and mask along channel dimension
        x = torch.cat([image, mask], dim=1)
        
        # Pass through conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling and projection
        x = self.global_pool(x).flatten(1)
        embedding = self.embedding_proj(x)
        
        return embedding

    def encode(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        mask: Union[str, Image.Image, torch.Tensor, np.ndarray],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode image and mask to embedding.
        
        Args:
            image: Input image (file path, PIL Image, tensor, or numpy array)
            mask: Input mask (file path, PIL Image, tensor, or numpy array)
            return_tensor: If True, return PyTorch tensor; if False, return numpy array
            
        Returns:
            Embedding tensor or numpy array
        """
        # Preprocess inputs
        image_tensor = self.preprocess_image(image)
        mask_tensor = self.preprocess_mask(mask)
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
            
        # Move to same device as model
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.forward(image_tensor, mask_tensor)
            
        if return_tensor:
            return embedding
        return embedding.cpu().numpy()

    def save_model(self, path: str):
        """Save model weights to file."""
        torch.save(self.state_dict(), path)
        
    def load_model(self, path: str):
        """Load model weights from file."""
        self.load_state_dict(torch.load(path))
        
# Example usage:
if __name__ == "__main__":
    # Create encoder
    encoder = ImageEncoder(
        embedding_dim=512,
        input_channels=3,
        base_channels=64,
        num_conv_blocks=4
    )
    
    # Create dummy inputs
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_mask = torch.randint(0, 2, (batch_size, 1, 224, 224)).float()
    
    # Get embeddings
    embeddings = encoder(dummy_image, dummy_mask)
    print(f"Embedding shape: {embeddings.shape}")  # Should be (2, 512)
    
    # Example with PIL image and mask
    """
    from PIL import Image
    import numpy as np
    
    # Load image and mask
    image = Image.open("example.jpg")
    mask = Image.open("mask.png").convert('L')
    
    # Get embedding
    embedding = encoder.encode(image, mask)
    print(f"Single image embedding shape: {embedding.shape}")  # Should be (1, 512)
    """