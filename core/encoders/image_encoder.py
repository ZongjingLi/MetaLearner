import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class PositionalEncoding2D(nn.Module):
    """
    Adds 2D positional encodings to the input features.
    Based on the sine-cosine positional encoding from the Transformer architecture.
    
    Args:
        channels (int): Number of channels to add positional encodings to
        height (int): Height of the feature map
        width (int): Width of the feature map
    """
    def __init__(self, channels, height, width):
        super().__init__()
        
        # Create positional encodings
        pos_h = torch.arange(height).unsqueeze(1).float()
        pos_w = torch.arange(width).unsqueeze(0).float()
        
        # Scale positions
        div_term = torch.exp(torch.arange(0, channels//4, 2).float() * (-math.log(10000.0) / (channels//4)))
        
        # Create empty encoding tensor [channels, height, width]
        pe = torch.zeros(channels, height, width)
        
        # Fill encoding tensor with sin/cos patterns
        for i in range(0, channels//4, 2):
            pe[i] = torch.sin(pos_h * div_term[i//2])
            pe[i+1] = torch.cos(pos_h * div_term[i//2])
            pe[i+channels//4] = torch.sin(pos_w * div_term[i//2])
            pe[i+1+channels//4] = torch.cos(pos_w * div_term[i//2])
        
        # Register buffer (persistent but not model parameters)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, channels, height, width]
        
        Returns:
            Output tensor with positional encodings added
        """
        # Add positional encodings to the input
        # Only add to the first channels//2 channels to preserve some original features
        x = x.clone()
        channels = min(x.size(1), self.pe.size(0))
        x[:, :channels, :, :] = x[:, :channels, :, :] + self.pe[:channels, :x.size(2), :x.size(3)]
        return x


class ImageEncoder(nn.Module):
    """
    An image encoder that incorporates positional encoding to help the model
    understand the spatial relationships of objects.
    
    Input:
    - images: tensor of shape [batch_size, 3, H, W]
    - masks: tensor of shape [batch_size, num_objects, H, W]
    
    Output:
    - embeddings: tensor of shape [batch_size, num_objects, embedding_dim]
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        
        # Input dimensions
        self.channels = 4  # RGB + mask
        self.embedding_dim = embedding_dim
        
        # CNN for processing masked regions (4 channels: RGB + mask)
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, embedding_dim, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(embedding_dim)
        
        # Positional encodings for each layer
        self.pe1 = None  # Will be initialized in forward pass
        self.pe2 = None
        self.pe3 = None
        self.pe4 = None
        
        # Global pooling and final MLP
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, images, masks):
        batch_size, num_objects, H, W = masks.shape
        device = images.device
        
        # Initialize positional encodings if not already done
        if self.pe1 is None or self.pe1.pe.device != device:
            self.pe1 = PositionalEncoding2D(32, H//2, W//2).to(device)
            self.pe2 = PositionalEncoding2D(64, H//4, W//4).to(device)
            self.pe3 = PositionalEncoding2D(128, H//8, W//8).to(device)
            self.pe4 = PositionalEncoding2D(self.embedding_dim, H//16, W//16).to(device)
        
        # Initialize output embeddings
        embeddings = torch.zeros(batch_size, num_objects, self.embedding_dim, device=device)
        
        # Process each object in each image
        for b in range(batch_size):
            img = images[b]  # [3, H, W]
            
            for n in range(num_objects):
                obj_mask = masks[b, n].unsqueeze(0)  # [1, H, W]
                
                # Concatenate image with mask
                masked_input = torch.cat([img, obj_mask], dim=0)  # [4, H, W]
                masked_input = masked_input.unsqueeze(0)  # [1, 4, H, W]
                
                # Apply CNN with positional encoding
                x = self.conv1(masked_input)
                x = self.pe1(x)
                x = F.relu(self.bn1(x))
                
                x = self.conv2(x)
                x = self.pe2(x)
                x = F.relu(self.bn2(x))
                
                x = self.conv3(x)
                x = self.pe3(x)
                x = F.relu(self.bn3(x))
                
                x = self.conv4(x)
                x = self.pe4(x)
                x = F.relu(self.bn4(x))
                
                # Global pooling
                x = self.pool(x).view(-1, self.embedding_dim)
                
                # MLP
                embeddings[b, n] = self.mlp(x)
        
        return embeddings

class ImageEncoderV2(nn.Module):
    """
    More efficient image encoder that processes all masks in parallel.
    
    Input:
    - image: tensor of shape [batch_size, 3, H, W]
    - masks: tensor of shape [batch_size, num_objects, H, W]
    
    Output:
    - embeddings: tensor of shape [batch_size, num_objects, embedding_dim]
    """
    def __init__(self, embedding_dim=128, backbone="resnet18", use_pretrained=True):
        super(ImageEncoderV2, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Get the backbone CNN model
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=use_pretrained)
            self.feature_dim = 512
        elif backbone == "resnet34":
            base_model = models.resnet34(pretrained=use_pretrained)
            self.feature_dim = 512
        elif backbone == "resnet50":
            base_model = models.resnet50(pretrained=use_pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Use the convolutional layers of the backbone
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Projection layer to reduce feature dimensions
        self.projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final MLP for object embedding
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, images, masks):
        batch_size, num_objects, H, W = masks.shape
        device = images.device
        
        # Extract global image features - [B, feature_dim, h, w]
        image_features = self.backbone(images)
        _, _, h, w = image_features.shape
        
        # Project to embedding dimension - [B, embedding_dim, h, w]
        image_features = self.projection(image_features)
        
        # Resize all masks to match feature map size
        masks_resized = F.interpolate(
            masks.view(batch_size * num_objects, 1, H, W), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).view(batch_size, num_objects, h, w)
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, num_objects, self.embedding_dim, device=device)
        
        for b in range(batch_size):
            # Expand image features for this batch to match number of objects
            # [embedding_dim, h, w] -> [num_objects, embedding_dim, h, w]
            batch_features = image_features[b].unsqueeze(0).expand(num_objects, -1, -1, -1)
            
            # Apply masks to features (element-wise multiplication)
            # [num_objects, embedding_dim, h, w] * [num_objects, 1, h, w]
            masked_features = batch_features * masks_resized[b].unsqueeze(1)
            
            # Global average pooling over spatial dimensions for each object
            mask_sums = masks_resized[b].reshape(num_objects, -1).sum(dim=1).unsqueeze(1) + 1e-8
            obj_features = masked_features.reshape(num_objects, self.embedding_dim, -1).sum(dim=2) / mask_sums
            
            # Apply final MLP to each object's features
            for n in range(num_objects):
                embeddings[b, n] = self.mlp(obj_features[n])
        
        return embeddings


class ImageEncoderEfficient(nn.Module):
    """
    Highly efficient image encoder that processes all objects in a batch simultaneously.
    
    Input:
    - image: tensor of shape [batch_size, 3, H, W]
    - masks: tensor of shape [batch_size, num_objects, H, W]
    
    Output:
    - embeddings: tensor of shape [batch_size, num_objects, embedding_dim]
    """
    def __init__(self, embedding_dim=128, backbone="resnet18", use_pretrained=True):
        super(ImageEncoderEfficient, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Get the backbone CNN model
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=use_pretrained)
            self.feature_dim = 512
        elif backbone == "resnet34":
            base_model = models.resnet34(pretrained=use_pretrained)
            self.feature_dim = 512
        elif backbone == "resnet50":
            base_model = models.resnet50(pretrained=use_pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Use the convolutional layers of the backbone
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Projection layer to reduce feature dimensions
        self.projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final MLP for object embedding
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, images, masks):
        batch_size, num_objects, H, W = masks.shape
        
        # Extract global image features
        image_features = self.backbone(images)  # [batch_size, feature_dim, h, w]
        _, _, h, w = image_features.shape
        
        # Project to embedding dimension
        image_features = self.projection(image_features)  # [batch_size, embedding_dim, h, w]
        
        # Resize all masks to match feature map size
        masks_resized = F.interpolate(
            masks.view(batch_size * num_objects, 1, H, W), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).view(batch_size, num_objects, h, w)
        
        # Prepare image features for batch processing
        # Repeat image features for each object: [batch_size, embedding_dim, h, w] -> [batch_size, num_objects, embedding_dim, h, w]
        repeated_features = image_features.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        
        # Prepare masks for multiplication: [batch_size, num_objects, h, w] -> [batch_size, num_objects, 1, h, w]
        masks_expanded = masks_resized.unsqueeze(2)
        
        # Apply masks to features (element-wise multiplication)
        masked_features = repeated_features * masks_expanded  # [batch_size, num_objects, embedding_dim, h, w]
        
        # Sum over spatial dimensions
        summed_features = masked_features.view(batch_size, num_objects, self.embedding_dim, -1).sum(dim=3)  # [batch_size, num_objects, embedding_dim]
        
        # Calculate mask sums for averaging
        mask_sums = masks_resized.view(batch_size, num_objects, -1).sum(dim=2).unsqueeze(2) + 1e-8  # [batch_size, num_objects, 1]
        
        # Normalize by mask area
        avg_features = summed_features / mask_sums  # [batch_size, num_objects, embedding_dim]
        
        # Reshape for MLP application
        flattened_features = avg_features.view(batch_size * num_objects, self.embedding_dim)
        
        # Apply MLP
        mlp_output = self.mlp(flattened_features)
        
        # Reshape back to [batch_size, num_objects, embedding_dim]
        embeddings = mlp_output.view(batch_size, num_objects, self.embedding_dim)
        
        return embeddings


def test_sprite_encoder():
    # Create dummy data
    batch_size = 2
    num_objects = 3
    image_size = 64
    embedding_dim = 128
    
    # Random images and masks
    images = torch.randn(batch_size, 3, image_size, image_size)
    masks = torch.zeros(batch_size, num_objects, image_size, image_size)
    
    # Create some simple masks for testing
    for b in range(batch_size):
        for n in range(num_objects):
            # Create a square mask at different positions
            start_h = (n * 15) % (image_size - 20)
            start_w = (n * 10) % (image_size - 20)
            masks[b, n, start_h:start_h+20, start_w:start_w+20] = 1.0
    
    # Initialize models
    encoder_v1 = ImageEncoder(embedding_dim=embedding_dim)
    encoder_v2 = ImageEncoderV2(embedding_dim=embedding_dim)
    encoder_eff = ImageEncoderEfficient(embedding_dim=embedding_dim)
    
    # Test forward pass
    with torch.no_grad():
        embeddings_v1 = encoder_v1(images, masks)
        embeddings_v2 = encoder_v2(images, masks)
        embeddings_eff = encoder_eff(images, masks)
    
    # Print output shapes
    print(f"Input image shape: {images.shape}")
    print(f"Input masks shape: {masks.shape}")
    print(f"Output embeddings shape (V1): {embeddings_v1.shape}")
    print(f"Output embeddings shape (V2): {embeddings_v2.shape}")
    print(f"Output embeddings shape (Efficient): {embeddings_eff.shape}")
    
    # Verify shape
    assert embeddings_v1.shape == (batch_size, num_objects, embedding_dim)
    assert embeddings_v2.shape == (batch_size, num_objects, embedding_dim)
    assert embeddings_eff.shape == (batch_size, num_objects, embedding_dim)
    
    print("All tests passed!")


if __name__ == "__main__":
    test_sprite_encoder()