import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union, Any
import re

class TypeSpaceBase:
    """Base class for type spaces that handle different kinds of data tensors"""
    
    @property
    def n_dim(self) -> int:
        """Return the number of dimensions for this type"""
        raise NotImplementedError
    
    @property
    def dtype(self) -> str:
        """Return the data type for this type space"""
        raise NotImplementedError
    
    def validate(self, value: Any) -> bool:
        """Check if the given value conforms to this type space"""
        raise NotImplementedError
    
    def create_empty(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Create an empty tensor of this type with optional batch dimension"""
        raise NotImplementedError
    
    @staticmethod
    def parse_type(type_str):
        """
        Parse a type string and return the corresponding TypeSpace object.
        
        Formats:
        - "bool" -> BooleanTypeSpace()
        - "vector[float,[128]]" -> VectorTypeSpace('float', [128])
        - "vector[int,[8,16]]" -> VectorTypeSpace('int', [8, 16])
        """
        # Handle boolean type
        if type_str.lower() == "bool":
            return BooleanTypeSpace()
        
        # Handle vector types with format: vector[dtype,[dim1,dim2,...]]
        vector_pattern = r"vector\[(.*?),\[(.*?)\]\]"
        match = re.match(vector_pattern, type_str)
        
        if match:
            # Extract dtype and dimensions
            dtype = match.group(1).strip()
            dims_str = match.group(2).strip()
            
            # Parse dimensions
            try:
                # Split by comma and convert to integers
                dims = [int(d.strip()) for d in dims_str.split(',')]
                return VectorTypeSpace(dtype, dims)
            except ValueError:
                raise ValueError(f"Invalid dimensions in vector type: {dims_str}. Expected integers.")
        
        # If we get here, the format is not recognized
        raise ValueError(f"Unrecognized type format: {type_str}. Expected 'bool' or 'vector[dtype,[dim1,dim2,...]]'")


class BooleanTypeSpace(TypeSpaceBase):
    """Type space for boolean values (represented as 1D vector)"""
    
    @property
    def n_dim(self) -> int:
        return 1
    
    @property
    def dtype(self) -> str:
        return 'bool'
    
    def validate(self, value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return value.dtype == torch.bool
        return isinstance(value, bool)
    
    def create_empty(self, batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            return torch.zeros((1,), dtype=torch.bool)
        return torch.zeros((batch_size, 1), dtype=torch.bool)
    
    def __str__(self) -> str:
        return "bool"

class VectorTypeSpace(TypeSpaceBase):
    """Type space for fixed-dimension vectors"""
    
    def __init__(self, dtype: str, dims: List[int]):
        self.dtype_str = dtype
        self.dims = dims
    
    @property
    def n_dim(self) -> int:
        return len(self.dims)
    
    @property
    def dtype(self) -> str:
        return self.dtype_str
    
    @property
    def num_elements(self) -> int:
        """Return the total number of elements in this type space"""
        return int(torch.prod(torch.tensor(self.dims)).item())
    
    def _torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype"""
        dtype_map = {
            'float': torch.float32,
            'float32': torch.float32,
            'float64': torch.float64,
            'int': torch.int32,
            'int32': torch.int32,
            'int64': torch.int64,
            'bool': torch.bool,
        }
        return dtype_map.get(self.dtype_str.lower(), torch.float32)



class TypeCaster(nn.Module):
    """
    Neural network that performs batched differentiable type casting between
    a specific source type space and target type space.
    """
    
    def __init__(self, source_type: TypeSpaceBase, target_type: TypeSpaceBase, hidden_dim=256):
        super().__init__()
        self.source_type = source_type
        self.target_type = target_type
        
        # Calculate total elements for source and target
        self.source_elements = self.source_type.num_elements
        self.target_elements = self.target_type.num_elements
        
        # Determine if this is a simple reshape or more complex transformation
        self.requires_transformation = (
            self.source_elements != self.target_elements or 
            self.source_type.dtype != self.target_type.dtype
        )
        
        # Main transformation network
        if self.requires_transformation:
            self.transformer = nn.Sequential(
                nn.Linear(self.source_elements, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.target_elements)
            )
        
        # Confidence prediction network (predicts probability of successful conversion)
        self.confidence_net = nn.Sequential(
            nn.Linear(self.target_elements, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform batched type casting operation
        
        Args:
            x: Input tensor or batch of tensors conforming to source_type
              - For single input: shape should match source_type.dims
              - For batched input: first dimension is batch size, rest should match source_type.dims
            
        Returns:
            Tuple of (cast_tensor, confidence)
            - cast_tensor: The transformed tensor(s) that conform to target_type
            - confidence: Confidence score (0-1) for each transformation
        """
        # Determine if input has batch dimension
        original_shape = x.shape
        is_batched = len(original_shape) > len(self.source_type.dims)
        
        if is_batched:
            batch_size = original_shape[0]
            # Reshape to (batch_size, source_elements)
            x_flat = x.reshape(batch_size, -1)
        else:
            # Add batch dimension for processing
            batch_size = 1
            x_flat = x.reshape(1, -1)
        
        # Validate input shape
        expected_flat_size = self.source_elements
        if x_flat.shape[1] != expected_flat_size:
            raise ValueError(
                f"Input tensor has wrong number of elements. "
                f"Got {x_flat.shape[1]}, expected {expected_flat_size}"
            )
        
        # Apply transformation if needed
        if self.requires_transformation:
            transformed = self.transformer(x_flat)
        else:
            # Simple reshape
            transformed = x_flat
        
        # Compute confidence score for each item in batch
        confidence = self.confidence_net(transformed).squeeze(-1)
        
        # Reshape to target dimensions
        target_shape = tuple(self.target_type.dims)
        if is_batched:
            reshaped = transformed.reshape(batch_size, *target_shape)
        else:
            # Remove batch dimension for single inputs
            reshaped = transformed.reshape(*target_shape)
        
        # Convert to target data type
        result = reshaped.to(self.target_type._torch_dtype())
        
        return result, confidence


if __name__ == "__main__":
    # Define source and target types
    source_type = VectorTypeSpace('float', [128])
    target_type = VectorTypeSpace('float32', [8,9])
    
    # Create the TypeCaster
    caster = TypeCaster(source_type, target_type, hidden_dim=256)
    
    # Create sample data
    input_tensor = torch.randn(128)
    
    output_tensor = caster(input_tensor)
    
    #print(output_tensor[0])
    #print(output_tensor[1])