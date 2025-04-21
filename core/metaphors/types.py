import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union, Any
import re

class TypeSpaceBase:
    """Base class for type spaces that handle different kinds of data tensors"""
    
    @property
    def n_dim(self) -> int:
        """return the number of dimensions for this type"""
        raise NotImplementedError
    
    @property
    def dtype(self) -> str:
        """return the data type for this type space"""
        raise NotImplementedError
    
    def validate(self, value: Any) -> bool:
        """check if the given value conforms to this type space"""
        raise NotImplementedError
    
    def create_empty(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """create an empty tensor of this type with optional batch dimension"""
        raise NotImplementedError
    
    @staticmethod
    def parse_type(type_str):
        """
        Parse a type string and return the corresponding TypeSpace object.
        
        Formats:
        - "bool" or "boolean" -> BooleanTypeSpace()
        - "vector[float,[128]]" -> VectorTypeSpace('float', [128])
        - "vector[int,[8,16]]" -> VectorTypeSpace('int', [8, 16])
        - "List[bool]" or "List[boolean]" -> ListTypeSpace(BooleanTypeSpace())
        - "List[vector[float,[64]]]" -> ListTypeSpace(VectorTypeSpace('float', [64]))
        - "List[Obj]" -> ListTypeSpace(ObjectTypeSpace())
        """
        # Remove any whitespace
        type_str = type_str.strip()
        
        # Handle simple types first
        if type_str.lower() in ["bool", "boolean"]:
            return BooleanTypeSpace()
            
        if type_str.lower() == "obj":
            return ObjectTypeSpace()
        
        # Check if it's a List type
        if type_str.lower().startswith("List["):
            # Need to find the correct closing bracket
            # Count open and close brackets to handle nested types
            open_count = 0
            close_count = 0
            end_pos = -1
            
            for i, char in enumerate(type_str):
                if char == '[':
                    open_count += 1
                elif char == ']':
                    close_count += 1
                    if open_count == close_count:
                        end_pos = i
                        break
            
            if end_pos == -1:
                raise ValueError(f"Unmatched brackets in type: {type_str}")
            
            # Extract the inner type (everything between List[ and the last ])
            inner_type_str = type_str[5:end_pos]
            
            # Recursively parse the inner type
            inner_type = TypeSpaceBase.parse_type(inner_type_str)
            return ListTypeSpace(inner_type)
        
        # Handle vector types with format: vector[dtype,[dim1,dim2,...]]
        if type_str.lower().startswith("vector["):
            # Find the comma that separates dtype from dimensions
            # Need to count brackets to handle nested parts
            bracket_depth = 0
            comma_pos = -1
            
            for i, char in enumerate(type_str):
                if char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                elif char == ',' and bracket_depth == 1:  # Only count commas at depth 1
                    comma_pos = i
                    break
            
            if comma_pos == -1:
                raise ValueError(f"Invalid vector format: {type_str}")
            
            # Extract dtype and dimensions part
            dtype_part = type_str[7:comma_pos].strip()
            dims_part = type_str[comma_pos+1:-1].strip()
            
            # Find the opening and closing brackets for the dims part
            if not dims_part.startswith('[') or not dims_part.endswith(']'):
                raise ValueError(f"Invalid dimensions format in vector: {dims_part}")
            
            # Extract the dimensions list (remove outer brackets)

            dims_str = dims_part[1:-1].strip()
            
            # Parse dimensions
            try:
                # Split by comma and convert to integers
                dims = [int(d.strip()) for d in dims_str.split(',')]
                return VectorTypeSpace(dtype_part, dims)
            except ValueError:
                raise ValueError(f"Invalid dimensions in vector type: {dims_str}. Expected integers.")
        
        # If we get here, the format is not recognized
        raise ValueError(f"Unrecognized type format: {type_str}. Expected 'bool', 'vector[dtype,[dim1,dim2,...]]', or 'List[...]'")


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
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, torch.Tensor):
            return False
        if value.dtype != self._torch_dtype():
            return False
        # Check if dimensions match
        expected_shape = tuple(self.dims)
        return value.shape == expected_shape
    
    def create_empty(self, batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            return torch.zeros(self.dims, dtype=self._torch_dtype())
        return torch.zeros((batch_size, *self.dims), dtype=self._torch_dtype())
    
    def __str__(self) -> str:
        dims_str = ','.join(str(d) for d in self.dims)
        return f"vector[{self.dtype_str},[{dims_str}]]"


class ObjectTypeSpace(TypeSpaceBase):
    """Type space for object references"""
    
    def __init__(self):
        # Objects are represented as indices (integers)
        pass
    
    @property
    def n_dim(self) -> int:
        return 1
    
    @property
    def dtype(self) -> str:
        return 'int64'
    
    def validate(self, value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return value.dtype == torch.int64 and value.ndim == 1 and value.size(0) == 1
        return isinstance(value, int)
    
    def create_empty(self, batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            return torch.zeros((1,), dtype=torch.int64)
        return torch.zeros((batch_size, 1), dtype=torch.int64)
    
    def __str__(self) -> str:
        return "Obj"


class ListTypeSpace(TypeSpaceBase):
    """Type space for variable-length lists of a specific type"""
    
    def __init__(self, element_type: TypeSpaceBase):
        self.element_type = element_type
    
    @property
    def n_dim(self) -> int:
        # List adds one dimension to the element type
        return 1 + self.element_type.n_dim
    
    @property
    def dtype(self) -> str:
        return self.element_type.dtype
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, (list, torch.Tensor)):
            return False
        
        if isinstance(value, torch.Tensor):
            # For tensors, the first dimension is the list length
            if value.ndim <= self.element_type.n_dim:
                return False
            
            # Check each item in the list
            for i in range(value.size(0)):
                item = value[i]
                if not self.element_type.validate(item):
                    return False
            return True
        else:  # It's a Python list
            for item in value:
                if not self.element_type.validate(item):
                    return False
            return True
    
    def create_empty(self, batch_size: Optional[int] = None, list_size: int = 0) -> torch.Tensor:
        """
        Create an empty tensor for this list type
        
        Args:
            batch_size: Optional batch dimension
            list_size: Size of the list dimension (can be 0 for empty list)
        """
        # Get the shape of a single element
        element_shape = self.element_type.create_empty().shape
        
        if batch_size is None:
            shape = (list_size, *element_shape)
        else:
            shape = (batch_size, list_size, *element_shape)
        
        return torch.zeros(shape, dtype=self.element_type._torch_dtype())
    
    def __str__(self) -> str:
        return f"List[{str(self.element_type)}]"


class TypeCaster(nn.Module):
    """
    Neural network that performs batched differentiable type casting between
    a specific source type space and target type space.
    """
    
    def __init__(self, source_type: TypeSpaceBase, target_type: TypeSpaceBase, hidden_dim=256):
        super().__init__()
        self.source_type = source_type
        self.target_type = target_type
        
        # Handle list types specially
        self.source_is_list = isinstance(source_type, ListTypeSpace)
        self.target_is_list = isinstance(target_type, ListTypeSpace)
        
        # If both are lists, we need to cast between their element types
        if self.source_is_list and self.target_is_list:
            self.element_caster = TypeCaster(
                source_type.element_type, 
                target_type.element_type,
                hidden_dim
            )
        else:
            # For non-list types or list-to-non-list conversions:
            # Calculate total elements for source and target
            if hasattr(source_type, 'num_elements'):
                self.source_elements = source_type.num_elements
            else:
                self.source_elements = 1  # Default for simple types like bool
                
            if hasattr(target_type, 'num_elements'):
                self.target_elements = target_type.num_elements
            else:
                self.target_elements = 1  # Default for simple types like bool
            
            # Determine if this is a simple reshape or more complex transformation
            self.requires_transformation = (
                self.source_elements != self.target_elements or 
                source_type.dtype != target_type.dtype
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
        # For list types, this will operate on the whole list
        if self.target_is_list:
            input_size = hidden_dim  # Use a fixed size for the confidence net
            self.list_encoder = nn.GRU(
                self.target_elements if not self.source_is_list else hidden_dim,
                hidden_dim,
                batch_first=True
            )
        else:
            input_size = self.target_elements
            
        self.confidence_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim // 2),
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
        # Special handling for list types
        if self.source_is_list and self.target_is_list:
            # Both source and target are lists
            # Process each element in the list using the element caster
            
            # Determine if input has batch dimension
            is_batched = x.dim() > self.source_type.n_dim
            batch_size = x.size(0) if is_batched else 1
            list_size = x.size(1) if is_batched else x.size(0)
            
            # Reshape to process each list element
            if is_batched:
                x_reshaped = x.reshape(batch_size * list_size, -1)
            else:
                x_reshaped = x.reshape(list_size, -1)
            
            # Process each element
            casted_elements, element_confidences = self.element_caster(x_reshaped)
            
            # Reshape back to list format
            element_shape = casted_elements.shape[1:] if casted_elements.dim() > 1 else (1,)
            if is_batched:
                result = casted_elements.reshape(batch_size, list_size, *element_shape)
                element_confidences = element_confidences.reshape(batch_size, list_size)
            else:
                result = casted_elements.reshape(list_size, *element_shape)
                element_confidences = element_confidences.reshape(list_size)
            
            # Aggregate confidences across the list
            if is_batched:
                # Use list_encoder to get a fixed-size representation
                _, hidden = self.list_encoder(result.reshape(batch_size, list_size, -1))
                confidence = self.confidence_net(hidden.squeeze(0)).squeeze(-1)
            else:
                # For single input, average the element confidences
                confidence = element_confidences.mean().unsqueeze(0)
            
            return result, confidence
            
        # For non-list types or list-to-non-list conversions:
        # Determine if input has batch dimension
        original_shape = x.shape
        is_batched = len(original_shape) > self.source_type.n_dim
        
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
        
        # If target is a list type but source is not, we need to reshape
        if self.target_is_list:
            # In this case, we're creating a list with one element
            target_element_shape = self.target_type.element_type.create_empty().shape
            if is_batched:
                reshaped = transformed.reshape(batch_size, 1, *target_element_shape)
            else:
                reshaped = transformed.reshape(1, *target_element_shape)
        else:
            # Reshape to target dimensions
            if hasattr(self.target_type, 'dims'):
                target_shape = tuple(self.target_type.dims)
            else:
                target_shape = (1,)  # Default shape for simple types
                
            if is_batched:
                reshaped = transformed.reshape(batch_size, *target_shape)
            else:
                # Remove batch dimension for single inputs
                reshaped = transformed.reshape(*target_shape)
        
        # Convert to target data type
        if hasattr(self.target_type, '_torch_dtype'):
            result = reshaped.to(self.target_type._torch_dtype())
        else:
            # For types that don't specify a torch dtype, keep as is
            result = reshaped
        
        return result, confidence


if __name__ == "__main__":
    # Test the type parser
    type_examples = [
        "bool",
        "boolean",
        "vector[float,[128]]",
        "vector[int,[8,16]]",
        "List[bool]",
        "List[boolean]",
        "List[vector[float,[64]]]",
        "List[Obj]"
    ]
    
    print("Testing type parser with examples:")
    for type_str in type_examples:
        try:
            type_space = TypeSpaceBase.parse_type(type_str)
            print(f"'{type_str}' parsed as: {type_space}")
        except ValueError as e:
            print(f"Error parsing '{type_str}': {e}")
    
    # Test type casting between different types
    print("\nTesting type casting:")
    
    # Example 1: Cast from vector to vector
    source_type = VectorTypeSpace('float', [128])
    target_type = VectorTypeSpace('float32', [8, 8])
    
    caster = TypeCaster(source_type, target_type, hidden_dim=256)
    
    input_tensor = torch.randn(128)
    output_tensor, confidence = caster(input_tensor)
    
    print(f"Cast from {source_type} to {target_type}")
    print(f"Input shape: {input_tensor.shape}, Output shape: {output_tensor.shape}")
    print(f"Confidence: {confidence.item():.4f}")
    
    # Example 2: Cast from List[vector] to List[vector]
    source_list_type = ListTypeSpace(VectorTypeSpace('float', [64]))
    target_list_type = ListTypeSpace(VectorTypeSpace('float', [32]))
    
    list_caster = TypeCaster(source_list_type, target_list_type, hidden_dim=256)
    
    # Create a list with 3 vectors
    input_list = torch.randn(3, 64)
    output_list, list_confidence = list_caster(input_list)
    
    print(f"\nCast from {source_list_type} to {target_list_type}")
    print(f"Input shape: {input_list.shape}, Output shape: {output_list.shape}")
    print(f"Confidence: {list_confidence.item():.4f}")