"""value type modeling the probablistic measures on the program context"""
from typing import Optional, Sequence, Any, Iterable, Mapping, Tuple, Dict, Union
from dataclasses import dataclass
from termcolor import colored

from helchriss.utils import stprint_str
from .dsl_types import ListType, ObjectType
import torch

class ProbValue(object):
    def __init__(self, vtype, value, prob = 1.0):
        self.vtype = vtype
        self.value = value
        self.prob = prob
    
    def __repr__(self):
        return (colored("Value", "cyan", attrs=["bold"]) + 
                colored(":[", "white") + 
                colored(f"{self.value}", "blue") + 
                colored("]-", "white") + 
                colored(f"{self.vtype}", "cyan") + " " +
                colored("P", "cyan", attrs=["bold"]) + 
                colored(":[", "white") + 
                colored(f"{self.prob}", "blue") + 
                colored("]", "white"))

class Value(ProbValue):
    def __init__(self, vtype, value):
        super().__init__(vtype, value, prob = 1.0)

class ListValue(Value):
    """A list of values."""

    dtype: ListType
    """The type of the value."""

    @property
    def element_type(self):
        return self.dtype.element_type

    values: tuple
    """The values."""

    def __init__(self, dtype: ListType, values: Sequence[Any]):
        """Initialize the Value object.

        Args:
            dtype: the type of the value.
            values: the values.
        """
        super().__init__(dtype)
        self.values = tuple(values)

    def __len__(self):
        return len(self.values)

    def __str__(self):
        elements_str = ', '.join(str(v) for v in self.values)
        return f'{{{elements_str}}}'

    def __repr__(self):
        elements_str = ', '.join(str(v) for v in self.values)
        return f'ListValue({{{elements_str}}}, dtype={self.dtype})'

"""the tensor state of measurement and otherwise constructed"""

QINDEX = slice(None)

class ObjectType(object):
    pass

@dataclass
class StateObjectReference(object):
    name : str
    idx : int
    dtype: Optional[ObjectType] = None

    def clone(self, dtype: Optional[ObjectType] = None) -> 'StateObjectReference':
        return StateObjectReference(self.name, self.index, dtype or self.dtype)

class StateObjectList(ListValue):
    def __init__(self, dtype: ListType, values: Union[Sequence[StateObjectReference], slice]):
        if isinstance(values, slice):
            assert values is QINDEX, 'Only QINDEX is allowed for the values of StateObjectList.'
            super().__init__(dtype, tuple())
            self.values = QINDEX
        else:
            super().__init__(dtype, values)

    dtype: ListType

    values: Union[Tuple[StateObjectReference, ...], slice]

    @property
    def element_type(self) -> ObjectType:
        return self.dtype.element_type

    @property
    def is_qindex(self) -> bool:
        return self.values == QINDEX

    @property
    def array_accessor(self) -> Union[Sequence[int], slice]:
        if isinstance(self.values, slice):
            return self.values
        return [v.index for v in self.values]

    def clone(self, dtype: Optional[ListType] = None) -> 'StateObjectList':
        if dtype is None:
            return StateObjectList(self.dtype, self.values)

        assert isinstance(dtype, ListType) or dtype is None, 'dtype should be a ListType.'
        if self.is_qindex:
            return StateObjectList(dtype, QINDEX)
        return StateObjectList(dtype, tuple(v.clone(dtype.element_type) for v in self.values))

    def __str__(self):
        if self.values == QINDEX:
            elements_str = 'QINDEX'
        else:
            elements_str = ', '.join(str(v.name) for v in self.values)

    def __repr__(self):
        if self.values == QINDEX:
            return f'LV(QINDEX, dtype={self.dtype})'
        elements_str = ', '.join(str(v.name) for v in self.values)
        return f'LV({{{elements_str}}}, dtype={self.dtype})'


class TensorValue(ProbValue):
    pass

class ProbTensorValue(ProbValue):
    pass

class MultidimensionalArrayInterface(object):
    """
    A multi-dimensional array inferface. At a high-level, this can be interpreted as a dictionary that maps
    feature names (keys) to multi-diemsntional tensors (value).
    """

    def __init__(self, all_feature_names: Iterable[str] = tuple()):
        self.all_feature_names = set(all_feature_names)

    def clone(self) -> 'MultidimensionalArrayInterface':
        """Clone the multidimensional array interface."""
        raise NotImplementedError()

    def get_feature(self, name: str) -> TensorValue:
        """Get the feature tensor with the given name."""
        raise NotImplementedError()

    def _set_feature_impl(self, name: str, feature: TensorValue):
        """Set the feature tensor with the given name. It is guaranteed that the name is in the all_feature_names."""
        raise NotImplementedError()

    def set_feature(self, name: str, feature: TensorValue):
        """Set the feature tensor with the given name."""
        if name not in self.all_feature_names:
            self.all_feature_names.add(name)
        self._set_feature_impl(name, feature)

    def update_feature(self, other_tensor_dict: Mapping[str, TensorValue]):
        """Update the feature tensors with the given tensor dict."""
        for key, value in other_tensor_dict.items():
            self.set_feature(key, value)

    def __contains__(self, item: str) -> bool:
        """Check if the given feature name is in the interface."""
        return item in self.all_feature_names

    def __getitem__(self, name: str) -> TensorValue:
        """Get the feature tensor with the given name."""
        return self.get_feature(name)

    def __setitem__(self, key, value):
        """Set the feature tensor with the given name."""
        self.set_feature(key, value)

    def keys(self) -> Iterable[str]:
        """Get the feature names."""
        return self.all_feature_names

    def values(self) -> Iterable[TensorValue]:
        """Get the feature tensors."""
        for key in self.all_feature_names:
            yield self.get_feature(key)

    def items(self) -> Iterable[Tuple[str, TensorValue]]:
        """Get the feature name-tensor pairs."""
        for key in self.all_feature_names:
            yield key, self.get_feature(key)

class TensorValueDict(MultidimensionalArrayInterface):
    """Basic tensor dict implementation."""

    def __init__(self, tensor_dict: Optional[Dict[str, TensorValue]] = None):
        if tensor_dict is None:
            tensor_dict = dict()

        super().__init__(tensor_dict.keys())
        self.tensor_dict = tensor_dict

    def clone(self) -> 'TensorValueDict':
        return type(self)({k: v.clone() for k, v in self.tensor_dict.items()})

    def get_feature(self, name: str) -> TensorValue:
        return self.tensor_dict[name]

    def _set_feature_impl(self, name, feature: TensorValue):
        self.tensor_dict[name] = feature


@dataclass
class MaskedTensorStorage(object):
    """A storage for quantized tensors."""

    value: Union[torch.Tensor, Any]
    """The unquantized tensor."""

    mask: Optional[torch.Tensor] = None
    """The mask of the value. If not None, entry = 1 if the value is valid, and 0 otherwise."""

    optimistic_values: Optional[torch.Tensor] = None
    """The optimistic values for the tensor. 0 for non-optimistic values."""

    quantized_values: Optional[torch.Tensor] = None
    """The quantized values for the tensor. -1 for non-quantized values."""

