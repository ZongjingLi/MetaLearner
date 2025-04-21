from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Iterable

class TypeBase(object):
    """Base class for all types."""

    def __init__(self, typename: str, alias: Optional[str] = None, parent_type: Optional['TypeBase'] = None):
        """Initialize the type.

        Args:
            typename: The name of the type.
            alias: The alias of the type.
        """

        self._typename = typename
        self._alias = alias
        self._parent_type = parent_type

    @property
    def typename(self) -> str:
        """The (full) typename of the type."""
        return self._typename

    @property
    def alias(self) -> Optional[str]:
        """An optional alias of the type."""
        return self._alias

    @property
    def parent_type(self) -> Optional['TypeBase']:
        """The parent type of the type."""
        return self._parent_type

    @property
    def element_type(self) -> Optional['TypeBase']:
        """The element type of the type."""
        raise TypeError(f'Type {self.typename} does not have an element type.')

    def set_parent_type(self, parent_type: 'TypeBase'):
        self._parent_type = parent_type

    @property
    def parent_typename(self):
        """Return the typename of the parent type."""
        return self._parent_type.typename

    @property
    def base_typename(self):
        """Return the typename of the base type."""
        if self._parent_type is None:
            return self.typename
        return self._parent_type.base_typename

    @property
    def is_wrapped_value_type(self) -> bool:
        return False

    @property
    def is_object_type(self) -> bool:
        """Return whether the type is an object type."""
        return False

    @property
    def is_value_type(self) -> bool:
        """Return whether the type is a value type."""
        return False

    @property
    def is_tensor_value_type(self) -> bool:
        """Return whether the type is a tensor value type."""
        return False

    @property
    def is_scalar_value_type(self) -> bool:
        return False

    @property
    def is_vector_value_type(self) -> bool:
        return False

    @property
    def is_pyobj_value_type(self) -> bool:
        """Return whether the type is a Python object value type."""
        return False

    @property
    def is_sequence_type(self) -> bool:
        """Return whether the type is a sequence type."""
        return False

    @property
    def is_tuple_type(self) -> bool:
        """Return whether the type is a tuple type."""
        return False

    @property
    def is_uniform_sequence_type(self) -> bool:
        return False

    @property
    def is_list_type(self) -> bool:
        """Return whether the type is a list type."""
        return False

    @property
    def is_batched_list_type(self) -> bool:
        """Return whether the type is a multidimensional list type."""
        return False

    def __str__(self) -> str:
        return self.short_str()

    def short_str(self) -> str:
        """Return the short string representation of the type."""
        if self.alias is not None:
            return self.alias
        return self.typename

    def long_str(self) -> str:
        """Return the long string representation of the type."""
        return f'Type[{self.short_str()}]'

    def assignment_type(self) -> 'TypeBase':
        """Return the value type for assignment."""
        return self

    def downcast_compatible(self, other: 'TypeBase', allow_self_list: bool = False, allow_list: bool = False) -> bool:
        """Check if the type is downcast-compatible with the other type; that is, if this type is a subtype of the other type.

        Args:
            other: the other type.
            allow_self_list: if True, this type can be a list type derived from the other type.
            allow_list: if True, the other type can be a list type derived from the type.
        """
        if self.typename == other.typename or other == AnyType or self == AutoType:
            return True
        if self.parent_type is not None and self.parent_type.downcast_compatible(other, allow_self_list=allow_self_list, allow_list=allow_list):
            return True
        if self.is_uniform_sequence_type and other.is_uniform_sequence_type:
            if not self.element_type.downcast_compatible(other.element_type, allow_self_list=False, allow_list=False):
                return False
            if self.is_batched_list_type and other.is_batched_list_type:
                if len(self.index_dtypes) != len(other.index_dtypes):
                    return False
                for i, j in zip(self.index_dtypes, other.index_dtypes):
                    if not i.downcast_compatible(j, allow_self_list=False, allow_list=False):
                        return False
                return True
            return True
        if allow_self_list and self.is_uniform_sequence_type:
            if self.element_type.downcast_compatible(other, allow_self_list=False, allow_list=False):
                return True
        if allow_list and self.is_uniform_sequence_type:
            if self.downcast_compatible(other.element_type, allow_self_list=False, allow_list=False):
                return True
        return False

    def unwrap_alias(self):
        return self

    def __eq__(self, other: 'TypeBase') -> bool:
        return self.typename == other.typename

    def __ne__(self, other: 'TypeBase') -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self.typename)

AnyType = TypeBase("AnyType") # the union of all types
AutoType = TypeBase("AutoType") # the type will be inferred later

class UnionType(TypeBase):
    """The UnionType is a type that is the union of multiple types."""

    def __init__(self, *types: TypeBase, alias: Optional[str] = None):
        """Initialize the union type.

        Args:
            types: The types in the union.
            alias: The alias of the union type.
        """
        self.types = tuple(types)
        super().__init__(self.long_str(), alias=alias)

    types: Tuple[TypeBase, ...]
    """The underlying types of the union type."""

    def short_str(self) -> str:
        return ' | '.join(t.short_str() for t in self.types)

    def long_str(self) -> str:
        return 'Union[' + ', '.join(t.long_str() for t in self.types) + ']'

    def downcast_compatible(self, other: TypeBase, allow_self_list: bool = False, allow_list: bool = False) -> bool:
        raise NotImplementedError('Cannot downcast to a union type.')

class ObjectType(TypeBase):
    """The ObjectType corresponds to the type of "real-world" objects."""

    def __init__(self, typename: str, parent_types: Optional[Sequence['ObjectType']] = None, alias: Optional[str] = None):
        """Initialize the object type.

        Args:
            typename: The name of the object type.
            alias: The alias of the object type.
        """
        super().__init__(typename, alias=alias)

        self.parent_types = tuple(parent_types) if parent_types is not None else tuple()

    @property
    def is_object_type(self):
        return True

    parent_types: Tuple['ObjectType', ...]
    """The parent types of the object type."""

    def iter_parent_types(self) -> Iterable['ObjectType']:
        """Iterate over all parent types.

        Yields:
            the parent types following the inheritance order.
        """
        for parent_type in self.parent_types:
            yield parent_type
            yield from parent_type.iter_parent_types()

    def long_str(self) -> str:
        if len(self.parent_types) == 0:
            return f'OT[{self.typename}]'

        return f'OT[{self.typename}, parent={", ".join(t.typename for t in self.parent_types)}]'



class SequenceType(TypeBase):
    """The basic sequence type. It has two forms: ListType and TupleType."""

    @property
    def is_sequence_type(self) -> bool:
        return True


class TupleType(SequenceType):
    def __init__(self, element_types: Sequence[TypeBase], alias: Optional[str] = None):
        super().__init__(f'Tuple[{", ".join(t.typename for t in element_types)}]', alias=alias)
        self.element_types = tuple(element_types)

    element_types: Tuple[TypeBase, ...]
    """The element types of the tuple."""

    @property
    def is_tuple_type(self) -> bool:
        return True


class UniformSequenceType(SequenceType):
    def __init__(self, typename: str, element_type: TypeBase, alias: Optional[str] = None):
        super().__init__(typename, alias=alias)
        self._element_type = element_type

    _element_type: TypeBase
    """The element type of the list."""

    @property
    def element_type(self) -> TypeBase:
        return self._element_type

    @property
    def is_uniform_sequence_type(self) -> bool:
        return True

    @property
    def is_object_type(self) -> bool:
        return self.element_type.is_object_type

    @property
    def is_value_type(self) -> bool:
        return self.element_type.is_value_type


class ListType(UniformSequenceType):
    def __init__(self, element_type: TypeBase, alias: Optional[str] = None):
        typename = f'List[{element_type.typename}]'
        super().__init__(typename, element_type, alias=alias)

    @property
    def is_list_type(self) -> bool:
        return True


class BatchedListType(UniformSequenceType):
    def __init__(self, element_type: TypeBase, index_dtypes: Sequence[ObjectType], alias: Optional[str] = None):
        typename = f'{element_type.typename}[{", ".join(t.typename for t in index_dtypes)}]'
        super().__init__(typename, element_type, alias=alias)
        self.index_dtypes = tuple(index_dtypes)

    element_type: TypeBase
    """The element type of the list."""

    index_dtypes: Tuple[ObjectType, ...]
    """The index types of the list."""

    def ndim(self) -> int:
        """The number of dimensions of the list."""
        return len(self.index_dtypes)

    @property
    def is_batched_list_type(self) -> bool:
        return True

    def iter_element_type(self) -> TypeBase:
        """Return the element type if we iterate over the list. Basically type(value[0])."""
        if len(self.index_dtypes) == 1:
            return self.element_type
        return BatchedListType(self.element_type, index_dtypes=self.index_dtypes[1:])
    
@dataclass
class FuncType(TypeBase):
    def __init__(self, parameters, return_type):
        self.parameters = parameters
        self.return_type = return_type


class ObjectType(TypeBase):
    """The ObjectType corresponds to the type of "real-world" objects."""

    def __init__(self, typename: str, parent_types: Optional[Sequence['ObjectType']] = None, alias: Optional[str] = None):
        """Initialize the object type.

        Args:
            typename: The name of the object type.
            alias: The alias of the object type.
        """
        super().__init__(typename, alias=alias)

        self.parent_types = tuple(parent_types) if parent_types is not None else tuple()

    @property
    def is_object_type(self):
        return True

    parent_types: Tuple['ObjectType', ...]
    """The parent types of the object type."""

    def iter_parent_types(self) -> Iterable['ObjectType']:
        """Iterate over all parent types.

        Yields:
            the parent types following the inheritance order.
        """
        for parent_type in self.parent_types:
            yield parent_type
            yield from parent_type.iter_parent_types()

    def long_str(self) -> str:
        if len(self.parent_types) == 0:
            return f'OT[{self.typename}]'

        return f'OT[{self.typename}, parent={", ".join(t.typename for t in self.parent_types)}]'
    

