
EPS = 1e-6
import numpy as np
from typing import Optional

numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
yes_or_no = ["yes", "no"]

def num2word(i):
    assert i < 10
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return numbers[i]
    
def copy_dict(d):
    return d if not isinstance(d, dict) else {k: copy_dict(v) for k, v in d.items()}


def head_and_paras(program):
    if len(program) == '' or program == None:return '',[]
    #assert program[-1] == ")", print(program)
    try:
        upper_index = program.index("(")
        assert program[-1] == ")", print(program)
        node_name = program[:upper_index]
        remain = program[upper_index+1:-1]
        args = [];count = 0
        last_start_index = 0

        for i in range(len(remain)):
            e = remain[i]
            if (e == "("):count += 1
            if (e == ")"):count -= 1
            if (count == 0 and e == ","):
                args.append(remain[last_start_index:i])
                last_start_index = i + 1
        args.append(remain[last_start_index:])
        return node_name, args
    except:
        return program, None

import numpy as np
import inspect
import re
import io
from contextlib import redirect_stdout
import dataclasses
from typing import Any, Dict, List, Tuple, Set

def stprint_str(data=None, var_name=None):
    """
    Returns a Lich King/Icecrown Citadel themed string representation of data structures.
    This function captures the output of stprint and returns it as a string.
    
    Args:
        data: Data to format. If None, uses a sample dictionary.
        var_name: Optional variable name to display. If None, attempts to detect it.
    
    Returns:
        A formatted string representation of the data.
    """
    # Capture the output in a string buffer
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        stprint(data, var_name, return_str=False)
    
    return buffer.getvalue()

def stprint(data=None, var_name=None, return_str=False):
    """
    Lich King/Icecrown Citadel themed pretty print function that formats various data 
    types including dataclasses, numpy arrays, PyTorch tensors, lists, and dictionaries.
    
    Args:
        data: Data to print. If None, uses a sample dictionary.
        var_name: Optional variable name to display. If None, attempts to detect it.
        return_str: If True, returns the formatted string instead of printing it.
    
    Returns:
        Formatted string if return_str is True, otherwise None.
    """
    # ANSI color codes - Icecrown Citadel theme
    ICY_BLUE = '\033[38;5;39m'    # For keys and containers
    FROST = '\033[38;5;45m'       # For strings
    SARONITE = '\033[38;5;240m'   # For brackets and structure
    PALE_BLUE = '\033[38;5;153m'  # For values
    DEATH_KNIGHT = '\033[38;5;63m' # For numpy/torch objects
    LICH_PURPLE = '\033[38;5;135m' # For special values and dataclasses
    SCOURGE_GREEN = '\033[38;5;77m' # For types and metadata
    FROSTFIRE = '\033[38;5;201m'  # For requires_grad=True
    BLOOD = '\033[38;5;160m'      # For requires_grad=False
    RUNIC = '\033[38;5;51m'       # For dataclass field names
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Prepare output buffer if returning a string
    if return_str:
        import sys
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
    
    # Sample data if none provided
    if data is None:
        from dataclasses import dataclass, field
        from typing import List, Tuple
        
        @dataclass
        class Item:
            color: str
            shape: str
            size: Tuple[float, float, float]
        
        @dataclass
        class Scene:
            items: List[Item]
        
        # Create sample data
        data = Scene(items=[
            Item(color="blue", shape="cube", size=(1.0, 1.0, 1.0)),
            Item(color="red", shape="sphere", size=(0.5, 0.5, 0.5))
        ])
    
    # Try to get the variable name if not provided
    if var_name is None:
        try:
            # Get the calling frame
            frame = inspect.currentframe().f_back
            # Get the source code of the calling line
            context = inspect.getframeinfo(frame).code_context[0].strip()
            # Extract the variable name using regex
            match = re.search(r'stprint(?:_str)?\s*\(\s*([^,)]+)', context)
            if match:
                var_name = match.group(1).strip()
        except:
            var_name = "data"  # Default if detection fails
    
    def _print_recursive(obj, indent=0, is_list_item=False, list_index=None, inline=False):
        indent_str = "  " * indent
        
        # Handle PyTorch tensors
        try:
            import torch
            is_torch_tensor = isinstance(obj, torch.Tensor)
        except ImportError:
            is_torch_tensor = False
        
        # Handle dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            class_name = type(obj).__name__
            if inline:
                print(f"{LICH_PURPLE}{class_name}{SARONITE}(...)")
            else:
                if is_list_item:
                    print(f"{indent_str}{ICY_BLUE}{list_index}: {LICH_PURPLE}{class_name}{SARONITE}{{")
                else:
                    print(f"{indent_str}{LICH_PURPLE}{class_name}{SARONITE}{{")
                
                # Get all fields
                fields = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
                
                # Print fields
                for key, value in fields.items():
                    print(f"{indent_str}  {RUNIC}{key}{PALE_BLUE}: ", end="")
                    if isinstance(value, (dict, list,)) or \
                       dataclasses.is_dataclass(value) or \
                       isinstance(value, np.ndarray) and len(value.shape) > 1 or \
                       is_torch_tensor and len(value.shape) > 1:
                        print()
                        _print_recursive(value, indent + 1)
                    else:
                        _print_recursive(value, indent, inline=True)
                
                print(f"{indent_str}{SARONITE}}}")
        
        # Handle dictionary
        elif isinstance(obj, dict):
            if indent == 0 or is_list_item:
                prefix = f"{indent_str}{ICY_BLUE}{list_index}: {SARONITE}" if is_list_item else f"{indent_str}{SARONITE}"
                print(f"{prefix}dict{PALE_BLUE}{{")
            else:
                print(f"{indent_str}{SARONITE}dict{PALE_BLUE}{{")
            
            for key, value in obj.items():
                print(f"{indent_str}  {ICY_BLUE}{key}{PALE_BLUE}: ", end="")
                if isinstance(value, (dict, list,)) or \
                   dataclasses.is_dataclass(value) or \
                   isinstance(value, np.ndarray) and len(value.shape) > 1 or \
                   is_torch_tensor and len(value.shape) > 1:
                    print()
                    _print_recursive(value, indent + 1)
                else:
                    _print_recursive(value, indent, inline=True)
            
            print(f"{indent_str}{SARONITE}}}")
        
        # Handle numpy ndarray
        elif isinstance(obj, np.ndarray):
            shape_str = f"shape={obj.shape}"
            dtype_str = f"dtype={obj.dtype}"
            
            if inline:
                print(f"{DEATH_KNIGHT}np.ndarray({SCOURGE_GREEN}{shape_str}, {dtype_str}{DEATH_KNIGHT})", end="")
                
                # For small 1D arrays, print inline
                if len(obj.shape) == 1 and obj.shape[0] <= 10:
                    array_str = np.array2string(obj, separator=', ')
                    print(f"{SARONITE}{{{PALE_BLUE}{array_str}{SARONITE}}}")
                else:
                    print()
            else:
                print(f"{indent_str}{DEATH_KNIGHT}np.ndarray({SCOURGE_GREEN}{shape_str}, {dtype_str}{DEATH_KNIGHT})", end="")
                
                # For 2D arrays, pretty print the values
                if len(obj.shape) <= 2:
                    array_str = np.array2string(obj, prefix=indent_str + '  ')
                    print(f"{SARONITE}{{{PALE_BLUE}")
                    for line in array_str.split('\n'):
                        print(f"{indent_str}  {PALE_BLUE}{line}")
                    print(f"{indent_str}{SARONITE}}}")
                else:
                    print()  # Just a newline for 3D+ arrays
        
        # Handle PyTorch tensor
        elif is_torch_tensor:
            shape_str = f"shape={tuple(obj.shape)}"
            dtype_str = f"dtype={obj.dtype}"
            device_str = f"device='{obj.device}'"
            
            # Add requires_grad information with color coding
            if hasattr(obj, 'requires_grad'):
                grad_color = FROSTFIRE if obj.requires_grad else BLOOD
                grad_str = f"requires_grad={grad_color}{obj.requires_grad}{SCOURGE_GREEN}"
            else:
                grad_str = ""
            
            if inline:
                print(f"{DEATH_KNIGHT}torch.Tensor({SCOURGE_GREEN}{shape_str}, {dtype_str}, {device_str}", end="")
                if grad_str:
                    print(f", {grad_str}{DEATH_KNIGHT})", end="")
                else:
                    print(f"{DEATH_KNIGHT})", end="")
                
                # For small 1D tensors, print inline
                if len(obj.shape) == 1 and obj.shape[0] <= 10:
                    tensor_str = str(obj.detach().cpu().numpy()).replace('\n', ' ')
                    print(f"{SARONITE}{{{PALE_BLUE}{tensor_str}{SARONITE}}}")
                else:
                    print()
            else:
                print(f"{indent_str}{DEATH_KNIGHT}torch.Tensor({SCOURGE_GREEN}{shape_str}, {dtype_str}, {device_str}", end="")
                if grad_str:
                    print(f", {grad_str}{DEATH_KNIGHT})", end="")
                else:
                    print(f"{DEATH_KNIGHT})", end="")
                
                # For 2D tensors, pretty print the values
                if len(obj.shape) <= 2:
                    # Convert to numpy for pretty printing
                    array_obj = obj.detach().cpu().numpy()
                    array_str = np.array2string(array_obj, prefix=indent_str + '  ')
                    print(f"{SARONITE}{{{PALE_BLUE}")
                    for line in array_str.split('\n'):
                        print(f"{indent_str}  {PALE_BLUE}{line}")
                    print(f"{indent_str}{SARONITE}}}")
                else:
                    print()  # Just a newline for 3D+ tensors
        
        # Handle list or tuple
        elif isinstance(obj, (list, tuple)):
            container_type = "list" if isinstance(obj, list) else "tuple"
            bracket_open = "[" if isinstance(obj, list) else "("
            bracket_close = "]" if isinstance(obj, list) else ")"
            
            if inline:
                if len(obj) <= 5 and all(not isinstance(x, (dict, list, tuple, np.ndarray)) and not dataclasses.is_dataclass(x) for x in obj):
                    items = []
                    for item in obj:
                        if isinstance(item, str):
                            items.append(f"{FROST}'{item}'")
                        else:
                            items.append(f"{PALE_BLUE}{item}")
                    items_str = f"{PALE_BLUE}, ".join(items)
                    print(f"{ICY_BLUE}{container_type}{SARONITE}{bracket_open}{items_str}{SARONITE}{bracket_close}")
                else:
                    print(f"{ICY_BLUE}{container_type}{PALE_BLUE} (length: {len(obj)})")
            else:
                print(f"{indent_str}{ICY_BLUE}{container_type}{PALE_BLUE} (length: {len(obj)}){SARONITE}{bracket_open}")
                for i, item in enumerate(obj):
                    _print_recursive(item, indent + 1, is_list_item=True, list_index=i)
                print(f"{indent_str}{SARONITE}{bracket_close}")
        
        # Handle string
        elif isinstance(obj, str):
            if inline:
                print(f"{FROST}'{obj}'")
            else:
                if is_list_item:
                    print(f"{indent_str}{ICY_BLUE}{list_index}: {FROST}'{obj}'")
                else:
                    print(f"{indent_str}{FROST}'{obj}'")
        
        # Handle other basic types (int, float, bool, None)
        else:
            if inline:
                if obj is None:
                    print(f"{LICH_PURPLE}None")
                elif isinstance(obj, bool):
                    print(f"{ICY_BLUE}{obj}")
                elif isinstance(obj, int):
                    print(f"{PALE_BLUE}{obj}")
                elif isinstance(obj, float):
                    print(f"{PALE_BLUE}{obj}")
                else:
                    print(f"{PALE_BLUE}{obj} {SCOURGE_GREEN}({type(obj).__name__})")
            else:
                if is_list_item:
                    prefix = f"{indent_str}{ICY_BLUE}{list_index}: "
                else:
                    prefix = indent_str
                
                if obj is None:
                    print(f"{prefix}{LICH_PURPLE}None")
                elif isinstance(obj, bool):
                    print(f"{prefix}{ICY_BLUE}{obj}")
                elif isinstance(obj, int):
                    print(f"{prefix}{PALE_BLUE}{obj}")
                elif isinstance(obj, float):
                    print(f"{prefix}{PALE_BLUE}{obj}")
                else:
                    print(f"{prefix}{PALE_BLUE}{obj} {SCOURGE_GREEN}({type(obj).__name__})")
    
    # Print variable name if available
    if var_name:
        print(f"{BOLD}{FROST}{var_name} {SARONITE}= {RESET}", end="")
    
    # Start the recursive printing
    try:
        _print_recursive(data)
        print(RESET, end="")  # Reset colors at the end
    except Exception as e:
        # Show error traceback similar to the example
        import traceback
        error_msg = traceback.format_exc()
        print(f"{LICH_PURPLE}{error_msg}{RESET}")
    
    # Return the string if requested
    if return_str:
        result = buffer.getvalue()
        sys.stdout = old_stdout
        return result

# Example usage
if __name__ == "__main__":
    import sys
    from dataclasses import dataclass, field
    from typing import List, Tuple
    
    # Define dataclasses for testing
    @dataclass
    class Item:
        color: str
        shape: str
        size: Tuple[float, float, float]
    
    @dataclass
    class Scene:
        items: List[Item]
    
    # Create a scene with some items
    my_scene = Scene(items=[
        Item(color="blue", shape="cube", size=(1.0, 1.0, 1.0)),
        Item(color="red", shape="sphere", size=(0.5, 0.5, 0.5)),
        Item(color="green", shape="pyramid", size=(2.0, 2.0, 3.0))
    ])
    
    # Print the scene
    print("\nExample with a dataclass:")
    stprint(my_scene)
    
    # Create a complex nested structure
    complex_data = {
        "scene": my_scene,
        "metadata": {
            "creator": "Arthas",
            "version": 2.5,
            "tags": ["icecrown", "citadel", "frozen"]
        },
        "statistics": {
            "item_count": 3,
            "colors": ["blue", "red", "green"],
            "largest_item": Item(color="green", shape="pyramid", size=(2.0, 2.0, 3.0))
        }
    }
    
    # Print the complex structure
    print("\nExample with nested dataclasses:")
    stprint(complex_data, "complex_data")


def indent_text(text: str, level = 1, indent_format: Optional[str] = None, tabsize: Optional[int] = None, indent_first: bool = True) -> str:
    """Indent the text by the given level.

    Args:
        text: the text to be indented.
        level: the indent level.
        indent_format: the indent format. If None, use the tabsize.
        tabsize: the tab size. If None, use the default tab size (2).
        indent_first: whether to indent the first line.

    Returns:
        The indented text.
    """
    text = str(text)
    if indent_format is not None:
        assert tabsize is None, 'Cannot provide both indent format and tabsize.'
    if tabsize is not None:
        assert indent_format is None, 'Cannot provide both indent format and tabsize.'
        indent_format = ' ' * tabsize
    if indent_format is None and tabsize is None:
        indent_format = '  '
    indent_format = indent_format * level
    if indent_first:
        return indent_format + text.replace('\n', '\n' + indent_format)
    return text.replace('\n', '\n' + indent_format)

