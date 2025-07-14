import torch
import torch.nn as nn
from abc import abstractmethod
from helchriss.dsl.dsl_types import TypeBase
from helchriss.dsl.dsl_values import Value
from typing import List, Tuple,Union, Any, Dict, Tuple, Optional, Callable, Type
from dataclasses import dataclass, field
from helchriss.dsl.dsl_types import VectorType, ListType, EmbeddingType, TupleType, FixedListType, ArrowType, BatchedListType
import copy

__all__ = ["PatternVar","match_pattern", "CasterRegistry", "type_dim", "fill_hole", "infer_caster"]


"""Match the Pattern for Tree Regular Language"""
class PatternVar(TypeBase):
    """variable in the pattern"""
    def __init__(self, var_name: str):
        super().__init__(f"${var_name}")  # $ as the mark for the variable
        self.var_name = var_name

    def __eq__(self, other: TypeBase) -> bool:  return True 

def match_pattern(
    target_type: TypeBase,
    pattern: TypeBase,
    bindings: Optional[Dict[str, TypeBase]] = None
) -> Optional[Dict[str, TypeBase]]:
    """ if the pattern matches the
    
    Args:
        target_type: the type to match
        pattern: pattern to match
        bindings: the known pattern to match
    
    Returns:
        variable binding that maps %var to the actual type or value
    """
    bindings = bindings or {}

    # Case 1：pattern is single variable -> binding
    if isinstance(pattern, PatternVar):
        var_name = pattern.var_name
        if var_name in bindings: ### check if the variable binding is consistent
            if bindings[var_name] != target_type: return None
        else: bindings[var_name] = target_type
        return bindings

    # Case 2：target type inconsistent with the  -> failed to match
    if type(target_type) != type(pattern): return None

    try: # compare the type name
        if not (hasattr(target_type, 'element_type') or hasattr(target_type, 'element_types')):
            return bindings if target_type == pattern else None
    except: return None

    # nested type -> recursive match the subtype
    # 4.1 ListType（uniform sequence, single element type）
    if isinstance(target_type, (ListType, FixedListType, VectorType, BatchedListType)):
        # check the subtyping
        sub_bindings = match_pattern(
            target_type.element_type,
            pattern.element_type,
            copy.deepcopy(bindings)
        )
        if sub_bindings is None: return None
        # 额外检查ListType特有属性（如VectorType的dim）
        if isinstance(target_type, VectorType):
            if target_type.dim != pattern.dim and not isinstance(pattern.dim, PatternVar):
                return None
        if isinstance(target_type, FixedListType):
            if target_type.typename != pattern.typename:  # 包含length信息
                return None
        return sub_bindings

    # 4.2 TupleType with multiple elemnt types
    if isinstance(target_type, TupleType):
        if len(target_type.element_types) != len(pattern.element_types):
            return None
        # recusrive match each element type
        new_bindings = copy.deepcopy(bindings)
        for t_elem, p_elem in zip(target_type.element_types, pattern.element_types):
            elem_bindings = match_pattern(t_elem, p_elem, new_bindings)
            if elem_bindings is None:
                return None
            new_bindings.update(elem_bindings)
        return new_bindings

    # 4.3 EmbeddingType : space_name and dim
    if isinstance(target_type, EmbeddingType):
        if (target_type.space_name != pattern.space_name or 
            target_type.dim != pattern.dim):
            return None
        return bindings

    # 4.4 ArrowType : match the firs second
    if isinstance(target_type, ArrowType):
        first_bindings = match_pattern(target_type.first, pattern.first, copy.deepcopy(bindings))
        if first_bindings is None:
            return None
        second_bindings = match_pattern(target_type.second, pattern.second, first_bindings)
        return second_bindings

    return bindings if target_type == pattern else None

# 3. rule of transformation class
class TransformRule:
    def __init__(self, source_pattern: TypeBase, transform_func: Callable[[Dict[str, TypeBase]], TypeBase], target_pattern = None):
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.transform_func = transform_func

    def apply(self, source_type, target_type):
        source_vars = match_pattern(source_type, self.source_pattern)
        if source_vars is None : return None
        if self.target_pattern is not None and not match_pattern(target_type, self.target_pattern): return None
        return self.transform_func(source_vars)

class CasterRegistry:
    def __init__(self, rules : List[TransformRule]):
        self.rules = rules

def find_transform_path(
    initial_type: TypeBase,
    target_type: TypeBase,
    rules: List[TransformRule],
    max_depth: int = 5
) -> Optional[List[Tuple[TransformRule, Dict[str, TypeBase]]]]:
    """
    查找从初始类型到目标类型的转换路径
    
    Args:
        initial_type: 初始类型
        target_type: 目标类型
        rules: 转换规则列表
        max_depth: 最大搜索深度（防止循环）
    
    Returns:
        转换路径（规则+绑定），无路径则返回None
    """
    # BFS队列：(当前类型, 已应用的规则路径, 深度)
    queue = [(initial_type, [], 0)]
    # 已访问的类型（避免重复搜索）
    visited = set()

    while queue:
        current_type, path, depth = queue.pop(0)

        # 检查是否达到目标类型
        if current_type == target_type:
            return path

        # 超过最大深度则停止
        if depth >= max_depth:
            continue

        # 记录已访问的类型（用typename作为标识）
        type_key = current_type.typename
        if type_key in visited:
            continue
        visited.add(type_key)

        # 尝试应用每条规则
        for rule in rules:
            # 匹配当前类型与规则模式
            bindings = match_pattern(current_type, rule.pattern)
            if bindings is None:
                continue  # 不匹配则跳过

            # 应用转换函数生成新类型
            try:
                new_type = rule.transform_func(bindings)
            except Exception:
                continue  # 转换函数执行失败则跳过

            # 新类型加入队列
            new_path = path + [(rule, bindings)]
            queue.append((new_type, new_path, depth + 1))

    return None  # 未找到路径

@dataclass
class CastingRule:
    name: str
    priority: int = 0 
    
    def can_apply(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> bool:
        raise NotImplementedError
    
    def create_caster(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> nn.Module:
        raise NotImplementedError


class MLPFiller(nn.Module):
    def __init__(self, input_types : List[TypeBase], out_type : Union[TypeBase, List[TypeBase]], net : nn.Module):
        super().__init__()
        self.input_types = input_types
        self.out_types = out_type
        self.net = net
    
    @property
    def singular(self): return len(self.out_types) == 1

    def forward(self, *args):
        neural_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor): neural_args.append(arg)
            else: neural_args.append(torch.tensor(arg))
        cat_args = torch.cat([arg.reshape([1,-1]) for arg in neural_args], dim = -1)
        output = self.net(cat_args).reshape([-1])
        return Value(self.out_types, output)


class DimensionMappingRule(CastingRule):
    def __init__(self):
        super().__init__("DimensionMapping")
    
    def can_apply(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> bool:
        return all(self._is_basic_type(t) for t in input_types + output_types)
    
    def _is_basic_type(self, tp: TypeBase) -> bool:
        return (isinstance(tp, TypeBase) and tp.typename in ["int", "float", "bool"]) or \
               isinstance(tp, VectorType)
    
    def create_caster(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> nn.Module:
        input_dims = [type_dim(tp) for tp in input_types]
        output_dims = [type_dim(tp) for tp in output_types]
        return MLPCaster(input_dims, output_dims)

# 新增：List类型映射规则
class ListMappingRule(CastingRule):
    def __init__(self, element_rule: CastingRule):
        super().__init__("ListMapping")
        self.element_rule = element_rule
    
    def can_apply(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> bool:
        if len(input_types) != 1 or len(output_types) != 1:
            return False
        
        in_type = input_types[0]
        out_type = output_types[0]
        
        # 检查是否均为List类型
        if not (isinstance(in_type, ListType) and isinstance(out_type, ListType)):
            return False
        
        # 检查元素类型是否可由子规则处理
        return self.element_rule.can_apply([in_type.element_type], [out_type.element_type])
    
    def create_caster(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> nn.Module:
        in_list = input_types[0]
        out_list = output_types[0]
        
        # 创建元素类型的caster
        element_caster = self.element_rule.create_caster([in_list.element_type], [out_list.element_type])
        
        return ListCaster(element_caster, in_list.element_type, out_list.element_type)

# 新增：Vector与List互转规则
class VectorListConversionRule(CastingRule):
    def __init__(self):
        super().__init__("VectorListConversion", priority=1)
    
    def can_apply(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> bool:
        if len(input_types) != 1 or len(output_types) != 1:
            return False
        
        in_type = input_types[0]
        out_type = output_types[0]
        
        # 检查是否为Vector <-> List转换
        return (isinstance(in_type, VectorType) and isinstance(out_type, ListType) and 
                self._check_element_type(in_type.elem_type, out_type.element_type)) or \
               (isinstance(in_type, ListType) and isinstance(out_type, VectorType) and 
                self._check_element_type(in_type.element_type, out_type.elem_type))
    
    def _check_element_type(self, t1: TypeBase, t2: TypeBase) -> bool:
        # 检查元素类型是否兼容
        return t1.typename == t2.typename
    
    def create_caster(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> nn.Module:
        in_type = input_types[0]
        out_type = output_types[0]
        
        if isinstance(in_type, VectorType) and isinstance(out_type, ListType):
            return VectorToListCaster(in_type, out_type)
        else:
            return ListToVectorCaster(in_type, out_type)

class RuleBasedCasterInferer:
    def __init__(self):
        self.rules = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        self.register_rule(VectorListConversionRule())
        
        element_rule = DimensionMappingRule()
        list_rule = ListMappingRule(element_rule)
        self.register_rule(list_rule)
        
        self.register_rule(DimensionMappingRule())
    
    def register_rule(self, rule: CastingRule):
        """注册新的转换规则"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def infer_caster(self, input_types: List[TypeBase], output_types: List[TypeBase]) -> nn.Module:
        #[print(a) for a in input_types]
        #[print(a) for a in output_types]
        return fallback_infer_caster(input_types, output_types)
        for rule in self.rules:
            if rule.can_apply(input_types, output_types):
                return rule.create_caster(input_types, output_types)
        
        return fallback_infer_caster(input_types, output_types)

def type_dim(tp : TypeBase) -> int:
    if isinstance(tp, TypeBase) and tp.typename in ["int", "float", "boolean", "bool"]:
        return 1
    elif isinstance(tp, VectorType):
        return int(tp.dim)
    elif isinstance(tp, EmbeddingType):
        return int(tp.dim)
    elif isinstance(tp, ListType):
        # 假设List长度为10（可改进为动态长度）
        return 10 * type_dim(tp.element_type)
    elif isinstance(tp, TupleType):
        return sum(type_dim(elem) for elem in tp.element_types)
    elif isinstance(tp, FixedListType):
        length = tp.length if isinstance(tp.length, int) else 10  # 默认长度
        return length * type_dim(tp.element_type)
    raise NotImplementedError(f"dim of type {tp} cannot be inferred")

# 新增：改进的infer_caster函数
def infer_caster(input_type : List[TypeBase], output_types : List[TypeBase]) -> nn.Module:
    """基于规则系统的智能caster推理"""
    inferer = RuleBasedCasterInferer()
    return inferer.infer_caster(input_type, output_types)

# 新增：改进的fill_hole函数
def fill_hole(arg_types : List[TypeBase], out_type : TypeBase) -> nn.Module:
    """基于类型结构的智能hole填充器"""
    # 分析输入和输出类型的复杂度
    complexity = analyze_type_complexity(arg_types + [out_type])
    
    # 根据复杂度调整网络规模
    hidden_size = min(64 * (complexity // 2 + 1), 512)
    num_layers = min(complexity // 3 + 2, 5)
    
    # 构建网络
    in_dim = sum([type_dim(tp) for tp in arg_types])
    out_dim = type_dim(out_type)
    
    layers = []
    layers.append(nn.Linear(in_dim, hidden_size))
    layers.append(nn.ReLU())
    
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_size, out_dim))
    
    net = nn.Sequential(*layers)
    filler = MLPFiller(arg_types, out_type, net)
    return filler

def analyze_type_complexity(types: List[TypeBase]) -> int:
    complexity = 0
    
    for tp in types:
        if isinstance(tp, ListType) or isinstance(tp, TupleType):
            complexity += 2
            complexity += analyze_type_complexity([tp.element_type]) if hasattr(tp, 'element_type') else 0
            complexity += analyze_type_complexity(tp.element_types) if hasattr(tp, 'element_types') else 0
        elif isinstance(tp, VectorType) or isinstance(tp, EmbeddingType):
            complexity += 1
        else:
            complexity += 0  # 基础类型复杂度为0
    
    return complexity

class MLPCaster(nn.Module):
    def __init__(self, input_dims : List[int], output_dims : List[int]):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.total_input_dim = sum(input_dims)
        self.total_output_dim = sum(output_dims)
        
        self.net = nn.Sequential(
            nn.Linear(self.total_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.total_output_dim)
        )
        self.logit_net = nn.Sequential(
            nn.Linear(self.total_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.output_dims))
        )
    
    def forward(self, *args):
        # 将输入张量展平并连接
        flat_args = [arg.value.reshape(-1) for arg in args[0]]
        cat_args = torch.cat(flat_args, dim=0)


        output = self.net(cat_args)
        outputs = [t.reshape([d]) for t, d in zip(torch.split(output, self.output_dims), self.output_dims)]

        logit_output = self.logit_net(cat_args)

        pupil_output = [(o, logit_output[i]) for i,o in enumerate(outputs)]
        return pupil_output

class ListCaster(nn.Module):
    def __init__(self, element_caster: nn.Module, in_elem_type: TypeBase, out_elem_type: TypeBase):
        super().__init__()
        self.element_caster = element_caster
        self.in_elem_type = in_elem_type
        self.out_elem_type = out_elem_type
    
    def forward(self, input_list):
        # 假设输入是张量列表
        output_list = []
        for element in input_list:
            # 对每个元素应用元素caster
            element_output = self.element_caster(element)
            output_list.append(element_output[0])  # 假设返回列表，取第一个元素
        
        return [torch.stack(output_list, dim=0)]

class VectorToListCaster(nn.Module):
    def __init__(self, vector_type: VectorType, list_type: ListType):
        super().__init__()
        self.vector_dim = vector_type.dim
        self.list_length = 10  # 假设列表长度
    
    def forward(self, vector):
        # 将向量分割为列表
        elements = torch.chunk(vector, self.list_length, dim=1)
        return [torch.stack(elements, dim=0)]

class ListToVectorCaster(nn.Module):
    def __init__(self, list_type: ListType, vector_type: VectorType):
        super().__init__()
        self.list_length = 10  # 假设列表长度
        self.vector_dim = vector_type.dim
    
    def forward(self, list_tensor):
        # 将列表连接为向量
        flat_list = list_tensor.reshape(1, -1)
        # 截断或填充到目标维度
        if flat_list.shape[1] > self.vector_dim:
            return [flat_list[:, :self.vector_dim]]
        else:
            padded = torch.zeros(1, self.vector_dim, device=flat_list.device)
            padded[:, :flat_list.shape[1]] = flat_list
            return [padded]

def fallback_infer_caster(input_type : List[TypeBase], output_types : List[TypeBase]) -> nn.Module:
    input_dims = [type_dim(tp) for tp in input_type]
    output_dims = [type_dim(tp) for tp in output_types]
    return MLPCaster(input_dims, output_dims)
