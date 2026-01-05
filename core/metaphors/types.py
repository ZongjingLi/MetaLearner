import copy
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass, field
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import FLOAT, ListType, VectorType, EmbeddingType, TypeBase
from helchriss.dsl.dsl_types import VectorType, ListType, EmbeddingType, TupleType, FixedListType, ArrowType, BatchedListType, BOOL
from typing import List, Tuple,Union, Any, Dict, Tuple, Optional, Callable, Type

#from .rules import default_constructor_rules, TypeTransformRule, TypeTransformRuleBackward, PatternVar, match_pattern

try:
    from .rules import default_constructor_rules, TypeTransformRule, TypeTransformRuleBackward, PatternVar, match_pattern
except:
    from rules import default_constructor_rules, TypeTransformRule, TypeTransformRuleBackward, PatternVar, match_pattern

import graphviz

__all__ = ["PatternVar","match_pattern", "Constructor", "default_constructor_rules"]



import torch
import torch.nn as nn
from typing import List, Union, Callable, Any, Optional

class ConvexConstruct(nn.Module):
    """
    A learnable convex combination module for functions or nested ConvexConstruct instances.
    This module maintains non-negative weights that sum to 1 (convex combination constraints)
    and supports end-to-end gradient-based optimization.

    Attributes:
        functions (List[Union[Callable, "ConvexConstruct"]]): List of base functions or nested ConvexConstruct modules
        num_functions (int): Number of functions in the convex combination
        logits (nn.Parameter): Learnable logits (raw unnormalized scores, converted to valid weights via softmax)
    """

    def __repr__(self)-> str:
        return f"convex_construct {self.num_functions}"

    def __init__(self, functions: List[Union[Callable, "ConvexConstruct"]], weights: torch.Tensor = None, input_types = None, output_type = None):
        super(ConvexConstruct, self).__init__()
        
        self.functions = nn.ModuleList(functions)  # Use regular list instead of ModuleList to support non-nn.Module
        self.num_functions = len(self.functions)
        assert self.num_functions > 0, "Function list cannot be empty"
        self.input_types = input_types
        self.output_type = output_type

        if weights is None:
            init_logits = torch.zeros(self.num_functions, dtype=torch.float32)
        else:
            assert weights.ndim == 1, f"Weights must be 1D tensor (got {weights.ndim}D)"
            assert len(weights) == self.num_functions, \
                f"Weight count ({len(weights)}) must match function count ({self.num_functions})"
            assert torch.all(weights >= 0), "All weights must be non-negative"

            normalized_weights = weights / torch.sum(weights)

            init_logits = torch.logit(normalized_weights + 1e-8)

        self.logits = nn.Parameter(init_logits)


    def get_normalized_weights(self) -> torch.Tensor:
        """
        Convert logits to valid convex combination weights (non-negative, sum to 1)
        using softmax activation 

        Returns:
            torch.Tensor: Normalized weights of shape (num_functions,)
        """
        return torch.softmax(self.logits, dim=0)

    def forward(self, x) -> Any:
        """
        Forward pass: compute convex combination of function outputs.

        Args:
            x: Input to pass to each function

        Returns:
            Any: Weighted sum of function outputs (matches output type of base functions)
        """
        # Get normalized convex weights
        weights = self.get_normalized_weights()

        outputs = []
        for func in self.functions:
            if isinstance(x, Value) and isinstance(func, ConvexConstruct):
                x = x.value

            val = func(x)
            if isinstance(val, Value): val = val.value
            outputs.append(val)

        assert len(outputs) > 0, "No function outputs to combine"
        first_output = outputs[0]
        for output in outputs[1:]:
            assert type(output) == type(first_output), \
                f"All function outputs must have the same type (got {type(output)} and {type(first_output)})"

        if isinstance(first_output, torch.Tensor):
            weighted_sum = torch.zeros_like(first_output)
        else:
            # Handle non-tensor outputs (e.g., numpy arrays, scalars)
            weighted_sum = type(first_output)(0)

        for weight, output in zip(weights, outputs):
            weighted_sum += weight * output
        return Value(self.output_type,  weighted_sum)

    def normalize_weights(self) -> None:
        """
        Explicitly re-normalize weights (in-place) to ensure they satisfy convex constraints.
        Useful for post-optimization calibration (though softmax maintains this during training).
        """
        with torch.no_grad():
            normalized = self.get_normalized_weights()
            self.logits.data = torch.log(normalized + 1e-8)

    def get_top_p_functions(self, p: float) -> "ConvexConstruct":
        """
        Prune the module to retain only the top P proportion of functions with highest weights,
        returning a new ConvexConstruct instance with these top functions.

        Args:
            p (float): Proportion of top functions to retain (must be in (0, 1])

        Returns:
            ConvexConstruct: New instance with top P functions and their normalized weights
        """
        assert 0 < p <= 1, f"p must be in (0, 1] (got {p})"
        
        weights = self.get_normalized_weights()
        
        top_k = int(self.num_functions * p)
        top_k = max(top_k, 1)  # Ensure at least 1 function is retained

        top_indices = torch.argsort(weights, descending=True)[:top_k]

        top_functions = [self.functions[i] for i in top_indices]
        top_weights = weights[top_indices]

        return ConvexConstruct(top_functions, top_weights)

    def _compute_function_length(self, func: Union[Callable, "ConvexConstruct", "ConvexComposer", "BackwardCombinedFn"]) -> int:
        """
        Recursively compute the computational length of a given function/module.
        Handles nested structures (ConvexConstruct, ConvexComposer, BackwardCombinedFn) recursively.
        
        Args:
            func: The function/module to compute the length for
            
        Returns:
            int: Total computational length of the input function/module
        """
        # Base case: basic callable (non-custom module) has length 1
        if not isinstance(func, (ConvexConstruct, ConvexComposer, BackwardCombinedFn)):
            return 1
        
        # Case 1: ConvexConstruct (sum of lengths of its child functions)
        if isinstance(func, ConvexConstruct):
            total_length = 0
            for child_func in func.functions:
                total_length += self._compute_function_length(child_func)
            return total_length
        
        # Case 2: ConvexComposer (length of fn + length of convex module)
        if isinstance(func, ConvexComposer):
            fn_length = self._compute_function_length(func.fn)
            convex_length = self._compute_function_length(func.convex)
            return fn_length + convex_length
        
        # Case 3: BackwardCombinedFn (sum of lengths of its sub convex modules)
        if isinstance(func, BackwardCombinedFn):
            total_length = 0
            for sub_convex in func.sub_convex_list:
                total_length += self._compute_function_length(sub_convex)
            # Add length for the combine_fn itself (base case 1)
            total_length += self._compute_function_length(func.combine_fn)
            return total_length

    def clear(self, p: Union[int, float]) -> None:
        """
        In-place clear function that removes all child functions whose computational length exceeds p.
        Updates the module's functions, num_functions, and logits (weights) to retain only valid functions.
        
        Args:
            p: Maximum allowed computational length for functions (non-negative)
        """
        # Validate input parameter p
        assert isinstance(p, (int, float)) and p >= 0, f"p must be a non-negative number (got {p})"
        
        # Step 1: Compute length for each function and collect valid (length <= p) functions
        valid_indices = []
        valid_functions = []
        for idx, func in enumerate(self.functions):
            func_length = self._compute_function_length(func)
            if func_length <= p:
                valid_indices.append(idx)
                valid_functions.append(func)
        
        # Step 2: Ensure at least one valid function is retained (avoid empty module)
        if len(valid_functions) == 0:
            # Fallback: retain the function with the smallest length
            func_lengths = [self._compute_function_length(func) for func in self.functions]
            min_length_idx = func_lengths.index(min(func_lengths))
            valid_indices = [min_length_idx]
            valid_functions = [self.functions[min_length_idx]]
        
        # Step 3: Update module attributes in-place (with no gradient tracking)
        self.functions = nn.ModuleList(valid_functions)
        self.num_functions = len(self.functions)
        
        # Step 4: Update logits to match valid functions (retain corresponding weights and re-normalize)
        with torch.no_grad():
            current_weights = self.get_normalized_weights()
            valid_weights = current_weights[valid_indices]
            # Re-initialize logits from valid weights (maintain convex constraints)
            normalized_valid_weights = valid_weights / torch.sum(valid_weights)
            self.logits = nn.Parameter(torch.logit(normalized_valid_weights + 1e-8))

class ConvexComposer(nn.Module):
    def __init__(self, fn: Callable, convex: ConvexConstruct):
        super().__init__()
        self.fn = fn
        self.convex = convex

    def forward(self, *args, **kwargs):
        res = self.fn(*args, **kwargs)
        if isinstance(res, Value):res = res.value

        return self.convex(res)

class BackwardCombinedFn(nn.Module):
    def __init__(self, sub_convex_list: List[ConvexConstruct], combine_fn: Callable):
        super().__init__()
        self.sub_convex_list = nn.ModuleList(sub_convex_list)
        self.combine_fn = combine_fn

    def forward(self, *args, **kwargs) -> Any:
        sub_results = []
        for sub_convex in self.sub_convex_list:
            res = sub_convex(*args, **kwargs) 
            if isinstance(res, Value): res = res.value
            sub_results.append(res)

        return self.combine_fn(*sub_results)

class ConstModule(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def foward(self, v): return Value(torch.tensor(self,v), FLOAT)

from helchriss.utils.tensor import Id

class ConvexRewriter(nn.Module):
    def __init__(self, rewrites : List[ConvexConstruct], conds : List[ConvexConstruct]):
        super().__init__()
        self.rewrites = nn.ModuleList(rewrites) # take arg to another arg as rewrite
        self.conditions = nn.ModuleList(conds) # take arg to float as logit of rewrite
 
    def forward(self, x : List[Value]):
        rewrite_values = []
        rewrite_probs = []
        for i,arg in enumerate(x):
            if isinstance(arg, Value): arg = arg.value

            rw_value    = self.rewrites[i](arg)
            rw_weight   = self.conditions[i](arg)

            
            rewrite_values.append(rw_value)
            rewrite_probs.append(rw_weight)
    
        return rewrite_values, rewrite_probs


class Constructor:

    def __init__(self, rules = []):
        self.max_depth = 2 # maximum recursion depth

        self.backward_rules = []
        self.forward_rules = []

        for rule in rules:
            self.add_rule(rule)


    def add_rule(self, rule: TypeTransformRule):
        if isinstance(rule, TypeTransformRule):
            self.forward_rules.append(rule)
        elif isinstance(rule, TypeTransformRuleBackward):

            self.backward_rules.append(rule)
        else: raise TypeError(f"Unsupported rule type: {type(rule)}. Only TypeTransformRule and TypeTransformRuleBackward are allowed.")

    def create_convex_construct(self, src_type : TypeBase,  tgt_type : TypeBase, function_registry):

        def bfs(src_type : TypeBase, tgt_type : TypeBase, depth = 0):
            if depth > self.max_depth : return None
            
            direct_functions = function_registry.get_functions(src_type, tgt_type)

            if not direct_functions and depth == self.max_depth: return None

            functions = [] # get non fill in function
            """get all the functions that can be directly computed"""
            functions.extend(function_registry.get_functions(src_type, tgt_type))

            intermediate_functions = []
            for rule in self.forward_rules:
                match_success, var_binds = rule.match(src_type)

                if not match_success: continue
                inter_type, inter_fn = rule.apply(var_binds)
                
                
                if inter_type == tgt_type:
                    intermediate_functions.append(inter_fn)
                    continue
                else:
                    
                    inter_convex = bfs(inter_type, tgt_type, depth = depth+1)


                    if inter_convex is not None:

                        intermediate_functions.append(ConvexComposer(inter_fn, inter_convex))
            
            functions = functions + intermediate_functions

            """Intermediate Backward Function Subgoal Search"""
            intermediate_backward_functions = []
            # traverse all the subgoals
            for backward_rule in self.backward_rules:
                # 3.1 the match target type cooresponds to the pattern
                match_success, var_binds = backward_rule.match(tgt_type)
                #print(match_success,backward_rule.name, tgt_type)
                
                if not match_success: continue
                # 3.2 create subgoals to learn with
                
                sub_goal_types = backward_rule.sub_goals(tgt_type)
                #print(sub_goal_types[0],"->",tgt_type)


                # 3.3 recursively solve each subgoal
                sub_goal_convex_list = []
                valid_sub_goals = True
                for sub_goal_type in sub_goal_types:
                    
                    sub_convex = bfs(src_type, sub_goal_type, depth=depth+1)
                    if sub_convex:
                        pass
                        #print(src_type, sub_goal_type, tgt_type)
                    #print(src_type, sub_goal_type, sub_convex)
                    if sub_convex is None:
                        valid_sub_goals = False
                        break

                    sub_goal_convex_list.append(sub_convex)
            
                if not valid_sub_goals: continue 


                backward_combine_fn = backward_rule.apply(var_binds)


                combined_fn = BackwardCombinedFn(sub_goal_convex_list, backward_combine_fn)
                intermediate_backward_functions.append(combined_fn)


            functions.extend(intermediate_backward_functions)
            weights = None # default as uniform distribution

            if not functions: return None

            return ConvexConstruct(functions, weights, src_type, tgt_type)
    
        return bfs(src_type, tgt_type, depth=0)
    
    def create_convex_arg_rewriter(self, src_types : List[TypeBase], tgt_types : List[TypeBase], function_registry):
        assert len(src_types) == len(tgt_types), f"{len(src_types)} != {len(tgt_types)}"
        rewrites = []
        conds    = []

        for i in range(len(src_types)):

            if 1:
                #src_type = EmbeddingType("color_wheel", 1)
                #tgt_type = EmbeddingType("object", 96)

                rewrite = self.create_convex_construct(src_types[i], tgt_types[i], function_registry)
                cond    = self.create_convex_construct(src_types[i], BOOL, function_registry)
            #if src_types[i] == src_type and tgt_types[i] == tgt_type:
            #    print("GO",src_types[i], tgt_types[i], rewrite)
            assert rewrite, f"rewrite is None {src_types[i]}, {tgt_types[i]}"
            assert cond,    f"cond is None {src_types[i]}, {tgt_types[i]}"
            rewrites.append(rewrite)
            conds.append(cond)

        return ConvexRewriter(rewrites, conds)


class FunctionRegistry:

    def __init__(self):
        self.functions: Dict[Tuple[TypeBase, TypeBase], List[Callable]] = {}

    def register_function(self, input_type: TypeBase, output_type: TypeBase, func: Callable):

        key = (input_type, output_type)
        if key not in self.functions:
            self.functions[key] = []
        self.functions[key].append(func)

    def get_functions(self,  input_type: TypeBase, output_type: TypeBase) -> List[Callable]:

        key = (input_type, output_type)
        return self.functions.get(key, [])


import networkx as nx
import graphviz
from typing import Union, Optional
import warnings
import matplotlib.pyplot as plt

def visualize_convex_construct(
    convex: "ConvexConstruct",
    graph_name: str = "ConvexConstruct_Dependency"
) -> graphviz.Digraph:
    """
    可视化 ConvexConstruct 的函数依赖关系，自动展开嵌套的 ConvexConstruct 和 ConvexComposer 节点。
    
    Args:
        convex: 待可视化的 ConvexConstruct 实例 
        graph_name: 可视化图的名称
    
    Returns:
        graphviz.Digraph: 生成的可视化图对象（可调用 .render() 保存为图片，.view() 直接查看）
    """
    # 初始化有向图，设置样式
    dot = graphviz.Digraph(
        name=graph_name,
        format="png",
        node_attr={"shape": "box", "style": "filled", "fillcolor": "lightblue"},
        edge_attr={"color": "gray"}
    )
    
    # 记录已处理的节点（避免循环引用重复绘制，实际中 ConvexConstruct 一般无循环）
    processed_nodes = set()

    def _recursive_add_nodes(
        current_module: Union["ConvexConstruct", "ConvexComposer"],
        parent_node_id: str,
        parent_weights: float = None
    ):
        # 生成当前模块的唯一标识
        module_id = id(current_module)
        if module_id in processed_nodes:
            # 已处理过的节点，仅添加边不重复展开
            if parent_node_id:
                dot.edge(parent_node_id, f"{type(current_module).__name__}_{module_id}")
            return
        processed_nodes.add(module_id)

        # ------------------------------
        # 1. 处理 ConvexConstruct 模块
        # ------------------------------
        if isinstance(current_module, ConvexConstruct):
            # 添加 ConvexConstruct 节点（标注权重、输入输出类型信息）
            weights = current_module.get_normalized_weights().detach().cpu().numpy()
            
            # 拼接输入输出类型信息
            input_types_repr =  f"\n{current_module.input_types}"
            output_type_repr = str(current_module.output_type)
            
            # 构建节点标签
            convex_label = (
                f"ConvexConstruct\n"
                f"Input Types:\n{input_types_repr}\n"
                f"Output Type:\n{output_type_repr}\n"
                f"Weight Sum=1.0"
            )
            if parent_weights is not None:
                convex_label = (
                    f"ConvexConstruct\n"
                    f"Parent Weight={parent_weights:.3f}\n"
                    f"Input Types:\n{input_types_repr}\n"
                    f"Output Type:\n{output_type_repr}\n"
                    f"Internal Sum=1.0"
                )
            
            dot.node(
                f"ConvexConstruct_{module_id}",
                label=convex_label,
                fillcolor="lightcoral"
            )

            # 连接父节点与当前 ConvexConstruct 节点
            if parent_node_id:
                dot.edge(parent_node_id, f"ConvexConstruct_{module_id}")

            # 遍历所有函数，递归添加子节点
            for idx, func in enumerate(current_module.functions):
                func_id = id(func)
                func_weight = weights[idx]
                
                # 优先使用 __repr__，其次 __str__，最后默认标识
                if hasattr(func, '__repr__'):
                    func_repr = func.__repr__()
                    # 简化过长的repr输出，避免节点标签溢出
                    func_display_name = func_repr[:50] + "..." if len(func_repr) > 50 else func_repr
                elif hasattr(func, '__str__'):
                    func_str = func.__str__()
                    func_display_name = func_str[:50] + "..." if len(func_str) > 50 else func_str
                else:
                    func_display_name = f"Func_{idx}"

                # 函数节点标签（包含显示名称和权重）
                func_label = f"{func_display_name}\nWeight={func_weight:.3f}"

                # 子函数是 ConvexConstruct：递归展开
                if isinstance(func, ConvexConstruct):
                    dot.node(f"Func_{func_id}", label=func_label, fillcolor="lightgreen")
                    dot.edge(f"ConvexConstruct_{module_id}", f"Func_{func_id}")
                    _recursive_add_nodes(func, f"Func_{func_id}", parent_weights=func_weight)
                
                # 子函数是 ConvexComposer：递归展开
                elif isinstance(func, ConvexComposer):
                    dot.node(f"Func_{func_id}", label=func_label, fillcolor="gold")
                    dot.edge(f"ConvexConstruct_{module_id}", f"Func_{func_id}")
                    _recursive_add_nodes(func, f"Func_{func_id}", parent_weights=func_weight)
                
                # 普通函数：直接添加节点
                else:
                    dot.node(f"Func_{func_id}", label=func_label)
                    dot.edge(f"ConvexConstruct_{module_id}", f"Func_{func_id}")

        # ------------------------------
        # 2. 处理 ConvexComposer 模块
        # ------------------------------
        elif isinstance(current_module, ConvexComposer):
            # 添加 ConvexComposer 节点（若有输入输出类型也可添加，此处保持原有基础上补充）
            composer_label = "ConvexComposer\n(fn → ConvexConstruct)"
            if parent_weights is not None:
                composer_label = f"ConvexComposer\nParent Weight={parent_weights:.3f}\n(fn → ConvexConstruct)"
            
            # 若 ConvexComposer 也有 input_types 和 output_type，可补充如下
            if hasattr(current_module, 'input_types') and hasattr(current_module, 'output_type'):
                input_types_repr = "\n".join([str(t) for t in current_module.input_types])
                output_type_repr = str(current_module.output_type)
                composer_label = (
                    f"ConvexComposer\n"
                    f"Parent Weight={parent_weights:.3f}\n"
                    f"Input Types:\n{input_types_repr}\n"
                    f"Output Type:\n{output_type_repr}\n"
                    f"(fn → ConvexConstruct)"
                ) if parent_weights is not None else (
                    f"ConvexComposer\n"
                    f"Input Types:\n{input_types_repr}\n"
                    f"Output Type:\n{output_type_repr}\n"
                    f"(fn → ConvexConstruct)"
                )

            dot.node(
                f"ConvexComposer_{module_id}",
                label=composer_label,
                fillcolor="gold"
            )

            # 连接父节点与当前 ConvexComposer 节点
            if parent_node_id:
                dot.edge(parent_node_id, f"ConvexComposer_{module_id}")

            # 添加 fn 节点（使用 __repr__/__str__ 作为名称）
            fn = current_module.fn
            fn_id = id(fn)
            
            # 优先使用 __repr__，其次 __str__，最后默认名称
            if hasattr(fn, '__repr__'):
                fn_display_name = fn.__repr__()[:50] + "..." if len(fn.__repr__()) > 50 else fn.__repr__()
            elif hasattr(fn, '__str__'):
                fn_display_name = fn.__str__()[:50] + "..." if len(fn.__str__()) > 50 else fn.__str__()
            else:
                fn_name = getattr(fn, "__name__", "UnknownFn")
                fn_display_name = fn_name

            fn_label = f"Pre-Process\n{fn_display_name}"
            dot.node(f"Fn_{fn_id}", label=fn_label, fillcolor="lightcyan")
            dot.edge(f"ConvexComposer_{module_id}", f"Fn_{fn_id}")

            # 递归处理嵌套的 ConvexConstruct
            nested_convex = current_module.convex
            _recursive_add_nodes(nested_convex, f"ConvexComposer_{module_id}")
            # 绘制 fn → nested_convex 的逻辑边（展示数据流向）
            nested_convex_id = id(nested_convex)
            dot.edge(f"Fn_{fn_id}", f"ConvexConstruct_{nested_convex_id}", style="dashed", color="darkblue")

    _recursive_add_nodes(convex, parent_node_id="")

    return dot


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Create a hierarchical tree layout.
    
    Parameters:
    - G: networkx graph (should be a tree)
    - root: root node (if None, finds one automatically)
    - width: horizontal space allocated for each level
    - vert_gap: gap between levels
    - vert_loc: vertical location of root
    - xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('Cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            # Find root (node with no predecessors)
            root = [n for n, d in G.in_degree() if d == 0]
            if not root:
                root = list(G.nodes())[0]  # fallback
            else:
                root = root[0]
        else:
            root = list(G.nodes())[0]  # pick arbitrary root for undirected

    def _hierarchy_pos(G, root, width=100., vert_gap=100, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        #print(root, parent, children)
        if parent is not None and parent in children:
            children.remove(parent)
        
        if not children:
            return pos
        
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                               vert_loc=vert_loc+vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)



def construct_convex_graph(model : ConvexConstruct):
    tree = nx.Graph()

    convex_sum_count = 0
    backward_fn_count = 0
    forward_fn_count = 0
    function_count = 0

    def _is_edge_valid_for_tree(graph, u, v):
        """校验添加边(u, v)后是否仍保持树结构（无环、连通性符合树要求）"""
        # 树的核心特性：无环，若两节点已连通，添加边会形成环，破坏树结构
        if nx.has_path(graph, u, v):
            return False
        # 树的边数始终等于节点数-1，提前预判（冗余校验）
        if len(graph.edges) + 1 > len(graph.nodes) - 1:
            return False
        return True

    def build(model, parent):
        nonlocal convex_sum_count, backward_fn_count, forward_fn_count, function_count
        current_node = None
        if isinstance(model, ConvexConstruct):
            node_name = f"vexsum_{convex_sum_count}"
            node_label = f"{model.output_type}\n{model.input_types}"
            if node_name not in tree:
                tree.add_node(node_name, label={"label": node_label})
            weights = model.get_normalized_weights()
            for i, fn in enumerate(model.functions):
                fn_node_name = f"fn{convex_sum_count}_{i}"
                if fn_node_name not in tree:
                    tree.add_node(fn_node_name)
                if isinstance(fn, (ConvexConstruct, ConvexComposer, BackwardCombinedFn)):
                    build(fn, fn_node_name)
                
                # 校验边有效性，无效则发出警告
                if _is_edge_valid_for_tree(tree, fn_node_name, node_name):
                    tree.add_edge(fn_node_name, node_name, label={"weight": weights[i] if i < len(weights) else None})
                else:
                    warnings.warn(f"警告：添加边 ({fn_node_name} <-> {node_name}) 会破坏树结构（形成环/冗余边），已跳过该边添加。")
            # 校验当前节点到父节点的边有效性
            if parent not in tree:
                tree.add_node(parent)
            if _is_edge_valid_for_tree(tree, node_name, parent):
                tree.add_edge(node_name, parent)
            else:
                warnings.warn(f"警告：添加边 ({node_name} <-> {parent}) 会破坏树结构（形成环/冗余边），已跳过该边添加。")
            current_node = node_name
            convex_sum_count += 1
        elif isinstance(model, ConvexComposer):
            fn_node = f"compose{forward_fn_count}_{model.fn}"
            if fn_node not in tree:
                tree.add_node(fn_node)
            build(model.convex, fn_node)
            # 校验边有效性，无效则发出警告
            if _is_edge_valid_for_tree(tree, fn_node, parent):
                tree.add_edge(fn_node, parent)
            else:
                warnings.warn(f"警告：添加边 ({fn_node} <-> {parent}) 会破坏树结构（形成环/冗余边），已跳过该边添加。")
            current_node = fn_node
            forward_fn_count += 1
        elif isinstance(model, BackwardCombinedFn):
            fn_node = f"compose{backward_fn_count}_{model.combine_fn}"
            if fn_node not in tree:
                tree.add_node(fn_node)
            # 校验当前节点到父节点的边有效性
            if _is_edge_valid_for_tree(tree, fn_node, parent):
                tree.add_edge(fn_node, parent)
            else:
                warnings.warn(f"警告：添加边 ({fn_node} <-> {parent}) 会破坏树结构（形成环/冗余边），已跳过该边添加。")
            sub_list = model.sub_convex_list
            for i, fn in enumerate(sub_list):
                sub_node = f"sub{backward_fn_count}_{i}"
                build(fn, sub_node)
                if sub_node not in tree:
                    tree.add_node(sub_node)
                # 校验边有效性，无效则发出警告
                if _is_edge_valid_for_tree(tree, sub_node, fn_node):
                    tree.add_edge(sub_node, fn_node)
                else:
                    warnings.warn(f"警告：添加边 ({sub_node} <-> {fn_node}) 会破坏树结构（形成环/冗余边），已跳过该边添加。")
            current_node = fn_node
            backward_fn_count += 1
        else:
            node_name = f"{str(model)}_{function_count}"
            if node_name not in tree:
                tree.add_node(node_name)
            # 校验边有效性，无效则发出警告
            if _is_edge_valid_for_tree(tree, node_name, parent):
                tree.add_edge(node_name, parent)
            else:
                warnings.warn(f"警告：添加边 ({node_name} <-> {parent}) 会破坏树结构（形成环/冗余边），已跳过该边添加。")
            current_node = node_name
            function_count += 1
        return current_node

    # 初始化根节点
    root_node = "root"
    if root_node not in tree:
        tree.add_node(root_node)
    build(model, root_node)

    # 绘制树形图
    tree_pos = hierarchy_pos(tree)
    nx.draw_networkx(tree, tree_pos)
    plt.show()

    return tree


if __name__ == "__main__":
    VAR = TupleType([BOOL, EmbeddingType("object", 96)] )

    constructor = Constructor(default_constructor_rules)


    function_registry = FunctionRegistry()
    constructor.max_depth = 2


    def dummy_embed_to_float(embed: torch.Tensor) -> torch.Tensor:
        return torch.mean(embed, dim = -1, keepdim=True)
    #constructor.function_registry.register_function(
    #    input_type=EmbeddingType("Latent",128),
    #    output_type=FLOAT,
    #    func=dummy_embed_to_float
    #)


    src_type = EmbeddingType("color_wheel", 1)
    tgt_type = EmbeddingType("object", 96)

    convex = constructor.create_convex_construct(src_type, tgt_type, function_registry)

    #convex.clear()
    #construct_convex_graph(convex)
    rewrite = constructor.create_convex_arg_rewriter(
        (src_type,), (tgt_type,), function_registry)

    if convex:
        print("ConvexConstruct built successfully!")
        print(f"Number of functions: {convex.num_functions}")
        #print(convex.functions)
        
        visualize_convex_construct(convex, "vis").view()
        test_embed = torch.randn([2,1])
        output = convex(test_embed).value.shape


        print(f"Test output: {output}")
    else:
        print("Failed to build ConvexConstruct.")

if __name__ == "___main__":

    class Id(nn.Module):
        def forward(self, x): return x

    class Quad(nn.Module):
        def forward(self, x): return x ** 2

    fns = [Id(), Quad()]
    ws = torch.tensor([1., 1.])
    conv = ConvexConstruct(fns, ws)

    print(conv(torch.tensor([4.])))
    
    optim = torch.optim.Adam(conv.parameters(), lr = 1e-1)

    for epoch in range(100):
        loss = torch.abs(conv(torch.tensor([4.])) - 16)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(conv.log_weights)