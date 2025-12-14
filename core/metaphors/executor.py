'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-03-23 00:19:33
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-03-23 00:19:35
 # @Description:
'''

import os
from typing import List, Union, Mapping, Dict, Any, Tuple, Callable
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor, CentralExecutor
from helchriss.knowledge.symbolic import Expression,FunctionApplicationExpression, ConstantExpression, VariableExpression
from helchriss.knowledge.symbolic import LogicalAndExpression, LogicalNotExpression,LogicalOrExpression
from helchriss.knowledge.symbolic import TensorState, concat_states
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import TypeBase, AnyType, INT, FLOAT
from core.metaphors.rewrite import NeuralRewriter, RewriteRule, Frame, pth_file
from helchriss.domain import Domain
from .rewrite import NeuralRewriter, LocalFrame
from helchriss.logger import get_logger
import inspect


# this is the for type space, basically a wrapper class
from .types import RuleBasedTransform, get_transform_rules
from dataclasses import dataclass
import networkx as nx
from pathlib import Path

__all__ = ["ExecutorGroup", "RewriteExecutor"]

def clamp_list(values, min_val=0.3, max_val=1.0):
    return [max(min(v, max_val), min_val) for v in values]

class UnificationFailure(Exception):
    """
    Exception raised when unification between feature structures fails.
    
    This class captures information about why unification failed,
    which structures failed to unify, and at which feature the failure occurred.
    """
    
    def __init__(self, message=None, feature=None, left_structure=None, right_structure=None):
        """
        Initialize a UnificationFailure exception.
        
        Args:
            message (str): Description of why unification failed
            feature (str): The feature name where unification failed
            left_structure (dict): The left feature structure that failed to unify
            right_structure (dict): The right feature structure that failed to unify
        """
        self.feature = feature
        self.left_structure = left_structure
        self.right_structure = right_structure
        
        # Create a detailed message if none provided
        if message is None:
            if feature:
                message = f"Unification failed at feature '{feature}'"
                if left_structure and right_structure:
                    message += f": Cannot unify '{left_structure.get(feature)}' with '{right_structure.get(feature)}'"
            else:
                message = "Unification failed between incompatible structures"
        
        super().__init__(message)
    
    def __str__(self):
        """Return a string representation of the unification failure."""
        result = self.args[0]
        
        if self.left_structure and self.right_structure:
            result += "\nLeft structure: " + str(self.left_structure)
            result += "\nRight structure: " + str(self.right_structure)
            
        return result
    
    def get_conflict_path(self):
        """
        Return the path to the conflicting feature.
        
        Returns:
            str: A dot-notation path to the conflicting feature
        """
        if not self.feature:
            return None
        
        return self.feature
    
    def is_type_mismatch(self):
        """
        Check if the failure was due to type mismatch.
        
        Returns:
            bool: True if the failure was due to type mismatch
        """
        if not (self.feature and self.left_structure and self.right_structure):
            return False
        
        left_val = self.left_structure.get(self.feature)
        right_val = self.right_structure.get(self.feature)
        
        return (left_val is not None and right_val is not None and 
                type(left_val) != type(right_val))


def value_types(values : List[Value]): return [v.vtype for v in values]

class ExecutorGroup(CentralExecutor):
    """Storage of Domain Executors or some Extention Functions"""
    def __init__(self, domains : List[Union[CentralExecutor, Domain]]):
        super().__init__(None)  

        self.executor_group = nn.ModuleList([])
        for domain_executor in domains:
            if isinstance(domain_executor, CentralExecutor):
                executor = domain_executor
            elif isinstance(domain_executor, Domain):
                executor = CentralExecutor(domain_executor)
            else : raise Exception(f"input {domain_executor} is not a Domain or FunctionExecutor")
            executor.refs["executor_parent"] = self
            self.executor_group.append(executor)

        self.extended_registry = nn.ModuleDict({})

    
    def save_ckpt(self, path: str):
        path = Path(path)
        (path / "domains").mkdir(parents=True, exist_ok=True)
        (path / "extended").mkdir(parents=True, exist_ok=True)
        
        for executor in self.executor_group:
            #print(executor.domain.domain_name)
            torch.save(executor.state_dict(), path / "domains" / f"{executor.domain.domain_name}.pth")
        
        for name, module in self.extended_registry.items():
            #print(name, isinstance(module, nn.Module))
            torch.save(module, path / "extended" / f"{name}.ckpt")

    def load_ckpt(self, path: str):
        path = Path(path)
        for executor_file in (path / "domains").glob("*.pth"):
            name = executor_file.stem
            for executor in self.executor_group:
                if executor.domain.domain_name == name:
                    executor.load_state_dict(torch.load(executor_file,  weights_only=True))
                    break

        for extended_file in (path / "extended").glob("*.ckpt"):
            name = extended_file.stem
            print(name)
            print(extended_file)
            self.extended_registry[name] = torch.load(extended_file,  weights_only=False)


    def format(self, function : str, domain : str) -> str: return f"{function}:{domain}"

    @staticmethod
    def signature(function : str, types : List[TypeBase]):
        typenames = [f"{tp.typename}" for tp in types]
        type_sign = "->".join(typenames)
        return f"{function}#{type_sign}"

    @staticmethod
    def parse_signature(signature: str) ->Tuple[str, List[TypeBase],TypeBase]:
        parts = signature.split('#')
        if len(parts) != 2: raise ValueError(f"Invalid signature format: {signature}")
    
        function_name = parts[0]
        type_signature = parts[1]
    
        type_specs = type_signature.split('->')
        all_types = []
        
        for type_spec in type_specs:
            type_parts = type_spec.split('-')
            if len(type_parts) != 1: raise ValueError(f"Invalid type specification: {type_spec}")
        
            typename = type_parts[0]
            all_types.append(TypeBase(typename))
    
        input_types = all_types[:-1]


        output_types = all_types[-1]  # Return as list as requested
        return function_name, input_types, output_types

    def domain_function(self, func : str) -> bool: return ":" in func

    def freeze_extended(self, freeze = True):
        for param in self.extended_registry.parameters():
            param.requires_grad = freeze

    @property
    def functions(self) -> List[Tuple[str, List[TypeBase], TypeBase]]:
        functions = []
        for sign in self.extended_registry:
            f_sign, in_types, out_type = self.parse_signature(sign)

            functions.append([f_sign, in_types, out_type])
        for executor in self.executor_group:
            assert isinstance(executor, CentralExecutor), f"{executor} is not a executor"
            for func_name in executor._function_registry:

                functions.append([
                    self.format(func_name,executor.domain.domain_name),
                    executor.function_input_types[func_name],
                    executor.function_output_type[func_name]])
        #print("MAX:",list(fn for fn in functions if "max" in fn[0]))
        return functions

    def function_signature(self, func : str) -> List[Tuple[List[TypeBase], TypeBase]]:
        hyp_sign = []
        for function in self.functions:
            f_sign, in_types, out_type = function
            if func == f_sign:
                #print("item:",f_sign, in_types)
                hyp_sign.append([in_types, out_type])
        return hyp_sign
        
    def gather_functions(self, input_types, output_type : Union[TypeBase, bool]) -> List[str]:
        compatible_fns = []
        for function in self.functions:
            fn_sign, in_types, out_type = function
            #print(fn_sign,"\n  ::", in_types, input_types,in_types == input_types)
            #print(in_types , input_types, in_types == input_types)
            if in_types == input_types and out_type == output_type:
                #print("select:",fn_sign, in_types)
                compatible_fns.append(fn_sign)
        #print("COMP:", compatible_fns)
        return compatible_fns

    def register_extended_function(self, func : str, in_types : List[TypeBase], out_type : TypeBase, implement : nn.Module):
        signature = self.signature(func, in_types + [out_type])
        self.extended_registry[signature] = implement

    def infer_domain(self, func: str) -> str:
        for executor in self.executor_group: 
            assert isinstance(executor, CentralExecutor), "not a function executor"
            if func in executor._function_registry: return executor.domain.domain_name

    def execute(self, func : str, args : List[Value], arg_types : List[TypeBase],  grounding = None) -> Value:
        self._grounding = grounding
        """a function could be in some domain or in extention registry"""
        arg_types = value_types(args)
        signature = self.signature(func, arg_types)
        func_call = None

        # 1) check if this is a domain function
        if self.domain_function(func):
            func_name, domain_name = func.split(":")
            for executor in self.executor_group:
                assert isinstance(executor, CentralExecutor), f"{executor} is not a executor"
                if executor.domain.domain_name == domain_name: # find the correct executor
                    executor._grounding = grounding
                    func_call = executor._function_registry[func_name]


        # 2) check if this is an extention function
        for sign in self.extended_registry:
            if signature in sign:func_call = self.extended_registry[sign]


        ### collect arguments and evaluate on the function call
        args = [arg.value for arg in args]
        kwargs = {"arg_types" : arg_types}
        
        sig = inspect.signature(func_call)
        has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD 
                     for param in sig.parameters.values())

        if func_call is not None:
            if has_kwargs: return func_call(*args, **kwargs)
            else: return func_call(*args)
        else:raise ModuleNotFoundError(f"{signature} is not found.")


@dataclass
class SearchNode:
    """not the frame search node, the eval search node"""
    fn    : str               # executable fn on value
    value : List[Value]       # the value of the node, args of some function
    src   : None              # the source rewriter
    maintain : bool = None           # the virtual node
    next = []
    next_weights = []
    distr : None = None                    # the distribution value on the search node
    reachable : None  = None           # whether the node is reachable
    name : str = None


class SearchVisualizer:
    """Visualizer for logging and graphing search tree operations"""
    def __init__(self):
        self.query_trees = []
        self.active = 1  # Using int for boolean (1 = active, 0 = inactive)

    def log(
        self,
        query_fn: str,
        args: List[Value],
        nodes: List[SearchNode],
        mode: str
    ) -> None:
        """
        Log search tree data if the visualizer is active.
        
        Args:
            query_fn: Name of the query function
            args: List of Value objects passed to the query
            nodes: List of root SearchNodes for the search tree
            mode: Operation mode (e.g., "search", "expand", "evaluate")
        """
        if self.active:
            self.query_trees.append((query_fn, args, nodes, mode))

    def create_graph(self, nodes: List[SearchNode]) -> nx.DiGraph:
        """
        Create a directed graph from a list of SearchNodes using DFS traversal.
        
        Args:
            nodes: List of root SearchNodes to build the graph from
        
        Returns:
            nx.DiGraph: NetworkX directed graph with node/edge metadata
        """
        graph = nx.DiGraph()

        def dfs(current_node: SearchNode) -> None:
            """Recursive DFS helper to traverse SearchNodes and build the graph"""
            # Ensure value is a list (fallback for None)
            if current_node.value is None:
                current_node.value = []

            # Build node label with function name, value types, and rounded values
            vtypes = [str(v.vtype) for v in current_node.value]
            rounded_values = [
                str((torch.round(torch.tensor(v.value) * 100) / 100).detach().numpy())
                for v in current_node.value
            ]
            node_label = f"{current_node.fn}\n{vtypes}\n{rounded_values}"

            # Add node with metadata (use string representation of node as unique ID)
            graph.add_node(
                str(current_node),
                fn=current_node.fn,
                value=[v.value for v in current_node.value],
                label=node_label
            )

            # Traverse child nodes and add edges with weights
            for next_node, weight in zip(current_node.next, current_node.next_weights):
                dfs(next_node)
                # Add edge with rounded weight (convert tensor to numpy if needed)
                edge_weight = torch.tensor(weight).detach().numpy() if not isinstance(weight, (int, float)) else weight
                graph.add_edge(str(current_node), str(next_node), weight=edge_weight)

        # Start DFS from the first node (assuming nodes list has one root)
        if nodes:
            dfs(nodes[0])

        return graph

class SearchExecutor(CentralExecutor):
    """stores local rewrite frames then """
    def __init__(self, executor):
        super().__init__(None)
        self.base_executor : CentralExecutor = executor
        assert isinstance(self.base_executor, CentralExecutor), "not an CentralExecutor"
        self.base_executor.refs["executor_parent"] = self
        self.frames = nn.ModuleDict({}) # a bundle of rewriter rules that shares the same rewriter
        self.storage = SearchVisualizer()
        self.unification_p = 0.001
        self.supressed = 0
        self._gather_format = "{}:{}"
 
        self.inferer = RuleBasedTransform(*get_transform_rules())
        self.logger = get_logger(name = "SearchExecutor")
        self.ban_list = []
        self.eval_tree = {}
        self.node_count = 0

    """stupid formatting stuff"""
    def gather_format(self, name, domain): return self._gather_format.format(name, domain)
    
    @staticmethod
    def format(function : str): return function.split(":")[0]

    @property
    def types(self): return self.base_executor.types

    @property
    def functions(self): return self.base_executor.functions

    def function_out_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")

        return self.base_executor.function_out_type(func_name, domain)

    def function_input_type(self, func_name, domain = None):
        if domain is None: func_name, domain = func_name.split(":")
        return self.base_executor.function_input_type(func_name, domain)











    """Save and Load Utils and Add Frames"""
    def save_ckpt(self, ckpt_path) -> int:
        if not os.path.exists(f"{ckpt_path}/frames"): os.makedirs(f"{ckpt_path}/frames")
        for frame_name in self.frames: torch.save(self.frames[frame_name], f"{ckpt_path}/frames/{frame_name}.pth")#.save_ckpt(f"{ckpt_path}/frames/{frame_name}")
        self.base_executor.save_ckpt(ckpt_path)
        return self

    def load_ckpt(self, ckpt_path) -> int:
        # load the mapping frames learned
        frames_dir = f"{ckpt_path}/frames"
        for filename in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, filename)
            if os.path.isfile(file_path) and pth_file(filename):
                self.frames[filename[:-4]] = torch.load(file_path, weights_only = False)
        # load the domain specific executor parameters learned
        self.base_executor.load_ckpt(ckpt_path)
        return self

    def add_frame(self,name : str, frame : Frame):
        """raw method of adding a frame to the dictionary"""
        self.frames[name] = frame



    








    """Rewrite Search Graph Implementations"""
    def edges(self, node : SearchNode) -> List[SearchNode]:
        src_fn  = node.fn
        src_val = node.value
        src_tp  = value_types(src_val)
        nodes = []
        edges = []
        for key, frame in self.frames.items():
            assert isinstance(frame, Frame), f"{frame} is not a `Frame`"

            if frame.source_type == src_tp:
                nxt_val, nxt_pr = frame.apply(src_val) # applied value and pr
                """enumerate matches that start with src_fn"""
                for fn_gn in frame.matches:
                    (fn, gn) = fn_gn.split("@")
                    if fn == src_fn:
                        fn_logit = frame.matches[fn_gn] # function match logit
                        nodes.append(SearchNode(gn, nxt_val, None))
                        edges.append(torch.sigmoid(nxt_pr + fn_logit))
        return nodes, edges

    def rewrite_search_tree(self, init_node : SearchNode,  max_iters = 100) :
        """graph start with value and end with fn by rewrites"""
        nodes : List[SearchNode] = []
        edges : List             = []

        #. start the bfs queue
        itrs = 0
        done = False
        queue : List[SearchNode] = [init_node]
        while not done:
            curr_node = queue.pop(0)
            frontier_nodes, frontier_edges = self.edges(curr_node)
            for node, edge_weight in zip(frontier_nodes, frontier_edges):
                # node is the nodes the curr_node connected ; edge is a weight
                assert isinstance(node, SearchNode), f"{node} not a search node"
                queue.append(node)
                curr_node.next.append(node)
                curr_node.next_weights.append(edge_weight)
                edges.append((curr_node, node, edge_weight))
            nodes.append(curr_node)
            itrs += 1
            done =  len(queue) == 0 or (itrs >= max_iters)
        return nodes
    
    def subtree_filter_target(self, init_node :SearchNode, nodes : List[SearchNode], target : str) -> List[SearchNode]:
        def dfs(node : SearchNode):
            node.maintain = 0
            for son in node.next:
                dfs(son)
                assert isinstance(son, SearchNode), f"{son} is not SearchNode"
                if son.maintain:
                    node.maintain = 1
                    break
            if node.fn == target: node.maintain = 1

        def subtree_select(node : SearchNode, df : int):
            if df: node.maintain = 1
            for son in node.next:
                assert isinstance(son, SearchNode), f"{son} is not SearchNode"
                if node.fn == target:
                    son.maintain = 1
                    subtree_select(son, 1)
        
        dfs(init_node) # select the parent of the target node
        subtree_select(init_node, 0) # select the whole subtree of the target node

        return [node for node in nodes if node.maintain == 1]

    def subtree_filter_margin(self, init_node : SearchNode, nodes : List[SearchNode], margin = 0.1):
        def dfs(node : SearchNode):
            for son, weight in zip(node.next, node.next_weights):
                assert isinstance(son, SearchNode)
                son.reachable = node.reachable * weight
                if son.reachable >= margin:
                    son.maintain = 1
                    dfs(son)
                else: son.maintain = 0
        init_node.reachable = 1.0
        dfs(init_node)
        subtree_nodes = [node for node in nodes]
        return subtree_nodes

    def rewrite_distr(self, value : List[Value], query_fn : str, mode = "eval") :
        """ return the rewrite paths from fn to target value
        Args:
            value : the input value to evaluate on
            fn    : the target fn to locate and evaluate
        Return:
            a distribution over search nodes
        """
        



        src_node = SearchNode("node_0", None, None, None, reachable = 1.)
        src_node.next = []
        src_node.next_weights = []
        rw_nodes : List[SearchNode] = [src_node]
        #print("distr:", query_fn)

        """vertex of the rewrite search path"""
        self.vertex_count = -1
        rw_edges = []

        try:
            output_type = self.base_executor.function_signature(query_fn)[0][-1]
            #print(query_fn, value_types(value))
            functions = self.base_executor.gather_functions(value_types(value), output_type)

        except:
            raise NotImplementedError(f"{query_fn} is not found")
        for fn in functions:

            nd = SearchNode(fn, value, None)
            nd.next = []
            nd.next_weights = []
            subtree_nodes = self.rewrite_search_tree(nd)
            if mode == "eval": # select the subtree contains the query fn
                subtree_nodes = self.subtree_filter_target(subtree_nodes[0], subtree_nodes, query_fn)
            if mode == "update": # select the subtree that is reachable by margin
                subtree_nodes = self.subtree_filter_margin(subtree_nodes[0], subtree_nodes, margin = 0.01)
    
            if subtree_nodes: # not an empty subtree
                src_node.next.append(subtree_nodes[0]) # add the subtree head
                src_node.next_weights.append(1.)       # add the transition 1
                rw_nodes.extend(subtree_nodes)



        def dfs(node : SearchNode, success_reach, prev = None):
            self.vertex_count += 1
            if prev is not None:
               node.name = f"vertex{self.vertex_count}"
               rw_edges.append((prev, f"vertex{self.vertex_count}",success_reach))
            
            # the node is reachable and have no applicable rewrite.
            if node.next_weights:
                node.distr = node.reachable * (1. - torch.max(torch.tensor(node.next_weights)))
            else:
                node.distr = node.reachable
            for son, weight in zip(node.next, node.next_weights):
                assert isinstance(son, SearchNode)
                son.reachable = node.reachable * weight
                if son.fn == query_fn :
                    success_reach = max(success_reach, son.reachable)
                
                success_reach = max(success_reach,dfs(son, success_reach, f"vertex{self.vertex_count}"))
            return success_reach

        success_reach = dfs(src_node, 0., None) # dfs on the super source node

        #self.storage.log(query_fn, value, rw_nodes, mode)

        return [(node.name,node.fn, node.value, node.distr) for node in rw_nodes], success_reach, rw_nodes, rw_edges

    def reduction_call(self, fn : str , args : List[Value], arg_types : List[TypeBase]):
        """ evaluate the fn calls on the distribution over all possible rewrites
        start with the init_node that contains the value and the 
        1. locate each node that terminates, find the subtree that trace back to the init_node
        2. for the subtree, evaluate the expectation over the subtree distribution
        """
        rewrite_pairs, success_reach, rw_nodes, serial_edges = self.rewrite_distr(args, fn, mode = "eval")
        execute_pairs = rewrite_pairs[1:]
        if not isinstance(success_reach, torch.Tensor): success_reach = torch.tensor(success_reach)
        if success_reach < self.unification_p and not self.supressed:
            raise UnificationFailure(f"failed to unify {fn}({args}->{success_reach})", 
                                     left_structure = fn,
                                     right_structure = args)
        #print("fn",fn,success_reach)
        serial_nodes = [
            {       
                    "id":"vertex0",
                    "fn": "super",
                    "value":"super start",
                    "type":"start type",
                    "weight": 1.0}
        ]


        """expectation over all the possible rewrite pairs"""
        expect_output = 0.
        assert execute_pairs, f"{fn} canont be executed on {value_types(args)}"
        for i,(node, fn, vargs, weight) in enumerate(execute_pairs):

            measure : Value = self.base_executor.execute(fn, vargs, arg_types, self.grounding)
            if isinstance(measure.value, torch.Tensor) and (measure.vtype == INT or measure.vtype == FLOAT):
                expect_output += measure.value.reshape([-1])[0] * weight
            else: expect_output += measure.value * weight

            serial_nodes.append(
                {   "id":node,
                    "fn": fn,
                    "value":str(measure.value),
                    "type": str(value_types(args))+"->"+str(measure.vtype),
                    "weight": weight}
                )


        search_tree = {"nodes": serial_nodes, "edges":serial_edges}
        #print("fn",fn,success_reach)
        return Value(measure.vtype, expect_output), -torch.log(success_reach), search_tree

    """Evaluate Chain of Expressions"""
    def evaluate(self, expression, grounding):
        if not isinstance(expression, Expression):
            expression = self.parse_expression(expression)
 

        grounding = grounding if grounding is not None else {}

        self.node_count = 0
        self.prev_node  = {"id":"node0","fn" : "output_fn", "value": "output", "type": "type"}
                         
        self.eval_info = {
            "tree":{"nodes":[], "edges": []},
            "paths":{}} # tree, paths


        with self.with_grounding(grounding):
            outputs, loss, _ = self._evaluate(expression)

        #print("eval",self.eval_info)
        return outputs, loss
    
    def _evaluate(self, expr : Expression) -> Tuple[Value, str]:
        #print("ct:", self.node_count)
        self.node_count += 1
        node_id = f"node{self.node_count}"
        if isinstance(expr, FunctionApplicationExpression):
            
            # recusive call self.evaluate(arg) to evaluate the args in the subtree
            fn = expr.func.name
            args : List[Value] = []
            arg_loss  = []

            for arg in expr.args: 
                arg_value, subloss, son_id = self._evaluate(arg) # A List of Values
                args.append(arg_value)
                arg_loss.append(subloss)

                edge_info = (node_id, son_id, {"weight":float(torch.exp(torch.tensor(-subloss)) )})

                self.eval_info["tree"]["edges"].append(edge_info)

            arg_types : List[TypeBase] = [arg.vtype for arg in args]
            output, loss, paths = self.reduction_call(fn, args, arg_types)

            # node info loggers
            node_info = {"id":node_id, "fn" : fn, "value": str(output.value), "type": str(output.vtype)}
            self.eval_info["tree"]["nodes"].append(node_info)
            
            self.eval_info["paths"][f"{node_id}"] = paths

            self.prev_node = node_info

            return output, sum(arg_loss) + loss, node_id

        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const, 0., node_id
        elif isinstance(expr, VariableExpression):
            from helchriss.dsl.dsl_types import STR
            if isinstance(expr.name, Value): return expr.name, 0., node_id
            elif isinstance(expr.name, str) : return Value(STR,str(expr.name)), 0., node_id

            return expr.name, 0., node_id
        else:
            raise NotImplementedError(f'Unknown expression type: {type(expr)}')
    












    """Update Search Rules and Purge/Merge Entries""" 
    def update_chain(self, args : List[Value], fn : str, topK = 4):
        """assert exist a path from the arg to the fn located space by some frames
        add frames and update fn on the margin types.
        Args:
            args : List[TypeBase] as the input arg types for the fn
            fn   : str as the input function name (specific to domain tag)
        """
        # 1. add rewriters that locate from the value node to the fn node.
        execute_pairs, reach_loss, paths = self.rewrite_distr(args, fn, mode = "update")
        execute_pairs = execute_pairs[1:]

        if not execute_pairs: self.logger.info(f"execution not found for {fn}({value_types(args)})")

        # for each node that not in the 
        for (gn, vargs, weight) in execute_pairs:
            for (in_tps, out_tps) in self.base_executor.function_signature(fn):
  
                arg_types = value_types(vargs)

                src_tps = arg_types
                tgt_tps = in_tps 
    
                caster = self.inferer.infer_args_caster(src_tps, tgt_tps)
                learn0_frame = Frame(src_tps, tgt_tps, caster)
                #(learn0_frame)
                learn0_frame.matches[(f"{gn}@{fn}")] = torch.tensor(0.)

                src_sig = 'x'.join([str(tp) for tp in src_tps])
                tgt_sig = 'x'.join([str(tp) for tp in tgt_tps])
                frame_sig = f"frame:{src_sig}->{tgt_sig}_"
                id_itr = 0
                done = 0
                while not done:
                    if (frame_sig + str(id_itr)) in self.frames:id_itr += 1
                    else: done = 1
                print(f"{gn}@{fn} by",frame_sig + str(id_itr))
                self.frames[frame_sig + str(id_itr)] = learn0_frame

        # 2. add the filler that defined on the value node.
        
        for (in_tps, out_tps) in self.base_executor.function_signature(fn):
            #print(value_types(args), out_tps)
            fillers = self.inferer.infer_fn_prototypes(value_types(args), out_tps)
            assert fillers, f"failed to create fillers {fillers} for {fn}"
            max_filler = fillers[0]
            assert isinstance(self.base_executor, ExecutorGroup), "not a group executor"
            self.base_executor.register_extended_function(fn, value_types(args), out_tps, max_filler)
            #print("added",fn, value_types(args))

        return self
    
    def purge_frames(self, p = 0.1):
        """purge the the last p percent of all the newly added connection
        the last p percent of the newly added frames need to be removed.
        Args:
            p : the percent of newly added frames needed to be removed
        """
        return self
    
    def merge_frames(self, p = 0.1):
        """merge frames that shares the     same rewriter, distinguish rewriter using subsets of data
        Args:
            p : the metric of how close two rewriter need to be merged together
        merge two frames with shared rewriter and combine the 
        """
        return self

    def additive_evaluation(self, query, grounding):

        """return the (output and loss) pair, also update the eval chain if unification failure"""
        try:

            out = self.evaluate(query, grounding = grounding)
        except UnificationFailure as error:
            query_fn = error.left_structure
            value = error.right_structure
            if (query_fn, value_types(value)) not in self.ban_list:
                self.ban_list.append((query_fn, value_types(value)))
                self.update_chain(value, query_fn)
                
                self.logger.critical(f"update chain : {query_fn} -> {value_types(value)}")

            out = self.evaluate(query, grounding = grounding)
        return out



















from collections import defaultdict, deque

def convert_graph_to_visualization_data(G):
    """
    Convert a NetworkX graph to visualization data format needed for D3.js
    
    Args:
        G: NetworkX graph (eval_graph from the model)
    
    Returns:
        dict: JSON-serializable dictionary with nodes and edges data
    """
    # Initialize nodes and edges lists
    nodes = []
    edges = []
    
    # Process nodes
    for node_id, node_data in G.nodes(data=True):
        # Create node entry
        node = {
            "id": str(node_id),
            "weight": float(node_data.get('weight', 1.0)),
            "color": node_data.get('color', '#3a5f7d'),
            "text_color": node_data.get('text_color', 'white')
        }
        
        # Add output information if available
        if 'output' in node_data:
            output_data = node_data['output']
            output_info = {
                "value": float(output_data.value) if hasattr(output_data, 'value') else str(output_data),
                "vtype": output_data.vtype.alias.split(':')[0] if hasattr(output_data.vtype, 'alias') else str(output_data.vtype)
            }
            node["output"] = output_info
        
        nodes.append(node)
    
    # process edges
    for source, target, edge_data in G.edges(data=True):
        # Create edge entry
        edge = {
            "source": str(source),
            "target": str(target),
            "weight": float(edge_data.get('weight', 1.0)),
            "color": edge_data.get('color', '#555555')
        }
        
        # Add output information if available
        if 'output' in edge_data:
            output_data = edge_data['output']
            output_info = {
                "value": float(output_data.value) if hasattr(output_data, 'value') else str(output_data),
                "vtype": output_data.vtype.alias.split(':')[0] if hasattr(output_data.vtype, 'alias') else str(output_data.vtype)
            }
            edge["output"] = output_info
        
        edges.append(edge)
    
    import matplotlib.pyplot as plt
    #nx.draw(G)
    #plt.show()

    return {
        "nodes": nodes,
        "edges": edges,
   
    }

def check_graph_node_names(graph):
    """
    Modify a NetworkX DiGraph to remove the part after ':' from node names.
    Also updates all edges to use the new node names.
    
    Args:
        graph (nx.DiGraph): Input directed graph
        
    Returns:
        nx.DiGraph: New graph with modified node names
    """
    # Create a new graph
    new_graph = nx.DiGraph()
    
    # Create mapping from old names to new names
    name_mapping = {}
    for node in graph.nodes():
        if ':' in str(node):
            new_name = str(node).replace(":","&")
        else:
            new_name = str(node)
        name_mapping[node] = new_name
    
    # Add nodes with new names and preserve node attributes
    for old_node, new_node in name_mapping.items():
        node_attrs = graph.nodes[old_node] if graph.nodes[old_node] else {}

        new_graph.add_node(new_node, **node_attrs)
    
    # Add edges with new node names and preserve edge attributes
    for u, v in graph.edges():
        new_u = name_mapping[u]
        new_v = name_mapping[v]
        edge_attrs = graph.edges[u, v] if graph.edges[u, v] else {}

        new_graph.add_edge(new_u, new_v, **edge_attrs)
    
    return new_graph


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

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if parent is not None:
            children.remove(parent)
        
        if not children:
            return pos
        
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                               vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def radial_tree_pos(G, root=None, scale=1):
    """
    Create a radial tree layout where nodes are positioned in concentric circles.
    """
    if root is None:
        if isinstance(G, nx.DiGraph):
            roots = [n for n, d in G.in_degree() if d == 0]
            root = roots[0] if roots else list(G.nodes())[0]
        else:
            root = list(G.nodes())[0]
    
    # BFS to assign levels
    levels = {}
    queue = deque([(root, 0)])
    visited = {root}
    
    while queue:
        node, level = queue.popleft()
        levels[node] = level
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))
    
    # Group nodes by level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    pos = {}
    import math
    
    for level, nodes in level_nodes.items():
        if level == 0:  # root
            pos[nodes[0]] = (0, 0)
        else:
            radius = level * scale
            angle_step = 2 * math.pi / len(nodes)
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos[node] = (x, y)
    
    return pos

def balanced_tree_pos(G, root=None, k=2):
    """
    Create a balanced binary/k-ary tree layout.
    Assumes the tree is roughly balanced.
    """
    
    if root is None:
        if isinstance(G, nx.DiGraph):
            G = G.reverse()
            roots = [n for n, d in G.in_degree() if d == 0]

            root = roots[-1] if roots else list(G.nodes())[0]

        else:
            # For undirected, find a good root (center-ish node)
            root = nx.center(G)[0]

    pos = {}
    
    def assign_positions(node, level, position, width, parent=None):
        pos[node] = (position, -level)
        #print(node)
        children = [n for n in G.neighbors(node) if n != parent]
        if not children:
            return
        
        child_width = width / len(children)
        start_pos = position - width/2 + child_width/2
        
        for i, child in enumerate(children):
            child_pos = start_pos + i * child_width
            assign_positions(child, level + 1, child_pos, child_width, node)
    
    assign_positions(root, 0, 0, 4.0)  # Start with width of 4
    return pos
