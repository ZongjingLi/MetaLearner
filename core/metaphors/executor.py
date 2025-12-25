'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-03-23 00:19:33
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-03-23 00:19:35
 # @Description:
'''
from typing import List, Union, Mapping, Dict, Any, Tuple, Callable

import os
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor, CentralExecutor
from helchriss.knowledge.symbolic import Expression,FunctionApplicationExpression, ConstantExpression, VariableExpression
from helchriss.dsl.dsl_values import Value
from helchriss.dsl.dsl_types import TypeBase, AnyType, INT, FLOAT, TupleType
from helchriss.dsl.dsl_values import value_types
from core.metaphors.rewrite import Frame, pth_file
from helchriss.logger import get_logger
from core.grammar.tree_parser import TreeParser


# this is the for type space, basically a wrapper class
from .types import Constructor, FunctionRegistry, ConvexConstruct
from .rules import default_constructor_rules
from .group import ExecutorGroup
from dataclasses import dataclass
import networkx as nx

__all__ = ["ExecutorGroup", "RewriteExecutor"]


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


def clamp_list(values, min_val=0.3, max_val=1.0):
    return [max(min(v, max_val), min_val) for v in values]


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


class SearchExecutor(CentralExecutor):
    """stores local rewrite frames then """
    def __init__(self, executor):
        super().__init__(None)
        self.base_executor : CentralExecutor = executor
        assert isinstance(self.base_executor, CentralExecutor), "not an CentralExecutor"
        self.base_executor.refs["executor_parent"] = self
        self.frames = nn.ModuleDict({}) # a bundle of rewriter rules that shares the same rewriter
        self.unification_p = 0.001
        self.supressed = 0
        self._gather_format = "{}:{}"
 
        self.constructor = Constructor(default_constructor_rules)
        #print(self.constructor.forward_rules, self.constructor.backward_rules)
        self.logger = get_logger(name = "SearchExecutor")
        self.ban_list = []
        self.eval_tree = {}
        self.node_count = 0

        self.parser = TreeParser()

        self.parser.load_entries(self.functions)

        """load the history frames"""
        self.transient_background_functions = [] # store the newly added functions

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


        """vertex of the rewrite search path"""
        self.vertex_count = -1
        rw_edges = []

        try:
            output_type = self.base_executor.function_signature(query_fn)[0][-1]

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

        return Value(measure.vtype, expect_output), -torch.log(success_reach), search_tree

    """Evaluate Chain of Expressions"""
    def evaluate(self, expression, grounding):
        if not isinstance(expression, Expression):
            #self.parser.supress = 1
            expression = expression.replace(" ","")
            #print(expression)
            #print(self.parser.parse(expression))
            

            expression = self.parser.parse(expression)[0].fn
            #print(expression)

            expression = Expression.parse_program_string(expression)


        grounding = grounding if grounding is not None else {}

        self.node_count = 0
        self.prev_node  = {"id":"node0","fn" : "output_fn", "value": "output", "type": "type"}
                         
        self.eval_info = {
            "tree":{"nodes":[], "edges": []},
            "paths":{}} # tree, paths


        with self.with_grounding(grounding):
            outputs, loss, _, _ = self._evaluate(expression)


        return outputs, loss
    
    def _evaluate(self, expr : Expression) -> Tuple[Value, str]:

        self.node_count += 1
        node_id = f"node{self.node_count}"
        paths = {"nodes":[], "edges":[]}
        if isinstance(expr, FunctionApplicationExpression):
            
            # recusive call self.evaluate(arg) to evaluate the args in the subtree
            fn = expr.func.name
            args : List[Value] = []
            arg_loss  = []

            for arg in expr.args: 
                arg_value, subloss, son_id, eval_path = self._evaluate(arg) # A List of Values
                args.append(arg_value)
                arg_loss.append(subloss)

                sub_loss = -subloss if isinstance(subloss, torch.Tensor) else torch.tensor(-subloss)
                edge_info = (node_id, son_id, {"weight":float(torch.exp(sub_loss) )})

                self.eval_info["tree"]["edges"].append(edge_info)

            arg_types : List[TypeBase] = [arg.vtype for arg in args]
            output, loss, paths = self.reduction_call(fn, args, arg_types)

            # node info loggers
            node_info = {"id":node_id, "fn" : fn, "value": str(output.value), "type": str(output.vtype)}
            self.eval_info["tree"]["nodes"].append(node_info)
            self.eval_info["paths"][f"{node_id}"] = paths

            #self.prev_node = node_info

            return output, sum(arg_loss) + loss, node_id, paths

        elif isinstance(expr, ConstantExpression):
            assert isinstance(expr.const, Value)
            return expr.const, 0., node_id, paths
        elif isinstance(expr, VariableExpression):
            from helchriss.dsl.dsl_types import STR
            #node_info = {"id":node_id, "fn" : str(expr.name), "value": str(expr.name), "type": str(expr.name)}
            #self.eval_info["tree"]["nodes"].append(node_info)
            #self.eval_info["paths"][f"{node_id}"] = {"nodes":{},"edges":{}}
            if isinstance(expr.name, Value):
                return expr.name, 0., node_id, paths
            elif isinstance(expr.name, str) :
                return Value(STR,str(expr.name)), 0., node_id, paths

            return expr.name, 0., node_id, paths
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
        #print("RW:", fn, value_types(args))
        execute_pairs, reach_loss, rw_nodes, rw_edges = self.rewrite_distr(args, fn, mode = "update")
        execute_pairs = execute_pairs[1:]

        if not execute_pairs: self.logger.info(f"execution not found for {fn}({value_types(args)})")

        # for each node that not in the 
        for (_,gn, vargs, weight) in execute_pairs:

            for (in_tps, out_tps) in self.base_executor.function_signature(fn):
                arg_types = value_types(vargs)
                src_tps = arg_types
                tgt_tps = in_tps 

                if isinstance(src_tps, List): src_tps = TupleType(src_tps)
                if isinstance(tgt_tps, List): tgt_tps = TupleType(tgt_tps)

                caster = self.constructor.create_convex_arg_rewriter(
                    src_tps.element_types, tgt_tps.element_types, self.base_executor)

                assert caster, f"caster not found for {src_tps}->{tgt_tps}"

                learn0_frame = Frame(src_tps, tgt_tps, caster)
                learn0_frame.matches[(f"{gn}@{fn}")] = torch.tensor(0.)

                src_sig = 'x'.join([str(tp) for tp in src_tps.element_types])
                tgt_sig = 'x'.join([str(tp) for tp in tgt_tps.element_types])
                frame_sig = f"frame:{src_sig}->{tgt_sig}_"
                
                id_itr = 0
                done = 0
                while not done:
                    if (frame_sig + str(id_itr)) in self.frames:id_itr += 1
                    else: done = 1
                #print(f"{gn}@{fn} by",frame_sig + str(id_itr))
                self.frames[frame_sig + str(id_itr)] = learn0_frame
    
        # 2. add the filler that defined on the value node.
        
        for (in_tps, out_tps) in self.base_executor.function_signature(fn):
            if len(args) != 1:
                in_tps = TupleType(value_types(args))
            else:
                in_tps = value_types(args)[0]
            convex_fillter = self.constructor.create_convex_construct(in_tps, out_tps, self.base_executor)
            assert convex_fillter, f"failed to create fillers {convex_fillter} for {fn} ({value_types(args)} -> {out_tps})"

            assert isinstance(self.base_executor, ExecutorGroup), "not a group executor"
            self.base_executor.register_extended_function(fn, value_types(args), out_tps, convex_fillter)
            self.transient_background_functions.append([fn, value_types(args), out_tps])
    

        return self

    def unfreeze_background_frame(self, name = None):
        """if name is None then unfreeze all the background frames"""
        return

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

