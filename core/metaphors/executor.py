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
from helchriss.dsl.dsl_types import TypeBase, AnyType, INT, FLOAT, TupleType, EmbeddingType
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
import matplotlib.pyplot as plt

from helchriss.utils.meta import timer
from core.utils import hierarchy_pos

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


def node_sn(node):
    if node.sign is not None: return node.sign
    if ":" in node.fn:
        fn, space = node.fn.split(":")
    else:
        fn = node.fn
        space = "*"
    distr = node.distr
    if distr is None: distr = -1
    fn = fn + str(node.id)
    vars = '\n'.join([str(t) for t in value_types(node.value)])
    return f"{fn}\n{float(distr):.2f}\n{space}\n{vars}"

def visualize_subtree(subtree_root: "SearchNode", node_size=800, font_size=5, query_fn = None):
    G = nx.DiGraph()
    node_to_search_node = {}
    edge_weights = {}

    def build_graph(node: "SearchNode"):
        if node.value is None: node.value = []
        node_id = node_sn(node)
        G.add_node(node_id, label = "*")
        node_to_search_node[node_id] = node

        if node.next is None: node.next = []
        if node.next_weights is None: node.next_weights = []
        for son, weight in zip(node.next, node.next_weights):
            son_id = node_sn(son)
            G.add_edge(node_id, son_id)
            edge_weights[(node_id, son_id)] = weight
            build_graph(son)
    
    build_graph(subtree_root)
    pos = hierarchy_pos(G, vert_gap=3.5)  
    
    if 1:
        node_colors = []
        query_node_ids = []
        subtree_nodes = []
        parent_nodes  =  []
        for node_id, search_node in node_to_search_node.items():
            #assert isinstance(search_node, SearchNode)

            if search_node.is_target:
                query_node_ids.append(node_id)
            if search_node.is_subtree:
                subtree_nodes.append(node_id)
            if search_node.is_parent:
                parent_nodes.append(node_id)

        for node_id in G.nodes:
            if node_id in query_node_ids:
                node_colors.append("#D01117")
            if node_id in subtree_nodes:
                node_colors.append( "#14AF36")
            if node_id in parent_nodes:
                node_colors.append("#06627E")
                
        
        edge_colors = []
        import matplotlib.cm as cm  
        import matplotlib.colors as colors 

        for edge in G.edges():
            weight = edge_weights.get(edge, 0)
            try:
                weight = weight.detach().numpy()
            except: weight = float(weight)

            norm = colors.Normalize(vmin=0, vmax=1)
            cmap = cm.get_cmap("Blues")

            edge_colors.append(cmap(norm(weight)))
        #print(G)
        #print(node_colors)
        #print(edge_colors)

        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
                node_size=node_size, font_size=font_size, font_weight="bold", 
                arrows=True, arrowsize=20)
        
        plt.title(f"{query_fn}", fontsize=14)
        plt.show()
        

@dataclass
class SearchNode:
    """not the frame search node, the eval search node"""
    fn    : str               # executable fn on value
    value : List[Value]       # the value of the node, args of some function
    src   : None              # the source rewriter
    id    : int
    maintain : bool = None           # the virtual node
    next = []
    next_weights = []
    distr : None = None                    # the distribution value on the search node
    reachable : None  = None           # whether the node is reachable
    is_subtree = 0
    is_parent = 0
    is_target = 0
    name : str = None
    sign : str = None


    def __str__(self): return f"{self.fn}"

    def __hash__(self): return hash(self.__str__())


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

        self.logger = get_logger(name = "SearchExecutor")
        self.ban_list = []
        self.eval_tree = {}
        self.node_count = 0

        self.parser = TreeParser()

        self.parser.load_entries(self.functions)

        """load the history frames"""
        self.transient_background_functions = [] # store the newly added functions
        self.default_freeze = 1
        self.verbose = 0
        self.cut = 1
        self.non_effective_nodes = []

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



    





    def rewrite_skeleton(self):
        """visualize the skeleton graph of rewrites"""
        init_nodes = []
        for fn, in_types, out_type in self.functions:
            init_nodes.append(SearchNode(fn))
        #TODO: write the skeleton version of the rewrite graph
        return


    """Rewrite Search Graph Implementations"""
    def edges(self, node : SearchNode) -> List[SearchNode]:
        src_fn  = node.fn
        src_val = node.value
        src_tp  = value_types(src_val)
        nodes = []
        edges = []
        for key, frame in self.frames.items():
            assert isinstance(frame, Frame), f"{frame} is not a `Frame`"

            if len(src_tp) != len(frame.source_type):
                type_match = False
            else:
                type_match = True
                for i in range(len(src_tp)):
                    if frame.source_type[i] != src_tp[i]:
                        type_match = False

            if type_match:
                nxt_val, nxt_pr = frame.apply(src_val) # applied value and pr
    
                """enumerate matches that start with src_fn"""
                for gn_fn in frame.matches:
                    (gn, fn) = gn_fn.split("@")
 
                    if gn == src_fn:
                        fn_logit = frame.matches[gn_fn] # function match logit
                        nodes.append(SearchNode(fn, nxt_val, None, id = 0))

                        sum_pr = 0.
                        for arg_pr in nxt_pr:
                            sum_pr += torch.sum(arg_pr.value)

                        rw_weight = torch.sigmoid(sum_pr.flatten() + fn_logit).reshape([])

                        edges.append(rw_weight)

        return nodes, edges

    def rewrite_search_tree(self, init_node : SearchNode,  max_iters = 10, id_count = None) :
        """graph start with value and end with fn by rewrites"""
        from collections import deque
        nodes: List[SearchNode] = []
        edges: List[Tuple[SearchNode, SearchNode, float]] = []
        visited: set[SearchNode] = set()

        #. start the bfs queue
        queue = deque([init_node])
        visited.add(init_node.fn + str(value_types(init_node.value)))
        itrs = 0
        done = False

        id_count = {} if id_count is None else id_count

        def label_nodes(nodes):
            for node in nodes:
                sn = node_sn(node)
                if sn in id_count:
                    id_count[sn] += 1
                    node.id = id_count[sn]
                else:
                    id_count[sn] = 0
                    node.id = 0
            return nodes

        while not done:
            if not queue:
                done = True
                break
            
            curr_node = queue.popleft()  
            frontier_nodes, frontier_edges = self.edges(curr_node)
            frontier_nodes = label_nodes(frontier_nodes) # label nodes by the appear id


            for node, edge_weight in zip(frontier_nodes, frontier_edges):
                assert isinstance(node, SearchNode), f"{node} not a search node"
                if node.fn + str(value_types(node.value)) not in visited:
                    visited.add(node.fn + str(value_types(node.value)))
                    queue.append(node)
                    curr_node.next.append(node)
                    curr_node.next_weights.append(edge_weight)
                    edges.append((curr_node, node, edge_weight))
  
            nodes.append(curr_node)
            itrs += 1
            
            if itrs >= max_iters:
                done = True

        return nodes, id_count
    
    def subtree_filter_target(self, init_node: SearchNode, target: str) -> List[SearchNode]:
        """
        Filter nodes that either eventually reach the target node or are reachable by the target node (in the target's subtree).
        Traverse the subtree rooted at init_node, no additional nodes list required.
        
        Args:
            init_node: Root node of the subtree to start traversal and filtering.
            target: The target query node's fn value.
        
        Returns:
            List of nodes that meet the maintenance criteria (maintain=1).
        """
        def is_query_node(node):
            if node.fn == target :
                node.maintain = 1
                node.is_target = 1
            else :
                node.maintain = 0
                node.is_target = 0
            for son in node.next:
                is_query_node(son)
        
        def is_query_parent(node):
            assert isinstance(node, SearchNode)
            has_query_son = 0
            for son in node.next:
                val = is_query_parent(son)
                has_query_son = max(has_query_son, val)
            
            
    

            if node.fn == target: has_query_son = 1

            node.maintain = has_query_son

            node.is_parent = int(has_query_son and not node.is_target)

            return has_query_son

        def mark_subtree(node: SearchNode, parent = 0):
            has_parent = max(parent, node.fn == target)
            node.maintain = max(has_parent, node.maintain)
            if (has_parent and not node.is_target):
                node.is_subtree = 1 
            else: node.is_subtree =  0
            for son in node.next:
                assert isinstance(son, SearchNode), f"{son} is not SearchNode"
                mark_subtree(son, has_parent)


         

        is_query_node(init_node)
        is_query_parent(init_node)
        mark_subtree(init_node, parent = 0)
 

        nodes = []
        def display(node):
            if node.maintain:
                nodes.append(node)
            for son in node.next:
                display(son)
        
        display(init_node)

        return nodes

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
    
    def filter_cycle_canon(self, init_node: "SearchNode", query_fn: str) -> List["SearchNode"]:

        def dfs_track_fn(node: "SearchNode", fn_occurrences, path_query_count: int):
            if path_query_count >= 1 and node.fn == query_fn:
                node.maintain = 0  # do not maintain
                node.next = []
                node.next_weights = []
                return  # cut the path
        
            arg_types = value_types(node.value)
            new_query_count = path_query_count + 1 if node.fn == query_fn else path_query_count
        
            node.maintain = 1
        
            for son in node.next:
                assert isinstance(son, SearchNode), f"{son} is not a valid SearchNode"
                dfs_track_fn(son, fn_occurrences, new_query_count)
    

        dfs_track_fn(init_node, set(), path_query_count=0)

        valid_nodes = []
        def collect_valid_nodes(node: "SearchNode"):
            if node.maintain == 1:
                valid_nodes.append(node)
            for son in node.next:
                collect_valid_nodes(son)
    
        collect_valid_nodes(init_node)

        return valid_nodes
    
    def filter_cycle(self, init_node: "SearchNode", query_fn: str) -> List["SearchNode"]:
        cycle_found = False
        cycle_details = None
        
        def dfs_detect_cycle(node: "SearchNode", path_combined_keys: set, path_nodes: List[SearchNode]):
            nonlocal cycle_found, cycle_details
            if cycle_found: return
    
            arg_types = value_types(node.value)
            current_combined_key = f"{node.fn}_{arg_types}"
            is_query_fn = 1#node.fn == query_fn
            
            if is_query_fn and current_combined_key in path_combined_keys:
                cycle_found = True
                full_cycle_path = path_nodes + [node]
                cycle_details = {
                    "duplicate_combined_key": current_combined_key,
                    "cycle_path_length": len(full_cycle_path),
                    "cycle_path_nodes": [
                        f"{n.fn}({value_types(n.value)})" for n in full_cycle_path
                    ]
                }
                node.next = []
                node.next_weights = []


            new_path_keys = set(path_combined_keys)
            new_path_nodes = path_nodes + [node]
            new_path_keys.add(current_combined_key)
            

            for son in node.next:
                assert isinstance(son, SearchNode), f"{son} is not a valid SearchNode"
                dfs_detect_cycle(son, new_path_keys, new_path_nodes)
        
        def dfs_track_combined_key(node: "SearchNode", path_combined_keys: set, path_query_comb_count: int):

            arg_types = value_types(node.value)
            current_combined_key = f"{node.fn}_{arg_types}"
            is_query_fn = 1#node.fn == query_fn
            
            if is_query_fn and path_query_comb_count >= 1:
                node.maintain = 0
                node.next = []
                node.next_weights = []
                return
            
            new_path_keys = set(path_combined_keys)
            new_query_comb_count = path_query_comb_count
            
            if is_query_fn:
                new_path_keys.add(current_combined_key)
                new_query_comb_count += 1
            
            node.maintain = 1

            for son in node.next:
                assert isinstance(son, SearchNode), f"{son} is not a valid SearchNode"
                dfs_track_combined_key(son, new_path_keys, new_query_comb_count)
        
        dfs_detect_cycle(init_node, set(), [])
        

        if cycle_found and cycle_details:
            print("=" * 60)
            print("CYCLE DETECTED (duplicate query_fn+arg_types)")
            print(f"Duplicate combined key: {cycle_details['duplicate_combined_key']}")
            print(f"Cycle path length: {cycle_details['cycle_path_length']}")
            print(f"Full cycle path: {' -> '.join(cycle_details['cycle_path_nodes'])}")
            print("=" * 60)
        else:
            pass

        valid_nodes = []
        def collect_valid_nodes(node: "SearchNode"):
            if node.maintain == 1:
                valid_nodes.append(node)
            for son in node.next:
                collect_valid_nodes(son)
        
        collect_valid_nodes(init_node)
        return valid_nodes


    def rewrite_distr(self, value : List[Value], query_fn : str, mode = "eval") :
        """ return the rewrite paths from fn to target value
        Args:
            value : the input value to evaluate on
            fn    : the target fn to locate and evaluate
        Return:
            a distribution over search nodes
        """
        src_node = SearchNode("*", [], None, 1, None, distr = 0, reachable = 1.)
        src_node.next = []
        src_node.next_weights = []
        rw_nodes : List[SearchNode] = [src_node]

        """vertex of the rewrite search path"""
        self.vertex_count = -1
        rw_edges = []
        
        id_count = {}
        try:
            output_type = self.base_executor.function_signature(query_fn)[0][-1]
            functions = self.base_executor.gather_functions(value_types(value), output_type)

        except:
            print(query_fn,self.base_executor.function_signature(query_fn))
            raise NotImplementedError(f"{query_fn} is not found")
        for fn in functions:

            nd = SearchNode(fn, value, None, id = self.vertex_count + 1) # input node
            nd.next = []
            nd.next_weights = []
            subtree_nodes, id_count = self.rewrite_search_tree(nd, id_count = id_count)


            # filter cycle
            self.filter_cycle(subtree_nodes[0], query_fn)
                
            if mode == "eval": # select the subtree contains the query fn
                subtree_nodes = self.subtree_filter_target(subtree_nodes[0], query_fn)

            if mode == "update": # select the subtree that is reachable by margin
                subtree_nodes = self.subtree_filter_margin(subtree_nodes[0], subtree_nodes, margin = 0.01)
            
    
            if subtree_nodes: # not an empty subtree
                src_node.next.append(subtree_nodes[0]) # add the subtree head
                src_node.next_weights.append(1.)       # add the transition 1
                rw_nodes.extend(subtree_nodes)


        # each node weight to be successful reach exists and add the reach path
        def filter_path(node):
            """a node is keep only any of the three mask is correct"""
            sub_nodes = [nd for nd in node.next if (nd.is_parent or nd.is_subtree or nd.is_target)]
            node.next = sub_nodes

            has_reach = node.is_target
            for i,nd in enumerate(node.next):
                son_reached = filter_path(nd)
                has_reach = max(has_reach, son_reached * node.next_weights[i])
            
            return has_reach
            

        def distr(node : SearchNode, prob = 1.0):
            """exists a path to node and no output edge"""
            if node.next_weights:
                max_reach = torch.tensor(0.)
                for weight in node.next_weights:
                    weight = torch.tensor(weight) if not isinstance(weight, torch.Tensor) else weight
                    max_reach = torch.max(weight, max_reach)
                node.distr = prob * (1. - max_reach)
            else:
                node.distr = prob

            for son, weight in zip(node.next, node.next_weights):
                distr(son, prob * weight)

        def effective(query, node : SearchNode, has_query_parent):
            """effective nodes are below certain query node"""
            for son in node.next:
                effective(query, son, node.fn == query)
            if node.fn == query: node.distr *= 1.0
            else: node.distr *= float(has_query_parent)

        def kill(node): # kill the nodes with zero distr before mask
            next = [nd for nd in node.next if (nd.maintain > 0)]
            node.next = next
            for son in node.next:
                kill(son)
    
        def sum_weights(node):
            weight = float(node.distr)

            for nd in node.next:
                weight += sum_weights(nd)
            return weight

        def normalize(node, total):
            node.name = f"{node.fn}{node.id}({value_types(node.value)})"
            for son in node.next:
                normalize(son, total)
            node.distr = node.distr / total
        
        def ban_freeze(node : SearchNode,  ban_list):
            """freezed nodes should be banned"""
            for son in node.next:
                ban_freeze(son, ban_list)
            
            match = False
            for (fn, types) in ban_list:
                if fn == node.fn and types == value_types(node.value):
                    match = True

            if match: node.distr *= 0.0
            else: node.distr *= 1.0

        def exist_distr_path(node):
            """exists a path to a distribution node"""
            has_reach = node.distr
            for i,nd in enumerate(node.next):
                son_reached = exist_distr_path(nd)
                has_reach = max(has_reach, son_reached * node.next_weights[i])
            
            return has_reach
        

        self.subtree_filter_target(src_node,  query_fn)
        self.filter_cycle(src_node, query_fn) # filter self loops
        filter_path(src_node) # dfs on the super source node

        distr(src_node, prob = 1.0) # construct lastest node distr
        
        effective(query_fn, src_node, 0.0) # mask out parent nodes
        
        
        #kill(src_node)

        ban_freeze(src_node, self.non_effective_nodes)

        success_reach = exist_distr_path(src_node)

        total_weights = sum_weights(src_node) + 1e-6

        normalize(src_node, total_weights)

        if mode == "update":
            pass
        
        
        
        if len(rw_nodes) > self.cut and self.verbose:
            visualize_subtree(src_node, query_fn = query_fn)
            #print("DO This", len(rw_nodes), query_fn)
            pass 

        rw_nodes = [node for node in rw_nodes if node.name]

        return [(node.name, node.fn, node.value, node.distr) for node in rw_nodes], success_reach, rw_nodes, rw_edges

    #@timer(custom_desc="reduction call")
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
            execute_loss = 0.
            if isinstance(measure.value, Tuple):
                execute_loss = measure.value[1]
                measure.value = measure.value[0]

            #print(isinstance(measure, Value))
            if isinstance(measure.value, torch.Tensor) and (measure.vtype == INT or measure.vtype == FLOAT):
                expect_output += measure.value.reshape([-1])[0] * weight
            else:
                if isinstance(weight, torch.Tensor):
                    weight = weight.reshape([])
                if isinstance(measure.value, torch.Tensor):
                    if len(measure.value.shape) > 0:
                        b = measure.value.shape[0]
                        measure.value = measure.value.reshape([b, -1])
                #print(fn, weight)
                #print(weight, measure.value)
                expect_output +=  weight * measure.value

            def repr(v : Value):
                assert isinstance(v, Value), f"{v} is not a Value"
                if isinstance(v.value, torch.Tensor):
                    v = v.value
                    if sum(list(v.shape)) > 10:
                        return "Emb"+str(list(v.shape))
                    else:
                        return str(v)
                if isinstance(v.vtype, EmbeddingType):
                    while isinstance(v.value, Value):
                        v = v.value
                    return str(v.value)
                return str(v.value)

            input_repr = ",".join([repr(v) for v in vargs])
            serial_nodes.append(
                {   "id":node,
                    "fn": fn,
                    "value":str(f"{input_repr}->{measure.value}"),
                    "type": str(value_types(vargs))+"->"+str(measure.vtype),
                    "weight": str(weight)}
                )


        search_tree = {"nodes": serial_nodes, "edges":serial_edges}
        if len(rw_nodes) > self.cut and self.verbose:
            print("internal loss:", -torch.log(success_reach))
        return Value(measure.vtype, expect_output), -torch.log(success_reach) + execute_loss, search_tree

    """Evaluate Chain of Expressions"""
    def evaluate(self, expression, grounding):
        if not isinstance(expression, Expression):
            expression = self.parser.parse(expression)[0].fn


            expression = Expression.parse_program_string(expression)


        grounding = grounding if grounding is not None else {}

        self.node_count = 0
        self.prev_node  = {"id":"node0","fn" : "output_fn", "value": "output", "type": "type"}
                         
        self.eval_info = {
            "tree":{"nodes":[], "edges": []},
            "paths":{}} # tree, paths


        with self.with_grounding(grounding):
            outputs, loss, _, _ = self._evaluate(expression)
            #print("reach loss:", loss)

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

            nodes = [(node["id"], node["fn"], node["weight"]) for node in paths["nodes"]]
            #print("rewrite paths nodes",len(nodes),nodes)
            #print("\nrewrite paths edges:",paths["edges"])
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
        
        # 1. add the filler that defined on the value node.
        for (in_tps, out_tp) in self.base_executor.function_signature(fn):
            if len(args) != 1:
                in_tps = TupleType(value_types(args))
            else:
                in_tps = value_types(args)[0]
            convex_fillter = self.constructor.create_convex_construct(in_tps, out_tp, self.base_executor)
            assert convex_fillter, f"failed to create fillers {convex_fillter} for {fn} ({value_types(args)} -> {out_tp})"

            assert isinstance(self.base_executor, ExecutorGroup), "not a group executor"
            self.base_executor.register_extended_function(fn, value_types(args), out_tp, convex_fillter)
            self.logger.critical(f"registerd background extention for {fn} on {value_types(args)} -> {out_tp}")
            self.transient_background_functions.append([fn, value_types(args), out_tp])
    

        # 2. add rewriters that locate from the value reachable node to the fn node.
        execute_pairs, reach_loss, rw_nodes, rw_edges = self.rewrite_distr(args, fn, mode = "update")
        execute_pairs = execute_pairs[1:]

        if not execute_pairs: self.logger.info(f"execution not found for {fn}({value_types(args)})")

        # for each node that not in the 
        for (_,gn, vargs, weight) in execute_pairs:

            for (in_tps, out_tp) in self.base_executor.function_signature(fn):
                self_loop = 1.
                if self_loop and weight > 0:
                    """BAN ALL the automorphisms"""
                    arg_types = value_types(vargs)
                    src_tps = arg_types
                    tgt_tps = in_tps 

                    print(gn,arg_types,"->", fn, tgt_tps, weight)

                    if isinstance(src_tps, List): src_tps = TupleType(src_tps)
                    if isinstance(tgt_tps, List): tgt_tps = TupleType(tgt_tps)

                    caster = self.constructor.create_convex_arg_rewriter(
                        src_tps.element_types, tgt_tps.element_types, self.base_executor)

                    if not caster:
                        self.logger.warn(f"caster not found for {src_tps}->{tgt_tps}")

                    else:
                        learn0_frame = Frame(src_tps, tgt_tps, caster)
                        learn0_frame.matches[(f"{gn}@{fn}")] = torch.tensor(0.32)


                        src_sig = 'x'.join([str(tp) for tp in src_tps.element_types])
                        tgt_sig = 'x'.join([str(tp) for tp in tgt_tps.element_types])
                        frame_sig = f"frame:{src_sig}->{tgt_sig}_"
                        self.logger.critical(f"rewrite frame :{gn}({src_sig})->{fn}({tgt_sig})")
                        id_itr = 0
                        done = 0
                        while not done:
                            if (frame_sig + str(id_itr)) in self.frames:id_itr += 1
                            else: done = 1

                        self.frames[frame_sig + str(id_itr)] = learn0_frame

       
    

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
                
                from helchriss.utils.tensor import freeze
                freeze(self.base_executor.extended_registry, freeze = self.default_freeze)
                self.extended_non_effective(effective=False)
                self.logger.critical(f"update chain : {query_fn} -> {value_types(value)}, freezed:{self.default_freeze}\n")
                self.logger.critical(f"non effective nodes : extended registry")
                
                
            out = self.evaluate(query, grounding = grounding)
        return out
    
    def extended_non_effective(self, effective = False):
        if effective:
            self.non_effective_nodes = []
        else:
            self.non_effective_nodes
            for sign in self.base_executor.extended_registry:
                fn, in_types, out_type = self.base_executor.parse_signature(sign)

                self.non_effective_nodes.append([fn, in_types])
