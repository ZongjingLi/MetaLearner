from helchriss.dsl.dsl_types import TypeBase
from helchriss.dsl.dsl_values import Value
from typing import List, Mapping, Tuple, Any, Union, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
import networkx as nx
import os
from .types import BaseCaster

__all__ = ["RewriteRule", "LocalFrame", "NeuralRewriter"]

def pth_file(filename): return ":" in filename and filename[-4:] == ".pth"

@dataclass
class RewriteRule:
    left_func   : str
    right_func  : str
    func_weight : Union[float, torch.Tensor] # the logit of connection

    left_type   : List[TypeBase]
    right_type  : List[TypeBase]
    rewriter    : BaseCaster

    forced      : bool = False # if the rule exist then must apply and ignore the previous term, used for type casting
    _quantized  : bool = False # if quantized then the rule applied with binary cut

    @property
    def quantized(self): return self._quantized

    def apply(self, func : str, args : List[Value]) -> Tuple[str, Value, Union[float, torch.Tensor]]:
        if self.left_func != func: return None, None, None # cannot apply
        arg_type : List[TypeBase] = [arg.vtype for arg in args]
        if self.left_type != arg_type: return None, None, None

        results = self.rewriter(args)
        right_type = self.right_type
        rw_values : List[Value] = [Value(right_type[i],r[0]) for i,r in enumerate(results)]
        rw_weight : torch.Tensor  = sum([r[1] for r in results])

        return self.right_func, rw_values, rw_weight + self.func_weight
    
    def applicable(self, func : str, args : List[Value]) -> Union[bool, float, torch.Tensor]:
        _, _, prob = self.apply(func, args)
        if prob is None : return False
        if self.quantized: return prob >= 0.0
        return prob

    def applicable(self, weight, quantized = None) -> Union[bool, float, torch.Tensor]:
        if weight is None : return False
        if quantized:  return weight >= 0.0
        else : return weight
 

class LocalFrame(nn.Module):
    """ A local frame is a collection of rewrite rules that shares the same NeuralWriter
    store 1) a nn.ModuleDict of type self casters that adapts 2) a nn.ModuleDict that perform type casting to target function"""
    def __init__(self, func_name, natural_type : List[TypeBase], source_type : List[TypeBase], caster : nn.Module = None):
        super().__init__()
        self.func_name      : str               = func_name             # name of the function in the local frame
        self.natural_types  : List[TypeBase]    = natural_type          # natural input type of the function stored
        self.source_types  : List[TypeBase]    = source_type           # source type is mapped to the natural type and `g` defined on source

        assert caster is not None, "need to provide a fixed caster from the `g` to `f`"
        self.arg_writer     : nn.Module         =  caster               # how to map the `s` argument to the `t` argument
        self.func_mappers   : nn.ParameterDict  = nn.ParameterDict({})  # how to map the g function to the `f` function entailment

    def add_source_caster(self, dest : str, weight : Union[float, torch.Tensor] = 0.0):
        # dest : str the `g` function that can reduce to `f`
        if not isinstance(weight, torch.Tensor): self.func_mappers[dest] = nn.Parameter(torch.tensor(weight))
        else: self.func_mappers[dest] = nn.Parameter(weight)

    def rewrite_rules(self) -> List[RewriteRule]:
        rules : List[RewriteRule] = []

        for src in self.func_mappers:
            local_rule = RewriteRule(
                left_func   =    src,
                right_func  =    self.func_name,
                func_weight =    self.func_mappers[src],
                left_type   =    self.source_types,
                right_type  =    self.natural_types,
                rewriter    =    self.arg_writer,
                forced      =    src,
            )
            rules.append(local_rule)
        return rules

def type_suffix(args : List[Value]):
    arg_types = [f"{arg.vtype.typename}-{arg.vtype.alias}" for arg in args]
    if len(arg_types) == 0: return "#"
    else: return "#"+"->".join(arg_types)

class NeuralRewriter(nn.Module):
    def __init__(self):
        super().__init__()
        self.frames = nn.ModuleDict({}) # a bundle of rewrites that shares parameters for writer

    def save_ckpt(self, ckpt_path) -> int:
        if not os.path.exists(f"{ckpt_path}/frames"): os.makedirs(f"{ckpt_path}/frames")
        for frame_name in self.frames: torch.save(self.frames[frame_name], f"{ckpt_path}/frames/{frame_name}.pth")#.save_ckpt(f"{ckpt_path}/frames/{frame_name}")

    def load_ckpt(self, ckpt_path) -> int:
        frames_dir = f"{ckpt_path}/frames"
        for filename in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, filename)
            if os.path.isfile(file_path) and pth_file(filename):
                self.frames[filename[:-4]] = torch.load(file_path, weights_only = False)

    def add_frame(self, name : str, frame : LocalFrame): self.frames[name] = frame

    def add_rewrite_rule(self, f : str, g : str, f_type : List[TypeBase], g_type : List[TypeBase], writer = None):
        """add the rewrite rule by add a local frame with only (f,g)"""
        rewrite_frame = LocalFrame()
        self.add_frame(rewrite_frame)

    def rewrite_rules(self) -> List[RewriteRule]:
        rules = []
        for name in self.frames:
            frame : LocalFrame = self.frames[name]
            rules.extend(frame.rewrite_rules())
        return rules

    def rewrite_edges(self, q_func : str, args : List[Value]) -> List[Tuple[str, List[Value], torch.Tensor]]:
        rules : List[RewriteRule] = self.rewrite_rules()
        edges : List[Tuple[str, List[Value], Union[float, torch.Tensor]]] = []
        for rule in rules:
            tfunc, targs, weight = rule.apply(q_func, args)
            if rule.applicable(weight):
                edges.append((tfunc, targs, weight))
        return edges

    def rewrite_graph(self, q_func : str, args : List[Value], depth = None):
        rewrite_nodes = set()        #  nodes in the dfs(q) component generated by the query
        rewrite_edges = list()       #  edges stored the connection proposed by each local frame
        rewrite_args  = dict()       #  store List[Value] a dict containing the most probable dict at each args
        reduce_weight = dict()       #  store torch.Tensor a dict store current node max weight
        visited_nodes = set()

        stack = [(q_func, args, 1.0)] # should this a `dfs` or a `bfs`?
        while stack:
            node, args, weight = stack.pop(0) # a tuple of qfunc an arg

            suffix_node = node + type_suffix(args)
            #node = suffix_node


            rewrite_nodes.add(suffix_node) # add this as visited
            visited_nodes.add(suffix_node)# add this as visited
            
            # update the reduce args with the maxium 
            if suffix_node not in reduce_weight: reduce_weight[suffix_node] = -1e9
            if weight > reduce_weight[suffix_node]:
                rewrite_args[suffix_node] = args
                reduce_weight[suffix_node] = weight

            arged_edges = self.rewrite_edges(node, args)

            for (qfunc, args, weight) in arged_edges:
                suffix_qfunc = qfunc + type_suffix(args)
                if not suffix_qfunc in visited_nodes: # check if the node is visited

                    rewrite_edges.append([suffix_node, suffix_qfunc , weight])

                    stack.append((q_func, args, weight))
        
        return rewrite_nodes, rewrite_edges, rewrite_args
    
    def rewrite_distr(self, q_func : str, args : List[Value]) -> List[Tuple[str, List[Value], Any]]:
        ### 1. here we find the reduce_nodes, reduce_edges"""
        reduce_nodes, reduce_edges, reduce_args  = self.rewrite_graph(q_func, args)

        """the distribution over possible rewrirtes over f(x)"""
        ### 2. weight each reduction by the far-reaching weight
        q_func = q_func + type_suffix(args)

        weights_map: Mapping[str, Any] = reduce_weight(q_func, reduce_nodes, reduce_edges)


        ### 3. collect (func, args, weights) tuples as lists of results"""
        rewrite_distr = list()
        for node in reduce_nodes: rewrite_distr.append((node, reduce_args[node], weights_map[node]))
        # collect the `reduced_binds` as a List[Tuple[str, List[Value], torch.Tensor]]


        return rewrite_distr, (reduce_nodes, reduce_edges)

    def _display_dot_graph(self, rules: List[RewriteRule]) -> str:
        """Display graph in DOT format for Graphviz"""
        lines = [
        "digraph RewriteRules {",
        "    rankdir=LR;",
        "    node [shape=box, style=rounded];",
        ]

        nodes = set()
        for rule in rules:
            left_key = f"{rule.left_func}({self._format_types(rule.left_type)})"
            right_key = f"{rule.right_func}({self._format_types(rule.right_type)})"
            nodes.add(left_key)
            nodes.add(right_key)
    
        for i, node in enumerate(sorted(nodes)):
            node_id = f"n{i}"
            lines.append(f'    {node_id} [label="{node}"];')

        node_to_id = {node: f"n{i}" for i, node in enumerate(sorted(nodes))}
    
        for rule in rules:
            left_key = f"{rule.left_func}({self._format_types(rule.left_type)})"
            right_key = f"{rule.right_func}({self._format_types(rule.right_type)})"
        
            left_id = node_to_id[left_key]
            right_id = node_to_id[right_key]
        
            # Edge styling based on rule properties
            style_attrs = []
            if rule.forced: style_attrs.append('color="red"', 'penwidth=2')
            elif rule.quantized: style_attrs.append('style="dashed"')
        
            style_str = f", {', '.join(style_attrs)}" if style_attrs else ""
        
            lines.append(f'    {left_id} -> {right_id} [label="w:{rule.func_weight}"{style_str}];')
    
        lines.append("}")
        return "\n".join(lines)


def reduce_weight(query : str,  nodes : List[str], edges : List[Tuple[str, str, Any]]) -> Mapping[str, Any]:

        """ for a query node of an input graph, output each node is the most far reaching node
        Args:
            query : the query function name
            nodes : List[str] the vertices in the reduction graph
            edges: List[Tuple[str, str, Any]] is a list of edges with edge weight (AnyType)
        Returns:
            a dict that maps each function node to the probability of it is the most far reaching node (normalized probability)
        """
        """0. build the weighted graph for the reduction"""
        graph = {}
        for node in nodes: graph[node] = []

        for src, dst, prob in edges: graph[src].append((dst, torch.sigmoid(prob)))

        """1. delete all the backward edges to make it an DAG"""
        visited = set()
        in_current_path = set()
        dag_edges = []

    
        def dfs(node):
            if node in visited: return
            visited.add(node)
            in_current_path.add(node)
            for neighbor, weight in graph[node]:
                
                if neighbor not in in_current_path:
                    dag_edges.append((node, neighbor, weight))
                    dfs(neighbor)
        
            in_current_path.remove(node)

        dfs(query)

        """2. calculate the max probability of exists a path from query to each node"""
        dag = {}
        for node in nodes: dag[node] = []
        
        for src, dst, prob in dag_edges: dag[src].append((dst, prob))


        max_probs = {node: 0. for node in nodes}
        max_probs[query] = 1.0 # the query node can always be calculated
    
        # Topological sort before the dp for the strongest path
        topo_order = []
        visited = set()
    
        def topo_dfs(node):
            if node in visited: return
            visited.add(node)

            for neighbor, _ in dag[node]: topo_dfs(neighbor)
            topo_order.append(node)
    
        for node in nodes:
            if node not in visited: topo_dfs(node)
        topo_order.reverse()

        for node in topo_order: # perform dp to update the 
            for neighbor, prob in dag[node]:

                max_probs[neighbor] = max(max_probs[neighbor], max_probs[node] * prob)

        """3. calculate each node is the most far reaching """
        far_reaching_probs = {}
        for node in nodes:
            outgoing_p = 1.0
            for neighbor, prob in dag[node]:
                outgoing_p *= (1 - prob) # the p of no edge is going out    
            far_reaching_probs[node] = max_probs[node] * outgoing_p # the probability of this node is the most far reaching node

        """4. finally normalize the probability of each node is the most far reaching node"""
        total_prob = sum([p for p in far_reaching_probs.values()])
        for node in nodes: far_reaching_probs[node] = far_reaching_probs[node] / total_prob
        
        return far_reaching_probs

import matplotlib.pyplot as plt
import networkx as nx

def visualize_weight_reaching(query: str, nodes: List[str], edges: List[Tuple[str, str, Any]], probabilities: Dict[str, float]):
    """
    Visualize the graph and the weight reaching probabilities
    
    Args:
        query: the query node
        nodes: list of all nodes
        edges: list of edges (src, dst, weight)
        probabilities: dictionary mapping nodes to their far-reaching probabilities
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node)
    
    # Add edges
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 8))
    
    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Draw nodes with size proportional to their probability
    node_sizes = [probabilities.get(node, 0) * 3000 + 300 for node in G.nodes()]
    node_colors = ['red' if node == query else 'lightblue' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    # Add edge weights
    edge_labels = {(src, dst): f"{weight:.2f}" for src, dst, weight in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title(f"Weight Reaching from Query Node '{query}'")
    plt.axis('off')
    plt.tight_layout()
    
    # Add probability values as text
    for node, prob in probabilities.items():
        x, y = pos[node]
        plt.text(x, y - 0.1, f"{prob:.3f}", ha='center', fontsize=9)
    
    plt.show()