from helchriss.dsl.dsl_types import TypeBase
from helchriss.dsl.dsl_values import Value
from typing import List, Mapping, Tuple, Any, Union, Dict
import torch
import torch.nn as nn
import networkx as nx
import os

def pth_file(filename): return ":" in filename and filename[-4:] == ".pth"

class LocalFrame(nn.Module):
    """store 1) a nn.ModuleDict of type self casters that adapts 2) a nn.ModuleDict that perform type casting to target function"""
    def __init__(self, func_name, natural_type : List[TypeBase], output_type : TypeBase, source_type : List[TypeBase], caster : nn.Module = None):
        super().__init__()
        self.func_name      : str               = func_name             # name of the function in the local frame
        self.output_type    : TypeBase          = output_type           # natural output type of the fuction stored
        self.natural_types  : List[TypeBase]    = natural_type          # natural input type of the function stored
        self.source_types  : List[TypeBase]    = source_type           # source type is mapped to the natural type and `g` defined on source

        assert caster is not None, "need to provide a fixed caster from the `g` to `f`"
        self.arg_caster     : nn.Module         =  caster               # how to map the `s` argument to the `t` argument
        self.func_mappers   : nn.ParameterDict  = nn.ParameterDict({})  # how to map the g function to the `f` function entailment

    def add_source_caster(self, dest : str, weight : Union[float, torch.Tensor] = 0.0):
        # dest : str the `g` function that can reduce to `f`
        if not isinstance(weight, torch.Tensor): self.func_mappers[dest] = nn.Parameter(torch.tensor(weight))
        else: self.func_mappers[dest] = nn.Parameter(weight)

class ReductiveUnifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.frames = nn.ModuleDict({}) # local frames of functions

    def save_ckpt(self, ckpt_path) -> int:
        if not os.path.exists(f"{ckpt_path}/frames"): os.makedirs(f"{ckpt_path}/frames")
        for frame_name in self.frames: torch.save(self.frames[frame_name], f"{ckpt_path}/frames/{frame_name}.pth")#.save_ckpt(f"{ckpt_path}/frames/{frame_name}")
        return 0

    def load_ckpt(self, ckpt_path) -> int:
        frames_dir = f"{ckpt_path}/frames"
        for filename in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, filename)
            if os.path.isfile(file_path) and pth_file(filename):
                self.frames[filename[:-4]] = torch.load(file_path, weights_only = False)
        return 0

    def add_function_frame(self, name : str, frame : LocalFrame): self.frames[name] = frame

    def edges(self, query) -> List[Tuple[str, torch.Tensor, nn.Module]]:
        """for the query node given `query` find all the tuples of """
        edges = []
        for _, frame in self.frames.items():
            assert isinstance(frame, LocalFrame), f"{frame} is not a local frame"
            if query in frame.func_mappers:
                dest_func   : str           =  frame.func_name
                dest_weight : torch.Tensor  =  torch.sigmoid(frame.func_mappers[query])
                dest_caster : nn.Module     =  frame.arg_caster
                dest_type   : str           =  frame.natural_types
                edges.append([dest_func, dest_type, dest_weight, dest_caster])
        return edges
    
    def arged_edges(self, query, args : List[Value]) -> List[Tuple[str, List[Value], torch.Tensor]]:
        edges = self.edges(query)
        arged_edges = []
        for edge in edges:
            dest_func, dest_types, func_weight, caster = edge
            mapped_args    =   caster(args)
            reduce_args    =   [Value(dest_types[i],arg[0]) for i,arg in enumerate(mapped_args)] # the list of args of reduce
            arg_weight     =   torch.sigmoid(sum([arg[1] for arg in mapped_args]))  # the total reduce weights
            arged_edges.append([dest_func, reduce_args, arg_weight * func_weight])
        return arged_edges
    
    def valued_edges(self, args : List[Value]) -> List[Tuple[List[Value], torch.Tensor]]:
        arg_type : List[TypeBase] = [v.vtype for v in args]
        map_args = []
        for _, frame in self.frames.items():
            assert isinstance(frame, LocalFrame),  f"{frame} is not a local frame"
            if frame.source_types == arg_type:
                dest_types     = frame.natural_types
                caster         = frame.arg_caster
                mapped_args    =   caster(args)

                reduce_args    =   [Value(dest_types[i],arg[0]) for i,arg in enumerate(mapped_args)]
                arg_weight     =   torch.sigmoid(sum([arg[1] for arg in mapped_args]))  # the total reduce weights

                map_args.append([reduce_args, arg_weight])
        return map_args

    
    def type_cast(self, values: List[Value], vtypes: List[TypeBase]) -> Tuple[List[Value], torch.Tensor, Any]:
        """Search a path from source types to target types by local transforms.
        
        Args:
            values: List of input values to be cast
            vtypes: List of target types to cast to
            
        Returns:
            Tuple containing:
            - List of casted values
            - Probability (confidence) of the cast
            - Info dictionary containing leaf nodes and their probabilities
        """
        info = {"leaf_nodes": {}}

        if all(value.vtype == vtype for value, vtype in zip(values, vtypes)):
            info["leaf_nodes"] = {tuple(v.vtype.alias for v in values): 1.0}
            return values, 1.0, info
        
        # Use BFS to find transformation paths
        queue = [(values, 1.0)]
        visited = set(tuple(v.vtype.alias for v in values))
        max_prob = 0.0
        best_values = values
        
        while queue:
            current_values, current_prob = queue.pop(0)
            
            # Check if current values match target types
            if all(TypeBase.downcast_compatible(value.vtype,vtype) for value, vtype in zip(current_values, vtypes)):
                type_key = tuple(value.vtype.alias for value in current_values)
                info["leaf_nodes"][type_key] = current_prob
                return current_values, current_prob, info
            
            transformations = self.valued_edges(current_values)

            for new_values, transform_prob in transformations:
                new_type_key = tuple(value.vtype.alias for value in new_values)

                if new_type_key in visited or transform_prob < 0.01:
                    continue
                
                visited.add(new_type_key)
                new_prob = current_prob * transform_prob
                
                # Check if this is a leaf node (can't be transformed further)
                if len(self.valued_edges(new_values)) == 0:
                    info["leaf_nodes"][new_type_key] = new_prob
                    
                    # Update best match if this is closer to target
                    match_count = sum(1 for value, vtype in zip(new_values, vtypes) if value.vtype == vtype)
                    if match_count > 0 and new_prob > max_prob:
                        max_prob = new_prob
                        best_values = new_values

                queue.append((new_values, new_prob))
        
        # If we reach here, no complete path found to target types
        # Return the best partial match if any leaf nodes were found
        if info["leaf_nodes"]: return best_values, max_prob, info

        return values, 0.0, info

    def reduce_args(self, q_func : str, args : List[Value]) -> Tuple[List[Tuple[str, List[Value], Any]], Any]:
        """ gather possible reductions of the function and the corresponding weights (logits)
        Args:
            func : as the name of the function in a str
            args : as a list of values (typed)
        Returns:
            a list of distribution, representing the possible reduction, and corresponding args, and weights
            as return the Any as a graph of reduction
        """
        func_nodes = self.frames.keys()

        """1. here we find the reduce_nodes, reduce_edges"""
        reduce_nodes = set()        #  nodes in the dfs(q) component generated by the query
        reduce_edges = list()       #  edges stored the connection proposed by each local frame
        reduce_args = dict()        #  store List[Value] a dict containing the most probable dict at each args
        reduce_weight = dict()     #  store torch.Tensor a dict store current node max weight
        visited_nodes = set()

        stack = [(q_func, args, 1.0)] # should this a `dfs` or a `bfs`?
        while stack:
            node, args, weight = stack.pop(0) # a tuple of qfunc an arg
            reduce_nodes.add(node) # add this as visited
            visited_nodes.add(node)# add this as visited
            
            # update the reduce args with the maxium 
            if node not in reduce_weight: reduce_weight[node] = 0.0
            if weight > reduce_weight[node]:
                reduce_args[node] = args
                reduce_weight[node] = weight

            arged_edges = self.arged_edges(node, args)
            for (qfunc, args, weight) in arged_edges:
                if not qfunc in visited_nodes: # check if the node is visited

                    reduce_edges.append([node, qfunc, weight])
                    #else: reduce_edges[node] = [(node, qfunc, weight)]
                    stack.append((qfunc, args, weight))

        """2. weight each reduction by the far-reaching weight"""
        weights_map: Mapping[str, Any] = self.reduce_weight(q_func, reduce_nodes, reduce_edges)


        """3. collect (func, args, weights) tuples as lists of results"""
        reduced_binds = list()
        for node in reduce_nodes: reduced_binds.append((node, reduce_args[node], weights_map[node]))
        # collect the `reduced_binds` as a List[Tuple[str, List[Value], torch.Tensor]]
        return reduced_binds, (reduce_nodes, reduce_edges)

    @staticmethod
    def reduce_weight(query : str,  nodes : List[str], edges : List[Tuple[str, str, Any]], inf = 1e6) -> Mapping[str, Any]:

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

        for src, dst, prob in edges: graph[src].append((dst, prob))

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