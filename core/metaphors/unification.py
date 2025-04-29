from helchriss.dsl.dsl_types import TypeBase
from helchriss.dsl.dsl_values import Value
from typing import List, Mapping, Tuple, Any, Union, Dict
import torch
import torch.nn as nn
import networkx as nx

class LocalFrame(nn.Module):
    """store 1) a nn.ModuleDict of type self casters that adapts 2) a nn.ModuleDict that perform type casting to target function"""
    def __init__(self, func_name, natural_type : List[TypeBase], output_type : TypeBase):
        super().__init__()
        self.func_name : str = func_name
        self.output_type : TypeBase = output_type
        self.natural_types : List[TypeBase] = natural_type # this is a natural type, any new type need to be casted to this type to perform evaluation

        self.endo_casters = nn.ModuleDict() # how a new type map the the current natural type
        self.meta_casters = nn.ModuleDict() # how to map current to function to other function
    
    def add_endo_caster(self, neotype, caster): self.endo_casters[neotype] = caster
    
    def add_meta_caster(self, dest, caster): self.meta_casters[dest] = caster

    def compatible(self, input_args : List[Value]) -> List[List[Tuple[Value, Union[float, torch.Tensor]]]]:
        for arg in input_args:
            if  arg.vtype not in self.endo_casters: return False
        return True
        
    def endo_cast_args(self, args : List[Value]) -> List[Tuple[Value, Union[float, torch.Tensor]]]:
        """execute the current frame function using the input, if not natural type then transform."""
        targs = list()
        for i,arg in enumerate(args):
            if arg.vtype == self.natural_types[i]: # if no type cast, input is natural type
                tvalue = arg.value
                logp_weight = 0.0
            else: # if exist type cast, but input can be casted to the natural args
                assert arg.vtype in self.endo_casters, f"{arg} is not a valid input, cannot transform by self casters"
                tvalue, logp_weight = self.endo_casters[arg.vtype](arg.value)
            
            targs.append((Value(self.natural_types[i], tvalue),logp_weight))
        return targs

    def meta_cast_args(self, args : List[Value]) -> List[Tuple[str, List[Tuple[Value, torch.Tensor]] ] ]:
        meta_args = []
        for caster_name in self.meta_casters:
            caster = self.meta_casters[caster_name]
            meta_arg_values = caster(args)

            meta_args.append((caster_name, meta_arg_values))
        return meta_args


class ReductiveUnifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.frames = nn.ModuleDict({}) # local frames of functions

    def add_function_frame(self, name : str, frame : LocalFrame): self.frames[name] = frame

    def reduce_args(self, func : str, args : List[Value]) -> Tuple[List[Tuple[str, List[Value], Any]], Any]:
        """ gather possible reductions of the function and the corresponding weights (logits)
        Args:
            func : as the name of the function in a str
            args : as a list of values (typed)
        Returns:
            a list of distribution, representing the possible reduction, and corresponding args, and weights
            as return the Any as a graph of reduction
        """
        
        func_nodes = self.frames.keys()
        if not str(func) in func_nodes: return [(func, args, 1.0)], [[],[]] # there are no possible reductions


        init_frame : LocalFrame = self.frames[func]
        iterate_frames : List[LocalFrame] = [init_frame] # initalize the iterate frames

        # create the reduction graph during the iteration of the local frames
        reduce_nodes : set = {func}
        reduce_edges = []
        reduce_args_map = {func : args} # init with func and current args 

        """1. gather the possible reductions of the local frame"""
        while iterate_frames:

            curr_frame : LocalFrame = iterate_frames.pop(0) #TODO: BFS or DFS difference

            curr_func =  curr_frame.func_name # the current node name for evaluation
            reduced_frames = curr_frame.meta_cast_args(args)

            for reduce_frame in reduced_frames:
                reduce_func = reduce_frame[0]                             # the reduced func name


                reduce_args = [Value("tp",arg[0]) for arg in reduce_frame[1]]         # the list of args of reduce
                """TODO: Sigmoid or EXP??"""
                reduce_weight = torch.sigmoid(sum([arg[1] for arg in reduce_frame[1]]))  # the total reduce weights

                ### add the func as node and args as values and reduce weights as edge
                reduce_nodes.add(reduce_func)
                reduce_args_map[reduce_func] = reduce_args
                reduce_edges.append([curr_func, reduce_func,reduce_weight])
                if reduce_func not in reduce_nodes:
                    iterate_frames.append(self.frames[reduce_func]) ### add the current node to the queue


        reduce_nodes = list(reduce_nodes) # cast the node set
        # gain a reduce graph with nodes, edges and weights of the transform edges


        """2. weight each reduction by the far-reaching weight"""

        weights_map: Mapping[str, Any] = self.reduce_weight(func, reduce_nodes, reduce_edges)



        """3. collect (func, args, weights) tuples as lists of results"""
        reduced_binds = []
        for node in reduce_nodes:
            reduced_binds.append((node, reduce_args_map[node], weights_map[node]))
        # collect the `reduced_binds` as a List[Tuple[str, List[Value], torch.Tensor]]

        return reduced_binds, (reduce_nodes, reduce_edges)
    
    @staticmethod
    def reduce_weight(query : str,  nodes : List[str], edges : List[Tuple[str, str, Any]], inf = 1e6) -> Mapping[str, Any]:
        from helchriss.utils import stprint
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