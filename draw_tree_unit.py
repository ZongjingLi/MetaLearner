import networkx as nx
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import List, Any
from helchriss.dsl.dsl_values import Value
import torch
from core.metaphors.executor import visualize_subtree


@dataclass
class Node:
    """not the frame search node, the eval search node"""
    fn    : str               # executable fn on value
    value : List[Value]       # the value of the node, args of some function
    src   : None = None             # the source rewriter
    id    : int = 0
    maintain : bool = None           # the virtual node
    next : None = None
    next_weights : None = None
    distr : None = None                    # the distribution value on the search node
    reachable : None  = None           # whether the node is reachable
    is_subtree = 0
    is_parent = 0
    is_target = 0
    name : str = None
    sign : str = None



    def __str__(self): return f"{self.fn}"

    def __hash__(self): return hash(self.__str__())

def node_sn(node):
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

class Operator:
    
    def subtree_filter_target(self, init_node: Node, target: str) -> List[Node]:
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
            #print(node.fn, target)
            if node.fn == target :
                node.maintain = 1
                node.is_target = 1
            else :
                node.maintain = 0
                node.is_target = 0
            #print(node.fn,len(node.next))
            for son in node.next:
                is_query_node(son)
        
        def is_query_parent(node):
            assert isinstance(node, Node)
            has_query_son = 0
            for son in node.next:
                val = is_query_parent(son)
                has_query_son = max(has_query_son, val)
            
            
    

            if node.fn == target: has_query_son = 1

            node.maintain = has_query_son

            node.is_parent = int(has_query_son and not node.is_target)

            return has_query_son

        def mark_subtree(node: Node, parent = 0):
            has_parent = max(parent, node.fn == target)
            node.maintain = max(has_parent, node.maintain)
            if (has_parent and not node.is_target):
                node.is_subtree = 1 
            else: node.is_subtree =  0
            for son in node.next:
                assert isinstance(son, Node), f"{son} is not SearchNode"
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
    
    def subtree_filter_margin(self, init_node : Node, nodes : List[Node], margin = 0.1):
        def dfs(node : Node):
            for son, weight in zip(node.next, node.next_weights):
                assert isinstance(son, Node)
                son.reachable = node.reachable * weight
                if son.reachable >= margin:
                    son.maintain = 1
                    dfs(son)
                else: son.maintain = 0
        init_node.reachable = 1.0
        dfs(init_node)
        subtree_nodes = [node for node in nodes]
        return subtree_nodes
    
    def filter_cycle(self, init_node: "Node", query_fn: str) -> List["Node"]:
        cycle_found = False
        cycle_details = None
        
        def dfs_detect_cycle(node: "Node", path_combined_keys: set, path_nodes: List[Node]):
            nonlocal cycle_found, cycle_details
            if cycle_found: return
    
            current_combined_key = f"{node.fn}_{node.value}"
            is_query_fn = 1#node.fn == query_fn
            
            if is_query_fn and current_combined_key in path_combined_keys:
                cycle_found = True
                full_cycle_path = path_nodes + [node]
                cycle_details = {
                    "duplicate_combined_key": current_combined_key,
                    "cycle_path_length": len(full_cycle_path),
                    "cycle_path_nodes": [
                        f"{n.fn}({node.value})" for n in full_cycle_path
                    ]
                }
                node.next = []
                node.next_weights = []


            new_path_keys = set(path_combined_keys)
            new_path_nodes = path_nodes + [node]
            new_path_keys.add(current_combined_key)
            

            for son in node.next:
                assert isinstance(son, Node), f"{son} is not a valid SearchNode"
                dfs_detect_cycle(son, new_path_keys, new_path_nodes)
        
        def dfs_track_combined_key(node: "Node", path_combined_keys: set, path_query_comb_count: int):

            current_combined_key = f"{node.fn}_{node.value}"
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
            
            # 6. 递归遍历子节点，传递更新后的路径状态
            for son in node.next:
                assert isinstance(son, Node), f"{son} is not a valid SearchNode"
                dfs_track_combined_key(son, new_path_keys, new_query_comb_count)

        dfs_detect_cycle(init_node, set(), [])
        
        if cycle_found and cycle_details:
            print("=" * 60)
            print("🔴 CYCLE DETECTED (duplicate query_fn+arg_types)")
            print(f"Duplicate combined key: {cycle_details['duplicate_combined_key']}")
            print(f"Cycle path length: {cycle_details['cycle_path_length']}")
            print(f"Full cycle path: {' -> '.join(cycle_details['cycle_path_nodes'])}")
            print("=" * 60)
        else:
            pass

        valid_nodes = []
        def collect_valid_nodes(node: "Node"):
            if node.maintain == 1:
                valid_nodes.append(node)
            for son in node.next:
                collect_valid_nodes(son)
        
        collect_valid_nodes(init_node)
        return valid_nodes


    def rewrite_distr(self, value, query_fn, mode = "eval"):
        def exists_path(node : Node, success_reach, prev = None, wt = None):
            self.vertex_count += 1
            curr_vertex = f"vertex{self.vertex_count}"
        
            node.name = curr_vertex

            # touch a query node then return 
            if node.fn == query_fn:
                node.reachable = 1.0
                return 1.0
    
            
            if prev is not None:
               #print("ADD edge", prev, curr_vertex)
               rw_edges.append((prev, curr_vertex, str(wt)))
            
            
            # the node is reachable and have no applicable rewrite.
            
            success_reach = 1. if  node.fn == query_fn  else 0. # does not exists success reach
            for son, weight in zip(node.next, node.next_weights):
     
                if isinstance(weight, torch.Tensor):weight = weight.reshape([])
                assert isinstance(son, Node),son

                
                son_connect = exists_path(son, success_reach, prev = curr_vertex, wt = weight)
                success_reach = max(success_reach, weight * son_connect)
                

            node.maintain = success_reach
            return success_reach
        


def distr(node : Node, prob = 1.0):
    """exists a path to node and no output edge"""
    if node.next_weights:
        max_reach = torch.tensor(0.)
        for weight in node.next_weights:
            weight = torch.tensor(weight) if not isinstance(weight, torch.Tensor) else weight
            max_reach = torch.max(weight, max_reach)
        node.distr = prob * (1. - max_reach)
    else: node.distr = prob

    if node.next is None: node.next = []
    if node.next_weights is None: node.next_weights = []

    for son, weight in zip(node.next, node.next_weights):
        distr(son, prob * weight)

def effective(query, node : Node, has_query_parent):
    """effective nodes are below certain query node"""
    for son in node.next:
        effective(query, son, node.fn == query)
    if node.fn == query: node.distr *= 1.0
    else: node.distr *= float(has_query_parent)

def normalize(node, total):
    for son in node.next:
        normalize(son, total)
    node.distr = node.distr / total

def sum_weights(node):
    weight = float(node.distr)
    for nd in node.next:
        weight += sum_weights(nd)
    return weight

def kill(node): # kill the nodes with zero distr before mask
    next = [nd for nd in node.next if (nd.maintain > 0)]
    node.next = next
    for son in node.next:
        kill(son)

def exists_path(node):
    """a node is keep only any of the three mask is correct"""
    sub_nodes = [nd for nd in node.next if (nd.is_parent or nd.is_subtree or nd.is_target)]
    node.next = sub_nodes
    
    has_reach = node.is_target
    for i,nd in enumerate(node.next):
        son_reached = exists_path(nd)
        #print(son_reached * node.next_weights[i])
        has_reach = max(has_reach, son_reached * node.next_weights[i])

    return has_reach

def add_node(node, weight,fn,nodes, tp = None, id = None):
    ref_nodes = []
    for nd in nodes:
        if (fn == nd.fn) and (tp == None or tp == nd.value) and (id == None or nd.id == id):
            ref_nodes.append(nd)

    for ref_nd in ref_nodes:
        if ref_nd.next is None: ref_nd.next = []
        if ref_nd.next_weights is None: ref_nd.next_weights = []

        ref_nd.next.append(node)
        ref_nd.next_weights.append(weight)

    if ref_nodes:
        nodes.append(node)
    else:
        print("not found node in nodes")
    return nodes

def ban_nodes(x, query, value_types):
    return

if __name__ == "__main__":
    op = Operator()

    n0 = Node("*", "t0")
    nodes = [n0]
    
    nodes = add_node(Node("n1", "t0", next = []), 1., "*", nodes = nodes)
    nodes = add_node(Node("n2", "t0", next = []), 1., "*", nodes = nodes)
    nodes = add_node(Node("n3", "t0", next = []), 1., "*", nodes = nodes)
    nodes = add_node(Node("gn", "t2", next = []), .7, "n2", nodes = nodes)
    nodes = add_node(Node("gn", "t1", next = []), .6, "n3", nodes = nodes)
    nodes = add_node(Node("f0", "t2", next = []), .8, "gn", tp = "t1", nodes = nodes)
    nodes = add_node(Node("f1", "t3", next = []), .7, "f0", nodes = nodes)
    nodes = add_node(Node("f2", "t3", next = []), .2, "f0", nodes = nodes)



    nodes = op.subtree_filter_target(nodes[0], "gn")
    op.filter_cycle(nodes[0], "gn")
    print(exists_path(nodes[0]))
    distr(nodes[0], 1.)
    #effective("gn",nodes[0], 0)

    total = sum_weights(nodes[0])
    normalize(nodes[0], total)
    

    for nd in nodes: nd.sign = nd.fn + f"({nd.value})" + f"\n{nd.distr:.3f}\n{(nd.is_target, nd.is_parent, nd.is_subtree)}"

    visualize_subtree(n0, query_fn = "gn", font_size= 7)
