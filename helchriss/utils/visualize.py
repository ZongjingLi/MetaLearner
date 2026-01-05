import numpy as np

def visualize_pointclouds():
    return 

def visualize_image_batch():
    # [B,W,H,C]
    return 

from termcolor import colored
import networkx as nx

class Colors:
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

def stmetaphors(metaphors) -> str:
    output = f"infering from {len(metaphors)} mismatch expressions\n"
    for rewrites in metaphors:
        if len(rewrites) == 0 : output += "expression literally makes sense, no metaphors detected.\n"
        if len(rewrites) == 2:  output += "expression suggest an extention, but no short cut inferred.\n"
        ### len == 2 suggests only extended version and origonal version is possible.
        else: output += "expression suggest extention and potential rewrite short cuts.\n"

        if len(rewrites) > 0: ### show the extention of the current concept.
            extend = rewrites[0]
            fname, domain = extend[0].split(":")
            fname = colored(fname,"cyan")
            s_domain = colored(domain,"blue", attrs=["bold"])
            stype = [str(t) for t in extend[1]]
            ttype = [str(t) for t in extend[2]]
            output += f"{fname} \
of domain {s_domain} defined on \
{Colors.SARONITE}{stype}{Colors.RESET} might be extended to \
{Colors.SARONITE}{ttype}{Colors.RESET}\n"

        if len(rewrites) > 2: ### show the short cut created
            short_cuts = [
            rewrite[0].split(":")[0] for rewrite in rewrites[2:]]
            _, domain = rewrites[-1][0].split(":")
            t_domain = colored(domain, "blue", attrs = ["bold"])

            output += f"{colored(short_cuts,'cyan')}\n of domain {t_domain} defined on \
{Colors.SARONITE}{ttype}{Colors.RESET} might be rewrite to {fname} of domain {s_domain} defined on \
{Colors.SARONITE}{stype}{Colors.RESET}\n"
        output += "\n"
    return output








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
