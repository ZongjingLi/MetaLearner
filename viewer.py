import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_node("Program")
G.add_node("Filter1")
G.add_node("Relate")
G.add_node("Filter2")
G.add_node("Query")

# Add edges
G.add_edge("Program", "Filter1")
G.add_edge("Filter1", "Relate")
G.add_edge("Relate", "Filter2")
G.add_edge("Filter2", "Query")

# Custom colors for nodes and edges
node_colors = ["lightblue", "lightgreen", "lightyellow", "lightgreen", "lightpink"]
edge_colors = ["black", "blue", "blue", "red"]

# Define layout
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color=node_colors,  # <-- node colors here
    edge_color=edge_colors,  # <-- edge colors here
    font_size=12,
    font_weight="bold"
)
plt.title("Program Flow with Custom Colors")
plt.show()
