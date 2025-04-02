import pybullet as p
import pybullet_data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def setup_physics_world():
    """Set up a basic PyBullet physics world with some objects."""
    p.connect(p.GUI)  # or p.DIRECT for headless mode
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load plane
    ground_id = p.loadURDF("plane.urdf")
    
    # Load some objects
    object_ids = {
        "ground": ground_id,
        "table": p.loadURDF("table/table.urdf", basePosition=[0, 0, 0]),
        "cube1": p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
            basePosition=[0, 0, 1]
        ),
        "cube2": p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
            basePosition=[0, 0, 1.3]
        ),
        "sphere": p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
            basePosition=[0.3, 0, 1]
        )
    }
    
    return object_ids

def get_contact_points(object_ids):
    """Get all contact points between objects in the simulation."""
    contacts = []
    
    # Get contacts for each pair of objects
    for name_a, id_a in object_ids.items():
        for name_b, id_b in object_ids.items():
            if id_a < id_b:  # Avoid duplicate checks
                contact_points = p.getContactPoints(id_a, id_b)
                if contact_points:
                    for point in contact_points:
                        normal_force = point[9]  # Normal force
                        contacts.append({
                            'object_a': name_a,
                            'object_b': name_b,
                            'id_a': id_a,
                            'id_b': id_b,
                            'position': point[5],  # Position on B
                            'normal': point[7],    # Normal from B to A
                            'normal_force': normal_force,
                            'distance': point[8],  # Contact distance, negative means penetration
                        })
    
    return contacts

def determine_support_relationships(contacts, gravity_direction=(0, 0, -1)):
    """
    Determine which objects are supporting others based on contact normals and gravity.
    
    Args:
        contacts: List of contact dictionaries
        gravity_direction: Direction of gravity as a unit vector
        
    Returns:
        Dictionary mapping supported objects to their supporters
    """
    gravity_direction = np.array(gravity_direction)
    gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)
    
    support_relations = defaultdict(list)
    
    for contact in contacts:
        normal = np.array(contact['normal'])
        
        # The dot product between normal and gravity determines support
        # If the normal points opposite to gravity, it's likely a support
        support_score = -np.dot(normal, gravity_direction)
        
        # Consider it a support if the normal has a component against gravity and there's a positive normal force
        if support_score > 0.3 and contact['normal_force'] > 0:  # Threshold for considering it support
            supporter = contact['object_a']
            supported = contact['object_b']
            
            # Check which object is on top based on contact normal direction
            if np.dot(normal, gravity_direction) > 0:
                supporter, supported = supported, supporter
                
            support_relations[supported].append({
                'supporter': supporter, 
                'position': contact['position'],
                'support_score': support_score,
                'normal_force': contact['normal_force']
            })
    
    return support_relations

def build_contact_graph(contacts):
    """Build a graph representation of object contacts."""
    G = nx.Graph()
    
    # Add nodes for all objects involved in contacts
    all_objects = set()
    for contact in contacts:
        all_objects.add(contact['object_a'])
        all_objects.add(contact['object_b'])
    
    for obj in all_objects:
        G.add_node(obj)
    
    # Add edges for contacts
    for contact in contacts:
        G.add_edge(
            contact['object_a'], 
            contact['object_b'], 
            position=contact['position'],
            normal=contact['normal'],
            normal_force=contact['normal_force'],
            distance=contact['distance']
        )
    
    return G

def build_support_graph(support_relations):
    """Build a directed graph representation of support relationships."""
    G = nx.DiGraph()
    
    # Add nodes for all objects
    all_objects = set()
    for supported, supporters in support_relations.items():
        all_objects.add(supported)
        for s in supporters:
            all_objects.add(s['supporter'])
    
    for obj in all_objects:
        G.add_node(obj)
    
    # Add directed edges from supporter to supported
    for supported, supporters in support_relations.items():
        for s in supporters:
            G.add_edge(
                s['supporter'], 
                supported, 
                position=s['position'],
                support_score=s['support_score'],
                normal_force=s['normal_force']
            )
    
    return G

def visualize_graph(G, title="Object Relationship Graph"):
    """Visualize the graph using matplotlib."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    nx.draw(
        G, 
        pos, 
        with_labels=True, 
        node_color='lightblue', 
        node_size=1500, 
        font_size=10, 
        font_weight='bold', 
        arrows=True if isinstance(G, nx.DiGraph) else False,
        edge_color='gray'
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_simulation_and_get_graphs():
    """Run the full simulation and generate the contact and support graphs."""
    object_ids = setup_physics_world()
    
    # Run simulation for a few steps to let objects settle
    for _ in range(800):
        p.stepSimulation()
    
    # Get contacts and build graphs
    contacts = get_contact_points(object_ids)
    support_relations = determine_support_relationships(contacts)
    
    contact_graph = build_contact_graph(contacts)
    support_graph = build_support_graph(support_relations)
    
    # Print the graph information
    print("\nContact Graph:")
    for u, v, data in contact_graph.edges(data=True):
        print(f"{u} contacts {v} with normal force: {data['normal_force']:.4f}")
    
    print("\nSupport Graph:")
    for u, v, data in support_graph.edges(data=True):
        print(f"{u} supports {v} with score: {data['support_score']:.4f}")
    
    # Visualize
    visualize_graph(contact_graph, "Contact Graph")
    visualize_graph(support_graph, "Support Graph (Direction: Supporter -> Supported)")
    
    return contact_graph, support_graph, contacts, support_relations

if __name__ == "__main__":
    contact_graph, support_graph, contacts, support_relations = run_simulation_and_get_graphs()
    
    # Disconnect when done
    input("Press Enter to disconnect...")
    p.disconnect()