import pybullet as p
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class ContactModel:
    """
    A class to build and analyze contact and support relationships in a PyBullet environment.
    Includes contact and support tensors and object attribute tracking.
    """
    
    def __init__(self, gravity_direction=(0, 0, -1)):
        """
        Initialize the contact model.
        
        Args:
            gravity_direction: Direction of gravity as a unit vector (default: down)
        """
        self.object_registry = {}
        self.object_attributes = {}  # Store object attributes
        self.gravity_direction = np.array(gravity_direction)
        self.gravity_direction = self.gravity_direction / np.linalg.norm(self.gravity_direction)
        
        # Store the latest results
        self.contacts = []
        self.support_relations = {}
        self.contact_graph = None
        self.support_graph = None
        
        # Tensors (matrices)
        self.contact_tensor = None
        self.support_tensor = None
        self.object_names = []  # List to track object names in order for tensor indices
    
    def register_object(self, name, object_id, attributes=None):
        """
        Register an object with a name for tracking.
        
        Args:
            name: Name of the object
            object_id: PyBullet ID of the object
            attributes: Dictionary of additional attributes to store
        """
        self.object_registry[name] = object_id
        
        # Add to ordered list if not already present
        if name not in self.object_names:
            self.object_names.append(name)
        
        # Store attributes
        if attributes is None:
            attributes = {}
        
        # Get basic attributes from PyBullet
        try:
            pos, orn = p.getBasePositionAndOrientation(object_id)
            dynamics_info = p.getDynamicsInfo(object_id, -1)  # -1 for base link
            
            # Extract useful properties
            mass = dynamics_info[0]
            friction = dynamics_info[1]
            
            # Store basic physical attributes
            basic_attrs = {
                'position': pos,
                'orientation': orn,
                'mass': mass,
                'friction': friction
            }
            
            # Merge with provided attributes
            attributes.update(basic_attrs)
        except:
            # If we can't get dynamics info (e.g., for non-base objects), just use provided attributes
            pass
        
        self.object_attributes[name] = attributes
        
        # Reset tensors since objects have changed
        self.contact_tensor = None
        self.support_tensor = None
        
        return self
    
    def register_objects(self, object_dict, attributes_dict=None):
        """
        Register multiple objects.
        
        Args:
            object_dict: Dictionary mapping names to object IDs
            attributes_dict: Optional dictionary mapping names to attribute dictionaries
        """
        if attributes_dict is None:
            attributes_dict = {}
            
        for name, obj_id in object_dict.items():
            attrs = attributes_dict.get(name, {})
            self.register_object(name, obj_id, attrs)
            
        return self
    
    def get_contact_points(self):
        """Get all contact points between objects in the simulation."""
        contacts = []
        
        # Get contacts for each pair of objects
        for name_a, id_a in self.object_registry.items():
            for name_b, id_b in self.object_registry.items():
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
        
        self.contacts = contacts
        
        # Reset tensors since contacts have changed
        self.contact_tensor = None
        self.support_tensor = None
        
        return contacts
    
    def determine_support_relationships(self, min_support_score=0.7, min_normal_force=0.01):
        """
        Determine which objects are supporting others based on contact normals and gravity.
        
        Args:
            min_support_score: Minimum dot product between normal and gravity for support
            min_normal_force: Minimum normal force for considering support
            
        Returns:
            Dictionary mapping supported objects to their supporters
        """
        if not self.contacts:
            self.get_contact_points()
            
        support_relations = defaultdict(list)
        
        for contact in self.contacts:
            normal = np.array(contact['normal'])
            
            # The dot product between normal and gravity determines support
            # If the normal points opposite to gravity, it's likely a support
            support_score = np.dot(normal, self.gravity_direction)
            
            # Consider it a support if the normal has a component against gravity and there's a positive normal force
            if support_score >= min_support_score and contact['normal_force'] > min_normal_force:
                supporter = contact['object_a']
                supported = contact['object_b']
                
                # Check which object is on top based on contact normal direction
                if np.dot(normal, self.gravity_direction) > 0:
                    supporter, supported = supported, supporter
                    
                support_relations[supported].append({
                    'supporter': supporter, 
                    'position': contact['position'],
                    'support_score': support_score,
                    'normal_force': contact['normal_force']
                })
        
        self.support_relations = support_relations
        
        # Reset tensors since support relations have changed
        self.support_tensor = None
        
        return support_relations
    
    def build_contact_graph(self):
        """Build a graph representation of object contacts."""
        if not self.contacts:
            self.get_contact_points()
            
        G = nx.Graph()
        
        # Add nodes for all objects involved in contacts
        all_objects = set()
        for contact in self.contacts:
            all_objects.add(contact['object_a'])
            all_objects.add(contact['object_b'])
        
        for obj in all_objects:
            G.add_node(obj)
        
        # Add edges for contacts
        for contact in self.contacts:
            G.add_edge(
                contact['object_a'], 
                contact['object_b'], 
                position=contact['position'],
                normal=contact['normal'],
                normal_force=contact['normal_force'],
                distance=contact['distance']
            )
        
        self.contact_graph = G
        return G
    
    def build_support_graph(self):
        """Build a directed graph representation of support relationships."""
        if not self.support_relations:
            self.determine_support_relationships()
            
        G = nx.DiGraph()
        
        # Add nodes for all objects
        all_objects = set()
        for supported, supporters in self.support_relations.items():
            all_objects.add(supported)
            for s in supporters:
                all_objects.add(s['supporter'])
        
        for obj in all_objects:
            G.add_node(obj)
        
        # Add directed edges from supporter to supported
        for supported, supporters in self.support_relations.items():
            for s in supporters:
                G.add_edge(
                    s['supporter'], 
                    supported, 
                    position=s['position'],
                    support_score=s['support_score'],
                    normal_force=s['normal_force']
                )
        
        self.support_graph = G
        return G
    
    def build_contact_tensor(self):
        """
        Build a contact tensor (matrix) representation.
        
        Returns:
            numpy.ndarray: 2D matrix where tensor[i, j] is the normal force between objects i and j
        """
        n = len(self.object_names)
        tensor = np.zeros((n, n))
        
        if not self.contacts:
            self.get_contact_points()
            
        # Build contact matrix
        for contact in self.contacts:
            obj_a = contact['object_a']
            obj_b = contact['object_b']
            
            # Get indices
            i = self.object_names.index(obj_a)
            j = self.object_names.index(obj_b)
            
            # Set matrix values (normal force)
            # Make it symmetric
            tensor[i, j] = contact['normal_force']
            tensor[j, i] = contact['normal_force']
            
        self.contact_tensor = tensor
        return tensor
    
    def build_support_tensor(self):
        """
        Build a support tensor (matrix) representation.
        
        Returns:
            numpy.ndarray: 2D matrix where tensor[i, j] = support score if i supports j, 0 otherwise
        """
        n = len(self.object_names)
        tensor = np.zeros((n, n))
        
        if not self.support_relations:
            self.determine_support_relationships()
            
        # Build support matrix
        for supported, supporters in self.support_relations.items():
            if supported in self.object_names:
                j = self.object_names.index(supported)
                
                for s in supporters:
                    supporter = s['supporter']
                    if supporter in self.object_names:
                        i = self.object_names.index(supporter)
                        
                        # supporter -> supported (i -> j)
                        tensor[i, j] = s['support_score']
            
        self.support_tensor = tensor
        return tensor
    
    def get_contact_tensor(self):
        """Get the contact tensor, building it if necessary."""
        if self.contact_tensor is None:
            self.build_contact_tensor()
        return self.contact_tensor
    
    def get_support_tensor(self):
        """Get the support tensor, building it if necessary."""
        if self.support_tensor is None:
            self.build_support_tensor()
        return self.support_tensor
    
    def get_object_attributes(self, name=None):
        """
        Get attributes for a specific object or all objects.
        
        Args:
            name: Name of the object or None to get all attributes
            
        Returns:
            Dictionary of attributes for the specified object or all objects
        """
        if name is not None:
            return self.object_attributes.get(name, {})
        return self.object_attributes
    
    def update_object_attributes(self):
        """
        Update object attributes with current physics state.
        """
        for name, obj_id in self.object_registry.items():
            try:
                pos, orn = p.getBasePositionAndOrientation(obj_id)
                
                # Update position and orientation
                if name in self.object_attributes:
                    self.object_attributes[name]['position'] = pos
                    self.object_attributes[name]['orientation'] = orn
            except:
                # Skip if can't get position/orientation
                pass
        
        return self
    
    def get_attributes_list(self, attribute_name):
        """
        Get a list of a specific attribute for all objects.
        
        Args:
            attribute_name: Name of the attribute to extract
            
        Returns:
            List of values for the specified attribute, in the same order as object_names
        """
        result = []
        
        for name in self.object_names:
            attrs = self.object_attributes.get(name, {})
            value = attrs.get(attribute_name, None)
            result.append(value)
            
        return result
    
    def set_object_attribute(self, name, attribute_name, value):
        """
        Set an attribute for a specific object.
        
        Args:
            name: Name of the object
            attribute_name: Name of the attribute to set
            value: Value to set
        """
        if name in self.object_attributes:
            self.object_attributes[name][attribute_name] = value
        else:
            self.object_attributes[name] = {attribute_name: value}
        
        return self
    
    def visualize_graph(self, graph=None, title="Object Relationship Graph", figsize=(10, 8)):
        """
        Visualize a graph using matplotlib.
        
        Args:
            graph: The graph to visualize (defaults to contact_graph if None)
            title: Title for the plot
            figsize: Figure size as (width, height)
        """
        if graph is None:
            if self.contact_graph is not None:
                graph = self.contact_graph
            elif self.support_graph is not None:
                graph = self.support_graph
            else:
                self.build_contact_graph()
                graph = self.contact_graph
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(graph)
        
        nx.draw(
            graph, 
            pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=1500, 
            font_size=10, 
            font_weight='bold', 
            arrows=True if isinstance(graph, nx.DiGraph) else False,
            edge_color='gray'
        )
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_tensor(self, tensor=None, title="Contact/Support Tensor", figsize=(10, 8)):
        """
        Visualize a tensor as a heatmap using matplotlib.
        
        Args:
            tensor: The tensor to visualize (defaults to contact_tensor if None)
            title: Title for the plot
            figsize: Figure size as (width, height)
        """
        if tensor is None:
            if self.contact_tensor is not None:
                tensor = self.contact_tensor
            elif self.support_tensor is not None:
                tensor = self.support_tensor
            else:
                self.build_contact_tensor()
                tensor = self.contact_tensor
        
        plt.figure(figsize=figsize)
        plt.imshow(tensor, cmap='viridis')
        plt.colorbar(label='Force/Support Score')
        
        # Add object names as labels
        plt.xticks(range(len(self.object_names)), self.object_names, rotation=45)
        plt.yticks(range(len(self.object_names)), self.object_names)
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def print_contact_info(self):
        """Print information about contacts."""
        if not self.contact_graph:
            self.build_contact_graph()
            
        print("\nContact Graph:")
        for u, v, data in self.contact_graph.edges(data=True):
            print(f"{u} contacts {v} with normal force: {data['normal_force']:.4f}")
    
    def print_support_info(self):
        """Print information about support relationships."""
        if not self.support_graph:
            self.build_support_graph()
            
        print("\nSupport Graph:")
        for u, v, data in self.support_graph.edges(data=True):
            print(f"{u} supports {v} with score: {data['support_score']:.4f}")
    
    def get_objects_below(self, object_name):
        """
        Get all objects that are below (supporting) the given object.
        
        Args:
            object_name: Name of the object to check
            
        Returns:
            List of object names that are below/supporting the given object
        """
        if not self.support_graph:
            self.build_support_graph()
            
        supporters = []
        
        # Check if object is in the graph
        if object_name in self.support_graph:
            # Get all incoming edges (supporters)
            for u, v in self.support_graph.in_edges(object_name):
                supporters.append(u)
                
        return supporters

    def unregister_object(self, name):
        """
        Unregister an object from the contact model.
        
        Args:
            name: Name of the object to unregister
            
        Returns:
            True if object was unregistered, False otherwise
        """
        if name not in self.object_registry:
            print(f"No object with name '{name}' found in registry")
            return False
        
        # Remove from registry
        del self.object_registry[name]
        
        # Remove from attributes
        if name in self.object_attributes:
            del self.object_attributes[name]
        
        # Remove from object names list
        if name in self.object_names:
            self.object_names.remove(name)
        
        # Reset tensors and graphs since objects have changed
        self.contact_tensor = None
        self.support_tensor = None
        self.contact_graph = None
        self.support_graph = None
        
        return True

    def get_objects_above(self, object_name):
        """
        Get all objects that are above (supported by) the given object.
        
        Args:
            object_name: Name of the object to check
            
        Returns:
            List of object names that are above/supported by the given object
        """
        if not self.support_graph:
            self.build_support_graph()
            
        supported = []
        
        # Check if object is in the graph
        if object_name in self.support_graph:
            # Get all outgoing edges (supported objects)
            for u, v in self.support_graph.out_edges(object_name):
                supported.append(v)
                
        return supported
    
    def analyze(self, visualize=True, include_tensors=True):
        """
        Run the full analysis pipeline and return the results.
        
        Args:
            visualize: Whether to show the visualization
            include_tensors: Whether to include tensor visualizations
            
        Returns:
            Dictionary containing analysis results
        """
        self.get_contact_points()
        self.determine_support_relationships()
        self.build_contact_graph()
        self.build_support_graph()
        self.build_contact_tensor()
        self.build_support_tensor()
        self.update_object_attributes()
        
        self.print_contact_info()
        self.print_support_info()
        
        if visualize:
            self.visualize_graph(self.contact_graph, "Contact Graph")
            self.visualize_graph(self.support_graph, "Support Graph (Direction: Supporter -> Supported)")
            
            if include_tensors:
                self.visualize_tensor(self.contact_tensor, "Contact Tensor")
                self.visualize_tensor(self.support_tensor, "Support Tensor")
        
        return {
            'contact_graph': self.contact_graph,
            'support_graph': self.support_graph,
            'contact_tensor': self.contact_tensor,
            'support_tensor': self.support_tensor,
            'object_attributes': self.object_attributes,
            'contacts': self.contacts,
            'support_relations': self.support_relations
        }


# Example of how to use the enhanced ContactModel
if __name__ == "__main__":
    import pybullet as p
    import pybullet_data
    import time
    
    # Start PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Create some objects
    ground_id = p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
    cube1_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
        basePosition=[0, 0, 1]
    )
    cube2_id = p.createMultiBody(
        baseMass=0.5,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
        basePosition=[0, 0, 1.3]
    )
    
    # Register objects with the contact model
    contact_model = ContactModel()
    
    # Register with custom attributes
    contact_model.register_object("ground", ground_id, {"type": "ground", "movable": False})
    contact_model.register_object("table", table_id, {"type": "furniture", "movable": False})
    contact_model.register_object("cube1", cube1_id, {"type": "block", "color": "red", "movable": True})
    contact_model.register_object("cube2", cube2_id, {"type": "block", "color": "blue", "movable": True})
    
    # Run simulation to let objects settle
    for _ in range(800):
        p.stepSimulation()
        time.sleep(1/240)
    
    # Analyze contacts and support including tensors
    results = contact_model.analyze()
    
    # Examples of using the new attributes and tensor functionality
    print("\nObject Attributes:")
    for name, attrs in contact_model.get_object_attributes().items():
        print(f"{name}: {attrs}")
    
    print("\nExtract a specific attribute for all objects:")
    types = contact_model.get_attributes_list("type")
    print(f"Types: {types}")
    
    positions = contact_model.get_attributes_list("position")
    print(f"Positions: {positions}")
    
    print("\nSupport Tensor:")
    print(contact_model.get_support_tensor())
    
    # Wait for user input
    input("Press Enter to disconnect...")
    p.disconnect()