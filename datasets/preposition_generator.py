import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from collections import namedtuple

# Define a simple class to represent a point cloud object
class PointCloudObject:
    def __init__(self, position, color, name, num_points=100, size=0.1):
        self.position = position
        self.color = color
        self.name = name
        self.points = []
        self.visual_ids = []
        self.size = size
        
        # Generate point cloud in a sphere shape around the position
        for _ in range(num_points):
            # Random point on a unit sphere
            theta = 2 * math.pi * np.random.random()
            phi = math.acos(2 * np.random.random() - 1)
            x = math.sin(phi) * math.cos(theta) * size
            y = math.sin(phi) * math.sin(theta) * size
            z = math.cos(phi) * size
            
            point = [
                position[0] + x,
                position[1] + y,
                position[2] + z
            ]
            self.points.append(point)
    
    def visualize(self):
        # Clear any existing visualization
        for vid in self.visual_ids:
            p.removeBody(vid)
        self.visual_ids = []
        
        # Create visual points for each point in the cloud
        for point in self.points:
            shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=self.color)
            visual_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=point)
            self.visual_ids.append(visual_id)
    
    def move_to(self, new_position):
        # Move the entire point cloud to a new position
        vector = np.array(new_position) - np.array(self.position)
        self.position = new_position
        
        # Update all points
        for i in range(len(self.points)):
            self.points[i] = list(np.array(self.points[i]) + vector)
        
        # Update visualization
        self.visualize()

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

# Create text visualization function
def create_text(text, position, size=1.0, color=[1, 1, 1, 1]):
    return p.addUserDebugText(text, position, textColorRGB=color, textSize=size)

# Define colors for different objects
colors = {
    "reference": [1, 1, 1, 1],  # white
    "subject": [1, 0, 0, 1],    # red
    "floor": [0.5, 0.5, 0.5, 1] # gray
}

# Create a reference object (white cube)
reference = PointCloudObject([0, 0, 0.1], colors["reference"], "Reference", num_points=200, size=0.1)
reference.visualize()
create_text("Reference Object", [0, 0, 0.25], size=1.5)

# Function to demonstrate all spatial relations
def demonstrate_spatial_relations():
    # Store created text IDs to remove them later
    text_ids = []
    
    # Define the spatial relations to demonstrate
    SpatialRelation = namedtuple('SpatialRelation', ['name', 'position', 'description'])
    
    spatial_relations = [
        # Position/Location relations
        SpatialRelation("On", [0, 0, 0.25], "Object ON the reference"),
        SpatialRelation("Above", [0, 0, 0.4], "Object ABOVE the reference"),
        SpatialRelation("Under", [0, 0, -0.1], "Object UNDER the reference"),
        SpatialRelation("Below", [0, 0, -0.2], "Object BELOW the reference"),
        SpatialRelation("In front of", [0.3, 0, 0.1], "Object IN FRONT OF the reference"),
        SpatialRelation("Behind", [-0.3, 0, 0.1], "Object BEHIND the reference"),
        SpatialRelation("Beside", [0, 0.3, 0.1], "Object BESIDE the reference"),
        SpatialRelation("Between", [0, 0, 0.1], "Object BETWEEN references (needs 2+ references)"),
        SpatialRelation("Near", [0.15, 0.15, 0.1], "Object NEAR the reference"),
        SpatialRelation("Inside", [0, 0, 0.1], "Object INSIDE the reference (hollow reference needed)"),
        
        # Direction/Movement relations
        SpatialRelation("Toward", [0.4, 0, 0.1], "Object moving TOWARD the reference"),
        SpatialRelation("Away from", [-0.4, 0, 0.1], "Object moving AWAY FROM the reference"),
        SpatialRelation("Along", [0, 0.4, 0.1], "Object ALONG the reference"),
        SpatialRelation("Across", [0, -0.4, 0.1], "Object ACROSS from the reference"),
        SpatialRelation("Through", [0, 0, 0.1], "Object THROUGH the reference (animation needed)"),
    ]
    
    # Create subject object (red)
    subject = PointCloudObject([0, 0, 0.4], colors["subject"], "Subject", num_points=100, size=0.05)
    
    # For the "between" demonstration, create another reference object
    reference2 = None
    
    # Demonstrate each relation
    for i, relation in enumerate(spatial_relations):
        # Clear previous texts
        for text_id in text_ids:
            p.removeUserDebugItem(text_id)
        text_ids = []
        
        # Clear additional reference if it exists
        if reference2:
            for vid in reference2.visual_ids:
                p.removeBody(vid)
            reference2 = None
        
        # Move subject to demonstrate the relation
        subject.move_to(relation.position)
        
        # Special cases for specific relations
        if relation.name == "Between":
            # Create two reference objects for "between" demonstration
            reference.move_to([-0.2, 0, 0.1])
            reference2 = PointCloudObject([0.2, 0, 0.1], colors["reference"], "Reference 2", num_points=200, size=0.1)
            reference2.visualize()
            subject.move_to([0, 0, 0.1])
        
        elif relation.name == "Inside":
            # Make reference bigger and subject smaller for "inside" demonstration
            for vid in reference.visual_ids:
                p.removeBody(vid)
            reference = PointCloudObject([0, 0, 0.1], colors["reference"], "Reference", num_points=300, size=0.15)
            reference.visualize()
            
            for vid in subject.visual_ids:
                p.removeBody(vid)
            subject = PointCloudObject([0, 0, 0.1], colors["subject"], "Subject", num_points=50, size=0.05)
            subject.visualize()
        
        elif relation.name == "Through":
            # Animation for "through" demonstration
            start_pos = [-0.3, 0, 0.1]
            end_pos = [0.3, 0, 0.1]
            steps = 20
            
            subject.move_to(start_pos)
            for step in range(steps + 1):
                t = step / steps
                pos = [
                    start_pos[0] + t * (end_pos[0] - start_pos[0]),
                    start_pos[1] + t * (end_pos[1] - start_pos[1]),
                    start_pos[2] + t * (end_pos[2] - start_pos[2])
                ]
                subject.move_to(pos)
                time.sleep(0.05)
                p.stepSimulation()
        
        # Add informative text
        title_id = create_text(f"Demonstrating: {relation.name}", [0, 0.5, 0.5], size=2.0)
        desc_id = create_text(relation.description, [0, 0.5, 0.4], size=1.5)
        text_ids.extend([title_id, desc_id])
        
        # Wait for user to observe
        input(f"Press Enter to continue to next relation ({i+1}/{len(spatial_relations)})")
        
        # Return reference to original position after special cases
        if relation.name in ["Between", "Inside"]:
            for vid in reference.visual_ids:
                p.removeBody(vid)
            reference = PointCloudObject([0, 0, 0.1], colors["reference"], "Reference", num_points=200, size=0.1)
            reference.visualize()
            
            for vid in subject.visual_ids:
                p.removeBody(vid)
            subject = PointCloudObject([0, 0, 0.4], colors["subject"], "Subject", num_points=100, size=0.05)
            subject.visualize()

if __name__ == "__main__":
    print("PyBullet Spatial Relations Demonstration")
    print("----------------------------------------")
    print("This script demonstrates various spatial prepositions using point clouds.")
    print("Press Enter after each demonstration to proceed to the next relation.")
    
    demonstrate_spatial_relations()
    
    input("Demonstration complete. Press Enter to exit.")
    p.disconnect()