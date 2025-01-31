import random
import open3d as o3d
import numpy as np

# Function to create random 3D object shapes (no need for external models)
def create_random_object(object_type, size_range=(0.2, 0.5)):
    # Random position in a 3D space (place objects on a flat surface, z = 0.1 to 0.5 to avoid overlap with the ground)
    x = random.uniform(-2, 2)
    y = random.uniform(-2, 2)
    z = random.uniform(0.1, 0.5)  # Slight height above the floor
    
    # Random size for the object
    size = random.uniform(*size_range)
    
    # Create a point cloud for the object
    if object_type == "Bin" or object_type == "Box":
        color = [random.random(), random.random(), random.random()]
        # Create a simple cube for containers like bin and box
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    elif object_type == "Cup" or object_type == "Bottle":
        color = [random.random(), random.random(), random.random()]
        # Create a simple cylinder for cup and bottle
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.4, height=size * 2)
    elif object_type == "Plate":
        color = [random.random(), random.random(), random.random()]
        # Create a simple disk for plate
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=size, height=0.05)  # Thin cylinder to simulate a plate
    elif object_type == "Hammer":
        color = [random.random(), random.random(), random.random()]
        # Create a simple cylinder for hammer
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.2, height=size * 3)
    else:
        color = [random.random(), random.random(), random.random()]
        # Use a sphere for other types of objects
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    
    # Translate the object to its random position
    mesh.translate([x, y, z])
    
    # Set the color of the object
    mesh.paint_uniform_color(color)
    
    # Sample points to create a point cloud
    pcd = mesh.sample_points_poisson_disk(number_of_points=500)
    
    return pcd

# Function to generate a random scene with 3-5 objects and realistic distribution
def generate_random_scene(num_objects=5):
    scene = []
    
    # Randomly select object types
    object_types = ["Bin", "Box", "Cup", "Bottle", "Plate", "Hammer"]
    selected_objects = random.sample(object_types, num_objects)
    
    for object_type in selected_objects:
        # Create a random 3D object as point cloud
        pcd = create_random_object(object_type)
        
        # Randomly position the object in the scene (x, y on a flat plane, z for height)
        translation = np.array([random.uniform(-2, 2),  # Random x
                               random.uniform(-2, 2),  # Random y
                               random.uniform(0.1, 0.5)])  # Slight height above the floor
        pcd.translate(translation)
        
        # Append the point cloud to the scene
        scene.append(pcd)
    
    return scene

# Generate a random scene with 3-5 objects
scene = generate_random_scene(num_objects=random.randint(3, 5))

# Visualize the scene
o3d.visualization.draw_geometries(scene, window_name="Realistic Scene with Point Clouds", width=800, height=600)
