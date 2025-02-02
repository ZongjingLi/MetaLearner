import open3d as o3d
import numpy as np
import random

def create_random_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=random.uniform(0.2, 0.5))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(np.random.rand(3))
    return sphere

def create_random_cube():
    cube = o3d.geometry.TriangleMesh.create_box(width=random.uniform(0.3, 0.7),
                                                height=random.uniform(0.3, 0.7),
                                                depth=random.uniform(0.3, 0.7))
    cube.compute_vertex_normals()
    cube.paint_uniform_color(np.random.rand(3))
    return cube

def create_random_cylinder():
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=random.uniform(0.2, 0.4),
                                                         height=random.uniform(0.5, 1.0))
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(np.random.rand(3))
    return cylinder

def create_random_cone():
    cone = o3d.geometry.TriangleMesh.create_cone(radius=random.uniform(0.2, 0.4),
                                                 height=random.uniform(0.5, 1.0))
    cone.compute_vertex_normals()
    cone.paint_uniform_color(np.random.rand(3))
    return cone

def create_random_torus():
    torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=random.uniform(0.5, 1.0),
                                                   tube_radius=random.uniform(0.1, 0.2))
    torus.compute_vertex_normals()
    torus.paint_uniform_color(np.random.rand(3))
    return torus

# List of object generation functions
object_creators = [
    create_random_sphere,
    create_random_cube,
    create_random_cylinder,
    create_random_cone,
    create_random_torus
]

# Stable objects (excluding cone and torus)
stable_object_creators = [
    create_random_sphere,
    create_random_cube,
    create_random_cylinder
]

def is_collision_free(new_obj, existing_objs):
    new_bbox = new_obj.get_axis_aligned_bounding_box()
    new_min = new_bbox.get_min_bound()
    new_max = new_bbox.get_max_bound()

    for obj in existing_objs:
        existing_bbox = obj.get_axis_aligned_bounding_box()
        exist_min = existing_bbox.get_min_bound()
        exist_max = existing_bbox.get_max_bound()

        # Check for overlap along all three axes
        if (new_min[0] < exist_max[0] and new_max[0] > exist_min[0] and
            new_min[1] < exist_max[1] and new_max[1] > exist_min[1] and
            new_min[2] < exist_max[2] and new_max[2] > exist_min[2]):
            return False  # Collision detected

    return True

def stack_objects(base_objects, k=2):
    stacked_objects = []
    stable_bases = [obj for obj in base_objects if obj in stable_object_creators]
    print(len(stable_bases))
    selected_objects = random.sample(stable_bases, min(k, len(stable_bases)))

    for obj in selected_objects:
        stacked_obj_func = random.choice(object_creators)
        stacked_obj = stacked_obj_func()

        # Get the top of the base object's bounding box
        base_bbox = obj.get_axis_aligned_bounding_box()
        base_top_z = base_bbox.get_max_bound()[2]

        # Align the stacked object with the base object's position
        base_center = base_bbox.get_center()
        stacked_obj.translate([
            base_center[0] - stacked_obj.get_center()[0],
            base_center[1] - stacked_obj.get_center()[1],
            base_top_z - stacked_obj.get_min_bound()[2]  # Place directly on top
        ])

        stacked_objects.append(stacked_obj)

    return stacked_objects

def generate_random_scene(num_objects=10, points_per_object=1000, k=2):
    scene = []
    attempts = 0
    max_attempts = 1000  # To prevent infinite loops

    while len(scene) < num_objects and attempts < max_attempts:
        obj_func = random.choice(object_creators)
        obj = obj_func()

        # Random translation on the surface
        translation = np.random.uniform(-2, 2, size=2)
        obj.translate([translation[0], translation[1], -obj.get_min_bound()[2]])

        if is_collision_free(obj, scene):
            scene.append(obj)

        attempts += 1

    # Stack objects
    stacked_objects = stack_objects(scene, k)
    scene.extend(stacked_objects)

    return [o.sample_points_poisson_disk(number_of_points=points_per_object) for o in scene]

# Generate and visualize the scene
random_scene = generate_random_scene(num_objects=10, points_per_object=2500, k=2)
o3d.visualization.draw_geometries(random_scene)
