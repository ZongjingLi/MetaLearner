from bulletarm import env_factory
import pybullet as p
import numpy as np
import trimesh

def mesh_to_point_cloud(mesh_path, num_points):
    # Load the mesh
    mesh = trimesh.load(mesh_path)
    
    # Sample points uniformly from the surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    return points

def get_object_contacts(object_id):
  contact_points = p.getContactPoints(bodyA=object_id)
  return contact_points

def get_mesh_data(object_id):
    visual_shape_data = p.getVisualShapeData(object_id)
    return object_id

def gather_object_info():
  objects = []
  num_objects = p.getNumBodies()

  for i in range(num_objects):
    body_info = p.getBodyInfo(i)
    object_name = body_info[1].decode('utf-8')  # Body name is at index 1, needs decoding
    objcloud = None
    objmesh = get_mesh_data(i)
    objects.append({
        'id': i,
        'name': object_name,
        'position': p.getBasePositionAndOrientation(i)[0],
        'orientation': p.getBasePositionAndOrientation(i)[1],
        "bounding_box" :p.getAABB(i),
        "pointcloud" : objcloud
    })



  return objects
