import bpy
import random
import math
import os

"""some basic helper functions for the clear and get color etc"""

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def random_color():
    """get an random color in the (r,g,b,a) format"""
    return (random.random(), random.random(), random.random(), 1)

def is_position_valid(new_pos, new_radius, existing_objects):
    for obj in existing_objects:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_pos, obj['position'])))
        if dist < (new_radius + obj['radius']) * 1.2:  # *1.2 margin
            return False
    return True

def create_material(color, material_type='plastic'):
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    bsdf.inputs['Base Color'].default_value = color
    
    if material_type == 'metal':
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 0.2
    elif material_type == 'plastic':
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.35
    elif material_type == 'glass_':
        if 'Transmission' in bsdf.inputs:
            bsdf.inputs['Transmission'].default_value = 1.0
        #bsdf.inputs['Roughness'].default_value = 0.05
    else:  # diffuse
        bsdf.inputs['Roughness'].default_value = 1.0
    
    return mat

def add_object(shape_type, location, size, material_type='diffuse'):
    x, y, z = location
    if shape_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(x, y, z))
    elif shape_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=size * 2, location=(x, y, z))
    elif shape_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(radius=size, depth=size*2, location=(x, y, z))
    
    obj = bpy.context.object
    bpy.ops.object.shade_smooth()
    mat = create_material(random_color(), material_type)
    obj.data.materials.append(mat)
    return obj

def setup_camera():
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, -12, 8)
    cam_obj.rotation_euler = (math.radians(60), 0, 0)
    bpy.context.scene.camera = cam_obj

def setup_light():
    light_data = bpy.data.lights.new(name='Light', type='SUN')
    light_obj = bpy.data.objects.new('Light', light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, -5, 8)
    light_data.energy = 2

def create_ground():
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    plane = bpy.context.object
    plane_mat = create_material((0.9, 0.9, 0.9, 1), material_type='plastic')
    plane.data.materials.append(plane_mat)

# ----------- Main Script --------------

# Output folder
output_dir = os.path.join(bpy.path.abspath("//"), "data/aluneth_data")
os.makedirs(output_dir, exist_ok=True)

# Clean the scene
clean_scene()

# Create ground
create_ground()

# Object placement
objects = []
for _ in range(10):
    attempts = 0
    while attempts < 50:
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        z = 0.5
        size = random.uniform(0.3, 0.5)
        if is_position_valid((x, y, z), size, objects):
            shape_type = random.choice(['sphere', 'cube', 'cylinder'])
            material_type = random.choice([ 'metal', 'glass'])
            add_object(shape_type, (x, y, z), size, material_type)
            objects.append({'position': (x, y, z), 'radius': size})
            break
        attempts += 1

# Setup camera and light
setup_camera()
setup_light()

# Render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = os.path.join(output_dir, "clevr_scene.png")

# Render the image
bpy.ops.render.render(write_still=True)
