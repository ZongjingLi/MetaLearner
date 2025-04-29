import bpy
import random
import math

# ----------- Helper Functions --------------

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def random_color():
    return (random.random(), random.random(), random.random(), 1)

def create_material(color):
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = color
    return mat

def add_random_object():
    shape_type = random.choice(['sphere', 'cube', 'cylinder'])
    x = random.uniform(-3, 3)
    y = random.uniform(-3, 3)
    z = 0.5  # On the ground
    size = random.uniform(0.4, 0.8)

    if shape_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(x, y, z))
    elif shape_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=size * 2, location=(x, y, z))
    elif shape_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(radius=size, depth=size*2, location=(x, y, z))
    
    obj = bpy.context.object
    mat = create_material(random_color())
    obj.data.materials.append(mat)

def setup_camera():
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, -8, 5)
    cam_obj.rotation_euler = (math.radians(60), 0, 0)
    bpy.context.scene.camera = cam_obj

def setup_light():
    light_data = bpy.data.lights.new(name='Light', type='POINT')
    light_obj = bpy.data.objects.new('Light', light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, -5, 8)
    light_data.energy = 1500

# ----------- Main Script --------------

# Clean the scene
clean_scene()

# Add ground plane
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
plane = bpy.context.object
plane_mat = create_material((0.9, 0.9, 0.9, 1))
plane.data.materials.append(plane_mat)

# Add random objects
for _ in range(6):
    add_random_object()

# Setup camera and lights
setup_camera()
setup_light()

# Set render settings
bpy.context.scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE'
bpy.context.scene.cycles.samples = 64  # Low samples for fast rendering
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = "/tmp/clevr_scene.png"

# Render
bpy.ops.render.render(write_still=True)
