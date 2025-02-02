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
#random_scene = generate_random_scene(num_objects=10, points_per_object=2500, k=2)
#o3d.visualization.draw_geometries(random_scene)

# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2024-10-14 09:26:27
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-02 16:45:00
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from datasets.ccsp_dataset import *
from core.spatial.energy_graph import TimeInputEnergyMLP, PointEnergyMLP
from core.spatial.diffusion import training_loop, samples,  ScheduleLogLinear, ScheduleSigmoid


num_pts = 1000
pose_dim = 3

def random_pose(space_scale = 10.0, scale_range = (0.5, 1.0)):
    smin, smax = scale_range[0], scale_range[1]
    return torch.cat([  
        (torch.rand([2]) - 0.5) * (2 * space_scale),
        torch.zeros([1]),
        torch.randn(3),
        torch.rand([1]) * (smax - smin) + smin
        ])[:pose_dim]


def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:,0], batch[:,1], marker='.')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

constraints = {"online" : 1}


dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
loader   = DataLoader(dataset, batch_size=32, collate_fn=collate_graph_batch)
model    = TimeInputEnergyMLP(hidden_dims=(16,128,128,128,128,16))
model    = PointEnergyMLP(constraints)
#model.load_state_dict(torch.load("checkpoints/state.pth"))
schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
trainer  = training_loop(loader, model, schedule, epochs=250)
losses   = [ns.loss.item() for ns in trainer]
#torch.save(model.state_dict(),"checkpoints/state.pth")

batchsize = 300
#model.load_state_dict(torch.load("checkpoints/state.pth"))

cond = {"edges":[(i,"online") for i in range(batchsize)]}
xt = torch.randn([1, batchsize, 2])
*xt, x0  = samples(model, schedule.sample_sigmas(20), gam=2, cond = cond, batchsize = batchsize, xt = xt)


#print(x0.shape)

plot_batch(x0[0])
plt.show()
"""


#plt.plot(dataset.vals[:,0], dataset.vals[:,1])
#plt.scatter(x0[:,0], x0[:,1])
plt.show()



dataset = Swissroll(np.pi/2, 5 * np.pi, 100)
schedule = ScheduleLogLinear(N=1000, sigma_min=0.01, sigma_max=10)
model =  TimeInputEnergyMLP(hidden_dims=(16,128,128,128,128,16))
model.load_state_dict(torch.load("states.pth"))

def build_grid(resolution):
    w, h = resolution
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    x, y = torch.meshgrid([x, y])
    return torch.cat([x[..., None], y[..., None]], dim = -1)
W, H = 64, 64
xt = build_grid([W,H]).reshape([W*H,2])
sigmas = schedule.sample_sigmas(20)
#xt = model.rand_input(100)
for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        model.eval()
        sigma_embeds = get_sigma_embeds(xt.shape[0], sig.squeeze()) # shape: b x 2
        nn_input = torch.cat([xt, sigma_embeds], dim=1)               # 
        energy = model.net(nn_input)
        plt.figure("vis", figsize=(5,5))
        plt.scatter(xt[:,0],xt[:,1],c = energy.detach())
        plt.plot(dataset.vals[:,0], dataset.vals[:,1])
        plt.pause(0.01)
        plt.cla()

plt.plot(dataset.vals[:,0], dataset.vals[:,1])

plt.show()
"""

"""create some default objects to visualize """
"""test the swissroll experiment

disk_xs, disk_ys, disk_zs = sample_sphere_surface(num_pts)
unit_disk = torch.stack([disk_xs, disk_ys, disk_zs])

square_xs, square_ys, square_zs = sample_square_region(num_pts)
unit_square = torch.stack([square_xs, square_ys, square_zs])


obj0 = GeometricObject(unit_disk, random_pose())
obj1 = GeometricObject(unit_disk, random_pose())
obj2 = GeometricObject(unit_square, random_pose())
obj3 = GeometricObject(unit_square, random_pose())

geom_graph = GeometricGraph([obj0, obj1, obj2, obj3], [(0, 1, "near"), (1, 2, "far")])

constraints = {
    "near" : 2,
    "far" : 2,
}

dataset = [geom_graph]

geom_graph.setup(False)
plot_geom_graph(geom_graph)
solution = constraint_ensemble.solve(geom_graph)
geom_graph.update_control_variable(solution.detach())
plot_geom_graph(geom_graph)
"""
