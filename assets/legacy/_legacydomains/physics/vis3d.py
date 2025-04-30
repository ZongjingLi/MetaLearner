'''
 # @ Author: Modified from Zongjing Li's 2D version
 # @ Description: 3D Material Point Method simulation
 # @ License: MIT
'''
import torch
import torch.nn as nn
import taichi as ti
import taichi.math as tm
import numpy as np
ti.init(arch=ti.gpu)

@ti.data_oriented
class MaterialPointModel3d(nn.Module):
    def __init__(self):
        super().__init__()
        quality = 1  # Use a larger value for higher-res simulations
        n_particles, n_grid = 27000 * quality**3, 64 * quality  # Adjusted for 3D
        self.n_particles = n_particles
        self.n_grid = n_grid
        dx, inv_dx = 1 / n_grid, float(n_grid)
        self.inv_dx = inv_dx
        self.dx = dx
        dt = 1e-4 / quality
        self.dt = dt
        p_vol, p_rho = (dx * 0.5) ** 3, 1.0  # Adjusted for 3D
        self.p_vol = p_vol
        self.p_rho = p_rho
        
        self.p_mass = p_vol * p_rho
        E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
        self.E = E
        self.nu = nu
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        
        dim = 3  # Changed to 3D
        self.x = ti.Vector.field(dim, dtype=float, shape=n_particles)  # position
        self.v = ti.Vector.field(dim, dtype=float, shape=n_particles)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # affine velocity field
        self.F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # deformation gradient
        self.material = ti.field(dtype=int, shape=n_particles)  # material id
        self.Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
        
        # 3D grid fields
        self.grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))
        self.grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))
        
        self.gravity = ti.Vector.field(dim, dtype=float, shape=())
        self.attractor_strength = ti.field(dtype=float, shape=())
        self.attractor_pos = ti.Vector.field(dim, dtype=float, shape=())

    @ti.kernel
    def substep(self):
        for i, j, k in self.grid_m:
            self.grid_v[i, j, k] = [0, 0, 0]
            self.grid_m[i, j, k] = 0
            
        for p in self.x:
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            
            # Quadratic kernels
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # Deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 3) + self.dt * self.C[p]) @ self.F[p]
            
            # Hardening coefficient
            h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - self.Jp[p]))))
            if self.material[p] == 1:  # jelly
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == 0:  # liquid
                mu = 0.0
                
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            
            for d in ti.static(range(3)):  # Changed to 3D
                new_sig = sig[d, d]
                if self.material[p] == 2:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
                
            if self.material[p] == 0:
                self.F[p] = ti.Matrix.identity(float, 3) * ti.sqrt(J)
            elif self.material[p] == 2:
                self.F[p] = U @ sig @ V.transpose()
                
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
            stress = stress + ti.Matrix.identity(float, 3) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[p]
            
            # P2G
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass
                
        # Grid operations
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                self.grid_v[i, j, k] = (1 / self.grid_m[i, j, k]) * self.grid_v[i, j, k]
                self.grid_v[i, j, k] += self.dt * self.gravity[None] * 30
                
                # Boundary conditions
                if i < 3 and self.grid_v[i, j, k][0] < 0: self.grid_v[i, j, k][0] = 0
                if i > self.n_grid - 3 and self.grid_v[i, j, k][0] > 0: self.grid_v[i, j, k][0] = 0
                if j < 3 and self.grid_v[i, j, k][1] < 0: self.grid_v[i, j, k][1] = 0
                if j > self.n_grid - 3 and self.grid_v[i, j, k][1] > 0: self.grid_v[i, j, k][1] = 0
                if k < 3 and self.grid_v[i, j, k][2] < 0: self.grid_v[i, j, k][2] = 0
                if k > self.n_grid - 3 and self.grid_v[i, j, k][2] > 0: self.grid_v[i, j, k][2] = 0
                
        # G2P
        for p in self.x:
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                weight = w[i][0] * w[j][1] * w[k][2]
                g_v = self.grid_v[base + ti.Vector([i, j, k])]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]

    @ti.func
    def in_sphere(self, pos, center, radius):
        return (pos - center).norm() <= radius

    @ti.func
    def in_cube(self, pos, center, size):
        diff = ti.abs(pos - center)
        return diff[0] <= size/2 and diff[1] <= size/2 and diff[2] <= size/2

    @ti.func
    def in_torus(self, pos, center, R, r):
        xz_dist = ti.sqrt((pos[0] - center[0])**2 + (pos[2] - center[2])**2)
        return (ti.sqrt((xz_dist - R)**2 + (pos[1] - center[1])**2) <= r)

    @ti.func
    def in_cylinder(self, pos, center, radius, height):
        xz_dist = ti.sqrt((pos[0] - center[0])**2 + (pos[2] - center[2])**2)
        y_dist = ti.abs(pos[1] - center[1])
        return xz_dist <= radius and y_dist <= height/2

    @ti.kernel
    def reset(self):
        group_size = self.n_particles // 3
        for i in range(self.n_particles):
            self.x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
                ti.random() * 0.2 + 0.3  # Added Z coordinate
            ]
            self.material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
            self.v[i] = [0, 0, 0]
            self.F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.Jp[i] = 1
            self.C[i] = ti.Matrix.zero(float, 3, 3)

"""
[Previous code remains exactly the same until the main section]
"""

if __name__ == "__main__":
    window = ti.ui.Window("3D MPM", (1024, 1024))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    # Initialize camera
    camera.position(2.5, 2.5, 3.5)
    camera.lookat(0.5, 0.5, 0.5)
    camera.up(0, 1, 0)
    
    # Create color field for particles
    
    mpm = MaterialPointModel3d()
    mpm.reset()
    mpm.gravity[None] = [0, -9.8, 0]
    
    # Set initial colors
    particle_colors = ti.Vector.field(3, dtype=float, shape=mpm.n_particles)
    
    @ti.kernel
    def set_colors():
        for i in range(mpm.n_particles):
            if mpm.material[i] == 0:  # Fluid
                particle_colors[i] = np.random.choice([
                    ti.Vector([0.1, 0.6, 0.9]),
                    ti.Vector([0.93, 0.33, 0.23]),
                    ti.Vector([1.0, 1.0, 1.0]),
                    ])
            elif mpm.material[i] == 1:  # Jelly
                particle_colors[i] = np.random.choice([
                    ti.Vector([0.1, 0.6, 0.9]),
                    ti.Vector([0.93, 0.33, 0.23]),
                    ti.Vector([1.0, 1.0, 1.0]),
                    ])
            else:  # Snow
                particle_colors[i] = ti.Vector([1.0, 1.0, 1.0])
    set_colors()
    
    while window.running:
        for s in range(int(2e-3 // mpm.dt)):
            mpm.substep()
        
        # Update camera
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # Set up scene
        scene.ambient_light((0.1, 0.1, 0.1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        
        # Visualize particles directly using Taichi fields
        scene.particles(mpm.x, radius=0.01, per_vertex_color=particle_colors)
        
        canvas.scene(scene)
        window.show()