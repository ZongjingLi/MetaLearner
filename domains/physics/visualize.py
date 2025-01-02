'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-11-17 05:03:51
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-11-17 05:03:59
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)  # Try to run on GPU

@ti.data_oriented
class MaterialPointModel2d(nn.Module): # it is actuall material point
    def __init__(self):
        super().__init__()        
        quality = 1  # Use a larger value for higher-res simulations
        n_particles, n_grid = 18000 * quality**2, 128 * quality
        self.n_particles = n_particles
        self.n_grid = n_grid
        dx, inv_dx = 1 / n_grid, float(n_grid)
        self.inv_dx = inv_dx
        self.dx = dx; 
        dt = 1e-4 / quality
        self.dt = dt
        p_vol, p_rho = (dx * 0.5) ** 2, 1.0
        self.p_vol = p_vol
        self.p_rho = p_rho
    
        self.p_mass = p_vol * p_rho
        E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
        self.E = E
        self.nu = nu
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
        dim = 2
        self.x = ti.Vector.field(dim, dtype=float, shape=n_particles)  # position
        self.v = ti.Vector.field(dim, dtype=float, shape=n_particles)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # affine velocity field
        self.F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # deformation gradient
        self.material = ti.field(dtype=int, shape=n_particles)  # material id
        self.Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
        self.grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
        self.gravity = ti.Vector.field(dim, dtype=float, shape=())
        self.attractor_strength = ti.field(dtype=float, shape=())
        self.attractor_pos = ti.Vector.field(dim, dtype=float, shape=())


    @ti.kernel
    def substep(self):
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0
        for p in self.x:  # Particle state update and scatter to grid (P2G)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)

            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            # deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            # Hardening coefficient: snow gets harder when compressed
            h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - self.Jp[p]))))
            if self.material[p] == 1:  # jelly, make it softer
                h = 0.2
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == 0:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if self.material[p] == 2:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.material[p] == 0:
                # Reset deformation gradient to avoid numerical instability
                self.F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif self.material[p] == 2:
                # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (
                J - 1
            )
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0:  # No need for epsilon here
                # Momentum to velocity
                self.grid_v[i, j] = (1 / self.grid_m[i, j]) * self.grid_v[i, j]
                self.grid_v[i, j] += self.dt * self.gravity[None] * 30  # gravity
                dist = self.attractor_pos[None] - self.dx * ti.Vector([i, j])
                self.grid_v[i, j] += dist / (0.01 + dist.norm()) * self.attractor_strength[None] * self.dt * 100
                if i < 3 and self.grid_v[i, j][0] < 0:
                    self.grid_v[i, j][0] = 0  # Boundary conditions
                if i > self.n_grid - 3 and self.grid_v[i, j][0] > 0:
                    self.grid_v[i, j][0] = 0
                if j < 3 and self.grid_v[i, j][1] < 0:
                    self.grid_v[i, j][1] = 0
                if j > self.n_grid - 3 and self.grid_v[i, j][1] > 0:
                    self.grid_v[i, j][1] = 0
        for p in self.x:  # grid to particle (G2P)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]  # advection

    @ti.kernel
    def reset(self):
        group_size = self.n_particles // 3

        for i in range(self.n_particles):
                self.x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
                ]
                self.material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
                self.v[i] = [0, 0]
                self.F[i] = ti.Matrix([[1, 0], [0, 1]])
                self.Jp[i] = 1
                self.C[i] = ti.Matrix.zero(float, 2, 2)


if __name__ == "__main__":

    gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)

    mpm = MaterialPointModel2d()
    mpm.reset()
    
    mpm.gravity[None] = [0, -9.8]

    for frame in range(20000):
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                mpm.reset()
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
 
        for s in range(int(2e-3 // mpm.dt)):
            mpm.substep()
            
            gui.circles(   
            mpm.x.to_numpy(),
            radius=1.5,
            palette=[0x068587, 0x292929, 0xEEEEF0],
            palette_indices=mpm.material,
        )


        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        gui.show()