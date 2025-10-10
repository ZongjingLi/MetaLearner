# Install required packages (uncomment if needed)
# !pip install taichi numpy matplotlib opencv-python pillow

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math

ti.init(arch=ti.cpu, debug=False)

num_gaussians = 10000
image_width, image_height = 512, 512

# Gaussian data structures
positions = ti.Vector.field(3, dtype=ti.f32, shape=num_gaussians)  # x, y, z coordinates
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_gaussians)     # RGB colors (0-1)
scales = ti.Vector.field(3, dtype=ti.f32, shape=num_gaussians)     # 3D scaling factors
rotations = ti.Vector.field(4, dtype=ti.f32, shape=num_gaussians)  # Quaternion rotations (x,y,z,w)

# Render buffers
image = ti.Vector.field(3, dtype=ti.f32, shape=(image_height, image_width))
depth_buffer = ti.field(dtype=ti.f32, shape=(image_height, image_width))
alpha_buffer = ti.field(dtype=ti.f32, shape=(image_height, image_width))

# Camera class
class Camera:
    def __init__(self, fov=60.0, position=[0, 0, -5], look_at=[0, 0, 0]):
        self.fov = fov  # Field of view in degrees
        self.position = ti.Vector(position, dt=ti.f32)
        self.look_at = ti.Vector(look_at, dt=ti.f32)
        self.up = ti.Vector([0, 1, 0], dt=ti.f32)

        self.forward = (self.look_at - self.position).normalized()
        self.right = self.forward.cross(self.up).normalized()
        self.up = self.right.cross(self.forward).normalized()
        
        self.focal = 0.5 * image_width / math.tan(math.radians(fov) * 0.5)

@ti.kernel
def initialize_gaussians():
    for i in positions:
        positions[i] = ti.Vector([
            ti.random() * 4.0 - 2.0,
            ti.random() * 4.0 - 2.0,
            ti.random() * 4.0 + 1.0
        ])
        
        # Random color
        colors[i] = ti.Vector([
            ti.random(),
            ti.random(),
            ti.random()
        ])
        
        scale = ti.random() * 0.2 + 0.05
        scales[i] = ti.Vector([scale, scale, scale])

        u1, u2, u3 = ti.random(), ti.random(), ti.random()
        rotations[i] = ti.Vector([
            ti.sqrt(1 - u1) * ti.sin(2 * math.pi * u2),
            ti.sqrt(1 - u1) * ti.cos(2 * math.pi * u2),
            ti.sqrt(u1) * ti.sin(2 * math.pi * u3),
            ti.sqrt(u1) * ti.cos(2 * math.pi * u3)
        ]).normalized()

@ti.func
def quat_to_mat(q):
    x, y, z, w = q
    norm = ti.sqrt(x**2 + y**2 + z**2 + w**2)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return ti.Matrix([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# Main rendering kernel
@ti.kernel
def render(camera: ti.template()):
    for i, j in image:
        image[i, j] = ti.Vector([0.0, 0.0, 0.0])
        depth_buffer[i, j] = -1e10
        alpha_buffer[i, j] = 0.0
    
    for g in positions:
        pos = positions[g]
        color = colors[g]
        scale = scales[g]
        rot = rotations[g]
        
        cam_pos = pos - camera.position
        cam_x = cam_pos.dot(camera.right)
        cam_y = cam_pos.dot(camera.up)
        cam_z = cam_pos.dot(camera.forward)
        if cam_z <= 0: continue

        x = camera.focal * cam_x / cam_z + image_width * 0.5
        y = camera.focal * cam_y / cam_z + image_height * 0.5
        rot_mat = quat_to_mat(rot)

        scale_mat = ti.Matrix([
            [scale[0], 0.0, 0.0],
            [0.0, scale[1], 0.0],
            [0.0, 0.0, scale[2]]
        ])
        cov_3d = rot_mat @ scale_mat @ scale_mat @ rot_mat.transpose()

        inv_z2 = 1.0 / (cam_z * cam_z)
        cov = ti.Matrix([
            [cov_3d[0,0] * camera.focal**2 * inv_z2, cov_3d[0,1] * camera.focal**2 * inv_z2],
            [cov_3d[1,0] * camera.focal**2 * inv_z2, cov_3d[1,1] * camera.focal**2 * inv_z2]
        ])

        det = cov[0,0] * cov[1,1] - cov[0,1] * cov[1,0]
        if det <= 0: continue

        inv_cov = (1.0 / det) * ti.Matrix([
            [cov[1,1], -cov[0,1]],
            [-cov[1,0], cov[0,0]]
        ])

        radius = int(3.0 * ti.sqrt(max(cov[0,0], cov[1,1])) + 1.0)
        x_min = max(0, int(x - radius))
        x_max = min(image_width - 1, int(x + radius))
        y_min = max(0, int(y - radius))
        y_max = min(image_height - 1, int(y + radius))

        for i in range(y_min, y_max + 1):
            for j in range(x_min, x_max + 1):
                dx = j - x
                dy = i - y

                dist = ti.Vector([dx, dy])
                weight = ti.exp(-0.5 * dist.dot(inv_cov @ dist))
                weight = weight / (2.0 * math.pi * ti.sqrt(det))
                if cam_z > depth_buffer[i, j]:
                    alpha = weight * (1.0 - alpha_buffer[i, j])
                    image[i, j] += alpha * color
                    alpha_buffer[i, j] += alpha
                    depth_buffer[i, j] = cam_z

def show_image():
    img_array = image.to_numpy()
    img_array = np.clip(img_array, 0.0, 1.0)
    img_array = (img_array * 255).astype(np.uint8)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_array)
    plt.title('3D Gaussian Splatting Render')
    plt.axis('off')
    plt.show()

def create_animation(num_frames=30):
    import os
    os.makedirs('animation', exist_ok=True)
    camera = Camera()
    
    for frame in range(num_frames):
        # Orbit camera around scene
        angle = 2 * math.pi * frame / num_frames
        camera.position = ti.Vector([
            5 * math.sin(angle),
            0,
            -5 * math.cos(angle)
        ])
        camera.forward = (camera.look_at - camera.position).normalized()
        camera.right = camera.forward.cross(camera.up).normalized()
        camera.up = camera.right.cross(camera.forward).normalized()
        
        # Render frame
        render(camera)
        
        # Save frame
        img_array = image.to_numpy()
        img_array = np.clip(img_array, 0.0, 1.0)
        img_array = (img_array * 255).astype(np.uint8)
        Image.fromarray(img_array).save(f'animation/frame_{frame:03d}.png')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('gaussian_animation.mp4', fourcc, 10, (image_width, image_height))
    
    for frame in range(num_frames):
        img = cv2.imread(f'animation/frame_{frame:03d}.png')
        video.write(img)
    
    video.release()
    print("Animation saved as 'gaussian_animation.mp4'")

if __name__ == "__main__":
    initialize_gaussians()
    camera = Camera(position=[0, 0, -5], look_at=[0, 0, 2])

    render(camera)
    show_image()
    
    # Uncomment to create animation (requires ffmpeg)
    #create_animation()