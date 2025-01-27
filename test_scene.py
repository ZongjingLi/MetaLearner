import open3d as o3d
import numpy as np

# Load saved point cloud
points = np.load('outputs/point_cloud.npy')
colors = np.load('outputs/point_cloud_colors.npy') / 255.0  # Normalize RGB values

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])
