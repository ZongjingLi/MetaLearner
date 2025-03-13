import numpy as np
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import matplotlib.animation as animation

class ShapeSortingDatasetGenerator:
    def __init__(self, output_dir="shape_sorting_dataset", num_samples=1000):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory to save the dataset
            num_samples: Number of shape-hole pairs to generate
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.shape_types = ["cube", "cylinder", "sphere", "pyramid", "star", "pentagon"]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "objects"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "holes"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "animations"), exist_ok=True)
        
        # Dictionary to store dataset metadata
        self.dataset_info = {
            "dataset_name": "Shape Sorting Point Cloud Dataset",
            "version": "1.0",
            "num_samples": num_samples,
            "shape_types": self.shape_types,
            "samples": []
        }
    
    def generate_cube_points(self, size=1.0, num_points=1000, noise_level=0.02):
        """Generate point cloud for a cube"""
        points = []
        # Generate points for each face of the cube
        for dim in range(3):  # x, y, z dimensions
            for sign in [-1, 1]:  # negative and positive direction
                face_points = np.random.rand(num_points // 6, 3) * size - size/2
                face_points[:, dim] = sign * size/2  # Set the dimension to the face position
                points.append(face_points)
        
        points = np.vstack(points)
        
        # Add some noise
        points += np.random.normal(0, noise_level, points.shape)
        
        return points
    
    def generate_cylinder_points(self, radius=0.5, height=1.0, num_points=1000, noise_level=0.02):
        """Generate point cloud for a cylinder"""
        num_circle_points = int(num_points * 0.8)  # Points on the circular surface
        num_cap_points = num_points - num_circle_points  # Points on the top and bottom caps
        
        # Generate points on the cylinder surface
        theta = np.random.uniform(0, 2*np.pi, num_circle_points)
        h = np.random.uniform(-height/2, height/2, num_circle_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = h
        
        cylinder_points = np.column_stack((x, y, z))
        
        # Generate points on the caps
        cap_points = []
        for sign in [-1, 1]:  # Bottom and top caps
            r = np.sqrt(np.random.uniform(0, radius**2, num_cap_points // 2))
            theta = np.random.uniform(0, 2*np.pi, num_cap_points // 2)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.ones(num_cap_points // 2) * sign * height/2
            cap_points.append(np.column_stack((x, y, z)))
        
        points = np.vstack([cylinder_points] + cap_points)
        
        # Add some noise
        points += np.random.normal(0, noise_level, points.shape)
        
        return points
    
    def generate_sphere_points(self, radius=0.5, num_points=1000, noise_level=0.02):
        """Generate point cloud for a sphere"""
        # Generate random points on a sphere
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2*np.pi, num_points)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        points = np.column_stack((x, y, z))
        
        # Add some noise
        points += np.random.normal(0, noise_level, points.shape)
        
        return points
    
    def generate_pyramid_points(self, base_size=1.0, height=1.0, num_points=1000, noise_level=0.02):
        """Generate point cloud for a square pyramid"""
        # Base points (square)
        num_base_points = int(num_points * 0.5)
        base_points = np.random.uniform(-base_size/2, base_size/2, (num_base_points, 2))
        base_points = np.column_stack((base_points, np.ones(num_base_points) * -height/2))
        
        # Side points (triangular faces)
        num_side_points = num_points - num_base_points
        side_points_per_face = num_side_points // 4
        
        side_points = []
        # Four triangular faces
        base_corners = [
            [-base_size/2, -base_size/2, -height/2],
            [base_size/2, -base_size/2, -height/2],
            [base_size/2, base_size/2, -height/2],
            [-base_size/2, base_size/2, -height/2]
        ]
        
        apex = [0, 0, height/2]  # Pyramid apex
        
        for i in range(4):
            corner1 = base_corners[i]
            corner2 = base_corners[(i+1)%4]
            
            # Generate random points on the triangular face
            u = np.random.uniform(0, 1, side_points_per_face)
            v = np.random.uniform(0, 1-u, side_points_per_face)
            
            # Compute the points as a weighted sum of the corners and apex
            face_points = np.outer(u, np.array(corner1)) + np.outer(v, np.array(corner2)) + np.outer(1-u-v, np.array(apex))
            side_points.append(face_points)
        
        points = np.vstack([base_points] + side_points)
        
        # Add some noise
        points += np.random.normal(0, noise_level, points.shape)
        
        return points
    
    def generate_pentagon_points(self, radius=0.5, height=0.2, num_points=1000, noise_level=0.02):
        """Generate point cloud for a regular pentagon prism"""
        num_vertices = 5
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        
        # Generate the vertices of the regular pentagon
        vertices_top = []
        vertices_bottom = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices_top.append([x, y, height/2])
            vertices_bottom.append([x, y, -height/2])
        
        vertices_top = np.array(vertices_top)
        vertices_bottom = np.array(vertices_bottom)
        
        # Generate points on the top and bottom faces
        top_bottom_points = []
        for vertices in [vertices_top, vertices_bottom]:
            center = np.mean(vertices, axis=0)
            for i in range(num_vertices):
                # Triangle from center to two adjacent vertices
                v1 = vertices[i]
                v2 = vertices[(i+1) % num_vertices]
                
                # Generate random points in this triangle
                num_tri_points = num_points // (2 * num_vertices)
                u = np.random.uniform(0, 1, num_tri_points)
                v = np.random.uniform(0, 1-u, num_tri_points)
                
                tri_points = np.outer(u, v1) + np.outer(v, v2) + np.outer(1-u-v, center)
                top_bottom_points.append(tri_points)
        
        # Generate points on the side faces (rectangles)
        side_points = []
        for i in range(num_vertices):
            v1_top = vertices_top[i]
            v2_top = vertices_top[(i+1) % num_vertices]
            v1_bottom = vertices_bottom[i]
            v2_bottom = vertices_bottom[(i+1) % num_vertices]
            
            # Generate random points on this rectangular face
            num_rect_points = num_points // num_vertices
            u = np.random.uniform(0, 1, num_rect_points)
            v = np.random.uniform(0, 1, num_rect_points)
            
            # Bilinear interpolation (properly broadcasting)
            rect_points = (1-u)[:,np.newaxis]*(1-v)[:,np.newaxis]*np.array(v1_bottom) + \
                         (1-u)[:,np.newaxis]*v[:,np.newaxis]*np.array(v1_top) + \
                         u[:,np.newaxis]*(1-v)[:,np.newaxis]*np.array(v2_bottom) + \
                         u[:,np.newaxis]*v[:,np.newaxis]*np.array(v2_top)
            
            side_points.append(rect_points)
        
        points = np.vstack(top_bottom_points + side_points)
        
        # Add some noise
        points += np.random.normal(0, noise_level, points.shape)
        
        return points
    
    def generate_star_points(self, outer_radius=0.5, inner_radius=0.25, height=0.2, num_points=1000, noise_level=0.02):
        """Generate point cloud for a 5-pointed star prism"""
        num_points_per_face = num_points // 3  # Top, bottom, and sides
        
        # Generate the vertices of a 5-pointed star
        num_vertices = 10
        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
        
        vertices_top = []
        vertices_bottom = []
        for i, angle in enumerate(angles):
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices_top.append([x, y, height/2])
            vertices_bottom.append([x, y, -height/2])
        
        vertices_top = np.array(vertices_top)
        vertices_bottom = np.array(vertices_bottom)
        
        # Generate points on the top and bottom faces
        top_bottom_points = []
        for vertices in [vertices_top, vertices_bottom]:
            center = np.mean(vertices, axis=0)
            for i in range(num_vertices):
                # Triangle from center to two adjacent vertices
                v1 = vertices[i]
                v2 = vertices[(i+1) % num_vertices]
                
                # Generate random points in this triangle
                num_tri_points = num_points_per_face // num_vertices
                u = np.random.uniform(0, 1, num_tri_points)
                v = np.random.uniform(0, 1-u, num_tri_points)
                
                tri_points = np.outer(u, v1) + np.outer(v, v2) + np.outer(1-u-v, center)
                top_bottom_points.append(tri_points)
        
        # Generate points on the side faces (rectangles)
        side_points = []
        for i in range(num_vertices):
            v1_top = vertices_top[i]
            v2_top = vertices_top[(i+1) % num_vertices]
            v1_bottom = vertices_bottom[i]
            v2_bottom = vertices_bottom[(i+1) % num_vertices]
            
            # Generate random points on this rectangular face
            num_rect_points = num_points_per_face // num_vertices
            u = np.random.uniform(0, 1, num_rect_points)
            v = np.random.uniform(0, 1, num_rect_points)
            
            # Bilinear interpolation (properly broadcasting)
            rect_points = (1-u)[:,np.newaxis]*(1-v)[:,np.newaxis]*np.array(v1_bottom) + \
                         (1-u)[:,np.newaxis]*v[:,np.newaxis]*np.array(v1_top) + \
                         u[:,np.newaxis]*(1-v)[:,np.newaxis]*np.array(v2_bottom) + \
                         u[:,np.newaxis]*v[:,np.newaxis]*np.array(v2_top)
            
            side_points.append(rect_points)
        
        points = np.vstack(top_bottom_points + side_points)
        
        # Add some noise
        points += np.random.normal(0, noise_level, points.shape)
        
        return points
        
    def generate_shape_points(self, shape_type, **kwargs):
        """Generate point cloud based on shape type"""
        if shape_type == "cube":
            return self.generate_cube_points(**kwargs)
        elif shape_type == "cylinder":
            return self.generate_cylinder_points(**kwargs)
        elif shape_type == "sphere":
            return self.generate_sphere_points(**kwargs)
        elif shape_type == "pyramid":
            return self.generate_pyramid_points(**kwargs)
        elif shape_type == "pentagon":
            return self.generate_pentagon_points(**kwargs)
        elif shape_type == "star":
            return self.generate_star_points(**kwargs)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
    
    def generate_hole_from_shape(self, shape_points, clearance=0.05, thickness=0.1, base_size=2.0):
        """
        Generate a point cloud for a receiving object with a hole that matches the shape
        
        Args:
            shape_points: Point cloud of the shape
            clearance: Additional clearance to ensure the shape fits
            thickness: Thickness of the container walls
            base_size: Size of the base container object
            
        Returns:
            hole_points: Point cloud of the container with hole
        """
        # Calculate the bounding box of the shape
        min_coords = np.min(shape_points, axis=0)
        max_coords = np.max(shape_points, axis=0)
        shape_size = max_coords - min_coords
        shape_center = (min_coords + max_coords) / 2
        
        # Scale the shape slightly to create the hole with clearance
        hole_shape = shape_points * (1 + clearance)
        
        # Create a cuboid container
        container_size = np.maximum(shape_size * 2, [base_size, base_size, base_size/2])
        container_min = shape_center - container_size/2
        container_max = shape_center + container_size/2
        
        # Number of points to generate for the container
        num_container_points = len(shape_points) * 5
        
        # Generate points for the container (cuboid with hole)
        # We'll create points for each face of the cuboid
        container_points = []
        
        # Bottom face (full)
        bottom_points = np.random.uniform(
            low=[container_min[0], container_min[1], container_min[2]],
            high=[container_max[0], container_max[1], container_min[2] + thickness],
            size=(num_container_points // 6, 3)
        )
        container_points.append(bottom_points)
        
        # Top face (with hole)
        # First generate a grid of points
        top_x = np.linspace(container_min[0], container_max[0], int(np.sqrt(num_container_points // 6)))
        top_y = np.linspace(container_min[1], container_max[1], int(np.sqrt(num_container_points // 6)))
        top_xx, top_yy = np.meshgrid(top_x, top_y)
        top_points = np.column_stack([top_xx.flatten(), top_yy.flatten(), 
                                      np.ones_like(top_xx.flatten()) * container_max[2]])
        
        # Remove points that are inside the hole shape's projection
        hole_min = np.min(hole_shape, axis=0)
        hole_max = np.max(hole_shape, axis=0)
        
        # Function to check if a point is inside the hole's projection
        def is_inside_hole_projection(point):
            # Project the hole shape onto the xy plane
            # For simplicity, we'll use the bounding box of the hole
            return (hole_min[0] <= point[0] <= hole_max[0] and 
                    hole_min[1] <= point[1] <= hole_max[1])
        
        # Filter out points that are inside the hole
        top_points = np.array([p for p in top_points if not is_inside_hole_projection(p)])
        container_points.append(top_points)
        
        # Side faces (4 sides)
        # Front face (y minimum)
        front_points = np.random.uniform(
            low=[container_min[0], container_min[1], container_min[2] + thickness],
            high=[container_max[0], container_min[1] + thickness, container_max[2]],
            size=(num_container_points // 6, 3)
        )
        container_points.append(front_points)
        
        # Back face (y maximum)
        back_points = np.random.uniform(
            low=[container_min[0], container_max[1] - thickness, container_min[2] + thickness],
            high=[container_max[0], container_max[1], container_max[2]],
            size=(num_container_points // 6, 3)
        )
        container_points.append(back_points)
        
        # Left face (x minimum)
        left_points = np.random.uniform(
            low=[container_min[0], container_min[1] + thickness, container_min[2] + thickness],
            high=[container_min[0] + thickness, container_max[1] - thickness, container_max[2]],
            size=(num_container_points // 6, 3)
        )
        container_points.append(left_points)
        
        # Right face (x maximum)
        right_points = np.random.uniform(
            low=[container_max[0] - thickness, container_min[1] + thickness, container_min[2] + thickness],
            high=[container_max[0], container_max[1] - thickness, container_max[2]],
            size=(num_container_points // 6, 3)
        )
        container_points.append(right_points)
        
        # Combine all the container points
        hole_points = np.vstack(container_points)
        
        # Add some noise to make it realistic
        hole_points += np.random.normal(0, 0.005, hole_points.shape)
        
        return hole_points
    
    def generate_success_transform(self, shape_points, hole_points, shape_type):
        """
        Generate a success transformation matrix (position and orientation)
        for inserting the shape into its hole
        
        Args:
            shape_points: Point cloud of the shape
            hole_points: Point cloud of the hole
            shape_type: Type of the shape
            
        Returns:
            transform: 4x4 transformation matrix
        """
        # Calculate the center of the shape
        shape_center = np.mean(shape_points, axis=0)
        
        # Calculate the center of the hole in the xy plane
        # We need to find where the hole is in the container
        hole_top_face = hole_points[hole_points[:, 2] > np.percentile(hole_points[:, 2], 90)]
        if len(hole_top_face) > 0:
            # Find the center of the "missing" region (hole) by looking at the extremes
            # of the top face and finding the center of that space
            x_coords = hole_top_face[:, 0]
            y_coords = hole_top_face[:, 1]
            
            # Get the min and max x, y coordinates
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # Estimate the hole center
            hole_center_xy = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
            
            # Get the maximum z-coordinate for the hole object
            hole_z_max = np.max(hole_points[:, 2])
            
            # Set position offset to place the shape centered above the hole
            pos_offset = np.array([
                hole_center_xy[0] - shape_center[0],
                hole_center_xy[1] - shape_center[1],
                hole_z_max - np.min(shape_points[:, 2]) + 0.1  # Position slightly above the hole
            ])
        else:
            # Fallback if we can't determine hole center properly
            hole_center = np.mean(hole_points, axis=0)
            pos_offset = hole_center - shape_center + np.array([0, 0, 1.0])  # 1.0 is arbitrary height above
        
        # Add small random jitter to make it realistic
        pos_offset += np.random.normal(0, 0.02, 3)
        
        # Random orientation around z-axis based on shape type
        if shape_type in ["cube", "pyramid", "pentagon", "star"]:
            # These shapes have rotational symmetry
            if shape_type == "cube":
                # Cube can rotate 90 degrees
                angle_z = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])
            elif shape_type == "pyramid" or shape_type == "pentagon":
                # Pentagon has 5-fold symmetry
                angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
                angle_z = np.random.choice(angles)
            elif shape_type == "star":
                # 5-pointed star has 5-fold symmetry
                angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
                angle_z = np.random.choice(angles)
        else:
            # Cylinder and sphere can rotate freely around z
            angle_z = np.random.uniform(0, 2*np.pi)
        
        # Create rotation matrix (only around z-axis for simplicity)
        rotation = Rotation.from_euler('z', angle_z)
        R = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = pos_offset
        
        return transform
    
    def get_aligned_hole_points(self, shape_points, hole_points, transform):
        """
        Transform the hole points to align with the shape points according to
        the success transformation
        
        Args:
            shape_points: Point cloud of the shape
            hole_points: Point cloud of the hole
            transform: 4x4 transformation matrix
            
        Returns:
            aligned_hole_points: Transformed hole points
        """
        # Homogeneous coordinates for hole points
        hole_homogeneous = np.hstack([hole_points, np.ones((hole_points.shape[0], 1))])
        
        # Apply transformation
        aligned_homogeneous = hole_homogeneous @ transform.T
        
        # Convert back to 3D points
        aligned_hole_points = aligned_homogeneous[:, :3]
        
        return aligned_hole_points
    
    def visualize_shape_hole_pair(self, shape_points, hole_points, transform, filename):
        """
        Visualize the shape and its corresponding hole
        
        Args:
            shape_points: Point cloud of the shape
            hole_points: Point cloud of the hole
            transform: 4x4 transformation matrix
            filename: Output filename for the visualization
        """
        # Transform the hole points to match the success configuration
        aligned_hole_points = self.get_aligned_hole_points(shape_points, hole_points, transform)
        
        fig = plt.figure(figsize=(12, 5))
        
        # Plot original shape and hole
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(shape_points[:, 0], shape_points[:, 1], shape_points[:, 2], 
                  c='blue', s=1, alpha=0.5, label='Shape')
        ax1.scatter(hole_points[:, 0], hole_points[:, 1], hole_points[:, 2], 
                  c='red', s=1, alpha=0.5, label='Hole')
        ax1.set_title('Original Shape and Hole')
        ax1.legend()
        
        # Plot shape and aligned hole (success configuration)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(shape_points[:, 0], shape_points[:, 1], shape_points[:, 2], 
                  c='blue', s=1, alpha=0.5, label='Shape')
        ax2.scatter(aligned_hole_points[:, 0], aligned_hole_points[:, 1], aligned_hole_points[:, 2], 
                  c='red', s=1, alpha=0.5, label='Aligned Hole')
        ax2.set_title('Success Configuration')
        ax2.legend()
        
        # Set consistent axis limits
        range_x = max([
            shape_points[:, 0].max() - shape_points[:, 0].min(),
            hole_points[:, 0].max() - hole_points[:, 0].min(),
            aligned_hole_points[:, 0].max() - aligned_hole_points[:, 0].min()
        ])
        range_y = max([
            shape_points[:, 1].max() - shape_points[:, 1].min(),
            hole_points[:, 1].max() - hole_points[:, 1].min(),
            aligned_hole_points[:, 1].max() - aligned_hole_points[:, 1].min()
        ])
        range_z = max([
            shape_points[:, 2].max() - shape_points[:, 2].min(),
            hole_points[:, 2].max() - hole_points[:, 2].min(),
            aligned_hole_points[:, 2].max() - aligned_hole_points[:, 2].min()
        ])
        max_range = max(range_x, range_y, range_z)
        
        for ax in [ax1, ax2]:
            # For the first subplot, use shape and hole
            if ax == ax1:
                mid_x = (shape_points[:, 0].mean() + hole_points[:, 0].mean()) / 2
                mid_y = (shape_points[:, 1].mean() + hole_points[:, 1].mean()) / 2
                mid_z = (shape_points[:, 2].mean() + hole_points[:, 2].mean()) / 2
            # For the second subplot, use shape and aligned hole
            else:
                mid_x = (shape_points[:, 0].mean() + aligned_hole_points[:, 0].mean()) / 2
                mid_y = (shape_points[:, 1].mean() + aligned_hole_points[:, 1].mean()) / 2
                mid_z = (shape_points[:, 2].mean() + aligned_hole_points[:, 2].mean()) / 2
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def create_animation(self, shape_points, hole_points, transform, filename, num_frames=20):
        """
        Create an animation of the shape being inserted into the hole
        
        Args:
            shape_points: Point cloud of the shape
            hole_points: Point cloud of the hole
            transform: 4x4 transformation matrix for success insertion
            filename: Output filename for the animation
            num_frames: Number of frames in the animation
        """
        # Create a figure for the animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a starting position above the hole
        start_transform = np.eye(4)
        start_transform[:3, :3] = transform[:3, :3]  # Keep the same rotation
        start_transform[:3, 3] = transform[:3, 3] + np.array([0, 0, 1.0])  # Start 1 unit above
        
        # Determine axis limits based on all expected positions
        shape_homogeneous = np.hstack([shape_points, np.ones((shape_points.shape[0], 1))])
        start_shape = np.dot(shape_homogeneous, start_transform.T)[:, :3]
        end_shape = np.dot(shape_homogeneous, transform.T)[:, :3]
        
        all_points = np.vstack([hole_points, start_shape, end_shape])
        x_min, y_min, z_min = np.min(all_points, axis=0)
        x_max, y_max, z_max = np.max(all_points, axis=0)
        
        # Add some padding
        padding = 0.2
        x_range = x_max - x_min + 2 * padding
        y_range = y_max - y_min + 2 * padding
        z_range = z_max - z_min + 2 * padding
        max_range = max(x_range, y_range, z_range)
        
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2
        
        # Function to initialize the animation
        def init():
            ax.clear()
            return []
        
        # Function to update the animation frame
        def update(frame):
            ax.clear()
            
            # Interpolate between start and end transforms
            t = frame / (num_frames - 1)
            current_transform = np.eye(4)
            current_transform[:3, 3] = (1 - t) * start_transform[:3, 3] + t * transform[:3, 3]
            
            # Apply the current transform to the shape
            current_shape = np.dot(shape_homogeneous, current_transform.T)[:, :3]
            
            # Plot the hole (container)
            ax.scatter(hole_points[:, 0], hole_points[:, 1], hole_points[:, 2], 
                      c='red', s=2, alpha=0.5, label='Container with Hole')
            
            # Plot the shape
            ax.scatter(current_shape[:, 0], current_shape[:, 1], current_shape[:, 2], 
                      c='blue', s=5, alpha=0.7, label='Shape')
            
            # Set axis limits
            ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
            ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
            ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Shape Insertion Animation - Frame {frame+1}/{num_frames}')
            
            # Fix aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Add legend
            ax.legend()
            
            return []
        
        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                      init_func=init, blit=True, interval=100)
        
        # Save the animation
        anim.save(filename, writer='pillow', fps=10)
        plt.close(fig)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib.animation as animation

class DatasetVisualizer:
    def __init__(self, dataset_dir="shape_sorting_dataset"):
        """
        Initialize the dataset visualizer.
        
        Args:
            dataset_dir: Directory containing the generated dataset
        """
        self.dataset_dir = dataset_dir
        self.current_index = 0
        self.animation_step = 0
        self.insert_steps = 20  # Number of steps for insertion animation
        self.view_mode = "3D"  # Either "3D" or "GIF"
        
        # Load dataset metadata
        with open(os.path.join(dataset_dir, "dataset_info.json"), "r") as f:
            self.dataset_info = json.load(f)
        
        self.num_samples = len(self.dataset_info["samples"])
        if self.num_samples == 0:
            raise ValueError("No samples found in the dataset")
        
        # Setup the figure for interactive visualization
        self.setup_figure()
        
        # Load the first sample
        self.load_current_sample()
        
    def setup_figure(self):
        """Set up the figure for interactive visualization"""
        self.fig = plt.figure(figsize=(12, 8))
        
        # Set up the 3D axis for visualization
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Add navigation buttons
        self.ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.21, 0.05, 0.1, 0.075])
        self.ax_animate = plt.axes([0.32, 0.05, 0.1, 0.075])
        self.ax_reset = plt.axes([0.43, 0.05, 0.1, 0.075])
        
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_animate = Button(self.ax_animate, 'Animate')
        self.btn_reset = Button(self.ax_reset, 'Reset View')
        
        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)
        self.btn_animate.on_clicked(self.toggle_animation)
        self.btn_reset.on_clicked(self.reset_view)
        
        # Add viewpoint sliders
        self.ax_elev = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.ax_azim = plt.axes([0.1, 0.1, 0.65, 0.03])
        
        self.slider_elev = Slider(self.ax_elev, 'Elevation', 0, 90, valinit=30)
        self.slider_azim = Slider(self.ax_azim, 'Azimuth', 0, 360, valinit=30)
        
        self.slider_elev.on_changed(self.update_elev)
        self.slider_azim.on_changed(self.update_azim)
        
        # Add view mode selector (3D vs animation)
        self.ax_view_mode = plt.axes([0.8, 0.1, 0.15, 0.15])
        self.radio_view_mode = RadioButtons(self.ax_view_mode, ('3D View', 'Animation'))
        self.radio_view_mode.on_clicked(self.change_view_mode)
        
        # Text info
        self.ax_text = plt.figtext(0.5, 0.01, "", ha="center")
        
        # Animation state
        self.animating = False
        self.anim_id = None
        self.gif_frames = []
        self.gif_idx = 0
        self.gif_anim = None
        
    def load_current_sample(self):
        """Load and display the current sample"""
        sample = self.dataset_info["samples"][self.current_index]
        
        # Update info text
        info_text = f"Sample {self.current_index + 1}/{self.num_samples} - Type: {sample['shape_type']}"
        self.ax_text.set_text(info_text)
        
        # Load shape and hole point clouds
        shape_file = os.path.join(self.dataset_dir, sample["shape_file"])
        hole_file = os.path.join(self.dataset_dir, sample["hole_file"])
        
        self.shape_points = np.loadtxt(shape_file, delimiter=',', skiprows=1)
        self.hole_points = np.loadtxt(hole_file, delimiter=',', skiprows=1)
        
        # Get the success transform
        self.transform = np.array(sample["success_transform"])
        
        # Load animation GIF if it exists
        self.gif_path = os.path.join(self.dataset_dir, sample["animation_file"])
        self.gif_frames = []
        if os.path.exists(self.gif_path):
            try:
                gif = Image.open(self.gif_path)
                for i in range(gif.n_frames):
                    gif.seek(i)
                    frame = np.array(gif.convert('RGBA'))
                    self.gif_frames.append(frame)
            except Exception as e:
                print(f"Error loading animation: {e}")
                self.gif_frames = []
        
        # Reset animation step
        self.animation_step = 0
        self.animating = False
        self.gif_idx = 0
        if self.anim_id is not None:
            self.fig.canvas.mpl_disconnect(self.anim_id)
            self.anim_id = None
        
        # Update the visualization based on current view mode
        self.update_visualization()
    
    def get_current_transform(self):
        """Get the current transform based on animation step"""
        if self.animation_step == 0:
            # Initial position (above the hole)
            start_transform = np.eye(4)
            start_transform[:3, 3] = self.transform[:3, 3] + np.array([0, 0, 1.0])  # Start 1 unit above
            return start_transform
        elif self.animation_step >= self.insert_steps:
            # Final position (fully inserted)
            return self.transform
        else:
            # Intermediate position (linear interpolation)
            start_transform = np.eye(4)
            start_transform[:3, 3] = self.transform[:3, 3] + np.array([0, 0, 1.0])
            
            # Interpolate between start and end transforms
            t = self.animation_step / self.insert_steps
            current_transform = np.eye(4)
            current_transform[:3, :3] = self.transform[:3, :3]  # Keep rotation the same
            current_transform[:3, 3] = (1 - t) * start_transform[:3, 3] + t * self.transform[:3, 3]
            return current_transform
    
    def update_visualization(self):
        """Update the visualization with the current sample and animation step"""
        # Handle different view modes
        if self.view_mode == "3D":
            self.update_3d_view()
        else:  # "GIF" mode
            self.update_gif_view()
        
        # Draw the updated figure
        self.fig.canvas.draw_idle()
    
    def update_3d_view(self):
        """Update the 3D point cloud visualization"""
        self.ax.clear()
        
        # Get the current transform for animation
        current_transform = self.get_current_transform()
        
        # Apply the current transform to the shape
        shape_homogeneous = np.hstack([self.shape_points, np.ones((self.shape_points.shape[0], 1))])
        transformed_shape = np.dot(shape_homogeneous, current_transform.T)[:, :3]
        
        # Plot the hole (container)
        self.ax.scatter(self.hole_points[:, 0], self.hole_points[:, 1], self.hole_points[:, 2], 
                        c='red', s=2, alpha=0.5, label='Container with Hole')
        
        # Plot the shape
        self.ax.scatter(transformed_shape[:, 0], transformed_shape[:, 1], transformed_shape[:, 2], 
                        c='blue', s=5, alpha=0.7, label='Shape')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        
        # Set consistent viewing angle
        self.ax.view_init(elev=self.slider_elev.val, azim=self.slider_azim.val)
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        sample = self.dataset_info["samples"][self.current_index]
        self.ax.set_title(f"Shape Type: {sample['shape_type']}")
        
        # Add legend
        self.ax.legend()
    
    def update_gif_view(self):
        """Update the view to show the pre-rendered animation GIF"""
        self.ax.clear()
        
        if len(self.gif_frames) > 0:
            # Display the current frame of the GIF
            self.ax.imshow(self.gif_frames[self.gif_idx])
            self.ax.axis('off')
            self.ax.set_title(f"Animation Frame {self.gif_idx + 1}/{len(self.gif_frames)}")
        else:
            # No GIF available, show a message
            self.ax.text(0.5, 0.5, "No Animation Available", 
                         ha='center', va='center', transform=self.ax.transAxes, fontsize=16)
            self.ax.axis('off')
    
    def next_sample(self, event):
        """Load the next sample"""
        self.current_index = (self.current_index + 1) % self.num_samples
        self.load_current_sample()
    
    def prev_sample(self, event):
        """Load the previous sample"""
        self.current_index = (self.current_index - 1) % self.num_samples
        self.load_current_sample()
    
    def toggle_animation(self, event):
        """Toggle the insertion animation"""
        self.animating = not self.animating
        
        if self.animating:
            # Start the animation
            if self.view_mode == "3D":
                self.animation_step = 0
                self.anim_id = self.fig.canvas.mpl_connect('draw_event', self.animate_3d)
                self.animate_3d(None)
            else:  # "GIF" mode
                self.gif_idx = 0
                self.anim_id = self.fig.canvas.mpl_connect('draw_event', self.animate_gif)
                self.animate_gif(None)
        else:
            # Stop the animation
            if self.anim_id is not None:
                self.fig.canvas.mpl_disconnect(self.anim_id)
                self.anim_id = None
    
    def animate_3d(self, event):
        """Update the 3D animation"""
        if self.animating:
            self.animation_step += 1
            if self.animation_step > self.insert_steps:
                self.animation_step = 0  # Reset to start position
            
            self.update_3d_view()
            
            # Continue the animation
            self.fig.canvas.draw_idle()
            plt.pause(0.1)  # Pause to allow for viewing
            
            if self.animating:  # Check if still animating
                self.fig.canvas.mpl_connect('draw_event', self.animate_3d)
    
    def animate_gif(self, event):
        """Update the GIF animation"""
        if self.animating and len(self.gif_frames) > 0:
            self.gif_idx = (self.gif_idx + 1) % len(self.gif_frames)
            
            self.update_gif_view()
            
            # Continue the animation
            self.fig.canvas.draw_idle()
            plt.pause(0.1)  # Pause to allow for viewing
            
            if self.animating:  # Check if still animating
                self.fig.canvas.mpl_connect('draw_event', self.animate_gif)
    
    def update_elev(self, val):
        """Update the elevation angle"""
        if self.view_mode == "3D":
            self.ax.view_init(elev=val, azim=self.ax.azim)
            self.fig.canvas.draw_idle()
    
    def update_azim(self, val):
        """Update the azimuth angle"""
        if self.view_mode == "3D":
            self.ax.view_init(elev=self.ax.elev, azim=val)
            self.fig.canvas.draw_idle()
    
    def reset_view(self, event):
        """Reset the view to default"""
        self.slider_elev.set_val(30)
        self.slider_azim.set_val(30)
        self.fig.canvas.draw_idle()
    
    def change_view_mode(self, label):
        """Change the view mode between 3D and Animation"""
        if label == '3D View':
            self.view_mode = "3D"
        else:  # Animation
            self.view_mode = "GIF"
        
        # Stop any running animation
        self.animating = False
        if self.anim_id is not None:
            self.fig.canvas.mpl_disconnect(self.anim_id)
            self.anim_id = None
        
        # Update visualization
        self.update_visualization()
    
    def show(self):
        """Show the interactive visualization"""
        plt.show()

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate and visualize a shape sorting dataset')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--visualize_only', action='store_true', help='Skip generation and visualize existing dataset')
    parser.add_argument('--dataset_dir', type=str, default='shape_sorting_dataset', help='Dataset directory')
    
    args = parser.parse_args()
    
    # Generate the dataset if not in visualize-only mode
    if not args.visualize_only:
        print(f"Generating a dataset with {args.samples} samples...")
        generator = ShapeSortingDatasetGenerator(output_dir=args.dataset_dir, num_samples=args.samples)
        generator.generate_dataset()
        print("Dataset generation complete.")
    
    # Launch the interactive visualizer
    print("Launching interactive visualizer...")
    visualizer = DatasetVisualizer(dataset_dir=args.dataset_dir)
    visualizer.show()

if __name__ == "__main__":
    main()