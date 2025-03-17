import os
import numpy as np
import pybullet as p
import time
import json

from scipy.spatial.transform import Rotation
import cv2
import matplotlib.pyplot as plt

import open3d as o3d

class SceneRecorder:
    def __init__(self, 
                 num_views=8,              # Number of camera views around the scene
                 camera_distance=2.0,      # Distance of cameras from center point
                 camera_height=1.0,        # Height of cameras
                 target_point=[0, 0, 0],   # Center point to look at
                 camera_up_vector=[0, 0, 1], # Up vector for camera orientation
                 image_width=512,          # Width of captured images
                 image_height=512,         # Height of captured images
                 near_plane=0.01,          # Near plane for depth capture
                 far_plane=10.0):          # Far plane for depth capture
        
        self.num_views = num_views
        self.camera_distance = camera_distance
        self.camera_height = camera_height
        self.target_point = target_point
        self.camera_up_vector = camera_up_vector
        self.image_width = image_width
        self.image_height = image_height
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.fov = 60  # Field of view in degrees
        
    def _calculate_camera_poses(self):
        """Calculate camera positions in a circle around the target point."""
        camera_poses = []
        
        for i in range(self.num_views):
            angle = 2 * np.pi * i / self.num_views
            
            # Calculate camera position in a circle around the target
            x = self.target_point[0] + self.camera_distance * np.cos(angle)
            y = self.target_point[1] + self.camera_distance * np.sin(angle)
            z = self.target_point[2] + self.camera_height
            
            camera_position = [x, y, z]
            
            # Create view matrix (camera extrinsics)
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_position,
                cameraTargetPosition=self.target_point,
                cameraUpVector=self.camera_up_vector
            )
            
            # Create projection matrix (camera intrinsics)
            aspect = float(self.image_width) / float(self.image_height)
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=self.fov,
                aspect=aspect,
                nearVal=self.near_plane,
                farVal=self.far_plane
            )
            
            # Calculate camera intrinsics matrix for point cloud generation
            fov_rad = np.deg2rad(self.fov)
            fx = fy = self.image_height / (2 * np.tan(fov_rad / 2))
            cx = self.image_width / 2
            cy = self.image_height / 2
            
            intrinsic_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # Calculate camera extrinsics (world to camera transform)
            # Convert PyBullet view matrix to a standard 4x4 transform matrix
            view_mat = np.array(view_matrix).reshape(4, 4, order='F')
            
            # The 3x3 rotation matrix (world to camera)
            rotation_matrix = view_mat[:3, :3]
            
            # The camera position can be derived from view matrix or used directly
            translation = -np.dot(rotation_matrix, np.array(camera_position))
            
            # Construct camera extrinsic matrix (world to camera transform)
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = rotation_matrix
            extrinsic_matrix[:3, 3] = translation
            
            # Also calculate camera to world transform for point cloud conversion
            camera_to_world = np.linalg.inv(extrinsic_matrix)
            
            camera_poses.append({
                'position': camera_position,
                'view_matrix': view_matrix,
                'projection_matrix': projection_matrix,
                'angle': angle,
                'intrinsic_matrix': intrinsic_matrix,
                'extrinsic_matrix': extrinsic_matrix,
                'camera_to_world': camera_to_world
            })
            
        return camera_poses
    
    def _capture_rgbd(self, camera_pose):
        """Capture RGB and depth images from a specific camera pose."""
        # Render image from the camera
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=camera_pose['view_matrix'],
            projectionMatrix=camera_pose['projection_matrix'],
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert RGB from RGBA to RGB
        rgb_img = np.array(rgb_img)
        rgb_img = rgb_img[:, :, :3]
        
        # Convert depth buffer to actual distances
        depth_real = self.far_plane * self.near_plane / (self.far_plane - (self.far_plane - self.near_plane) * depth_img)
        
        return rgb_img, depth_real, seg_img
    
    def _depth_to_point_cloud(self, depth_img, rgb_img, camera_pose):
        """
        Convert depth image to point cloud in world coordinates using accurate camera transforms.
        
        This implementation uses accurate camera intrinsics and extrinsics to transform
        from pixel coordinates to world coordinates.
        """
        # Get camera intrinsics
        intrinsic = camera_pose['intrinsic_matrix']
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        # Get camera to world transform
        camera_to_world = camera_pose['camera_to_world']
        
        # Create pixel coordinate grid
        height, width = depth_img.shape
        v, u = np.mgrid[0:height, 0:width]
        
        # Convert to flattened 1D arrays for vectorized operations
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_img.flatten()
        
        # Filter out invalid or unreliable depth values
        valid_depth_mask = (depth_flat > self.near_plane) & (depth_flat < self.far_plane)
        u_valid = u_flat[valid_depth_mask]
        v_valid = v_flat[valid_depth_mask]
        z_valid = depth_flat[valid_depth_mask]
        
        # Convert pixel coordinates to camera coordinates
        x_camera = (u_valid - cx) * z_valid / fx
        y_camera = (v_valid - cy) * z_valid / fy
        
        # Create homogeneous points in camera coordinate system
        # Note: In camera coordinates, Z is forward, X right, Y down
        # so we need to adjust to match Open3D's coordinate system
        camera_points = np.vstack((
            x_camera,       # X (right)
            -y_camera,      # Y (up) - flipped from camera's Y which is down
            -z_valid,       # Z (forward) - negative because camera looks along negative Z
            np.ones_like(x_camera)
        )).T
        
        # Transform from camera to world coordinates
        world_points = np.dot(camera_points, camera_to_world.T)[:, :3]
        
        # Get colors for each point
        rgb_colors = rgb_img.reshape(-1, 3)[valid_depth_mask] / 255.0
        
        return world_points, rgb_colors
    
    def _depth_to_segmented_point_cloud(self, depth_img, rgb_img, seg_img, camera_pose):
        """
        Convert depth image to point cloud in world coordinates with segmentation info.
        
        This extends the original _depth_to_point_cloud method to also track object IDs
        from the segmentation image for each point.
        
        Parameters:
        depth_img (np.ndarray): Depth image from PyBullet
        rgb_img (np.ndarray): RGB image from PyBullet
        seg_img (np.ndarray): Segmentation image from PyBullet
        camera_pose (dict): Camera pose information
        
        Returns:
        tuple: (world_points, rgb_colors, object_ids)
        """
        # Get camera intrinsics
        intrinsic = camera_pose['intrinsic_matrix']
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        # Get camera to world transform
        camera_to_world = camera_pose['camera_to_world']
        
        # Create pixel coordinate grid
        height, width = depth_img.shape
        v, u = np.mgrid[0:height, 0:width]
        
        # Convert to flattened 1D arrays for vectorized operations
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_img.flatten()
        
        # Get object IDs from segmentation image
        seg_flat = seg_img.flatten()
        
        # Filter out invalid or unreliable depth values
        valid_depth_mask = (depth_flat > self.near_plane) & (depth_flat < self.far_plane)
        u_valid = u_flat[valid_depth_mask]
        v_valid = v_flat[valid_depth_mask]
        z_valid = depth_flat[valid_depth_mask]
        seg_valid = seg_flat[valid_depth_mask]
        
        # Convert pixel coordinates to camera coordinates
        x_camera = (u_valid - cx) * z_valid / fx
        y_camera = (v_valid - cy) * z_valid / fy
        
        # Create homogeneous points in camera coordinate system
        # Note: In camera coordinates, Z is forward, X right, Y down
        # so we need to adjust to match Open3D's coordinate system
        camera_points = np.vstack((
            x_camera,       # X (right)
            -y_camera,      # Y (up) - flipped from camera's Y which is down
            -z_valid,       # Z (forward) - negative because camera looks along negative Z
            np.ones_like(x_camera)
        )).T
        
        # Transform from camera to world coordinates
        world_points = np.dot(camera_points, camera_to_world.T)[:, :3]
        
        # Get colors for each point
        rgb_colors = rgb_img.reshape(-1, 3)[valid_depth_mask] / 255.0
        
        # Get object IDs for each point
        object_ids = seg_valid
        
        return world_points, rgb_colors, object_ids
    
    def _segment_point_cloud(self, points, colors, object_ids):
        """
        Segment the point cloud by object ID.
        
        Parameters:
        points (np.ndarray): Point coordinates
        colors (np.ndarray): Point colors
        object_ids (np.ndarray): Object IDs for each point
        
        Returns:
        dict: Dictionary mapping object IDs to segmented point clouds
        """
        # Get unique object IDs (excluding background ID 0)
        unique_ids = np.unique(object_ids)
        
        # Create dictionary to store segmented point clouds
        segmented_pcds = {}
        
        # For each object ID, create a separate point cloud
        for obj_id in unique_ids:
            # Skip background (usually ID 0)
            if obj_id == 0:
                continue
                
            # Create mask for this object ID
            mask = (object_ids == obj_id)
            
            # Extract points and colors for this object
            obj_points = points[mask]
            obj_colors = colors[mask]
            
            # Create Open3D point cloud for this object
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
            obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
            
            # Estimate normals
            if len(obj_points) > 10:  # Need enough points for normal estimation
                obj_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                obj_pcd.orient_normals_towards_camera_location(np.array(self.target_point))
            
            # Store in dictionary
            segmented_pcds[int(obj_id)] = obj_pcd
        
        return segmented_pcds
    
    def _create_o3d_point_cloud_with_normals(self, points, colors):
        """Create an Open3D point cloud with normal estimation."""
        if len(points) == 0:
            return None
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals - this is important for better registration and visualization
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Orient normals to point approximately towards the camera/viewpoint
        pcd.orient_normals_towards_camera_location(np.array(self.target_point))
        
        return pcd
    
    def _get_scene_objects(self):
        """Get all objects in the scene and their properties."""
        scene_objects = []
        
        # Get the number of bodies in the simulation
        num_bodies = p.getNumBodies()
        
        for body_id in range(num_bodies):
            # Get basic body information
            body_info = p.getBodyInfo(body_id)
            
            # Get position and orientation
            position, orientation = p.getBasePositionAndOrientation(body_id)
            
            # Get linear and angular velocity
            linear_vel, angular_vel = p.getBaseVelocity(body_id)
            
            # Get all joints and their states
            num_joints = p.getNumJoints(body_id)
            joints = []
            
            for joint_idx in range(num_joints):
                joint_info = p.getJointInfo(body_id, joint_idx)
                joint_state = p.getJointState(body_id, joint_idx)
                
                joint_data = {
                    'joint_index': joint_idx,
                    'joint_name': joint_info[1].decode('utf-8') if isinstance(joint_info[1], bytes) else str(joint_info[1]),
                    'joint_type': joint_info[2],
                    'position': joint_state[0],
                    'velocity': joint_state[1],
                    'reaction_forces': joint_state[2],
                    'applied_torque': joint_state[3]
                }
                
                joints.append(joint_data)
            
            # Compile all information for this object

            object_data = {
                'id': body_id,
                'name': body_info[1].decode('utf-8') if isinstance(body_info[1], bytes) else str(body_info[1]),
                'position': position,
                'orientation': orientation,
                'linear_velocity': linear_vel,
                'angular_velocity': angular_vel,
                'num_joints': num_joints,
                'joints': joints
            }
            
            scene_objects.append(object_data)
            
        return scene_objects
    
    def _merge_point_clouds(self, point_clouds, colors, camera_poses):
        """
        Merge multiple point clouds by simply combining them.
        
        This function:
        1. Concatenates all point clouds from different views
        2. Filters points within a specified distance range from the target point
        3. Downsamples the point cloud to reduce redundancy
        """
        # Filter out empty point clouds
        all_points = []
        all_colors = []
        
        for points, colors_i in zip(point_clouds, colors):
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors_i)
        
        if not all_points:
            return None
            
        # Concatenate all points and colors
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # Create Open3D point cloud
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(combined_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # Filter points by distance from target point
        # Calculate distances from each point to the target point
        target_point = np.array(self.target_point)
        points_array = np.asarray(merged_pcd.points)
        distances = np.linalg.norm(points_array - target_point, axis=1)
        
        # Set a reasonable maximum distance (e.g., 1.5 times the camera distance)
        max_distance = self.camera_distance * 1.0
        
        # Create a mask for points within the specified range
        mask = distances < max_distance
        
        # Filter the point cloud
        filtered_points = points_array[mask]
        filtered_colors = np.asarray(merged_pcd.colors)[mask]
        
        # Create a new point cloud with filtered points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        # Downsample the point cloud using voxel grid
        downsampled_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.01)
        
        # Optionally limit to a maximum number of points
        max_points = 100000  # Adjust based on your requirements
        points_array = np.asarray(downsampled_pcd.points)
        colors_array = np.asarray(downsampled_pcd.colors)
        
        if len(points_array) > max_points:
            # Randomly sample points without replacement
            indices = np.random.choice(len(points_array), max_points, replace=False)
            points_array = points_array[indices]
            colors_array = colors_array[indices]
            
            # Create final point cloud with limited points
            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(points_array)
            final_pcd.colors = o3d.utility.Vector3dVector(colors_array)
            return final_pcd
        
        return downsampled_pcd
    
    def _merge_segmented_point_clouds(self, all_segmented_point_clouds):
        """
        Merge segmented point clouds from multiple views, maintaining object segmentation.
        
        Parameters:
        all_segmented_point_clouds (list): List of dictionaries mapping object IDs to point clouds
        
        Returns:
        dict: Dictionary mapping object IDs to merged point clouds
        tuple: (merged_points, merged_colors, merged_object_ids)
        """
        # Dictionary to store merged point clouds for each object ID
        merged_segmented_pcds = {}
        
        # Get all unique object IDs across all views
        all_obj_ids = set()
        for seg_pcds in all_segmented_point_clouds:
            all_obj_ids.update(seg_pcds.keys())
        
        # For each object ID, merge point clouds from all views
        for obj_id in all_obj_ids:
            # Collect all point clouds for this object ID
            obj_pcds = []
            for seg_pcds in all_segmented_point_clouds:
                if obj_id in seg_pcds:
                    obj_pcds.append(seg_pcds[obj_id])
            
            if not obj_pcds:
                continue
                
            # Combine all point clouds for this object
            combined_pcd = o3d.geometry.PointCloud()
            
            for pcd in obj_pcds:
                combined_pcd += pcd
            
            # Filter by distance from target point
            target_point = np.array(self.target_point)
            points_array = np.asarray(combined_pcd.points)
            
            if len(points_array) == 0:
                continue
                
            # Calculate distances from target point
            distances = np.linalg.norm(points_array - target_point, axis=1)
            
            # Filter points within reasonable distance
            max_distance = self.camera_distance * 1.0
            mask = distances < max_distance
            
            # Apply filter
            filtered_points = points_array[mask]
            filtered_colors = np.asarray(combined_pcd.colors)[mask]
            
            if len(filtered_points) == 0:
                continue
                
            # Create filtered point cloud
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
            
            # Downsample
            downsampled_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.01)
            
            # Store in dictionary
            merged_segmented_pcds[obj_id] = downsampled_pcd
        
        # Also create a combined point cloud with object IDs
        all_points = []
        all_colors = []
        all_obj_ids = []
        
        for obj_id, pcd in merged_segmented_pcds.items():
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            ids = np.ones(len(points), dtype=int) * obj_id
            
            all_points.append(points)
            all_colors.append(colors)
            all_obj_ids.append(ids)
        
        if not all_points:
            return merged_segmented_pcds, (None, None, None)
            
        # Concatenate all data
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        merged_object_ids = np.hstack(all_obj_ids)
        
        return merged_segmented_pcds, (merged_points, merged_colors, merged_object_ids)
    
    def record_scene(self, output_dir=None, frame_idx=0, save=False):
        """
        Record the current PyBullet scene from multiple viewpoints.
        
        Args:
            output_dir: Directory to save the results (if save=True)
            frame_idx: Frame index for saving
            save: Whether to save the results to disk
            
        Returns:
            scene_data: Dictionary containing all scene information
        """
        # Calculate camera poses
        camera_poses = self._calculate_camera_poses()
        
        # Get scene objects and their properties
        scene_objects = self._get_scene_objects()
        
        # Capture RGB-D images and construct point clouds from each view
        rgb_images = []
        depth_images = []
        seg_images = []
        point_clouds = []
        point_colors = []
        
        for i, camera_pose in enumerate(camera_poses):
            # Capture RGB-D image
            rgb_img, depth_img, seg_img = self._capture_rgbd(camera_pose)
            
            # Convert depth image to point cloud
            points, colors = self._depth_to_point_cloud(depth_img, rgb_img, camera_pose)
            
            # Store the results
            rgb_images.append(rgb_img)
            depth_images.append(depth_img)
            seg_images.append(seg_img)
            point_clouds.append(points)
            point_colors.append(colors)
        
        # Merge point clouds from all views
        merged_point_cloud = self._merge_point_clouds(point_clouds, point_colors, camera_poses)
        
        # Create scene data dictionary
        scene_data = {
            'frame_idx': frame_idx,
            'scene_objects': scene_objects,
            'camera_poses': [
                {
                    'position': pose['position'],
                    'angle': pose['angle'],
                    'view_matrix': pose['view_matrix'],
                    'projection_matrix': pose['projection_matrix'],
                    'intrinsic_matrix': pose['intrinsic_matrix'].tolist(),
                    'extrinsic_matrix': pose['extrinsic_matrix'].tolist(),
                } for pose in camera_poses
            ],
            'rgb_images': rgb_images,
            'depth_images': depth_images,
            'segmentation_images': seg_images,
            'point_clouds': point_clouds,
            'point_colors': point_colors,
            'merged_point_cloud': merged_point_cloud
        }
        
        # Save the results if requested
        if save and output_dir is not None:
            self._save_scene_data(scene_data, output_dir, frame_idx)
        
        return scene_data
    
    def record_scene_with_segmentation(self, output_dir=None, frame_idx=0, save=False):
        """
        Record the current PyBullet scene with object segmentation from multiple viewpoints.
        
        Args:
            output_dir: Directory to save the results (if save=True)
            frame_idx: Frame index for saving
            save: Whether to save the results to disk
            
        Returns:
            scene_data: Dictionary containing all scene information with segmentation
        """
        # Calculate camera poses
        camera_poses = self._calculate_camera_poses()
        
        # Get scene objects and their properties
        scene_objects = self._get_scene_objects()
        
        # Capture RGB-D images and construct point clouds from each view
        rgb_images = []
        depth_images = []
        seg_images = []
        point_clouds = []
        point_colors = []
        point_object_ids = []
        segmented_point_clouds = []
        
        for i, camera_pose in enumerate(camera_poses):
            # Capture RGB-D image
            rgb_img, depth_img, seg_img = self._capture_rgbd(camera_pose)
            
            # Convert depth image to point cloud with segmentation
            points, colors, object_ids = self._depth_to_segmented_point_cloud(depth_img, rgb_img, seg_img, camera_pose)
            
            # Segment point cloud by object ID
            seg_pcds = self._segment_point_cloud(points, colors, object_ids)
            
            # Store the results
            rgb_images.append(rgb_img)
            depth_images.append(depth_img)
            seg_images.append(seg_img)
            point_clouds.append(points)
            point_colors.append(colors)
            point_object_ids.append(object_ids)
            segmented_point_clouds.append(seg_pcds)
        
        # Merge segmented point clouds
        merged_segmented_pcds, (merged_points, merged_colors, merged_object_ids) = self._merge_segmented_point_clouds(segmented_point_clouds)
        
        # Also create the regular merged point cloud
        merged_point_cloud = self._merge_point_clouds(point_clouds, point_colors, camera_poses)
        
        # Create scene data dictionary
        scene_data = {
            'frame_idx': frame_idx,
            'scene_objects': scene_objects,
            'camera_poses': [
                {
                    'position': pose['position'],
                    'angle': pose['angle'],
                    'view_matrix': pose['view_matrix'],
                    'projection_matrix': pose['projection_matrix'],
                    'intrinsic_matrix': pose['intrinsic_matrix'].tolist(),
                    'extrinsic_matrix': pose['extrinsic_matrix'].tolist(),
                } for pose in camera_poses
            ],
            'rgb_images': rgb_images,
            'depth_images': depth_images,
            'segmentation_images': seg_images,
            'point_clouds': point_clouds,
            'point_colors': point_colors,
            'point_object_ids': point_object_ids,
            'segmented_point_clouds': segmented_point_clouds,
            'merged_point_cloud': merged_point_cloud,
            'merged_segmented_point_clouds': merged_segmented_pcds,
            'merged_points_with_ids': (merged_points, merged_colors, merged_object_ids)
        }
        
        # Save the results if requested
        if save and output_dir is not None:
            self._save_scene_data_with_segmentation(scene_data, output_dir, frame_idx)
        
        return scene_data
    
    def _save_scene_data(self, scene_data, output_dir, frame_idx):
        """Save all scene data to disk."""
        # Create frame directory
        frame_dir = os.path.join(output_dir, f'scene_frame_{frame_idx}')
        os.makedirs(frame_dir, exist_ok=True)
        
        # Create subdirectories
        images_dir = os.path.join(frame_dir, 'images')
        point_clouds_dir = os.path.join(frame_dir, 'point_clouds')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(point_clouds_dir, exist_ok=True)
        
        # Save RGB and depth images
        for i in range(len(scene_data['rgb_images'])):
            # Save RGB image
            rgb_path = os.path.join(images_dir, f'view_{i}_rgb.png')
            cv2.imwrite(rgb_path, cv2.cvtColor(scene_data['rgb_images'][i], cv2.COLOR_RGB2BGR))
            
            # Save depth image (normalized for visualization)
            depth = scene_data['depth_images'][i]
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_path = os.path.join(images_dir, f'view_{i}_depth.png')
            cv2.imwrite(depth_path, depth_norm)
            
            # Save depth image as numpy array (for precise values)
            depth_np_path = os.path.join(images_dir, f'view_{i}_depth.npy')
            np.save(depth_np_path, depth)
            
            # Save segmentation image
            seg_path = os.path.join(images_dir, f'view_{i}_seg.png')
            cv2.imwrite(seg_path, scene_data['segmentation_images'][i])
            
            # Save individual point cloud
            if len(scene_data['point_clouds'][i]) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(scene_data['point_clouds'][i])
                pcd.colors = o3d.utility.Vector3dVector(scene_data['point_colors'][i])
                pcd_path = os.path.join(point_clouds_dir, f'view_{i}_points.ply')
                o3d.io.write_point_cloud(pcd_path, pcd)
        
        # Save merged point cloud
        if scene_data['merged_point_cloud'] is not None:
            merged_pcd_path = os.path.join(point_clouds_dir, 'merged_point_cloud.ply')
            o3d.io.write_point_cloud(merged_pcd_path, scene_data['merged_point_cloud'])
        
        # Save camera poses and scene objects as JSON
        metadata = {
            'frame_idx': frame_idx,
            'camera_poses': [
                {
                    'position': pose['position'],
                    'angle': pose['angle'],
                    # Convert matrices to lists for JSON serialization
                    'view_matrix': np.array(pose['view_matrix']).reshape(4, 4, order='F').tolist(),
                    'projection_matrix': np.array(pose['projection_matrix']).reshape(4, 4, order='F').tolist(),
                    'intrinsic_matrix': pose['intrinsic_matrix'],
                    'extrinsic_matrix': pose['extrinsic_matrix']
                } for pose in scene_data['camera_poses']
            ],
            'scene_objects': scene_data['scene_objects']
        }
        
        metadata_path = os.path.join(frame_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            
        print(f"Scene data saved to {frame_dir}")
    
    def _save_scene_data_with_segmentation(self, scene_data, output_dir, frame_idx):
        """Save all scene data to disk, including segmented point clouds."""
        # Create frame directory
        frame_dir = os.path.join(output_dir, f'scene_frame_{frame_idx}')
        os.makedirs(frame_dir, exist_ok=True)
        
        # Create subdirectories
        images_dir = os.path.join(frame_dir, 'images')
        point_clouds_dir = os.path.join(frame_dir, 'point_clouds')
        segmented_pcds_dir = os.path.join(point_clouds_dir, 'segmented')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(point_clouds_dir, exist_ok=True)
        os.makedirs(segmented_pcds_dir, exist_ok=True)
        
        # Save RGB, depth, and segmentation images
        for i in range(len(scene_data['rgb_images'])):
            # Save RGB image
            rgb_path = os.path.join(images_dir, f'view_{i}_rgb.png')
            cv2.imwrite(rgb_path, cv2.cvtColor(scene_data['rgb_images'][i], cv2.COLOR_RGB2BGR))
            
            # Save depth image (normalized for visualization)
            depth = scene_data['depth_images'][i]
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_path = os.path.join(images_dir, f'view_{i}_depth.png')
            cv2.imwrite(depth_path, depth_norm)
            
            # Save depth image as numpy array (for precise values)
            depth_np_path = os.path.join(images_dir, f'view_{i}_depth.npy')
            np.save(depth_np_path, depth)
            
            # Save segmentation image
            seg_path = os.path.join(images_dir, f'view_{i}_seg.png')
            cv2.imwrite(seg_path, scene_data['segmentation_images'][i])
            
            # Save individual point cloud with object IDs
            if len(scene_data['point_clouds'][i]) > 0:
                # Save complete point cloud for this view
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(scene_data['point_clouds'][i])
                pcd.colors = o3d.utility.Vector3dVector(scene_data['point_colors'][i])
                pcd_path = os.path.join(point_clouds_dir, f'view_{i}_points.ply')
                o3d.io.write_point_cloud(pcd_path, pcd)
                
                # Save object IDs as numpy array
                obj_ids_path = os.path.join(point_clouds_dir, f'view_{i}_object_ids.npy')
                np.save(obj_ids_path, scene_data['point_object_ids'][i])
                
                # Save view's segmented point clouds by object ID
                view_seg_dir = os.path.join(segmented_pcds_dir, f'view_{i}')
                os.makedirs(view_seg_dir, exist_ok=True)
                
                for obj_id, obj_pcd in scene_data['segmented_point_clouds'][i].items():
                    obj_pcd_path = os.path.join(view_seg_dir, f'object_{obj_id}.ply')
                    o3d.io.write_point_cloud(obj_pcd_path, obj_pcd)
        
        # Save merged point cloud
        if scene_data['merged_point_cloud'] is not None:
            merged_pcd_path = os.path.join(point_clouds_dir, 'merged_point_cloud.ply')
            o3d.io.write_point_cloud(merged_pcd_path, scene_data['merged_point_cloud'])
        
        # Save merged segmented point clouds
        for obj_id, obj_pcd in scene_data['merged_segmented_point_clouds'].items():
            obj_pcd_path = os.path.join(segmented_pcds_dir, f'merged_object_{obj_id}.ply')
            o3d.io.write_point_cloud(obj_pcd_path, obj_pcd)
        
        # Save merged points with object IDs if available
        if scene_data['merged_points_with_ids'][0] is not None:
            merged_points, merged_colors, merged_object_ids = scene_data['merged_points_with_ids']
            
            # Save as a single colored point cloud
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
            merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            merged_pcd_path = os.path.join(segmented_pcds_dir, 'merged_all_objects.ply')
            o3d.io.write_point_cloud(merged_pcd_path, merged_pcd)
            
            # Save object IDs separately
            merged_ids_path = os.path.join(segmented_pcds_dir, 'merged_object_ids.npy')
            np.save(merged_ids_path, merged_object_ids)
        
        # Save camera poses and scene objects as JSON
        metadata = {
            'frame_idx': frame_idx,
            'camera_poses': [
                {
                    'position': pose['position'],
                    'angle': pose['angle'],
                    # Convert matrices to lists for JSON serialization
                    'view_matrix': np.array(pose['view_matrix']).reshape(4, 4, order='F').tolist(),
                    'projection_matrix': np.array(pose['projection_matrix']).reshape(4, 4, order='F').tolist(),
                    'intrinsic_matrix': pose['intrinsic_matrix'],
                    'extrinsic_matrix': pose['extrinsic_matrix']
                } for pose in scene_data['camera_poses']
            ],
            'scene_objects': scene_data['scene_objects'],
            # Add mapping from object IDs to object names
            'object_id_mapping': {
                str(obj['id']): obj['name'] for obj in scene_data['scene_objects']
            }
        }
        
        metadata_path = os.path.join(frame_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            
        print(f"Scene data with segmentation saved to {frame_dir}")

# Helper class for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


# Example usage
if __name__ == "__main__":
    import pybullet as p
    import pybullet_data
    
    # Initialize PyBullet
    #p.connect(p.GUI)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load some objects
    plane_id = p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
    cube_id = p.loadURDF("cube.urdf", [0, 0, 1])
    
    # Create the scene recorder
    recorder = SceneRecorder(
        num_views=16,
        camera_distance=2.0,
        camera_height=1.5,
        target_point=[0, 0, 0.5]
    )
    
    # Record the scene with segmentation
    scene_data = recorder.record_scene_with_segmentation(
        output_dir="./scene_data_segmented", 
        frame_idx=0, 
        save=True
    )
    
    # Visualize one of the RGB images
    #plt.figure(figsize=(10, 8))
    #plt.imshow(scene_data['rgb_images'][0])
    #plt.title("View 0 RGB Image")
    #plt.show()
    
    # Optionally visualize the merged point cloud
    if scene_data['merged_point_cloud'] is not None:
        o3d.visualization.draw_geometries([scene_data['merged_point_cloud']])
    
    # Visualize segmented point clouds
    #for obj_id, obj_pcd in scene_data['merged_segmented_point_clouds'].items():
    #    print(f"Visualizing object {obj_id}")
    #    o3d.visualization.draw_geometries([obj_pcd])
    
    # Disconnect from PyBullet
    p.disconnect()