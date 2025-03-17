# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 01:14:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-09 23:54:31
import os
import numpy as np
import glob
from pathlib import Path
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader

# ----- Directory Setup -----
BASE_DATA_DIR = "data"

# ----- Scene Dataset -----
class SceneDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 scene_pattern="scene_frame_*",
                 transform=None,
                 n_points=None,
                 load_views=False):
        """
        Args:
            root_dir (str): Path to the data directory
            scene_pattern (str): Pattern to match scene directories (default: "scene_frame_*")
            transform (callable, optional): Optional transform to be applied on the point clouds
            n_points (int, optional): If specified, sample this number of points from each point cloud
            load_views (bool): Whether to load individual view directories (default: False)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.n_points = n_points
        self.load_views = load_views
        
        # Find all scene directories
        self.scene_dirs = sorted(glob.glob(str(self.root_dir / "mechanism_data" / scene_pattern)))
        
        # Build an index of all scenes and their components
        self.scenes_index = self._build_index()
    
    def _build_index(self):
        """Build an index of all scenes and their components"""
        scenes_index = []
        
        for scene_dir in self.scene_dirs:
            scene_path = Path(scene_dir)
            scene_name = scene_path.name
            
            # Find paths to point cloud files
            point_cloud_dir = scene_path / "point_clouds" / "segmented"
            
            if not point_cloud_dir.exists():
                continue
                
            # Get merged component files
            merged_component_files = sorted(point_cloud_dir.glob("merged_object_*.ply"))
            merged_all_file = point_cloud_dir / "merged_all_objects.ply"
            merged_all_exists = merged_all_file.exists()
            
            # Get view directories if needed
            view_dirs = []
            if self.load_views:
                view_dirs = sorted([d for d in point_cloud_dir.glob("view_*") if d.is_dir()])
            
            scenes_index.append({
                "scene_name": scene_name,
                "scene_dir": str(scene_path),
                "point_cloud_dir": str(point_cloud_dir),
                "merged_components": [str(f) for f in merged_component_files],
                "merged_all": str(merged_all_file) if merged_all_exists else None,
                "view_dirs": [str(d) for d in view_dirs],
                "num_components": len(merged_component_files)
            })
            
        return scenes_index
    
    def __len__(self):
        return len(self.scenes_index)
    
    def load_point_cloud(self, file_path):
        """Load a point cloud from a file path"""
        if not os.path.exists(file_path):
            return None
            
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            
            if self.n_points is not None and points.shape[0] > self.n_points:
                # Randomly sample n_points from the point cloud
                indices = np.random.choice(points.shape[0], self.n_points, replace=False)
                points = points[indices]
                
            # Check if the point cloud has normals
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
                if self.n_points is not None and normals.shape[0] > self.n_points:
                    normals = normals[indices]
                result = {"points": points, "normals": normals}
            else:
                result = {"points": points}
                
            # Check if the point cloud has colors
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                if self.n_points is not None and colors.shape[0] > self.n_points:
                    colors = colors[indices]
                result["colors"] = colors
                
            return result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        """
        Get the point clouds for a specific scene.
        
        Returns:
            dict: Dictionary containing:
                - scene_name: Name of the scene
                - components: List of component point clouds
                - merged_all: Merged point cloud of all components
                - views: List of view directories (if load_views=True)
        """
        scene_info = self.scenes_index[idx]
        
        # Load merged component point clouds
        components = []
        for comp_file in scene_info["merged_components"]:
            pcd_data = self.load_point_cloud(comp_file)
            if pcd_data is not None:
                components.append(pcd_data)
        
        # Load merged all point cloud
        merged_all = None
        if scene_info["merged_all"] is not None:
            merged_all = self.load_point_cloud(scene_info["merged_all"])
        
        # Load views if needed
        views = []
        if self.load_views and scene_info["view_dirs"]:
            # For simplicity, we're just listing the view directories here
            # You can extend this to load specific files from each view
            views = scene_info["view_dirs"]
        
        result = {
            "scene_name": scene_info["scene_name"],
            "components": components,
            "merged_all": merged_all,
            "num_components": scene_info["num_components"],
        }
        
        if self.load_views:
            result["views"] = views
            
        if self.transform:
            result = self.transform(result)
            
        return result


def collate_variable_components(batch):
    """
    Custom collate function to handle batches with variable numbers of components.
    Each item is a dictionary with variable-length lists.
    """
    batch_dict = {
        "scene_names": [],
        "components": [],
        "components_count": [],
        "merged_all": []
    }
    
    for item in batch:
        batch_dict["scene_names"].append(item["scene_name"])
        batch_dict["components"].append(item["components"])
        batch_dict["components_count"].append(len(item["components"]))
        
        if item["merged_all"] is not None:
            # Convert numpy arrays to tensors if needed
            if isinstance(item["merged_all"]["points"], np.ndarray):
                item["merged_all"]["points"] = torch.from_numpy(item["merged_all"]["points"]).float()
            batch_dict["merged_all"].append(item["merged_all"]["points"])
        else:
            batch_dict["merged_all"].append(None)
    
    # Convert counts to tensor
    batch_dict["components_count"] = torch.tensor(batch_dict["components_count"])
    
    return batch_dict


# Example usage
if __name__ == "__main__":
    # Example data directory path - adjust as needed
    data_dir = "data"
    
    # Create the dataset
    dataset = SceneDataset(
        root_dir=data_dir,
        n_points=1024,  # Sample 1024 points per point cloud
        load_views=False  # Don't load view directories
    )
    
    print(f"Found {len(dataset)} scenes")
    
    # Print information about the first scene
    if len(dataset) > 0:
        scene_data = dataset[0]
        print(f"Scene name: {scene_data['scene_name']}")
        print(f"Number of components: {scene_data['num_components']}")
        
        for i, comp in enumerate(scene_data['components']):
            print(f"Component {i+1} point cloud shape: {comp['points'].shape}")
            
        if scene_data['merged_all'] is not None:
            print(f"Merged point cloud shape: {scene_data['merged_all']['points'].shape}")
    
    # Create a data loader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_variable_components
    )
    
    # Iterate through the first batch
    for batch in dataloader:
        print(f"Batch size: {len(batch['scene_names'])}")
        print(f"Scene names: {batch['scene_names']}")
        print(f"Components per scene: {batch['components_count']}")
        print(batch.keys())
        for components in batch["components"]:
            for comp in components:
                print(comp["points"].shape, comp["colors"].shape)
        break