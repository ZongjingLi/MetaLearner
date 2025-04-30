import numpy as np
import pybullet as p
import pybullet_data
import random
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.stats import multivariate_normal
from pathlib import Path
import json
import pickle

from scipy.stats import multivariate_normal
from pathlib import Path
import json
import pickle
from tqdm import tqdm

@dataclass
class ObjectState:
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # quaternion [x, y, z, w]
    point_cloud: np.ndarray  # Nx3 array of points
    velocity: np.ndarray = None  # [vx, vy, vz] for temporal relations
    scale: np.ndarray = None  # [sx, sy, sz] object scale

    def to_dict(self):
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'point_cloud': self.point_cloud.tolist(),
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'scale': self.scale.tolist() if self.scale is not None else None
        }

class PrepositionFunctions:
    @staticmethod
    def inside(reference_cloud: np.ndarray, target_cloud: np.ndarray, 
              reference_scale: np.ndarray) -> np.ndarray:
        """Calculate probability distribution for 'inside' relationship"""
        center = np.mean(reference_cloud, axis=0)
        cov = np.diag(reference_scale ** 2 / 4)
        x, y, z = np.mgrid[-1:1:.1, -1:1:.1, -1:1:.1]
        pos = np.dstack((x, y, z))
        rv = multivariate_normal(center, cov)
        return rv.pdf(pos)

    @staticmethod
    def through(initial_state: ObjectState, 
               reference_pos: np.ndarray,
               timesteps: int) -> np.ndarray:
        """Generate trajectory for 'through' relationship"""
        start_pos = initial_state.position
        direction = reference_pos - start_pos
        direction = direction / np.linalg.norm(direction)
        
        trajectory = np.zeros((timesteps, 3))
        t = np.linspace(0, 1, timesteps)
        
        for i in range(timesteps):
            trajectory[i] = (
                start_pos + 
                direction * t[i] * 2.0 +  # 2.0 meters distance
                np.array([0, 0, np.sin(t[i] * np.pi) * 0.1])  # Small vertical curve
            )
        
        return trajectory

class PrepositionDatasetGenerator:
    def __init__(self, num_scenes: int = 1000, output_dir: str = "preposition_dataset",
                 visualize: bool = False):
        # Connect to PyBullet in GUI mode if visualization is enabled
        self.physics_client = p.connect(p.GUI if visualize else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.num_scenes = num_scenes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualize = visualize
        
        if visualize:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0]
            )

    def _create_point_cloud(self, obj_id: int, num_points: int = 1000) -> np.ndarray:
        """Generate point cloud using PyBullet ray casting"""
        aabb_min, aabb_max = p.getAABB(obj_id)
        points = []
        
        for _ in range(num_points):
            # Random ray start position around the object
            start = np.random.uniform(
                np.array(aabb_min) - 0.1,
                np.array(aabb_max) + 0.1
            )
            end = np.random.uniform(
                np.array(aabb_min) - 0.1,
                np.array(aabb_max) + 0.1
            )
            
            results = p.rayTest(start, end)
            
            if results[0][0] == obj_id:
                hit_pos = results[0][3]
                points.append(hit_pos)
        
        return np.array(points) if points else np.zeros((1, 3))

    def spawn_object(self, object_type: str, size: np.ndarray, 
                    position: np.ndarray, color: np.ndarray = None) -> int:
        """Spawn an object with specified parameters"""
        if color is None:
            color = [random.random(), random.random(), random.random(), 1]
            
        if object_type == "box":
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=size/2,
                rgbaColor=color
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=size/2
            )
        elif object_type == "sphere":
            radius = size[0]/2
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=color
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius
            )
        
        obj_id = p.createMultiBody(
            baseMass=1,
            baseVisualShapeIndex=visual_shape,
            baseCollisionShapeIndex=collision_shape,
            basePosition=position
        )
        
        return obj_id

    def generate_scene(self, preposition: str) -> Dict:
        """Generate a scene for a specific preposition"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        plane_id = p.loadURDF("plane.urdf")
        
        if preposition == "inside":
            # Create container
            container_size = np.array([0.5, 0.5, 0.5])
            container_pos = np.array([0, 0, 0.25])
            container_id = self.spawn_object(
                "box", container_size, container_pos, [0.8, 0.8, 0.8, 0.5]
            )
            
            # Create target object
            target_size = np.array([0.1, 0.1, 0.1])
            target_pos = container_pos + np.array([0, 0, 0.1])
            target_id = self.spawn_object(
                "sphere", target_size, target_pos, [1, 0, 0, 1]
            )
            
            # Let physics settle
            for _ in range(100):
                p.stepSimulation()
                if self.visualize:
                    time.sleep(1/240)
            
            # Get object states
            container_state = self._get_object_state(container_id, container_size)
            target_state = self._get_object_state(target_id, target_size)
            
            # Calculate distribution
            distribution = PrepositionFunctions.inside(
                container_state.point_cloud,
                target_state.point_cloud,
                container_size
            )
            
            scene_data = {
                "type": "spatial",
                "preposition": preposition,
                "reference_state": container_state.to_dict(),
                "target_state": target_state.to_dict(),
                "distribution": distribution.tolist()
            }
            
        elif preposition == "through":
            # Create gateway
            gateway_size = np.array([0.1, 1.0, 1.0])
            gateway_pos = np.array([0, 0, 0.5])
            gateway_id = self.spawn_object(
                "box", gateway_size, gateway_pos, [0.8, 0.8, 0.8, 0.5]
            )
            
            # Create moving object
            object_size = np.array([0.2, 0.2, 0.2])
            object_pos = np.array([-1.0, 0, 0.5])
            object_id = self.spawn_object(
                "sphere", object_size, object_pos, [1, 0, 0, 1]
            )
            
            # Set initial velocity
            p.resetBaseVelocity(object_id, linearVelocity=[1, 0, 0])
            
            # Record trajectory
            timesteps = 50
            trajectory = []
            
            for _ in range(timesteps):
                p.stepSimulation()
                if self.visualize:
                    time.sleep(1/240)
                pos, _ = p.getBasePositionAndOrientation(object_id)
                trajectory.append(pos)
            
            trajectory = np.array(trajectory)
            
            scene_data = {
                "type": "temporal",
                "preposition": preposition,
                "reference_state": self._get_object_state(gateway_id, gateway_size).to_dict(),
                "target_state": self._get_object_state(object_id, object_size).to_dict(),
                "trajectory": trajectory.tolist()
            }
        
        return scene_data

    def _get_object_state(self, obj_id: int, scale: np.ndarray) -> ObjectState:
        """Get complete state of an object"""
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        lin_vel, ang_vel = p.getBaseVelocity(obj_id)
        point_cloud = self._create_point_cloud(obj_id)
        
        return ObjectState(
            position=np.array(pos),
            orientation=np.array(orn),
            point_cloud=point_cloud,
            velocity=np.array(lin_vel),
            scale=scale
        )

    def generate_dataset(self):
        """Generate complete dataset"""
        prepositions = ["inside", "through"]  # Add more as needed
        dataset = {
            "spatial": {},
            "temporal": {},
            "metadata": {
                "date_generated": time.strftime("%Y%m%d-%H%M%S"),
                "num_scenes": self.num_scenes
            }
        }
        
        for prep in prepositions:
            print(f"Generating scenes for '{prep}'...")
            scenes = []
            for i in range(self.num_scenes):
                scene_data = self.generate_scene(prep)
                scenes.append(scene_data)
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{self.num_scenes}")
            
            if scene_data["type"] == "spatial":
                dataset["spatial"][prep] = scenes
            else:
                dataset["temporal"][prep] = scenes
        
        # Save dataset
        with open(self.output_dir / "dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
        
        # Save metadata
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(dataset["metadata"], f, indent=2)
        
        return dataset



def generate_and_visualize_dataset():
    """Generate and visualize the complete preposition dataset"""
    # List of all prepositions to generate data for
    spatial_prepositions = [
        "in", "on", "at", "near", "under", "over", 
        "below", "above", "around", "between", 
        "behind", "in_front_of"
    ]
    
    temporal_prepositions = [
        "through", "along", "across", "towards",
        "up", "down", "into", "out_of"
    ]
    
    # Create output directories
    base_dir = Path("preposition_dataset")
    vis_dir = base_dir / "visualizations"
    for prep in spatial_prepositions + temporal_prepositions:
        (vis_dir / prep).mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    generator = PrepositionDatasetGenerator(
        num_scenes=20,  # Number of scenes per preposition
        output_dir=str(base_dir),
        visualize=False  # Set to True to see PyBullet visualization
    )
    
    # Create summary figure for all prepositions
    fig_spatial = plt.figure(figsize=(20, 15))
    fig_spatial.suptitle("Spatial Prepositions", size=16)
    
    fig_temporal = plt.figure(figsize=(20, 15))
    fig_temporal.suptitle("Temporal Prepositions", size=16)
    
    # Generate and visualize spatial prepositions
    print("\nGenerating spatial prepositions...")
    for idx, prep in enumerate(tqdm(spatial_prepositions)):
        # Generate scenes
        scenes = [generator.generate_scene(prep) for _ in range(5)]
        
        # Create individual visualizations
        visualizer = PrepositionVisualizer()
        for i, scene in enumerate(scenes):
            # Save individual scene visualization
            fig = visualizer.visualize_scene(scene, show=False)
            fig.savefig(vis_dir / prep / f"scene_{i}.png")
            plt.close(fig)
            
            # Add to summary plot (first scene only)
            if i == 0:
                ax = fig_spatial.add_subplot(3, 4, idx + 1, projection='3d')
                visualizer.visualize_scene(scene, ax=ax, show=False)
                ax.set_title(prep)
    
    # Generate and visualize temporal prepositions
    print("\nGenerating temporal prepositions...")
    for idx, prep in enumerate(tqdm(temporal_prepositions)):
        # Generate scenes
        scenes = [generator.generate_scene(prep) for _ in range(5)]
        
        # Create individual visualizations
        visualizer = PrepositionVisualizer()
        for i, scene in enumerate(scenes):
            # Save individual scene visualization
            fig = visualizer.visualize_scene(scene, show=False)
            fig.savefig(vis_dir / prep / f"scene_{i}.png")
            plt.close(fig)
            
            # Add to summary plot (first scene only)
            if i == 0:
                ax = fig_temporal.add_subplot(2, 4, idx + 1, projection='3d')
                visualizer.visualize_scene(scene, ax=ax, show=False)
                ax.set_title(prep)
    
    # Save summary figures
    fig_spatial.tight_layout()
    fig_spatial.savefig(vis_dir / "spatial_summary.png", dpi=300, bbox_inches='tight')
    
    fig_temporal.tight_layout()
    fig_temporal.savefig(vis_dir / "temporal_summary.png", dpi=300, bbox_inches='tight')
    
    # Generate statistics
    stats = generate_dataset_statistics(base_dir)
    
    # Save statistics
    with open(base_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset generation complete!")
    print(f"Output directory: {base_dir}")
    print("\nDataset statistics:")
    print(json.dumps(stats, indent=2))

def generate_dataset_statistics(dataset_dir: Path) -> Dict:
    """Generate statistics for the dataset"""
    stats = {
        "total_scenes": 0,
        "spatial_prepositions": {},
        "temporal_prepositions": {},
        "visualization_files": 0
    }
    
    # Count visualization files
    vis_dir = dataset_dir / "visualizations"
    for path in vis_dir.rglob("*.png"):
        stats["visualization_files"] += 1
    
    # Load dataset
    with open(dataset_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    # Generate statistics
    for prep_type in ["spatial", "temporal"]:
        for prep, scenes in dataset[prep_type].items():
            stats[f"{prep_type}_prepositions"][prep] = {
                "num_scenes": len(scenes),
                "avg_points": np.mean([
                    len(scene["reference_state"]["point_cloud"]) 
                    for scene in scenes
                ]),
            }
            
            if prep_type == "temporal":
                stats[f"{prep_type}_prepositions"][prep]["avg_trajectory_length"] = np.mean([
                    len(scene["trajectory"]) for scene in scenes
                ])
            
            stats["total_scenes"] += len(scenes)
    
    return stats

class PrepositionVisualizer:
    def visualize_scene(self, scene_data: Dict, ax=None, show=True):
        """Visualize a scene using matplotlib"""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        if scene_data["type"] == "spatial":
            # Plot reference object
            ref_points = np.array(scene_data["reference_state"]["point_cloud"])
            ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2],
                      c='blue', alpha=0.5, s=20, label='Reference')
            
            # Plot target object
            target_points = np.array(scene_data["target_state"]["point_cloud"])
            ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                      c='red', alpha=0.5, s=20, label='Target')
            
            # Plot distribution if available
            if "distribution" in scene_data:
                dist = np.array(scene_data["distribution"])
                x, y, z = np.where(dist > np.max(dist) * 0.1)
                scatter = ax.scatter(x/10-1, y/10-1, z/10-1, 
                                  c=dist[x, y, z], cmap='viridis',
                                  alpha=0.2, s=10, label='Distribution')
        
        elif scene_data["type"] == "temporal":
            # Plot reference object
            ref_points = np.array(scene_data["reference_state"]["point_cloud"])
            ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2],
                      c='blue', alpha=0.5, s=20, label='Reference')
            
            # Plot trajectory
            trajectory = np.array(scene_data["trajectory"])
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   'r-', linewidth=2, label='Trajectory')
            
            # Plot start and end points
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                      c='green', s=50, label='Start')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                      c='red', s=50, label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(fontsize='x-small')
        
        # Set consistent view limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 2])
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        
        if show:
            plt.show()
        
        return fig

if __name__ == "__main__":
    generate_and_visualize_dataset()