import numpy as np
from scipy.spatial import KDTree, ConvexHull
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

class PointCloudOperations:
    def __init__(self, points):
        """Initialize with a set of points as numpy array."""
        self.points = np.array(points)
        self.kdtree = KDTree(self.points)
    
    # Basic Spatial Predicates
    def is_empty(self):
        """Check if point cloud is empty."""
        return len(self.points) == 0
    
    def contains_point(self, point, tolerance=1e-6):
        """Check if point cloud contains a specific point."""
        dist, _ = self.kdtree.query(point)
        return dist <= tolerance
    
    def is_connected(self, threshold):
        """Check if point cloud forms a single connected component."""
        # Create adjacency matrix based on distance threshold
        distances = cdist(self.points, self.points)
        adjacency = distances <= threshold
        n_components, _ = connected_components(adjacency)
        return n_components == 1
    
    # Topological Operations
    def find_connected_components(self, threshold):
        """Find connected components in point cloud."""
        clustering = DBSCAN(eps=threshold, min_samples=1).fit(self.points)
        return clustering.labels_
    
    def compute_boundary(self):
        """Compute boundary points using convex hull."""
        if len(self.points) < 4:
            return self.points
        hull = ConvexHull(self.points)
        return self.points[hull.vertices]
    
    # Metric Operations
    def compute_diameter(self):
        """Compute diameter (maximum distance between any two points)."""
        if len(self.points) < 2:
            return 0
        distances = cdist(self.points, self.points)
        return np.max(distances)
    
    def compute_volume(self):
        """Estimate volume using convex hull."""
        if len(self.points) < 4:
            return 0
        hull = ConvexHull(self.points)
        return hull.volume
    
    def compute_surface_area(self):
        """Estimate surface area using convex hull."""
        if len(self.points) < 4:
            return 0
        hull = ConvexHull(self.points)
        return hull.area
    
    # Set Operations
    def union(self, other_points):
        """Combine two point clouds."""
        return np.unique(np.vstack([self.points, other_points]), axis=0)
    
    def intersection(self, other_points, tolerance=1e-6):
        """Find common points between two point clouds."""
        other_tree = KDTree(other_points)
        common_points = []
        for point in self.points:
            dist, _ = other_tree.query(point)
            if dist <= tolerance:
                common_points.append(point)
        return np.array(common_points)
    
    # Geometric Operations
    def compute_centroid(self):
        """Compute centroid of point cloud."""
        return np.mean(self.points, axis=0)
    
    def translate(self, vector):
        """Translate point cloud by vector."""
        self.points += vector
        self.kdtree = KDTree(self.points)
    
    def rotate(self, angle, axis='z'):
        """Rotate point cloud around specified axis."""
        theta = np.radians(angle)
        if axis.lower() == 'z':
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        elif axis.lower() == 'y':
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis.lower() == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        self.points = np.dot(self.points, rotation_matrix.T)
        self.kdtree = KDTree(self.points)
    
    # Path and Curve Operations
    def find_shortest_path(self, start_idx, end_idx, threshold):
        """Find shortest path between two points using graph approach."""
        distances = cdist(self.points, self.points)
        adjacency = distances <= threshold
        
        # Using Floyd-Warshall algorithm for shortest path
        n_points = len(self.points)
        dist_matrix = np.full((n_points, n_points), np.inf)
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix[adjacency] = distances[adjacency]
        
        for k in range(n_points):
            for i in range(n_points):
                for j in range(n_points):
                    if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
        
        return dist_matrix[start_idx, end_idx]
    
    def estimate_curvature(self, point_idx, k=5):
        """Estimate local curvature at a point using k nearest neighbors."""
        distances, indices = self.kdtree.query(self.points[point_idx], k=k+1)
        neighbors = self.points[indices[1:]]  # Exclude the point itself
        
        # Fit a sphere to the point and its neighbors
        # Return the inverse of the radius as an estimate of curvature
        centroid = np.mean(neighbors, axis=0)
        distances_to_centroid = np.linalg.norm(neighbors - centroid, axis=1)
        return 1.0 / np.mean(distances_to_centroid)

    def segment_by_curvature(self, threshold):
        """Segment point cloud based on local curvature."""
        curvatures = np.array([self.estimate_curvature(i) for i in range(len(self.points))])
        return curvatures > threshold