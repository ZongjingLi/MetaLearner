import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random

@dataclass
class ShapeParams:
    """Parameters for shape generation"""
    size: float = 1.0
    secondary_size: float = 0.5  # For shapes with multiple dimensions
    height: float = 1.0
    radius: float = 0.5
    segments: int = 32
    thickness: float = 0.1
    num_points: int = 1000
    noise: float = 0.02  # For adding surface irregularity
    twist: float = 0.0  # For twisted shapes
    bend: float = 0.0   # For bent shapes

class EnhancedShapeGenerator:
    """Enhanced shape generator with various geometric and everyday objects"""

    SHAPE_TYPES = {
        # Basic Geometric Shapes
        'cube': 'Basic cube shape',
        'sphere': 'Perfect sphere',
        'cylinder': 'Vertical cylinder',
        'cone': 'Regular cone',
        'torus': 'Donut-shaped torus',
        'plane': 'Flat rectangular plane',
        'line': '3D line segment',
        'ellipsoid': 'Stretched sphere',
        'pyramid': 'Square-based pyramid',
        'mobius': 'Mobius strip',
        'helix': 'Helical spiral',
        'prism': 'Triangular prism',

        # Complex Mathematical Shapes
        'klein_bottle': 'Non-orientable surface',
        'trefoil_knot': 'Basic knot shape',
        'hyperboloid': 'Hyperbolic surface',
        'dodecahedron': 'Regular dodecahedron',

        # Everyday Objects
        'mug': 'Cylindrical mug with handle',
        'bottle': 'Cylindrical bottle with neck',
        'bowl': 'Rounded bowl shape',
        'plate': 'Flat circular plate',
        'chair': 'Basic chair shape',
        'table': 'Four-legged table',
        'lamp': 'Desk lamp shape',
        'book': 'Rectangular book',
        'vase': 'Curved vase shape',
        'teapot': 'Classic teapot shape',
        'glass': 'Drinking glass',
        'fork': 'Eating utensil',
        'spoon': 'Curved spoon',
        'knife': 'Table knife',
        'box': 'Rectangular box',
        'frame': 'Picture frame',

        # Natural Shapes
        'leaf': 'Basic leaf shape',
        'shell': 'Spiral shell',
        'tree': 'Simple tree structure',
        'rock': 'Irregular rock shape',
        'cloud': 'Puffy cloud shape',
        'wave': 'Ocean wave form'
    }

    def __init__(self):
        """Initialize the shape generator"""
        self.params = ShapeParams()

    def generate_shapes(self) -> Dict[str, np.ndarray]:
        """
        Generate all available 3D shapes and return them in a dictionary.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping shape names to point clouds
        """
        shapes = {
            # Basic Geometric Shapes
            "cube": self._generate_cube(),
            "sphere": self._generate_sphere(),
            "cylinder": self._generate_cylinder(),
            "cone": self._generate_cone(),
            "torus": self._generate_torus(),
            "plane": self._generate_plane(),
            "line": self._generate_line(),
            "ellipsoid": self._generate_ellipsoid(),
            "pyramid": self._generate_pyramid(),
            "mobius": self._generate_mobius(),
            "helix": self._generate_helix(),
            "prism": self._generate_prism(),

            # Complex Mathematical Shapes
            "klein_bottle": self._generate_klein_bottle(),
            "trefoil_knot": self._generate_trefoil_knot(),
            "hyperboloid": self._generate_hyperboloid(),
            "dodecahedron": self._generate_dodecahedron(),

            # Everyday Objects
            "mug": self._generate_mug(),
            "bottle": self._generate_bottle(),
            "bowl": self._generate_bowl(),
            "plate": self._generate_plate(),
            "chair": self._generate_chair(),
            "table": self._generate_table(),
            "lamp": self._generate_lamp(),
            "book": self._generate_book(),
            "vase": self._generate_vase(),
            "teapot": self._generate_teapot(),
            "glass": self._generate_glass(),
            "fork": self._generate_fork(),
            "spoon": self._generate_spoon(),
            "knife": self._generate_knife(),
            "box": self._generate_box(),
            "frame": self._generate_frame(),

            # Natural Shapes
            "leaf": self._generate_leaf(),
            "shell": self._generate_shell(),
            "tree": self._generate_tree(),
            "rock": self._generate_rock(),
            "cloud": self._generate_cloud(),
            "wave": self._generate_wave()
        }

        return shapes

    def generate_shape_batch(self, shape_names: List[str], batch_size: int = 1) -> Dict[str, List[np.ndarray]]:
        """
        Generate multiple instances of specified shapes.

        Args:
            shape_names (List[str]): List of shape names to generate
            batch_size (int): Number of instances of each shape to generate

        Returns:
            Dict[str, List[np.ndarray]]: Dictionary mapping shape names to lists of point clouds
        """
        batch_shapes = {}

        for shape_name in shape_names:
            if shape_name not in self.SHAPE_TYPES:
                raise ValueError(f"Unknown shape type: {shape_name}")

            shape_batch = []
            for _ in range(batch_size):
                # Randomize parameters slightly for variation
                random_params = ShapeParams(
                    size=self.params.size * np.random.uniform(0.8, 1.2),
                    secondary_size=self.params.secondary_size * np.random.uniform(0.8, 1.2),
                    height=self.params.height * np.random.uniform(0.8, 1.2),
                    radius=self.params.radius * np.random.uniform(0.8, 1.2),
                    noise=self.params.noise * np.random.uniform(0.5, 1.5)
                )

                # Generate shape with randomized parameters
                points = self.generate_shape(shape_name, params=random_params)
                shape_batch.append(points)

            batch_shapes[shape_name] = shape_batch

        return batch_shapes

    def generate_shape(self, shape_type: str, params: Optional[ShapeParams] = None) -> np.ndarray:
        """Generate points for the specified shape"""
        if params:
            self.params = params

        if shape_type not in self.SHAPE_TYPES:
            raise ValueError(f"Unknown shape type: {shape_type}")

        method_name = f"_generate_{shape_type}"
        if hasattr(self, method_name):
            points = getattr(self, method_name)()
        else:
            # Default to cube if shape not implemented
            points = self._generate_cube()

        # Add surface noise if specified
        if self.params.noise > 0:
            points += np.random.normal(0, self.params.noise, points.shape)

        return points

    def _generate_cube(self) -> np.ndarray:
        """Generate points for a cube"""
        points = []
        size = self.params.size
        points_per_face = self.params.num_points // 6

        for axis in range(3):
            for sign in [-1, 1]:
                face_points = np.random.uniform(-size/2, size/2, (points_per_face, 3))
                face_points[:, axis] = sign * size/2
                points.append(face_points)

        return np.vstack(points)

    def _generate_sphere(self) -> np.ndarray:
        """Generate points for a sphere"""
        num_points = self.params.num_points
        radius = self.params.radius

        phi = np.random.uniform(0, 2*np.pi, num_points)
        theta = np.arccos(np.random.uniform(-1, 1, num_points))

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        return np.column_stack((x, y, z))

    def _generate_cylinder(self) -> np.ndarray:
        """Generate points for a cylinder"""
        points = []
        height = self.params.height
        radius = self.params.radius
        num_points = self.params.num_points

        # Side points
        theta = np.random.uniform(0, 2*np.pi, int(0.7*num_points))
        h = np.random.uniform(-height/2, height/2, int(0.7*num_points))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append(np.column_stack((x, y, h)))

        # Cap points
        for sign in [-1, 1]:
            theta = np.random.uniform(0, 2*np.pi, num_points//6)
            r = np.sqrt(np.random.uniform(0, radius**2, num_points//6))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = sign * height/2 * np.ones(num_points//6)
            points.append(np.column_stack((x, y, z)))

        return np.vstack(points)

    def _generate_torus(self) -> np.ndarray:
        """Generate points for a torus"""
        num_points = self.params.num_points
        R = self.params.size  # Major radius
        r = self.params.secondary_size  # Minor radius

        u = np.random.uniform(0, 2*np.pi, num_points)
        v = np.random.uniform(0, 2*np.pi, num_points)

        x = (R + r*np.cos(v)) * np.cos(u)
        y = (R + r*np.cos(v)) * np.sin(u)
        z = r * np.sin(v)

        return np.column_stack((x, y, z))

    def _generate_cone(self) -> np.ndarray:
        """Generate points for a cone"""
        points = []
        height = self.params.height
        radius = self.params.radius
        num_points = self.params.num_points

        # Side points
        theta = np.random.uniform(0, 2*np.pi, int(0.7*num_points))
        h = np.random.uniform(0, height, int(0.7*num_points))
        r = radius * (1 - h/height)  # Radius decreases with height
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append(np.column_stack((x, y, h)))

        # Base points
        theta = np.random.uniform(0, 2*np.pi, num_points//3)
        r = np.sqrt(np.random.uniform(0, radius**2, num_points//3))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(num_points//3)
        points.append(np.column_stack((x, y, z)))

        return np.vstack(points)

    def _generate_mobius(self) -> np.ndarray:
        """Generate points for a Mobius strip"""
        num_points = self.params.num_points
        R = self.params.size  # Major radius
        w = self.params.secondary_size  # Width

        u = np.random.uniform(0, 2*np.pi, num_points)
        v = np.random.uniform(-w/2, w/2, num_points)

        x = (R + v*np.cos(u/2)) * np.cos(u)
        y = (R + v*np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)

        return np.column_stack((x, y, z))

    def _generate_mug(self) -> np.ndarray:
        """Generate points for a mug"""
        points = []

        # Main cylinder
        cylinder_points = self._generate_cylinder()
        points.append(cylinder_points)

        # Handle
        handle_radius = self.params.radius * 0.3
        handle_center = np.array([self.params.radius, 0, 0])
        theta = np.linspace(0, 2*np.pi, int(0.2*self.params.num_points))
        phi = np.random.uniform(0, 2*np.pi, int(0.2*self.params.num_points))

        x = handle_center[0] + handle_radius * np.cos(theta)
        y = handle_radius * np.sin(theta)
        z = np.random.uniform(-self.params.height/3, self.params.height/3, len(theta))

        handle_points = np.column_stack((x, y, z))
        points.append(handle_points)

        return np.vstack(points)

    def _generate_teapot(self) -> np.ndarray:
        """Generate points for a stylized teapot"""
        points = []

        # Body (squashed sphere)
        body_points = self._generate_sphere()
        body_points[:, 0] *= 1.5  # Stretch horizontally
        body_points[:, 2] *= 0.8  # Squash vertically
        points.append(body_points)

        # Spout
        spout_points = self._generate_cone()
        spout_points = spout_points * 0.3  # Scale down
        spout_points = self._rotate_points(spout_points, np.pi/2, axis='z')
        spout_points += np.array([self.params.size, 0, 0])
        points.append(spout_points)

        # Handle
        handle_points = self._generate_torus()
        handle_points = handle_points * 0.3
        handle_points = self._rotate_points(handle_points, np.pi/2, axis='x')
        handle_points += np.array([-self.params.size, 0, 0])
        points.append(handle_points)

        return np.vstack(points)

    def _generate_tree(self) -> np.ndarray:
        """Generate points for a simple tree"""
        points = []

        # Trunk (cylinder)
        trunk_params = ShapeParams(
            height=self.params.height,
            radius=self.params.radius * 0.2,
            num_points=int(self.params.num_points * 0.3)
        )
        trunk_points = self._generate_cylinder()
        points.append(trunk_points)

        # Crown (multiple spheres)
        crown_centers = [
            [0, 0, self.params.height * 0.7],
            [self.params.radius * 0.5, 0, self.params.height * 0.6],
            [-self.params.radius * 0.5, 0, self.params.height * 0.6],
            [0, self.params.radius * 0.5, self.params.height * 0.6],
            [0, -self.params.radius * 0.5, self.params.height * 0.6]
        ]

        for center in crown_centers:
            sphere_params = ShapeParams(
                radius=self.params.radius * 0.4,
                num_points=int(self.params.num_points * 0.14)
            )
            sphere_points = self._generate_sphere()
            sphere_points += np.array(center)
            points.append(sphere_points)

        return np.vstack(points)

    def _rotate_points(self, points: np.ndarray, angle: float, axis: str = 'z') -> np.ndarray:
        """Rotate points around specified axis"""
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)

        if axis == 'x':
            R = np.array([[1, 0, 0],
                         [0, cos_t, -sin_t],
                         [0, sin_t, cos_t]])
        elif axis == 'y':
            R = np.array([[cos_t, 0, sin_t],
                         [0, 1, 0],
                         [-sin_t, 0, cos_t]])
        else:  # z
            R = np.array([[cos_t, -sin_t, 0],
                         [sin_t, cos_t, 0],
                         [0, 0, 1]])

        return np.dot(points, R.T)

# Add to EnhancedShapeGenerator class

    def _generate_line(self) -> np.ndarray:
        """Generate points for a 3D line segment"""
        num_points = self.params.num_points
        size = self.params.size

        # Create points along a line with some thickness
        t = np.random.uniform(0, 1, num_points)
        noise = np.random.normal(0, 0.02*size, (num_points, 2))

        x = size * t
        y = noise[:, 0]
        z = noise[:, 1]

        return np.column_stack((x, y, z)) - np.array([size/2, 0, 0])

    def _generate_plane(self) -> np.ndarray:
        """Generate points for a rectangular plane"""
        num_points = self.params.num_points
        size = self.params.size

        # Generate points on a plane with slight thickness
        x = np.random.uniform(-size/2, size/2, num_points)
        y = np.random.uniform(-size/2, size/2, num_points)
        z = np.random.normal(0, 0.01*size, num_points)

        return np.column_stack((x, y, z))


    def _generate_klein_bottle(self) -> np.ndarray:
        """Generate points for a Klein bottle"""
        num_points = self.params.num_points
        scale = self.params.size

        u = np.random.uniform(0, 2*np.pi, num_points)
        v = np.random.uniform(0, 2*np.pi, num_points)

        x = (2/3 * scale * (1 + np.sin(u) * np.sin(v/2)) * np.cos(v))
        y = (2/3 * scale * (1 + np.sin(u) * np.sin(v/2)) * np.sin(v))
        z = (2/3 * scale * (np.sin(u) + np.cos(u) * np.cos(v/2)))

        return np.column_stack((x, y, z))

    def _generate_hyperboloid(self) -> np.ndarray:
        """Generate points for a hyperboloid surface"""
        num_points = self.params.num_points
        scale = self.params.size

        u = np.random.uniform(0, 2*np.pi, num_points)
        v = np.random.uniform(-1, 1, num_points)

        x = scale * np.sqrt(1 + v**2) * np.cos(u)
        y = scale * np.sqrt(1 + v**2) * np.sin(u)
        z = scale * v

        return np.column_stack((x, y, z))

    def _generate_dodecahedron(self) -> np.ndarray:
        """Generate points for a regular dodecahedron"""
        points = []
        size = self.params.size
        num_points = self.params.num_points

        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2

        # Vertices of a dodecahedron
        vertices = np.array([
            # Cube vertices
            [x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]
        ] + [
            # Additional vertices
            [0, x*1/phi, y*phi] for x in [-1, 1] for y in [-1, 1]
        ] + [
            [x*1/phi, y*phi, 0] for x in [-1, 1] for y in [-1, 1]
        ] + [
            [x*phi, 0, y*1/phi] for x in [-1, 1] for y in [-1, 1]
        ]) * size

        # Generate points near each vertex
        for vertex in vertices:
            vertex_points = np.random.normal(vertex, 0.1*size, (num_points//len(vertices), 3))
            points.append(vertex_points)

        return np.vstack(points)

    def _generate_plate(self) -> np.ndarray:
        """Generate points for a circular plate"""
        num_points = self.params.num_points
        radius = self.params.radius
        thickness = self.params.thickness

        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = np.sqrt(np.random.uniform(0, radius**2, num_points))
        z = np.random.uniform(-thickness/2, thickness/2, num_points)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.column_stack((x, y, z))

    def _generate_glass(self) -> np.ndarray:
        """Generate points for a drinking glass"""
        points = []
        height = self.params.height
        radius = self.params.radius
        num_points = self.params.num_points

        # Generate points along the glass profile
        h = np.random.uniform(0, height, num_points)
        # Radius varies with height (wider at top)
        r = radius * (0.7 + 0.3 * h/height)
        theta = np.random.uniform(0, 2*np.pi, num_points)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h

        # Add bottom
        bottom_points = self._generate_plate()
        bottom_points *= np.array([0.7, 0.7, 0.1])  # Scale for bottom

        return np.vstack([np.column_stack((x, y, z)), bottom_points])

    def _generate_book(self) -> np.ndarray:
        """Generate points for a book with cover, pages, and spine"""
        points = []
        width = self.params.size          # Width of the book (x)
        height = self.params.height * 0.7  # Height of the book (y)
        thickness = self.params.secondary_size * 0.2  # Thickness of the book (z)
        cover_thickness = thickness * 0.1  # Thickness of the covers

        # Generate front and back covers (slightly larger than pages)
        for z_offset in [-thickness/2, thickness/2]:  # Back and front covers
            # Generate points for covers with rounded corners
            cover_points = int(self.params.num_points * 0.2)

            # Create grid of points
            x = np.random.uniform(-width/2, width/2, cover_points)
            y = np.random.uniform(-height/2, height/2, cover_points)
            z = np.random.uniform(-cover_thickness/2, cover_thickness/2, cover_points) + z_offset

            # Round the corners by removing points
            corner_radius = height * 0.05
            corners = [
                (-width/2, -height/2), (-width/2, height/2),
                (width/2, -height/2), (width/2, height/2)
            ]
            for corner_x, corner_y in corners:
                dist_from_corner = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)
                valid_points = dist_from_corner > corner_radius
                x, y, z = x[valid_points], y[valid_points], z[valid_points]

            cover = np.column_stack((x, y, z))
            # Add slight warping to covers
            cover[:, 2] += 0.02 * np.sin(np.pi * cover[:, 0]/width) * np.sin(np.pi * cover[:, 1]/height)
            points.append(cover)

        # Generate spine with rounded edges
        spine_points = int(self.params.num_points * 0.2)
        spine_x = np.full(spine_points, -width/2)
        spine_y = np.random.uniform(-height/2, height/2, spine_points)
        spine_z = np.random.uniform(-thickness/2, thickness/2, spine_points)
        # Add curved profile to spine
        spine_curve = 0.05 * thickness * np.sin(np.pi * spine_y/height)
        spine_x += spine_curve
        spine = np.column_stack((spine_x, spine_y, spine_z))
        points.append(spine)

        # Generate pages with slight curve and rough edge
        page_points = int(self.params.num_points * 0.4)
        num_pages = 50
        for i in range(num_pages):
            t = i / num_pages  # Parameter for page position

            # Generate points for each page
            points_per_page = page_points // num_pages
            x = np.random.uniform(-width/2, width/2, points_per_page)
            y = np.random.uniform(-height/2, height/2, points_per_page)
            # Add roughness to page edges
            edge_roughness = 0.02 * height * np.random.rand(points_per_page)
            y += edge_roughness * np.exp(-5 * (x + width/2)**2/width**2)  # More roughness near spine

            # Calculate z position with slight curve
            z = (t - 0.5) * (thickness - 2*cover_thickness)
            # Add page curvature
            page_curve = 0.1 * thickness * (1 - np.exp(-3 * (x + width/2)/width))
            z += page_curve

            page = np.column_stack((x, y, z))
            points.append(page)

        # Add texture details to spine (ridges or title embossing)
        num_ridges = 5
        ridge_points = int(self.params.num_points * 0.1)
        points_per_ridge = ridge_points // num_ridges

        for i in range(num_ridges):
            y_pos = (i - num_ridges/2) * height/num_ridges

            # Generate ridge points
            ridge_y = np.random.normal(y_pos, height*0.02, points_per_ridge)
            ridge_x = np.random.normal(-width/2, width*0.01, points_per_ridge)
            ridge_z = np.random.uniform(-thickness/2, thickness/2, points_per_ridge)

            # Add slight protrusion for ridge
            ridge_x += 0.02 * thickness * np.exp(-((ridge_y - y_pos)/(height*0.02))**2)

            ridge = np.column_stack((ridge_x, ridge_y, ridge_z))
            points.append(ridge)

        return np.vstack(points)


    def _generate_pyramid(self) -> np.ndarray:
        """Generate points for a square pyramid"""
        points = []
        height = self.params.height
        base_size = self.params.size
        num_points = self.params.num_points

        # Base vertices
        base_vertices = np.array([
            [base_size/2, base_size/2, 0],
            [base_size/2, -base_size/2, 0],
            [-base_size/2, -base_size/2, 0],
            [-base_size/2, base_size/2, 0]
        ])
        apex = np.array([0, 0, height])

        # Generate base points
        base_points = np.random.uniform(-base_size/2, base_size/2, (num_points//2, 2))
        base_points = np.column_stack([base_points, np.zeros(num_points//2)])
        points.append(base_points)

        # Generate points on triangular faces
        for i in range(4):
            v1 = base_vertices[i]
            v2 = base_vertices[(i+1)%4]

            # Generate random points in triangle
            face_points = np.random.uniform(0, 1, (num_points//8, 2))
            # Ensure sum <= 1 for barycentric coordinates
            mask = np.sum(face_points, axis=1) > 1
            face_points[mask] = 1 - face_points[mask]

            # Convert to 3D points using barycentric coordinates
            triangle_points = (
                apex[None, :] * face_points[:, 0:1] +
                v1[None, :] * face_points[:, 1:2] +
                v2[None, :] * (1 - face_points[:, 0:1] - face_points[:, 1:2])
            )
            points.append(triangle_points)

        return np.vstack(points)

    def _generate_helix(self) -> np.ndarray:
        """Generate points for a helical spiral"""
        num_points = self.params.num_points
        radius = self.params.radius
        height = self.params.height
        num_turns = 3 # Number of complete turns
        thickness = self.params.thickness * 0.1

        # Generate points along the helix curve
        t = np.linspace(0, num_turns * 2 * np.pi, num_points)

        # Core helix points
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = height * t / (2 * np.pi * num_turns)

        # Add thickness by generating points around the core curve
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = np.random.uniform(0, thickness, num_points)

        # Create circular cross-section
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)

        # Add cross-section to core points
        x = x + dx
        y = y + dy

        # Center the helix vertically
        z = z - height/2

        return np.column_stack((x, y, z))

    def _generate_trefoil_knot(self) -> np.ndarray:
        """Generate points for a trefoil knot"""
        num_points = self.params.num_points
        scale = self.params.size
        thickness = self.params.thickness * 0.1

        # Generate points along the trefoil curve
        t = np.linspace(0, 2*np.pi, num_points)

        # Trefoil knot parametric equations
        x = scale * (np.sin(t) + 2*np.sin(2*t))
        y = scale * (np.cos(t) - 2*np.cos(2*t))
        z = scale * -np.sin(3*t)

        # Calculate tangent vectors for circular cross-section
        dx = scale * (np.cos(t) + 4*np.cos(2*t))
        dy = scale * (-np.sin(t) + 4*np.sin(2*t))
        dz = scale * -3*np.cos(3*t)

        # Normalize tangent vectors
        tangent = np.column_stack((dx, dy, dz))
        tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]

        # Generate random points in circular cross-section
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = np.random.uniform(0, thickness, num_points)

        # Create basis vectors for circular cross-section
        # First basis vector is random perpendicular to tangent
        random_vec = np.random.randn(3)
        basis1 = np.cross(tangent, random_vec)
        basis1 = basis1 / np.linalg.norm(basis1, axis=1)[:, None]
        # Second basis vector completes orthonormal basis
        basis2 = np.cross(tangent, basis1)

        # Add thickness using circular cross-section
        offset = (r[:, None] * (np.cos(theta)[:, None] * basis1 +
                               np.sin(theta)[:, None] * basis2))

        points = np.column_stack((x, y, z)) + offset
        return points

    def _generate_ellipsoid(self) -> np.ndarray:
        """Generate points for an ellipsoid"""
        num_points = self.params.num_points
        a = self.params.size  # x-axis radius
        b = self.params.secondary_size  # y-axis radius
        c = self.params.height  # z-axis radius

        # Generate points using spherical coordinates
        phi = np.random.uniform(0, 2*np.pi, num_points)
        theta = np.arccos(np.random.uniform(-1, 1, num_points))

        # Convert to cartesian coordinates with different scaling in each dimension
        x = a * np.sin(theta) * np.cos(phi)
        y = b * np.sin(theta) * np.sin(phi)
        z = c * np.cos(theta)

        # Optional: add surface perturbations for more natural look
        if self.params.noise > 0:
            perturbation = np.random.normal(0, self.params.noise, (num_points, 3))
            perturbation *= np.array([a, b, c]) / np.max([a, b, c])
            return np.column_stack((x, y, z)) + perturbation

        return np.column_stack((x, y, z))

    def _generate_composite_shape(self,
                                shapes: List[Tuple[str, Dict[str, float], np.ndarray]]) -> np.ndarray:
        """
        Generate a composite shape from multiple basic shapes

        Args:
            shapes: List of tuples (shape_type, params_dict, transform_matrix)

        Returns:
            np.ndarray: Combined point cloud
        """
        points = []
        num_points_per_shape = self.params.num_points // len(shapes)

        for shape_type, params_dict, transform in shapes:
            # Create temporary params
            temp_params = ShapeParams(num_points=num_points_per_shape, **params_dict)

            # Generate shape with temporary params
            original_params = self.params
            self.params = temp_params
            shape_points = self.generate_shape(shape_type)
            self.params = original_params

            # Apply transformation
            if transform is not None:
                shape_points = np.dot(shape_points, transform[:3, :3].T)
                if transform.shape[1] > 3:
                    shape_points += transform[:3, 3]

            points.append(shape_points)

        return np.vstack(points)

    def _generate_frame(self) -> np.ndarray:
        """Generate points for a picture frame"""
        points = []
        size = self.params.size
        thickness = self.params.thickness
        depth = self.params.secondary_size * 0.2

        # Generate frame edges
        edges = [
            [[-size/2, -size/2, 0], [size/2, -size/2, 0]],
            [[size/2, -size/2, 0], [size/2, size/2, 0]],
            [[size/2, size/2, 0], [-size/2, size/2, 0]],
            [[-size/2, size/2, 0], [-size/2, -size/2, 0]]
        ]

        for edge in edges:
            start, end = np.array(edge)
            t = np.random.uniform(0, 1, self.params.num_points//4)[:, None]
            edge_points = start[None, :] * (1-t) + end[None, :] * t

            # Add thickness and depth
            noise_xy = np.random.uniform(-thickness/2, thickness/2, (len(t), 2))
            noise_z = np.random.uniform(0, depth, len(t))

            points.append(np.column_stack([
                edge_points[:, :2] + noise_xy,
                edge_points[:, 2] + noise_z
            ]))

        return np.vstack(points)

    def _generate_rock(self) -> np.ndarray:
        """Generate points for an irregular rock"""
        points = self._generate_sphere()

        # Add random displacement to create irregular surface
        displacement = np.random.uniform(-0.3, 0.3, points.shape)
        displacement *= np.linalg.norm(points, axis=1)[:, None]
        points += displacement * self.params.noise

        return points

    def _generate_bottle(self) -> np.ndarray:
        """Generate points for a bottle with body and neck"""
        points = []
        height = self.params.height
        radius = self.params.radius
        num_points = self.params.num_points

        # Body (wider at bottom, narrower at top)
        body_height = height * 0.7
        h = np.random.uniform(0, body_height, int(0.7 * num_points))
        theta = np.random.uniform(0, 2*np.pi, len(h))
        # Radius varies with height for bottle shape
        r = radius * (1.2 - 0.4 * h/body_height)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h
        body_points = np.column_stack((x, y, z))
        points.append(body_points)

        # Neck (thinner cylinder)
        neck_points = self._generate_cylinder()
        neck_points = neck_points * np.array([0.3, 0.3, 0.3])  # Scale down
        neck_points += np.array([0, 0, body_height])  # Move up
        points.append(neck_points)

        return np.vstack(points)

    def _generate_bowl(self) -> np.ndarray:
        """Generate points for a bowl (half-sphere with thickness)"""
        points = []
        radius = self.params.radius
        thickness = self.params.thickness
        num_points = self.params.num_points

        # Generate points on bowl surface
        phi = np.arccos(np.random.uniform(0, 1, num_points))  # Upper hemisphere only
        theta = np.random.uniform(0, 2*np.pi, num_points)

        # Outer surface
        r_outer = radius * (1 + np.random.normal(0, 0.02, num_points))  # Slight irregularity
        x_outer = r_outer * np.sin(phi) * np.cos(theta)
        y_outer = r_outer * np.sin(phi) * np.sin(theta)
        z_outer = r_outer * np.cos(phi)
        points.append(np.column_stack((x_outer, y_outer, z_outer)))

        # Inner surface (slightly smaller)
        r_inner = (radius - thickness) * (1 + np.random.normal(0, 0.02, num_points))
        x_inner = r_inner * np.sin(phi) * np.cos(theta)
        y_inner = r_inner * np.sin(phi) * np.sin(theta)
        z_inner = r_inner * np.cos(phi)
        points.append(np.column_stack((x_inner, y_inner, z_inner)))

        # Add rim points
        rim_theta = np.random.uniform(0, 2*np.pi, num_points//4)
        rim_r = np.random.uniform(radius-thickness, radius, num_points//4)
        rim_x = rim_r * np.cos(rim_theta)
        rim_y = rim_r * np.sin(rim_theta)
        rim_z = np.zeros(num_points//4)
        points.append(np.column_stack((rim_x, rim_y, rim_z)))

        return np.vstack(points)


    def _generate_chair(self) -> np.ndarray:
        """Generate points for a basic chair"""
        points = []
        size = self.params.size
        height = self.params.height

        # Seat
        seat = self._generate_cube()
        seat = seat * np.array([1.0, 1.0, 0.1])  # Flatten for seat
        seat = seat + np.array([0, 0, height * 0.4])  # Raise to sitting height
        points.append(seat)

        # Backrest
        back = self._generate_cube()
        back = back * np.array([0.1, 1.0, 0.8])  # Thin, tall back
        back = back + np.array([-size/2, 0, height * 0.8])  # Position at back
        points.append(back)

        # Legs
        leg_positions = [
            [size/2 - 0.1, size/2 - 0.1, height * 0.2],
            [size/2 - 0.1, -size/2 + 0.1, height * 0.2],
            [-size/2 + 0.1, size/2 - 0.1, height * 0.2],
            [-size/2 + 0.1, -size/2 + 0.1, height * 0.2]
        ]

        for pos in leg_positions:
            leg = self._generate_cylinder()
            leg = leg * np.array([0.1, 0.1, 0.4])  # Thin, tall legs
            leg = leg + np.array(pos)
            points.append(leg)

        return np.vstack(points)

    def _generate_table(self) -> np.ndarray:
        """Generate points for a basic table"""
        points = []
        size = self.params.size
        height = self.params.height

        # Table top
        top = self._generate_cube()
        top = top * np.array([1.5, 1.0, 0.1])  # Flatten for table top
        top = top + np.array([0, 0, height])  # Raise to table height
        points.append(top)

        # Legs
        leg_positions = [
            [size * 0.6, size * 0.4, height/2],
            [size * 0.6, -size * 0.4, height/2],
            [-size * 0.6, size * 0.4, height/2],
            [-size * 0.6, -size * 0.4, height/2]
        ]

        for pos in leg_positions:
            leg = self._generate_cylinder()
            leg = leg * np.array([0.1, 0.1, 1.0])  # Thin legs
            leg = leg + np.array(pos)
            points.append(leg)

        return np.vstack(points)

    def _generate_lamp(self) -> np.ndarray:
        """Generate points for a desk lamp"""
        points = []
        size = self.params.size

        # Base
        base = self._generate_cylinder()
        base = base * np.array([0.3, 0.3, 0.1])
        points.append(base)

        # Arm segments
        arm_points = []
        segments = 2
        for i in range(segments):
            segment = self._generate_cylinder()
            segment = segment * np.array([0.8, 0.1, 0.1])
            segment = self._rotate_points(segment, np.pi/4 * (i+1), 'y')
            segment = segment + np.array([0, 0, size * 0.3 * (i+1)])
            arm_points.append(segment)
        points.extend(arm_points)

        # Lamp head (cone)
        head = self._generate_cone()
        head = head * np.array([0.3, 0.3, 0.3])
        head = self._rotate_points(head, -np.pi/4, 'y')
        head = head + np.array([size * 0.4, 0, size * 0.8])
        points.append(head)

        return np.vstack(points)

    def _generate_vase(self) -> np.ndarray:
        """Generate points for a decorative vase"""
        points = []
        height = self.params.height
        radius = self.params.radius
        num_points = self.params.num_points

        # Generate points along a curved profile
        h = np.random.uniform(0, height, num_points)
        theta = np.random.uniform(0, 2*np.pi, num_points)

        # Create curved profile using sine function
        r = radius * (1 + 0.3 * np.sin(np.pi * h/height))

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h

        # Add neck
        neck_points = int(0.2 * num_points)
        neck_h = np.random.uniform(height * 0.8, height, neck_points)
        neck_theta = np.random.uniform(0, 2*np.pi, neck_points)
        neck_r = radius * 0.5  # Narrower at top

        neck_x = neck_r * np.cos(neck_theta)
        neck_y = neck_r * np.sin(neck_theta)

        points.append(np.column_stack((x, y, z)))
        points.append(np.column_stack((neck_x, neck_y, neck_h)))

        return np.vstack(points)

    def _generate_box(self) -> np.ndarray:
        """Generate points for a box with lid"""
        points = []
        size = self.params.size
        height = self.params.height
        thickness = self.params.thickness

        # Main box body
        body = self._generate_cube()
        body = body * np.array([1.0, 1.0, 0.8])
        points.append(body)

        # Lid (slightly larger than body)
        lid = self._generate_cube()
        lid = lid * np.array([1.05, 1.05, 0.2])
        lid = lid + np.array([0, 0, height * 0.8])
        points.append(lid)

        # Add rim detail
        rim_points = []
        for i in range(4):
            rim = self._generate_cube()
            rim = rim * np.array([0.05, 1.0, 0.1])
            rim = self._rotate_points(rim, i * np.pi/2, 'z')
            rim = rim + np.array([0, 0, height * 0.7])
            rim_points.append(rim)
        points.extend(rim_points)

        return np.vstack(points)

    def _generate_knife(self) -> np.ndarray:
        """Generate points for a knife"""
        points = []
        length = self.params.size * 1.5

        # Handle
        handle = self._generate_cylinder()
        handle = handle * np.array([0.3, 0.15, 0.15])
        handle = self._rotate_points(handle, np.pi/2, 'y')
        handle = handle - np.array([length/2, 0, 0])
        points.append(handle)

        # Blade
        num_blade_points = self.params.num_points // 2
        blade_length = length * 0.7

        # Generate blade points with tapering thickness and width
        x = np.random.uniform(0, blade_length, num_blade_points)
        y = np.random.uniform(-0.1, 0.1, num_blade_points) * (1 - x/blade_length)
        z = np.random.uniform(-0.02, 0.02, num_blade_points) * (1 - x/blade_length)

        blade_points = np.column_stack((x, y, z))
        blade_points = blade_points + np.array([0, 0, 0])
        points.append(blade_points)


        return np.vstack(points)

    def _generate_fork(self) -> np.ndarray:
        """Generate points for a fork with handle and prongs"""
        points = []
        length = self.params.size * 1.5
        handle_radius = self.params.radius * 0.15
        prong_length = length * 0.3

        # Handle (tapered cylinder)
        handle_points = int(self.params.num_points * 0.4)
        t = np.linspace(0, length * 0.7, handle_points)
        theta = np.random.uniform(0, 2*np.pi, handle_points)
        # Radius varies slightly along length
        r = handle_radius * (1 + 0.2 * t/length)

        x = -t  # Handle extends in negative x direction
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        handle = np.column_stack((x, y, z))
        points.append(handle)

        # Prongs
        prong_points = int(self.params.num_points * 0.15)  # Points per prong
        num_prongs = 4
        prong_spacing = handle_radius * 0.8

        for i in range(num_prongs):
            # Calculate prong position
            offset_y = ((i - (num_prongs-1)/2) * prong_spacing)

            # Generate prong points (tapered towards tip)
            t = np.linspace(0, prong_length, prong_points)
            # Taper the thickness and width
            thickness = 0.05 * handle_radius * (1 - 0.7*t/prong_length)

            # Random points within the tapered prong shape
            dx = np.random.uniform(-thickness, thickness, prong_points)
            dy = np.random.uniform(-thickness, thickness, prong_points)

            prong = np.column_stack((t, dy + offset_y, dx))
            points.append(prong)

        return np.vstack(points)

    def _generate_knife(self) -> np.ndarray:
        """Generate points for a knife with handle and blade"""
        points = []
        length = self.params.size * 1.5
        handle_length = length * 0.4
        blade_length = length * 0.6
        handle_radius = self.params.radius * 0.15

        # Handle
        handle_points = int(self.params.num_points * 0.4)
        t = np.linspace(-handle_length, 0, handle_points)
        theta = np.random.uniform(0, 2*np.pi, handle_points)

        # Slightly oval handle shape
        r_x = handle_radius * np.cos(theta)
        r_y = handle_radius * 1.2 * np.sin(theta)

        x = t
        y = r_x
        z = r_y
        handle = np.column_stack((x, y, z))
        points.append(handle)

        # Blade
        blade_points = int(self.params.num_points * 0.6)
        t = np.linspace(0, blade_length, blade_points)

        # Blade profile (height varies along length)
        blade_height = handle_radius * 2 * (1 - 0.9*t/blade_length)
        # Blade thickness (tapers towards tip and edge)
        max_thickness = handle_radius * 0.3

        # Generate points distributed through blade volume
        for i in range(len(t)):
            num_slice_points = int(blade_points * blade_height[i] / np.max(blade_height))
            if num_slice_points < 2:
                continue

            # Points distributed in vertical slice
            h = np.random.uniform(-blade_height[i]/2, blade_height[i]/2, num_slice_points)
            # Thickness varies with height (thinner towards edge)
            thick = max_thickness * (1 - np.abs(h)/(blade_height[i]/2)) * (1 - 0.7*t[i]/blade_length)
            w = np.random.uniform(-thick/2, thick/2, num_slice_points)

            slice_points = np.column_stack((
                np.full(num_slice_points, t[i]),
                h,
                w
            ))
            points.append(slice_points)

        return np.vstack(points)

    def _generate_prism(self) -> np.ndarray:
        """Generate points for a triangular prism"""
        points = []
        height = self.params.height
        size = self.params.size
        num_points = self.params.num_points

        # Define the triangular base vertices
        base_vertices = np.array([
            [0, size/2, 0],
            [-size/2, -size/2, 0],
            [size/2, -size/2, 0]
        ])

        # Generate points for the triangular ends
        for z in [-height/2, height/2]:
            # Generate random barycentric coordinates
            bary_coords = np.random.uniform(0, 1, (num_points//3, 3))
            bary_coords = bary_coords / bary_coords.sum(axis=1, keepdims=True)

            # Convert to Cartesian coordinates
            end_points = np.dot(bary_coords, base_vertices)
            end_points = np.column_stack((end_points, np.full(len(end_points), z)))
            points.append(end_points)

        # Generate points for the rectangular faces
        for i in range(3):
            v1 = base_vertices[i]
            v2 = base_vertices[(i+1)%3]

            # Generate random points on the rectangular face
            u = np.random.uniform(0, 1, (num_points//3, 1))  # Along edge
            v = np.random.uniform(0, 1, (num_points//3, 1))  # Along height

            # Compute points
            face_points = (
                v1[None, :] * (1-u) + v2[None, :] * u
            )
            z = (v * height) - height/2

            face_points = np.column_stack((face_points, z))
            points.append(face_points)

        return np.vstack(points)

    def _generate_spoon(self) -> np.ndarray:
        """Generate points for a spoon with handle and bowl"""
        points = []
        length = self.params.size * 1.5
        handle_radius = self.params.radius * 0.15
        bowl_length = length * 0.25
        bowl_width = length * 0.15

        # Handle (tapered cylinder)
        handle_points = int(self.params.num_points * 0.4)
        t = np.linspace(0, length * 0.75, handle_points)
        theta = np.random.uniform(0, 2*np.pi, handle_points)
        # Radius varies slightly along length
        r = handle_radius * (1 + 0.2 * t/length)

        x = -t  # Handle extends in negative x direction
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        handle = np.column_stack((x, y, z))
        points.append(handle)

        # Bowl
        bowl_points = int(self.params.num_points * 0.6)
        u = np.random.uniform(0, 1, bowl_points)
        v = np.random.uniform(0, 1, bowl_points)

        # Elliptical bowl shape with depth
        x = bowl_length * u
        y = bowl_width * (v - 0.5)
        # Bowl curvature (parabolic)
        z = -0.15 * length * (1 - (2*v-1)**2) * (1 - (u-0.5)**2)

        bowl = np.column_stack((x, y, z))
        # Add random small variations for more natural look
        bowl += np.random.normal(0, 0.01 * length, bowl.shape)
        points.append(bowl)

        return np.vstack(points)

    def _generate_plate(self) -> np.ndarray:
        """Generate points for a circular plate with rim"""
        points = []
        radius = self.params.radius
        thickness = self.params.thickness
        num_points = self.params.num_points

        # Main surface points (slight concave shape)
        surface_points = int(num_points * 0.7)
        r = np.sqrt(np.random.uniform(0, radius**2, surface_points))
        theta = np.random.uniform(0, 2*np.pi, surface_points)

        # Create slight concave shape
        depth_factor = 0.02 * radius
        z = -depth_factor * (1 - (r/radius)**2)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        surface = np.column_stack((x, y, z))
        points.append(surface)

        # Bottom surface (flat with small gap)
        bottom_z = np.full_like(z, -thickness)
        bottom = np.column_stack((x, y, bottom_z))
        points.append(bottom)

        # Rim points
        rim_points = num_points - surface_points
        rim_theta = np.random.uniform(0, 2*np.pi, rim_points)
        rim_r = np.random.uniform(0.9 * radius, radius, rim_points)

        # Create rim with curved profile
        t = np.linspace(0, np.pi, rim_points)
        rim_x = rim_r * np.cos(rim_theta)
        rim_y = rim_r * np.sin(rim_theta)
        rim_z = -thickness * np.cos(t)  # Curved profile

        rim = np.column_stack((rim_x, rim_y, rim_z))
        points.append(rim)

        return np.vstack(points)

    def _generate_bottle(self) -> np.ndarray:
        """Generate points for a bottle"""
        points = []

        # Body (stretched cylinder)
        body_height = self.params.height * 0.7
        body_points = self._generate_cylinder()
        body_points[:, 2] *= 0.7  # Scale height
        points.append(body_points)

        # Neck (thinner cylinder)
        neck_params = ShapeParams(
            height=self.params.height * 0.3,
            radius=self.params.radius * 0.3,
            num_points=self.params.num_points // 4
        )
        neck_points = self._generate_cylinder()
        neck_points += np.array([0, 0, body_height * 0.7])
        points.append(neck_points)

        return np.vstack(points)

    def _generate_cloud(self) -> np.ndarray:
        """Generate points for a puffy cloud"""
        points = []
        base_size = self.params.size

        # Create cloud from multiple overlapping spheres
        centers = [
            [0, 0, 0],
            [base_size*0.5, 0, 0],
            [-base_size*0.5, 0, 0],
            [0, base_size*0.5, 0],
            [0, -base_size*0.5, 0],
            [0, 0, base_size*0.3]
        ]

        sizes = np.random.uniform(0.7, 1.0, len(centers)) * base_size

        for center, size in zip(centers, sizes):
            sphere = self._generate_sphere()
            sphere = sphere * size
            sphere += np.array(center)
            points.append(sphere)

        return np.vstack(points)

    def _generate_shell(self) -> np.ndarray:
        """Generate points for a spiral shell"""
        num_points = self.params.num_points
        scale = self.params.size

        # Logarithmic spiral
        t = np.random.uniform(0, 4*np.pi, num_points)
        a = 0.1
        r = scale * np.exp(a*t)

        x = r * np.cos(t)
        y = r * np.sin(t)
        z = scale * t/(4*np.pi)

        return np.column_stack((x, y, z))

    def _generate_wave(self) -> np.ndarray:
        """Generate points for an ocean wave"""
        num_points = self.params.num_points
        size = self.params.size

        x = np.random.uniform(-size, size, num_points)
        y = np.random.uniform(-size, size, num_points)

        # Multiple wave components
        z = (0.5 * np.sin(2*np.pi*x/size) +
             0.3 * np.sin(2*np.pi*y/size) +
             0.2 * np.sin(4*np.pi*np.sqrt(x**2 + y**2)/size))

        return np.column_stack((x, y, z))

    def _generate_leaf(self) -> np.ndarray:
        """Generate points for a leaf shape"""
        num_points = self.params.num_points
        size = self.params.size

        t = np.random.uniform(0, 2*np.pi, num_points)
        r = size * (1 + np.sin(t)) * np.abs(np.sin(2*t))

        x = r * np.cos(t)
        y = r * np.sin(t)
        z = 0.1 * size * np.sin(8*t)  # Add some waviness

        return np.column_stack((x, y, z))

# Example usage
def visualize_shape_examples():
    generator = EnhancedShapeGenerator()
    shapes = list(generator.SHAPE_TYPES.keys())

    # Create visualization for each shape
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplots for the first 16 shapes
    fig = make_subplots(
        rows=4, cols=4,
        specs=[[{'type': 'scene'}]*4]*4,
        subplot_titles=shapes[:16]
    )

    for i, shape in enumerate(shapes[:16], 1):
        row = (i-1) // 4 + 1
        col = (i-1) % 4 + 1

        points = generator.generate_shape(shape)

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points[:, 2],
                    colorscale='Viridis',
                ),
                name=shape
            ),
            row=row, col=col
        )

    fig.update_layout(height=1000, width=1000, title_text="Shape Examples (1-16)")
    fig.show()

    # Create subplots for the next 16 shapes
    fig2 = make_subplots(
        rows=4, cols=4,
        specs=[[{'type': 'scene'}]*4]*4,
        subplot_titles=shapes[16:32]
    )

    for i, shape in enumerate(shapes[16:32], 1):
        row = (i-1) // 4 + 1
        col = (i-1) % 4 + 1

        points = generator.generate_shape(shape)

        fig2.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points[:, 2],
                    colorscale='Viridis',
                ),
                name=shape
            ),
            row=row, col=col
        )

    fig2.update_layout(height=1000, width=1000, title_text="Shape Examples (17-32)")
    fig2.show()

def visualize_remaining_shapes():
    generator = EnhancedShapeGenerator()
    remaining_shapes = ['shell', 'tree', 'rock', 'cloud', 'leaf', 'wave']

    # Create visualization using plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplots for the remaining shapes
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scene'}]*3]*2,
        subplot_titles=remaining_shapes
    )

    for i, shape in enumerate(remaining_shapes, 1):
        row = (i-1) // 3 + 1
        col = (i-1) % 3 + 1

        # Generate shape points
        points = generator.generate_shape(shape)

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points[:, 2],
                    colorscale='Viridis',
                ),
                name=shape
            ),
            row=row, col=col
        )

        # Update layout for each subplot
        fig.update_scenes(
            dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=800,
        width=1200,
        title_text="Remaining Shape Examples",
        showlegend=False
    )

    fig.show()

# Add this to the main visualization function
def visualize_all_shapes():
    # Original code for first 32 shapes...
    visualize_shape_examples()
    # Add visualization for remaining shapes
    visualize_remaining_shapes()


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math

class GeometricDataset(Dataset):
    def __init__(self, num_samples: int = 1000, num_points: int = 1024):
        """
        Initialize the geometric shapes dataset.

        Args:
            num_samples (int): Number of total samples in the dataset
            num_points (int): Number of points in each shape point cloud
        """
        # Initialize shape generator with specified number of points
        self.generator = EnhancedShapeGenerator()
        self.generator.params.num_points = num_points
        self.num_points = num_points

        # Get all available shape types
        self.shape_names = list(self.generator.SHAPE_TYPES.keys())

        # Generate dataset
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            # Randomly select a shape type
            shape_name = np.random.choice(self.shape_names)

            # Generate shape with random variations
            random_params = ShapeParams(
                size=np.random.uniform(0.8, 1.2),
                secondary_size=np.random.uniform(0.8, 1.2),
                height=np.random.uniform(0.8, 1.2),
                radius=np.random.uniform(0.8, 1.2),
                noise=np.random.uniform(0.01, 0.03),
                num_points=num_points
            )

            # Generate shape points
            shape_points = self.generator.generate_shape(shape_name, params=random_params)

            # Ensure exact number of points through random sampling or padding
            shape_points = self._ensure_point_count(shape_points)

            # Normalize points to [-1, 1] range
            shape_points = self._normalize_points(shape_points)

            self.data.append(shape_points[:,:3])
            self.labels.append(self.shape_names.index(shape_name))

        # Convert to torch tensors
        total = 0
        for i,d in enumerate(self.data):
          if d.shape != (320, 3):
            total += 1
            #print(i, d.shape)
        #print(total)
        self.data = torch.FloatTensor(np.stack(self.data))
        self.labels = torch.LongTensor(self.labels)

        # Store shape name to label mapping
        self.shape_to_label = {shape: idx for idx, shape in enumerate(self.shape_names)}
        self.label_to_shape = {idx: shape for idx, shape in enumerate(self.shape_names)}

    def _ensure_point_count(self, points: np.ndarray) -> np.ndarray:
        """
        Ensure the point cloud has exactly num_points points through sampling or padding.

        Args:
            points (np.ndarray): Input point cloud

        Returns:
            np.ndarray: Point cloud with exactly num_points points
        """
        if len(points) == self.num_points:
            return points

        if len(points) > self.num_points:
            # Randomly sample points
            indices = np.random.choice(len(points), self.num_points, replace=False)
            return points[indices]
        else:
            # Pad with repeated points
            padding_indices = np.random.choice(len(points), self.num_points - len(points))
            padding = points[padding_indices]
            return np.vstack([points, padding])

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud to fit in a [-1, 1] cube.

        Args:
            points (np.ndarray): Point cloud of shape (N, 3)

        Returns:
            np.ndarray: Normalized point cloud
        """
        # Center points
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # Scale to [-1, 1]
        scale = np.max(np.abs(points))
        if scale > 0:
            points = points / scale

        return points

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (point_cloud, label)
        """
        return self.data[idx], self.labels[idx]

    def get_shape_name(self, label: int) -> str:
        """Get shape name from label."""
        return self.label_to_shape[label]

    def get_label(self, shape_name: str) -> int:
        """Get label from shape name."""
        return self.shape_to_label[shape_name]

    def get_num_classes(self) -> int:
        """Get the number of shape classes."""
        return len(self.shape_names)

class PointCloudVisualizer:
    """Utility class for visualizing point clouds"""

    @staticmethod
    def visualize_single_shape(points: np.ndarray, title: str = "Point Cloud"):
        """
        Visualize a single point cloud.

        Args:
            points (np.ndarray): Point cloud of shape (N, 3)
            title (str): Title for the plot
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        # Make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0.1)
        ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0.1)
        ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0.1)

        plt.show()

    @staticmethod
    def visualize_batch(batch: Dict[str, torch.Tensor], dataset):
        """
        Visualize a batch of point clouds.

        Args:
            batch (dict): Batch dictionary containing 'points' and 'label'
            dataset (GeometricPointCloudDataset): Dataset object for label mapping
        """
        points = batch['points'].numpy()
        labels = batch['label'].numpy()

        batch_size = points.shape[0]
        rows = int(np.ceil(np.sqrt(batch_size)))
        cols = int(np.ceil(batch_size / rows))

        fig = plt.figure(figsize=(4*cols, 4*rows))

        for i in range(batch_size):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], c='b', marker='.')

            shape_name = dataset.get_shape_name(labels[i])
            ax.set_title(f'{shape_name}')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Make the plot more visually appealing
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0.1)
            ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0.1)
            ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0.1)

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    pass
    visualize_all_shapes()