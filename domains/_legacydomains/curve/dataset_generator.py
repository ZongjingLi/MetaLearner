import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from scipy.integrate import quad

import math

class GeometricShapeGenerator:
    def __init__(self, num_points: int = 100):
        self.num_points = num_points

    def add_noise(self, points: np.ndarray, scale: float = 0.02) -> np.ndarray:
        return points + np.random.normal(0, scale, points.shape)

    def generate_shapes(self) -> Dict[str, np.ndarray]:
        shapes = {
            # Basic shapes
            "circle": self._generate_circle(),
            "ellipse": self._generate_ellipse(),
            "rectangle": self._generate_rectangle(),
            "triangle": self._generate_triangle(),
            "pentagon": self._generate_pentagon(),
            "hexagon": self._generate_hexagon(),
            "star": self._generate_star(),
            "crescent": self._generate_crescent(),

            # Curves
            "flat_line" : self._generate_flat_line(),
            "line" : self._generate_line(),
            "sine_wave": self._generate_sine_wave(),
            "cosine_wave": self._generate_cosine_wave(),
            "spiral": self._generate_spiral(),
            "helix": self._generate_helix(),
            "lemniscate": self._generate_lemniscate(),
            "cardioid": self._generate_cardioid(),
            "astroid": self._generate_astroid(),
            "epicycloid": self._generate_epicycloid(),

            # Complex shapes
            "butterfly": self._generate_butterfly(),
            "infinity": self._generate_infinity(),
            "trefoil": self._generate_trefoil(),
            "rose_curve": self._generate_rose_curve(),
            "hypocycloid": self._generate_hypocycloid(),
            "limacon": self._generate_limacon(),
            "nephroid": self._generate_nephroid(),
            "deltoid": self._generate_deltoid(),

            # Mathematical curves
            "parabola": self._generate_parabola(),
            "hyperbola": self._generate_hyperbola(),
            "catenary": self._generate_catenary(),
            "cycloid": self._generate_cycloid(),
            "involute": self._generate_involute(),
            "archimedean_spiral": self._generate_archimedean_spiral(),
            "logarithmic_spiral": self._generate_logarithmic_spiral(),
            "cochleoid": self._generate_cochleoid(),

            # Additional shapes
            "heart": self._generate_heart(),
            "cloud": self._generate_cloud(),
            "drop": self._generate_drop(),
            "cross": self._generate_cross(),

            "superellipse": self._generate_superellipse(random_params=True),
            "hypotrochoid": self._generate_hypotrochoid(random_params=True),
            "epitrochoid": self._generate_epitrochoid(random_params=True),
            "lissajous": self._generate_lissajous(random_params=True),
            "fourier_curve": self._generate_fourier_curve(random_params=True),
            "power_curve": self._generate_power_curve(random_params=True),
            "butterfly_curve": self._generate_butterfly_curve(random_params=True),
            "hyperbolic_spiral": self._generate_hyperbolic_spiral(random_params=True),
            "clothoid": self._generate_clothoid(random_params=True),
            "fibonacci_spiral": self._generate_fibonacci_spiral(random_params=True),
            "tanh_curve": self._generate_tanh_curve(random_params=True),
            "gaussian_curve": self._generate_gaussian_curve(random_params=True),
            "witch_of_agnesi": self._generate_witch_of_agnesi(random_params=True),
            #"cissoid": self._generate_cissoid(random_params=True),
            #"conchoid": self._generate_conchoid(random_params=True),
            #"folium": self._generate_folium(random_params=True),
            "toric_section": self._generate_toric_section(random_params=True),
            "kappa_curve": self._generate_kappa_curve(random_params=True),
            "cayley_curve": self._generate_cayley_curve(random_params=True),
            #"kampyle": self._generate_kampyle(random_params=True)
        }
        return shapes
    def _generate_superellipse(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            n = np.random.uniform(0.5, 3.0)
        else:
            n = 2.5

        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/n)
        y = np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/n)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_hypotrochoid(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            R = np.random.uniform(2.0, 4.0)
            r = np.random.uniform(0.5, 1.5)
            d = np.random.uniform(0.2, 0.8)
        else:
            R, r, d = 3, 1, 0.5

        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = (R-r)*np.cos(theta) + d*np.cos((R-r)*theta/r)
        y = (R-r)*np.sin(theta) - d*np.sin((R-r)*theta/r)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_epitrochoid(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            R = np.random.uniform(2.0, 4.0)
            r = np.random.uniform(0.5, 1.5)
            d = np.random.uniform(0.2, 0.8)
        else:
            R, r, d = 3, 1, 0.5

        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = (R+r)*np.cos(theta) - d*np.cos((R+r)*theta/r)
        y = (R+r)*np.sin(theta) - d*np.sin((R+r)*theta/r)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_lissajous(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.randint(1, 5)
            b = np.random.randint(1, 5)
            delta = np.random.uniform(0, np.pi/2)
        else:
            a, b, delta = 3, 2, np.pi/4

        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = np.sin(a*theta + delta)
        y = np.sin(b*theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_fourier_curve(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            num_terms = np.random.randint(2, 5)
            coeffs = np.random.uniform(-1, 1, (num_terms, 2))
        else:
            num_terms = 3
            coeffs = np.array([[1, 0], [0.5, 0.5], [0.25, -0.25]])

        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = np.zeros_like(theta)
        y = np.zeros_like(theta)

        for i in range(num_terms):
            x += coeffs[i, 0] * np.cos((i+1)*theta)
            y += coeffs[i, 1] * np.sin((i+1)*theta)

        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_power_curve(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            power = np.random.uniform(0.2, 2.0)
        else:
            power = 0.5

        x = np.linspace(-1, 1, self.num_points)
        y = np.sign(x) * np.abs(x)**power
        return self.add_noise(np.column_stack([x, y]))

    def _generate_line(self) -> np.ndarray:
        x = np.linspace(-1, 1, self.num_points)
        return self.add_noise(np.column_stack([x,x]))

    def _generate_flat_line(self) -> np.ndarray:
        x = np.linspace(-1, 1, self.num_points)
        return self.add_noise(np.column_stack([x,x * 0.0]))

    def _generate_butterfly_curve(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(1.0, 3.0)
            b = np.random.uniform(0.5, 1.5)
        else:
            a, b = 2.0, 1.0

        theta = np.linspace(0, 12*np.pi, self.num_points)
        r = np.exp(np.cos(theta)) - a*np.cos(b*theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_hyperbolic_spiral(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(0.1, 8*np.pi, self.num_points)  # Avoid theta=0
        r = a / theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_clothoid(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.1, 0.3)
        else:
            a = 0.2

        t = np.linspace(0, 8, self.num_points)
        x = np.zeros_like(t)
        y = np.zeros_like(t)

        for i in range(len(t)):
            x[i] = quad(lambda s: np.cos(a*s**2), 0, t[i])[0]
            y[i] = quad(lambda s: np.sin(a*s**2), 0, t[i])[0]

        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_fibonacci_spiral(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            growth_factor = np.random.uniform(1.1, 1.3)
        else:
            growth_factor = 1.618034  # Golden ratio

        theta = np.linspace(0, 8*np.pi, self.num_points)
        r = growth_factor**(theta/(2*np.pi))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_tanh_curve(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            scale = np.random.uniform(0.5, 2.0)
        else:
            scale = 1.0

        x = np.linspace(-3, 3, self.num_points)
        y = np.tanh(scale * x)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_gaussian_curve(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            mu = np.random.uniform(-0.5, 0.5)
            sigma = np.random.uniform(0.2, 0.8)
        else:
            mu, sigma = 0, 0.4

        x = np.linspace(-2, 2, self.num_points)
        y = np.exp(-((x - mu)**2)/(2*sigma**2))
        return self.add_noise(np.column_stack([x, y]))

    def _generate_witch_of_agnesi(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        x = np.linspace(-3, 3, self.num_points)
        y = (8 * a**3) / (x**2 + 4*a**2)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_cissoid(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, self.num_points)
        r = 2 * a * np.sin(theta)**2 / np.cos(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_conchoid(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
            b = np.random.uniform(0.3, 1.0)
        else:
            a, b = 1.0, 0.5

        theta = np.linspace(0, 2*np.pi, self.num_points)
        r = a / np.cos(theta) + b
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_folium(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        t = np.linspace(-2, 2, self.num_points)
        x = (3 * a * t) / (1 + t**3)
        y = (3 * a * t**2) / (1 + t**3)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_toric_section(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(1.5, 3.0)
            b = np.random.uniform(0.5, 1.5)
        else:
            a, b = 2.0, 1.0

        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = (a + b*np.cos(theta)) * np.cos(theta)
        y = (a + b*np.cos(theta)) * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_kappa_curve(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(0, 4*np.pi, self.num_points)
        r = a * (1 + 2*np.sin(theta/2))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_cayley_curve(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = a * np.cos(theta) * (1 + np.cos(theta))
        y = a * np.sin(theta) * (1 + np.cos(theta))
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_kampyle(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        x = np.linspace(-2, 2, self.num_points)
        y = a**2 / x
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_cochleoid_spiral(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(0.1, 8*np.pi, self.num_points)
        r = a * np.sin(theta) / theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_bernoulli_lemniscate(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(0, 2*np.pi, self.num_points)
        r = np.sqrt(2 * a**2 * np.cos(2*theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_strophoid(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, self.num_points)
        x = a * (np.cos(2*theta) / np.cos(theta))
        y = a * (np.sin(2*theta) / np.cos(theta))
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_sturmian_spiral(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(1.5, 3.0)
            b = np.random.uniform(0.5, 1.5)
        else:
            a, b = 2.0, 1.0

        theta = np.linspace(0, 8*np.pi, self.num_points)
        r = a / (theta**b)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def _generate_fermat_spiral(self, random_params: bool = False) -> np.ndarray:
        if random_params:
            a = np.random.uniform(0.5, 2.0)
        else:
            a = 1.0

        theta = np.linspace(0, 8*np.pi, self.num_points)
        r = a * np.sqrt(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(self._normalize_points(np.column_stack([x, y])))

    def generate_random_batch(self, batch_size: int = 64) -> np.ndarray:
        """Generate a batch of random curves with random parameters."""
        curves = []
        generators = [method for method in dir(self) if method.startswith('_generate_')]

        for _ in range(batch_size):
            generator = np.random.choice(generators)
            curve = getattr(self, generator)(random_params=True)
            curves.append(curve)

        return np.stack(curves)

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to fit in [-1, 1] range."""
        max_abs = np.max(np.abs(points))
        if max_abs > 0:
            return points / max_abs
        return points

    def _generate_circle(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = np.cos(theta)
        y = np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_ellipse(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        a, b = 1.5, 1.0
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_rectangle(self) -> np.ndarray:
        t = np.linspace(0, 4, self.num_points)
        x = np.where(t <= 1, t,
            np.where(t <= 2, 1,
                np.where(t <= 3, 3-t, 0)))
        y = np.where(t <= 1, 0,
            np.where(t <= 2, t-1,
                np.where(t <= 3, 1, 4-t)))
        return self.add_noise(np.column_stack([x, y]))

    def _generate_triangle(self) -> np.ndarray:
        t = np.linspace(0, 3, self.num_points)
        x = np.where(t <= 1, t,
            np.where(t <= 2, 2-t, t-3))
        y = np.where(t <= 1, t,
            np.where(t <= 2, 2-t, 0))
        return self.add_noise(np.column_stack([x, y]))

    def _generate_pentagon(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack([x, y])
        angles = 2 * np.pi * np.arange(5) / 5
        vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        return self.add_noise(points)

    def _generate_hexagon(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack([x, y])
        angles = 2 * np.pi * np.arange(6) / 6
        vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        return self.add_noise(points)

    def _generate_star(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        r = 0.5 + 0.5 * np.sin(5*theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_crescent(self) -> np.ndarray:
        theta = np.linspace(-np.pi/2, np.pi/2, self.num_points)
        r1, r2 = 1.0, 0.8
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        x2 = r2 * np.cos(theta) + 0.2
        y2 = r2 * np.sin(theta)
        points = np.concatenate([
            np.column_stack([x1, y1]),
            np.column_stack([x2[::-1], y2[::-1]])
        ])
        indices = np.linspace(0, len(points)-1, self.num_points).astype(int)
        return self.add_noise(points[indices])

    def _generate_sine_wave(self) -> np.ndarray:
        x = np.linspace(-2*np.pi, 2*np.pi, self.num_points)
        y = np.sin(x)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_cosine_wave(self) -> np.ndarray:
        x = np.linspace(-2*np.pi, 2*np.pi, self.num_points)
        y = np.cos(x)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_spiral(self) -> np.ndarray:
        theta = np.linspace(0, 6*np.pi, self.num_points)
        r = theta/6
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_helix(self) -> np.ndarray:
        t = np.linspace(0, 4*np.pi, self.num_points)
        x = np.cos(t)
        y = np.sin(t)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_lemniscate(self) -> np.ndarray:
        t = np.linspace(0, 2*np.pi, self.num_points)
        x = np.cos(t) / (1 + np.sin(t)**2)
        y = np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_cardioid(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        a = 1
        x = a * (2*np.cos(theta) - np.cos(2*theta))
        y = a * (2*np.sin(theta) - np.sin(2*theta))
        return self.add_noise(np.column_stack([x, y]))

    def _generate_astroid(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = np.cos(theta)**3
        y = np.sin(theta)**3
        return self.add_noise(np.column_stack([x, y]))

    def _generate_epicycloid(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        a, b = 1, 0.3
        x = (a+b)*np.cos(theta) - b*np.cos((a+b)*theta/b)
        y = (a+b)*np.sin(theta) - b*np.sin((a+b)*theta/b)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_butterfly(self) -> np.ndarray:
        theta = np.linspace(0, 24*np.pi, self.num_points)
        r = np.exp(np.cos(theta)) - 2*np.cos(4*theta) + np.sin(theta/12)**5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_infinity(self) -> np.ndarray:
        t = np.linspace(0, 2*np.pi, self.num_points)
        x = np.cos(t) / (1 + np.sin(t)**2)
        y = np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_trefoil(self) -> np.ndarray:
        t = np.linspace(0, 2*np.pi, self.num_points)
        x = np.sin(3*t) * np.cos(t)
        y = np.sin(3*t) * np.sin(t)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_rose_curve(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        k = 3
        r = np.cos(k*theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_hypocycloid(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        a, b = 5, 3
        x = (a-b)*np.cos(theta) + b*np.cos((a-b)*theta/b)
        y = (a-b)*np.sin(theta) - b*np.sin((a-b)*theta/b)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_limacon(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        a, b = 1, 0.5
        r = a + b*np.cos(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_nephroid(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        a = 1
        x = a * (3*np.cos(theta) - np.cos(3*theta))
        y = a * (3*np.sin(theta) - np.sin(3*theta))
        return self.add_noise(np.column_stack([x, y]))

    def _generate_deltoid(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        a = 1
        x = 2*a*np.cos(theta) + a*np.cos(2*theta)
        y = 2*a*np.sin(theta) - a*np.sin(2*theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_parabola(self) -> np.ndarray:
        x = np.linspace(-2, 2, self.num_points)
        y = x**2
        return self.add_noise(np.column_stack([x, y]))

    def _generate_hyperbola(self) -> np.ndarray:
        x = np.linspace(-2, 2, self.num_points)
        x = x[x != 0]  # Avoid division by zero
        y = (1/(x*x + 1.0))
        return self.add_noise(np.column_stack([x, y]))

    def _generate_catenary(self) -> np.ndarray:
        x = np.linspace(-2, 2, self.num_points)
        y = np.cosh(x)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_cycloid(self) -> np.ndarray:
        theta = np.linspace(0, 4*np.pi, self.num_points)
        x = theta - np.sin(theta)
        y = 1 - np.cos(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_involute(self) -> np.ndarray:
        theta = np.linspace(0, 4*np.pi, self.num_points)
        x = np.cos(theta) + theta*np.sin(theta)
        y = np.sin(theta) - theta*np.cos(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_archimedean_spiral(self) -> np.ndarray:
        theta = np.linspace(0, 6*np.pi, self.num_points)
        r = theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_logarithmic_spiral(self) -> np.ndarray:
        theta = np.linspace(0, 4*np.pi, self.num_points)
        a = 0.1
        r = np.exp(a * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_cochleoid(self) -> np.ndarray:
        theta = np.linspace(0.1, 8*np.pi, self.num_points)  # Avoid theta=0
        r = np.sin(theta) / theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.add_noise(np.column_stack([x, y]))

    def _generate_heart(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = 16 * np.sin(theta)**3
        y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
        # Normalize to similar scale as other shapes
        x = x / 16
        y = y / 16
        return self.add_noise(np.column_stack([x, y]))

    def _generate_cloud(self) -> np.ndarray:
        # Generate cloud using multiple circles
        circles = []
        centers = [(0, 0), (0.4, 0), (0.8, 0), (0.4, 0.3), (0.4, -0.3)]
        radii = [0.5, 0.4, 0.3, 0.3, 0.3]
        points_per_circle = self.num_points // len(centers)

        for (cx, cy), r in zip(centers, radii):
            theta = np.linspace(0, 2*np.pi, points_per_circle)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            circles.append(np.column_stack([x, y]))

        points = np.concatenate(circles)
        indices = np.linspace(0, len(points)-1, self.num_points).astype(int)
        return self.add_noise(points[indices])

    def _generate_drop(self) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, self.num_points)
        r = 1 - np.sin(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta) - 0.5
        return self.add_noise(np.column_stack([x, y]))

    def _generate_cross(self) -> np.ndarray:
        # Generate points for a cross shape
        vertical = np.linspace(-1, 1, self.num_points // 2)
        horizontal = np.linspace(-0.5, 0.5, self.num_points // 4)

        v_points = np.column_stack([np.zeros_like(vertical), vertical])
        h_points = np.column_stack([horizontal, np.zeros_like(horizontal)])

        points = np.concatenate([v_points, h_points])
        indices = np.linspace(0, len(points)-1, self.num_points).astype(int)
        return self.add_noise(points[indices])