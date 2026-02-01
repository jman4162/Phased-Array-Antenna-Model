"""
Array geometry definitions and generation functions.

Includes rectangular, circular, conformal, sparse, and subarray architectures.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ArrayGeometry:
    """
    Container for phased array element positions and orientations.

    Attributes
    ----------
    x : ndarray
        Element x-positions in meters
    y : ndarray
        Element y-positions in meters
    z : ndarray, optional
        Element z-positions in meters (for 3D/conformal arrays)
    nx : ndarray, optional
        Element normal x-components (unit vectors)
    ny : ndarray, optional
        Element normal y-components
    nz : ndarray, optional
        Element normal z-components
    element_indices : ndarray, optional
        Original indices before any thinning (for tracking)
    """
    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray] = None
    nx: Optional[np.ndarray] = None
    ny: Optional[np.ndarray] = None
    nz: Optional[np.ndarray] = None
    element_indices: Optional[np.ndarray] = None

    @property
    def n_elements(self) -> int:
        """Number of elements in the array."""
        return len(self.x)

    @property
    def is_planar(self) -> bool:
        """Check if array is planar (all z and normals the same)."""
        if self.z is None:
            return True
        return np.allclose(self.z, self.z[0])

    @property
    def is_conformal(self) -> bool:
        """Check if array has varying element orientations."""
        if self.nx is None or self.ny is None or self.nz is None:
            return False
        # Check if all normals point the same direction
        if np.allclose(self.nx, self.nx[0]) and \
           np.allclose(self.ny, self.ny[0]) and \
           np.allclose(self.nz, self.nz[0]):
            return False
        return True

    def copy(self) -> 'ArrayGeometry':
        """Create a deep copy of the geometry."""
        return ArrayGeometry(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy() if self.z is not None else None,
            nx=self.nx.copy() if self.nx is not None else None,
            ny=self.ny.copy() if self.ny is not None else None,
            nz=self.nz.copy() if self.nz is not None else None,
            element_indices=self.element_indices.copy() if self.element_indices is not None else None
        )


@dataclass
class SubarrayArchitecture:
    """
    Defines subarray partitioning of a phased array.

    Supports both non-overlapped and overlapped subarray architectures.
    In overlapped architectures, elements can belong to multiple subarrays
    with potentially different weights.

    Attributes
    ----------
    geometry : ArrayGeometry
        Full array geometry
    subarray_assignments : ndarray, optional
        Subarray index for each element (shape: n_elements,)
        None for overlapped subarrays
    n_subarrays : int
        Number of subarrays
    subarray_centers : ndarray, optional
        Center positions of each subarray (n_subarrays, 2 or 3)
    overlapped : bool
        True if this is an overlapped subarray architecture
    subarray_elements : list of ndarray, optional
        For overlapped: list of element indices for each subarray
    overlap_weights : list of ndarray, optional
        For overlapped: amplitude weights for shared elements in each subarray
    """
    geometry: ArrayGeometry
    subarray_assignments: Optional[np.ndarray]
    n_subarrays: int
    subarray_centers: Optional[np.ndarray] = None
    overlapped: bool = False
    subarray_elements: Optional[List[np.ndarray]] = None
    overlap_weights: Optional[List[np.ndarray]] = None

    def get_subarray_elements(self, subarray_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get element indices and mask for a specific subarray."""
        if self.overlapped and self.subarray_elements is not None:
            indices = self.subarray_elements[subarray_idx]
            mask = np.zeros(self.geometry.n_elements, dtype=bool)
            mask[indices] = True
            return indices, mask
        else:
            mask = self.subarray_assignments == subarray_idx
            indices = np.where(mask)[0]
            return indices, mask

    def get_element_subarrays(self, element_idx: int) -> np.ndarray:
        """
        Get indices of all subarrays that contain a given element.

        For non-overlapped architectures, returns single subarray index.
        For overlapped, may return multiple indices.

        Parameters
        ----------
        element_idx : int
            Index of the element

        Returns
        -------
        subarray_indices : ndarray
            Array of subarray indices containing this element
        """
        if self.overlapped and self.subarray_elements is not None:
            subarrays = []
            for sub_idx, elements in enumerate(self.subarray_elements):
                if element_idx in elements:
                    subarrays.append(sub_idx)
            return np.array(subarrays)
        else:
            return np.array([self.subarray_assignments[element_idx]])


def create_rectangular_array(
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    wavelength: float = 1.0,
    center: bool = True
) -> ArrayGeometry:
    """
    Create a rectangular planar array.

    Parameters
    ----------
    Nx : int
        Number of elements in x
    Ny : int
        Number of elements in y
    dx : float
        Element spacing in x (in wavelengths)
    dy : float
        Element spacing in y (in wavelengths)
    wavelength : float
        Wavelength in meters (converts spacing to meters)
    center : bool
        If True, center array at origin

    Returns
    -------
    geometry : ArrayGeometry
        Array geometry with element positions

    Examples
    --------
    Create a 16x16 array with half-wavelength spacing:

    >>> import phased_array as pa
    >>> geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
    >>> geom.n_elements
    256
    >>> geom.x.shape
    (256,)

    Create a 10x10 array at 10 GHz (3 cm wavelength):

    >>> geom = pa.create_rectangular_array(10, 10, dx=0.5, dy=0.5, wavelength=0.03)
    >>> geom.n_elements
    100

    Access element positions:

    >>> geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
    >>> geom.is_planar
    True
    """
    # Convert spacing to meters
    dx_m = dx * wavelength
    dy_m = dy * wavelength

    # Create grid
    x_1d = np.arange(Nx) * dx_m
    y_1d = np.arange(Ny) * dy_m

    if center:
        x_1d -= x_1d.mean()
        y_1d -= y_1d.mean()

    x_grid, y_grid = np.meshgrid(x_1d, y_1d, indexing='ij')
    x = x_grid.ravel()
    y = y_grid.ravel()

    return ArrayGeometry(
        x=x, y=y,
        z=np.zeros_like(x),
        nx=np.zeros_like(x),
        ny=np.zeros_like(x),
        nz=np.ones_like(x),
        element_indices=np.arange(len(x))
    )


def create_triangular_array(
    Nx: int,
    Ny: int,
    dx: float,
    dy: Optional[float] = None,
    wavelength: float = 1.0,
    center: bool = True
) -> ArrayGeometry:
    """
    Create an offset triangular (hexagonal) array.

    Rows are offset by half the x-spacing for optimal packing.

    Parameters
    ----------
    Nx : int
        Approximate number of elements in x
    Ny : int
        Number of rows in y
    dx : float
        Element spacing in x (in wavelengths)
    dy : float, optional
        Row spacing in y. Default: dx * sqrt(3)/2 for equilateral
    wavelength : float
        Wavelength in meters
    center : bool
        If True, center array at origin

    Returns
    -------
    geometry : ArrayGeometry

    Examples
    --------
    Create a triangular grid array (hexagonal packing):

    >>> import phased_array as pa
    >>> geom = pa.create_triangular_array(10, 10, dx=0.5)
    >>> geom.n_elements  # Slightly fewer than 10x10 due to offset rows
    95
    >>> geom.is_planar
    True

    Triangular grids provide ~13% better packing efficiency:

    >>> rect = pa.create_rectangular_array(10, 10, dx=0.5, dy=0.5)
    >>> tri = pa.create_triangular_array(10, 10, dx=0.5)
    >>> rect.n_elements > tri.n_elements  # Rect has more for same Nx, Ny
    True
    """
    if dy is None:
        dy = dx * np.sqrt(3) / 2

    dx_m = dx * wavelength
    dy_m = dy * wavelength

    x_list = []
    y_list = []

    for row in range(Ny):
        offset = (dx_m / 2) if (row % 2 == 1) else 0
        n_in_row = Nx if (row % 2 == 0) else (Nx - 1)

        for col in range(n_in_row):
            x_list.append(col * dx_m + offset)
            y_list.append(row * dy_m)

    x = np.array(x_list)
    y = np.array(y_list)

    if center:
        x -= x.mean()
        y -= y.mean()

    return ArrayGeometry(
        x=x, y=y,
        z=np.zeros_like(x),
        nx=np.zeros_like(x),
        ny=np.zeros_like(x),
        nz=np.ones_like(x),
        element_indices=np.arange(len(x))
    )


def create_elliptical_array(
    a: float,
    b: float,
    dx: float,
    dy: Optional[float] = None,
    grid_type: str = 'rectangular',
    wavelength: float = 1.0
) -> ArrayGeometry:
    """
    Create an array within an elliptical boundary.

    Parameters
    ----------
    a : float
        Ellipse semi-axis in x (in wavelengths)
    b : float
        Ellipse semi-axis in y (in wavelengths)
    dx : float
        Element spacing in x (in wavelengths)
    dy : float, optional
        Element spacing in y (default: same as dx)
    grid_type : str
        'rectangular' or 'triangular'
    wavelength : float
        Wavelength in meters

    Returns
    -------
    geometry : ArrayGeometry
    """
    if dy is None:
        dy = dx

    a_m = a * wavelength
    b_m = b * wavelength
    dx_m = dx * wavelength
    dy_m = dy * wavelength if grid_type == 'rectangular' else dx * np.sqrt(3) / 2 * wavelength

    # Generate grid larger than ellipse
    nx_max = int(np.ceil(2 * a / dx)) + 2
    ny_max = int(np.ceil(2 * b / dy)) + 2

    x_list = []
    y_list = []

    for row in range(ny_max):
        if grid_type == 'triangular':
            offset = (dx_m / 2) if (row % 2 == 1) else 0
        else:
            offset = 0

        for col in range(nx_max):
            x_pos = col * dx_m + offset - a_m
            y_pos = row * dy_m - b_m

            # Check if inside ellipse
            if (x_pos / a_m) ** 2 + (y_pos / b_m) ** 2 <= 1.0:
                x_list.append(x_pos)
                y_list.append(y_pos)

    x = np.array(x_list)
    y = np.array(y_list)

    # Center
    x -= x.mean()
    y -= y.mean()

    return ArrayGeometry(
        x=x, y=y,
        z=np.zeros_like(x),
        nx=np.zeros_like(x),
        ny=np.zeros_like(x),
        nz=np.ones_like(x),
        element_indices=np.arange(len(x))
    )


def create_circular_array(
    n_elements: int,
    radius: float,
    wavelength: float = 1.0,
    start_angle: float = 0.0
) -> ArrayGeometry:
    """
    Create a circular (ring) array in the xy-plane.

    Parameters
    ----------
    n_elements : int
        Number of elements
    radius : float
        Ring radius in wavelengths
    wavelength : float
        Wavelength in meters
    start_angle : float
        Starting angle in radians

    Returns
    -------
    geometry : ArrayGeometry
    """
    r_m = radius * wavelength
    angles = np.linspace(start_angle, start_angle + 2*np.pi, n_elements, endpoint=False)

    x = r_m * np.cos(angles)
    y = r_m * np.sin(angles)

    # Normals point radially outward for circular array
    nx = np.cos(angles)
    ny = np.sin(angles)
    nz = np.zeros_like(x)

    return ArrayGeometry(
        x=x, y=y,
        z=np.zeros_like(x),
        nx=nx, ny=ny, nz=nz,
        element_indices=np.arange(n_elements)
    )


def create_concentric_rings_array(
    n_rings: int,
    elements_per_ring: Union[int, List[int]],
    ring_spacing: float,
    wavelength: float = 1.0,
    include_center: bool = True
) -> ArrayGeometry:
    """
    Create a concentric rings array.

    Parameters
    ----------
    n_rings : int
        Number of rings
    elements_per_ring : int or list
        Elements per ring (int applies to all, list for custom)
    ring_spacing : float
        Spacing between rings in wavelengths
    wavelength : float
        Wavelength in meters
    include_center : bool
        Include a center element

    Returns
    -------
    geometry : ArrayGeometry
    """
    x_list = []
    y_list = []
    nx_list = []
    ny_list = []

    if include_center:
        x_list.append(0.0)
        y_list.append(0.0)
        nx_list.append(0.0)
        ny_list.append(0.0)

    if isinstance(elements_per_ring, int):
        elements_per_ring = [elements_per_ring] * n_rings

    for ring_idx, n_elem in enumerate(elements_per_ring[:n_rings]):
        radius = (ring_idx + 1) * ring_spacing * wavelength
        angles = np.linspace(0, 2*np.pi, n_elem, endpoint=False)

        for angle in angles:
            x_list.append(radius * np.cos(angle))
            y_list.append(radius * np.sin(angle))
            nx_list.append(np.cos(angle))
            ny_list.append(np.sin(angle))

    x = np.array(x_list)
    y = np.array(y_list)

    return ArrayGeometry(
        x=x, y=y,
        z=np.zeros_like(x),
        nx=np.array(nx_list),
        ny=np.array(ny_list),
        nz=np.zeros_like(x),
        element_indices=np.arange(len(x))
    )


def create_cylindrical_array(
    n_azimuth: int,
    n_vertical: int,
    radius: float,
    height: float,
    wavelength: float = 1.0
) -> ArrayGeometry:
    """
    Create a cylindrical conformal array.

    Elements are distributed on a cylinder surface with normals
    pointing radially outward.

    Parameters
    ----------
    n_azimuth : int
        Number of elements around circumference
    n_vertical : int
        Number of vertical rings
    radius : float
        Cylinder radius in wavelengths
    height : float
        Cylinder height in wavelengths
    wavelength : float
        Wavelength in meters

    Returns
    -------
    geometry : ArrayGeometry
    """
    r_m = radius * wavelength
    h_m = height * wavelength

    phi = np.linspace(0, 2*np.pi, n_azimuth, endpoint=False)
    z_vals = np.linspace(-h_m/2, h_m/2, n_vertical)

    phi_grid, z_grid = np.meshgrid(phi, z_vals, indexing='ij')
    phi_flat = phi_grid.ravel()
    z_flat = z_grid.ravel()

    x = r_m * np.cos(phi_flat)
    y = r_m * np.sin(phi_flat)
    z = z_flat

    # Normals point radially outward
    nx = np.cos(phi_flat)
    ny = np.sin(phi_flat)
    nz = np.zeros_like(x)

    return ArrayGeometry(
        x=x, y=y, z=z,
        nx=nx, ny=ny, nz=nz,
        element_indices=np.arange(len(x))
    )


def create_spherical_array(
    n_theta: int,
    n_phi: int,
    radius: float,
    theta_min: float = 0.0,
    theta_max: float = np.pi/2,
    wavelength: float = 1.0
) -> ArrayGeometry:
    """
    Create a spherical conformal array.

    Parameters
    ----------
    n_theta : int
        Number of elements in theta (elevation)
    n_phi : int
        Number of elements in phi (azimuth)
    radius : float
        Sphere radius in wavelengths
    theta_min : float
        Minimum theta in radians (0 = north pole)
    theta_max : float
        Maximum theta in radians
    wavelength : float
        Wavelength in meters

    Returns
    -------
    geometry : ArrayGeometry
    """
    r_m = radius * wavelength

    theta = np.linspace(theta_min, theta_max, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()

    # Spherical to Cartesian
    x = r_m * np.sin(theta_flat) * np.cos(phi_flat)
    y = r_m * np.sin(theta_flat) * np.sin(phi_flat)
    z = r_m * np.cos(theta_flat)

    # Normals point radially outward
    nx = np.sin(theta_flat) * np.cos(phi_flat)
    ny = np.sin(theta_flat) * np.sin(phi_flat)
    nz = np.cos(theta_flat)

    return ArrayGeometry(
        x=x, y=y, z=z,
        nx=nx, ny=ny, nz=nz,
        element_indices=np.arange(len(x))
    )


# ============== Sparse/Thinned Arrays ==============

def thin_array_random(
    geometry: ArrayGeometry,
    thinning_factor: float,
    seed: Optional[int] = None
) -> ArrayGeometry:
    """
    Randomly thin an array by removing elements.

    Parameters
    ----------
    geometry : ArrayGeometry
        Original array geometry
    thinning_factor : float
        Fraction of elements to keep (0 to 1)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    geometry : ArrayGeometry
        Thinned array geometry
    """
    if seed is not None:
        np.random.seed(seed)

    n_keep = int(geometry.n_elements * thinning_factor)
    keep_indices = np.random.choice(geometry.n_elements, n_keep, replace=False)
    keep_indices = np.sort(keep_indices)

    return ArrayGeometry(
        x=geometry.x[keep_indices],
        y=geometry.y[keep_indices],
        z=geometry.z[keep_indices] if geometry.z is not None else None,
        nx=geometry.nx[keep_indices] if geometry.nx is not None else None,
        ny=geometry.ny[keep_indices] if geometry.ny is not None else None,
        nz=geometry.nz[keep_indices] if geometry.nz is not None else None,
        element_indices=keep_indices
    )


def thin_array_density_tapered(
    geometry: ArrayGeometry,
    density_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    seed: Optional[int] = None
) -> ArrayGeometry:
    """
    Thin array with position-dependent density (statistical thinning).

    Parameters
    ----------
    geometry : ArrayGeometry
        Original array geometry
    density_func : callable
        Function(x, y) -> probability of keeping each element (0 to 1)
    seed : int, optional
        Random seed

    Returns
    -------
    geometry : ArrayGeometry
        Thinned geometry
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute keep probability for each element
    keep_prob = density_func(geometry.x, geometry.y)
    keep_prob = np.clip(keep_prob, 0, 1)

    # Random selection based on probability
    keep_mask = np.random.random(geometry.n_elements) < keep_prob
    keep_indices = np.where(keep_mask)[0]

    return ArrayGeometry(
        x=geometry.x[keep_indices],
        y=geometry.y[keep_indices],
        z=geometry.z[keep_indices] if geometry.z is not None else None,
        nx=geometry.nx[keep_indices] if geometry.nx is not None else None,
        ny=geometry.ny[keep_indices] if geometry.ny is not None else None,
        nz=geometry.nz[keep_indices] if geometry.nz is not None else None,
        element_indices=keep_indices
    )


def thin_array_genetic_algorithm(
    geometry: ArrayGeometry,
    n_target: int,
    objective_func: Callable[[ArrayGeometry], float],
    population_size: int = 50,
    n_generations: int = 100,
    mutation_rate: float = 0.1,
    seed: Optional[int] = None
) -> ArrayGeometry:
    """
    Optimize element selection using a genetic algorithm.

    Parameters
    ----------
    geometry : ArrayGeometry
        Original full array geometry
    n_target : int
        Target number of elements to keep
    objective_func : callable
        Function(ArrayGeometry) -> float to minimize
    population_size : int
        GA population size
    n_generations : int
        Number of generations
    mutation_rate : float
        Probability of mutation per gene
    seed : int, optional
        Random seed

    Returns
    -------
    geometry : ArrayGeometry
        Optimized thinned geometry
    """
    if seed is not None:
        np.random.seed(seed)

    n_elements = geometry.n_elements

    def create_individual():
        """Create a random selection of n_target elements."""
        return np.random.choice(n_elements, n_target, replace=False)

    def evaluate(individual):
        """Evaluate fitness of an individual."""
        geom = ArrayGeometry(
            x=geometry.x[individual],
            y=geometry.y[individual],
            z=geometry.z[individual] if geometry.z is not None else None,
            nx=geometry.nx[individual] if geometry.nx is not None else None,
            ny=geometry.ny[individual] if geometry.ny is not None else None,
            nz=geometry.nz[individual] if geometry.nz is not None else None,
            element_indices=individual
        )
        return objective_func(geom)

    def crossover(parent1, parent2):
        """Single-point crossover maintaining n_target elements."""
        # Combine unique elements from both parents
        combined = np.unique(np.concatenate([parent1, parent2]))
        if len(combined) >= n_target:
            return np.random.choice(combined, n_target, replace=False)
        else:
            # Fill with random elements if needed
            remaining = np.setdiff1d(np.arange(n_elements), combined)
            additional = np.random.choice(remaining, n_target - len(combined), replace=False)
            return np.concatenate([combined, additional])

    def mutate(individual):
        """Mutate by swapping elements."""
        individual = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                # Swap with a random element not in the array
                available = np.setdiff1d(np.arange(n_elements), individual)
                if len(available) > 0:
                    individual[i] = np.random.choice(available)
        return individual

    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    fitness = [evaluate(ind) for ind in population]

    for gen in range(n_generations):
        # Selection (tournament)
        new_population = []
        elite_idx = np.argmin(fitness)
        new_population.append(population[elite_idx])  # Elitism

        while len(new_population) < population_size:
            # Tournament selection
            i1, i2 = np.random.choice(population_size, 2, replace=False)
            parent1 = population[i1] if fitness[i1] < fitness[i2] else population[i2]

            i3, i4 = np.random.choice(population_size, 2, replace=False)
            parent2 = population[i3] if fitness[i3] < fitness[i4] else population[i4]

            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        fitness = [evaluate(ind) for ind in population]

    # Return best individual
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]

    return ArrayGeometry(
        x=geometry.x[best_individual],
        y=geometry.y[best_individual],
        z=geometry.z[best_individual] if geometry.z is not None else None,
        nx=geometry.nx[best_individual] if geometry.nx is not None else None,
        ny=geometry.ny[best_individual] if geometry.ny is not None else None,
        nz=geometry.nz[best_individual] if geometry.nz is not None else None,
        element_indices=best_individual
    )


# ============== Subarray Architectures ==============

def create_rectangular_subarrays(
    Nx_total: int,
    Ny_total: int,
    Nx_sub: int,
    Ny_sub: int,
    dx: float,
    dy: float,
    wavelength: float = 1.0
) -> SubarrayArchitecture:
    """
    Create rectangular subarrays from a rectangular array.

    Parameters
    ----------
    Nx_total : int
        Total elements in x
    Ny_total : int
        Total elements in y
    Nx_sub : int
        Elements per subarray in x
    Ny_sub : int
        Elements per subarray in y
    dx, dy : float
        Element spacing in wavelengths
    wavelength : float
        Wavelength in meters

    Returns
    -------
    architecture : SubarrayArchitecture
    """
    # Create full array
    geometry = create_rectangular_array(Nx_total, Ny_total, dx, dy, wavelength)

    # Number of subarrays
    n_sub_x = Nx_total // Nx_sub
    n_sub_y = Ny_total // Ny_sub
    n_subarrays = n_sub_x * n_sub_y

    # Assign elements to subarrays
    assignments = np.zeros(geometry.n_elements, dtype=int)
    subarray_centers = []

    # Element positions on the original grid
    for sub_idx in range(n_subarrays):
        sub_ix = sub_idx % n_sub_x
        sub_iy = sub_idx // n_sub_x

        # Elements in this subarray
        x_start = sub_ix * Nx_sub
        y_start = sub_iy * Ny_sub

        center_x = 0.0
        center_y = 0.0
        count = 0

        for ix in range(Nx_sub):
            for iy in range(Ny_sub):
                global_ix = x_start + ix
                global_iy = y_start + iy
                if global_ix < Nx_total and global_iy < Ny_total:
                    elem_idx = global_ix * Ny_total + global_iy
                    if elem_idx < geometry.n_elements:
                        assignments[elem_idx] = sub_idx
                        center_x += geometry.x[elem_idx]
                        center_y += geometry.y[elem_idx]
                        count += 1

        if count > 0:
            subarray_centers.append([center_x / count, center_y / count])
        else:
            subarray_centers.append([0, 0])

    return SubarrayArchitecture(
        geometry=geometry,
        subarray_assignments=assignments,
        n_subarrays=n_subarrays,
        subarray_centers=np.array(subarray_centers)
    )


def compute_subarray_weights(
    architecture: SubarrayArchitecture,
    k: float,
    theta0_deg: float,
    phi0_deg: float,
    intra_subarray_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute element weights for subarray-level beamforming.

    Phase is quantized to subarray centers (one phase shifter per subarray).

    Parameters
    ----------
    architecture : SubarrayArchitecture
        Subarray architecture
    k : float
        Wavenumber
    theta0_deg, phi0_deg : float
        Steering direction in degrees
    intra_subarray_weights : ndarray, optional
        Amplitude taper applied within each subarray

    Returns
    -------
    weights : ndarray
        Complex weights for all elements
    """
    from .core import steering_vector

    geom = architecture.geometry
    weights = np.ones(geom.n_elements, dtype=complex)

    # Apply intra-subarray amplitude weights if provided
    if intra_subarray_weights is not None:
        weights *= intra_subarray_weights

    # Compute steering phase for each subarray center
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)
    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    for sub_idx in range(architecture.n_subarrays):
        center = architecture.subarray_centers[sub_idx]
        phase = k * (center[0] * u0 + center[1] * v0)

        # Apply this phase to all elements in the subarray
        mask = architecture.subarray_assignments == sub_idx
        weights[mask] *= np.exp(-1j * phase)

    return weights


def array_factor_conformal(
    theta: np.ndarray,
    phi: np.ndarray,
    geometry: ArrayGeometry,
    weights: np.ndarray,
    k: float,
    element_pattern_func: Optional[Callable] = None,
    **element_kwargs
) -> np.ndarray:
    """
    Compute array factor for conformal arrays with element orientation.

    Each element's contribution is weighted by its projected pattern
    in the observation direction.

    Parameters
    ----------
    theta : ndarray
        Observation theta angles in radians
    phi : ndarray
        Observation phi angles in radians
    geometry : ArrayGeometry
        Array geometry with element normals
    weights : ndarray
        Complex element weights
    k : float
        Wavenumber
    element_pattern_func : callable, optional
        Element pattern function
    **element_kwargs
        Arguments for element pattern function

    Returns
    -------
    AF : ndarray
        Array factor (same shape as theta)
    """
    original_shape = theta.shape
    theta_flat = theta.ravel()
    phi_flat = phi.ravel()
    n_angles = len(theta_flat)

    # Observation direction unit vectors
    obs_x = np.sin(theta_flat) * np.cos(phi_flat)
    obs_y = np.sin(theta_flat) * np.sin(phi_flat)
    obs_z = np.cos(theta_flat)

    AF = np.zeros(n_angles, dtype=complex)

    for i in range(geometry.n_elements):
        # Phase from element position
        phase = k * (geometry.x[i] * obs_x + geometry.y[i] * obs_y)
        if geometry.z is not None:
            phase += k * geometry.z[i] * obs_z

        # Element pattern correction for orientation
        if geometry.nx is not None and geometry.ny is not None and geometry.nz is not None:
            # Cosine of angle between observation direction and element normal
            cos_angle = (geometry.nx[i] * obs_x +
                        geometry.ny[i] * obs_y +
                        geometry.nz[i] * obs_z)
            cos_angle = np.maximum(cos_angle, 0)  # Only forward hemisphere

            if element_pattern_func is not None:
                # Local theta for this element (angle from its normal)
                local_theta = np.arccos(np.clip(cos_angle, -1, 1))
                element_gain = element_pattern_func(local_theta, phi_flat, **element_kwargs)
            else:
                element_gain = cos_angle  # Simple cosine pattern
        else:
            element_gain = 1.0

        AF += weights[i] * element_gain * np.exp(1j * phase)

    return AF.reshape(original_shape)


# ============== Overlapped Subarrays ==============

def create_overlapped_subarrays(
    Nx_total: int,
    Ny_total: int,
    Nx_sub: int,
    Ny_sub: int,
    overlap_x: int,
    overlap_y: int,
    dx: float,
    dy: float,
    wavelength: float = 1.0,
    taper_overlap: bool = True
) -> SubarrayArchitecture:
    """
    Create overlapped subarrays from a rectangular array.

    Overlapped subarrays share elements between adjacent subarrays,
    which can improve pattern characteristics and reduce grating lobes
    while maintaining independent subarray beam control.

    Parameters
    ----------
    Nx_total : int
        Total elements in x
    Ny_total : int
        Total elements in y
    Nx_sub : int
        Elements per subarray in x
    Ny_sub : int
        Elements per subarray in y
    overlap_x : int
        Number of overlapping elements in x between adjacent subarrays
    overlap_y : int
        Number of overlapping elements in y between adjacent subarrays
    dx, dy : float
        Element spacing in wavelengths
    wavelength : float
        Wavelength in meters
    taper_overlap : bool
        If True, apply amplitude taper to overlapping elements

    Returns
    -------
    architecture : SubarrayArchitecture
        Overlapped subarray architecture

    Examples
    --------
    Create 50% overlapped 4x4 subarrays:

    >>> import phased_array as pa
    >>> arch = pa.create_overlapped_subarrays(
    ...     Nx_total=16, Ny_total=16,
    ...     Nx_sub=4, Ny_sub=4,
    ...     overlap_x=2, overlap_y=2,
    ...     dx=0.5, dy=0.5
    ... )
    >>> arch.overlapped
    True
    >>> arch.n_subarrays > 0
    True

    Notes
    -----
    Subarray stride (non-overlapping portion):
        stride_x = Nx_sub - overlap_x
        stride_y = Ny_sub - overlap_y

    Elements in the overlap region receive contributions from
    multiple subarrays, typically with tapered weights.
    """
    # Create full array geometry
    geometry = create_rectangular_array(Nx_total, Ny_total, dx, dy, wavelength)

    # Compute strides
    stride_x = Nx_sub - overlap_x
    stride_y = Ny_sub - overlap_y

    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("Overlap cannot be >= subarray size")

    # Number of subarrays in each direction
    n_sub_x = (Nx_total - Nx_sub) // stride_x + 1
    n_sub_y = (Ny_total - Ny_sub) // stride_y + 1
    n_subarrays = n_sub_x * n_sub_y

    # Build subarray element lists and overlap weights
    subarray_elements = []
    overlap_weights = []
    subarray_centers = []

    for sub_iy in range(n_sub_y):
        for sub_ix in range(n_sub_x):
            # Starting indices for this subarray
            start_x = sub_ix * stride_x
            start_y = sub_iy * stride_y

            elements = []
            weights_list = []
            center_x = 0.0
            center_y = 0.0

            for local_ix in range(Nx_sub):
                for local_iy in range(Ny_sub):
                    global_ix = start_x + local_ix
                    global_iy = start_y + local_iy

                    if global_ix < Nx_total and global_iy < Ny_total:
                        elem_idx = global_ix * Ny_total + global_iy
                        elements.append(elem_idx)
                        center_x += geometry.x[elem_idx]
                        center_y += geometry.y[elem_idx]

                        # Compute overlap weight
                        if taper_overlap:
                            # Taper based on position within subarray
                            # 1.0 in center, reduced at edges if overlapping
                            wx = 1.0
                            wy = 1.0

                            # Left overlap region
                            if local_ix < overlap_x and sub_ix > 0:
                                wx = (local_ix + 0.5) / overlap_x

                            # Right overlap region
                            if local_ix >= Nx_sub - overlap_x and sub_ix < n_sub_x - 1:
                                wx = (Nx_sub - local_ix - 0.5) / overlap_x

                            # Bottom overlap region
                            if local_iy < overlap_y and sub_iy > 0:
                                wy = (local_iy + 0.5) / overlap_y

                            # Top overlap region
                            if local_iy >= Ny_sub - overlap_y and sub_iy < n_sub_y - 1:
                                wy = (Ny_sub - local_iy - 0.5) / overlap_y

                            weights_list.append(wx * wy)
                        else:
                            weights_list.append(1.0)

            subarray_elements.append(np.array(elements, dtype=int))
            overlap_weights.append(np.array(weights_list))

            if len(elements) > 0:
                subarray_centers.append([center_x / len(elements), center_y / len(elements)])
            else:
                subarray_centers.append([0.0, 0.0])

    return SubarrayArchitecture(
        geometry=geometry,
        subarray_assignments=None,  # Not used for overlapped
        n_subarrays=n_subarrays,
        subarray_centers=np.array(subarray_centers),
        overlapped=True,
        subarray_elements=subarray_elements,
        overlap_weights=overlap_weights
    )


def overlapped_subarray_weights(
    architecture: SubarrayArchitecture,
    k: float,
    theta0_deg: float,
    phi0_deg: float,
    subarray_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute element weights for overlapped subarray beamforming.

    Parameters
    ----------
    architecture : SubarrayArchitecture
        Overlapped subarray architecture
    k : float
        Wavenumber
    theta0_deg, phi0_deg : float
        Steering direction in degrees
    subarray_weights : ndarray, optional
        Complex weights for each subarray (n_subarrays,).
        If None, uses uniform amplitude with steering phase.

    Returns
    -------
    weights : ndarray
        Complex weights for all elements

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> arch = pa.create_overlapped_subarrays(
    ...     Nx_total=16, Ny_total=16,
    ...     Nx_sub=4, Ny_sub=4,
    ...     overlap_x=2, overlap_y=2,
    ...     dx=0.5, dy=0.5
    ... )
    >>> k = pa.wavelength_to_k(1.0)
    >>> weights = pa.overlapped_subarray_weights(arch, k, 15, 0)
    >>> weights.shape
    (256,)

    Notes
    -----
    For overlapped architectures, each element's weight is the sum
    of contributions from all subarrays it belongs to, weighted by
    the overlap weight for that subarray.
    """
    if not architecture.overlapped:
        # Fall back to standard subarray weights
        from .beamforming import compute_subarray_weights
        return compute_subarray_weights(architecture, k, theta0_deg, phi0_deg)

    geom = architecture.geometry
    element_weights = np.zeros(geom.n_elements, dtype=complex)

    # Compute steering phases for subarray centers
    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)
    u0 = np.sin(theta0) * np.cos(phi0)
    v0 = np.sin(theta0) * np.sin(phi0)

    for sub_idx in range(architecture.n_subarrays):
        elements = architecture.subarray_elements[sub_idx]
        overlap_w = architecture.overlap_weights[sub_idx]
        center = architecture.subarray_centers[sub_idx]

        # Subarray steering phase
        phase = k * (center[0] * u0 + center[1] * v0)
        subarray_phase = np.exp(-1j * phase)

        # Apply subarray weight if provided
        if subarray_weights is not None:
            subarray_phase *= subarray_weights[sub_idx]

        # Add contribution to each element
        for local_idx, elem_idx in enumerate(elements):
            element_weights[elem_idx] += overlap_w[local_idx] * subarray_phase

    return element_weights


def compute_overlapped_pattern(
    architecture: SubarrayArchitecture,
    k: float,
    theta0_deg: float,
    phi0_deg: float,
    theta_range: Tuple[float, float] = (0, np.pi/2),
    n_points: int = 361,
    phi_cut_deg: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radiation pattern for overlapped subarray architecture.

    Parameters
    ----------
    architecture : SubarrayArchitecture
        Overlapped subarray architecture
    k : float
        Wavenumber
    theta0_deg, phi0_deg : float
        Steering direction in degrees
    theta_range : tuple
        (min, max) theta range in radians
    n_points : int
        Number of pattern points
    phi_cut_deg : float
        Phi angle for the pattern cut

    Returns
    -------
    theta_deg : ndarray
        Theta angles in degrees
    pattern_dB : ndarray
        Normalized pattern in dB

    Examples
    --------
    >>> import numpy as np
    >>> import phased_array as pa
    >>> arch = pa.create_overlapped_subarrays(
    ...     Nx_total=16, Ny_total=16,
    ...     Nx_sub=4, Ny_sub=4,
    ...     overlap_x=2, overlap_y=2,
    ...     dx=0.5, dy=0.5
    ... )
    >>> k = pa.wavelength_to_k(1.0)
    >>> theta_deg, pattern_dB = pa.compute_overlapped_pattern(
    ...     arch, k, theta0_deg=0, phi0_deg=0
    ... )
    >>> len(theta_deg) == 361
    True
    """
    from .core import array_factor_vectorized
    from .utils import linear_to_db

    # Get element weights
    weights = overlapped_subarray_weights(architecture, k, theta0_deg, phi0_deg)

    # Compute pattern
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    phi = np.full_like(theta, np.deg2rad(phi_cut_deg))

    theta_grid = theta.reshape(-1, 1)
    phi_grid = phi.reshape(-1, 1)

    geom = architecture.geometry
    AF = array_factor_vectorized(
        theta_grid, phi_grid,
        geom.x, geom.y, weights, k
    ).ravel()

    # Convert to dB and normalize
    pattern_dB = linear_to_db(np.abs(AF)**2)
    pattern_dB -= np.max(pattern_dB)

    theta_deg = np.rad2deg(theta)

    return theta_deg, pattern_dB
