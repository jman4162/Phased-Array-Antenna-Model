Array Geometries
================

This guide covers the various array geometries supported by the library,
from simple rectangular grids to complex conformal arrays.

Planar Arrays
-------------

Rectangular Grid
^^^^^^^^^^^^^^^^

The most common array configuration. Elements are arranged on a regular
rectangular grid with specified spacing.

.. code-block:: python

   import phased_array as pa

   # 16x16 array with half-wavelength spacing
   geom = pa.create_rectangular_array(
       Nx=16, Ny=16,
       dx=0.5, dy=0.5  # spacing in wavelengths
   )
   print(f"Elements: {geom.n_elements}")  # 256

   # Access element positions
   print(f"X range: {geom.x.min():.2f} to {geom.x.max():.2f}")
   print(f"Y range: {geom.y.min():.2f} to {geom.y.max():.2f}")

**Key parameters:**

- ``Nx, Ny``: Number of elements in each dimension
- ``dx, dy``: Element spacing in wavelengths
- ``wavelength``: Physical wavelength (converts spacing to meters)
- ``center``: If True (default), center array at origin

Triangular Grid
^^^^^^^^^^^^^^^

Offset triangular (hexagonal) lattice provides ~13% better packing efficiency
than rectangular grids while maintaining similar grating lobe performance.

.. code-block:: python

   geom = pa.create_triangular_array(
       Nx=16, Ny=16,
       dx=0.5  # dy is automatically set for equilateral spacing
   )
   print(f"Elements: {geom.n_elements}")  # ~240 (fewer than 16x16)

The row offset is ``dx/2``, and default ``dy = dx * sqrt(3)/2`` for equilateral
triangular cells.

Elliptical Boundary
^^^^^^^^^^^^^^^^^^^

Arrays within elliptical (or circular) boundaries are common in radar systems.

.. code-block:: python

   # Circular array (a = b)
   geom_circle = pa.create_elliptical_array(
       a=4.0, b=4.0,  # semi-axes in wavelengths
       dx=0.5,
       grid_type='rectangular'
   )

   # Elliptical with triangular grid
   geom_ellipse = pa.create_elliptical_array(
       a=5.0, b=3.0,
       dx=0.5,
       grid_type='triangular'
   )

Ring and Circular Arrays
------------------------

Circular Array (Single Ring)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elements arranged in a single ring, useful for direction finding.

.. code-block:: python

   geom = pa.create_circular_array(
       n_elements=16,
       radius=2.0,  # in wavelengths
       start_angle=0.0  # radians
   )

Concentric Rings
^^^^^^^^^^^^^^^^

Multiple concentric rings, often used for low-sidelobe designs.

.. code-block:: python

   geom = pa.create_concentric_rings_array(
       n_rings=4,
       elements_per_ring=[8, 12, 16, 20],  # increasing with radius
       ring_spacing=0.5,
       include_center=True
   )

Conformal Arrays
----------------

Conformal arrays conform to curved surfaces. The key difference from planar
arrays is that elements have different orientations (normal vectors).

Cylindrical Array
^^^^^^^^^^^^^^^^^

Elements on a cylindrical surface, normals pointing radially outward.

.. code-block:: python

   geom = pa.create_cylindrical_array(
       n_azimuth=32,  # around circumference
       n_vertical=8,  # vertical rings
       radius=5.0,    # wavelengths
       height=4.0     # wavelengths
   )
   print(f"Is conformal: {geom.is_conformal}")  # True

Spherical Array
^^^^^^^^^^^^^^^

Elements on a spherical cap, useful for hemispherical coverage.

.. code-block:: python

   geom = pa.create_spherical_array(
       n_theta=8,
       n_phi=32,
       radius=5.0,
       theta_min=0.0,       # start at pole
       theta_max=np.pi/2    # hemisphere
   )

Computing Patterns for Conformal Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conformal arrays require special pattern computation that accounts for
element orientations:

.. code-block:: python

   import numpy as np

   # Create cylindrical array
   geom = pa.create_cylindrical_array(16, 4, radius=3.0, height=2.0)
   k = pa.wavelength_to_k(1.0)

   # Steering weights
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=0, phi0_deg=0, z=geom.z)

   # Use conformal array factor (accounts for element normals)
   theta = np.linspace(0, np.pi/2, 91)
   phi = np.linspace(0, 2*np.pi, 181)
   theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

   AF = pa.array_factor_conformal(
       theta_grid, phi_grid, geom, weights, k,
       element_pattern_func=pa.element_pattern
   )

Sparse/Thinned Arrays
---------------------

Sparse arrays remove elements from a full grid to reduce cost while
maintaining the same aperture size. The tradeoff is higher sidelobes.

Random Thinning
^^^^^^^^^^^^^^^

Simple random element removal:

.. code-block:: python

   # Start with full array
   full_geom = pa.create_rectangular_array(32, 32, dx=0.5, dy=0.5)

   # Keep 50% of elements randomly
   sparse_geom = pa.thin_array_random(
       full_geom,
       thinning_factor=0.5,
       seed=42  # for reproducibility
   )
   print(f"Full: {full_geom.n_elements}, Sparse: {sparse_geom.n_elements}")

Density Taper Thinning
^^^^^^^^^^^^^^^^^^^^^^

Remove elements with probability based on position (more elements at center):

.. code-block:: python

   def taylor_density(x, y):
       # Higher probability near center
       r = np.sqrt(x**2 + y**2)
       r_max = np.sqrt(x.max()**2 + y.max()**2)
       return 1.0 - 0.7 * (r / r_max)**2

   sparse_geom = pa.thin_array_density_tapered(
       full_geom,
       density_func=taylor_density,
       seed=42
   )

Genetic Algorithm Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize element selection to minimize an objective (e.g., peak sidelobe):

.. code-block:: python

   def sidelobe_objective(geom):
       # Compute peak sidelobe level
       k = pa.wavelength_to_k(1.0)
       weights = np.ones(geom.n_elements)
       theta, phi, pattern_dB = pa.compute_full_pattern(
           geom.x, geom.y, weights, k, n_theta=91, n_phi=1
       )
       # Return peak sidelobe (excluding main beam)
       main_beam_idx = np.argmax(pattern_dB[:, 0])
       sidelobes = np.concatenate([
           pattern_dB[:main_beam_idx-5, 0],
           pattern_dB[main_beam_idx+5:, 0]
       ])
       return np.max(sidelobes)

   optimized_geom = pa.thin_array_genetic_algorithm(
       full_geom,
       n_target=500,  # keep 500 elements
       objective_func=sidelobe_objective,
       population_size=30,
       n_generations=50,
       seed=42
   )

Subarray Architectures
----------------------

Subarrays group elements that share a common phase shifter, reducing hardware
cost at the expense of beam quality.

.. code-block:: python

   # Create 64x64 array divided into 8x8 subarrays
   architecture = pa.create_rectangular_subarrays(
       Nx_total=64, Ny_total=64,
       Nx_sub=8, Ny_sub=8,  # 8x8 elements per subarray
       dx=0.5, dy=0.5
   )

   print(f"Total elements: {architecture.geometry.n_elements}")  # 4096
   print(f"Number of subarrays: {architecture.n_subarrays}")     # 64

   # Compute subarray-level steering
   k = pa.wavelength_to_k(1.0)
   weights = pa.compute_subarray_weights(
       architecture, k,
       theta0_deg=20, phi0_deg=0
   )

ArrayGeometry Class
-------------------

All geometry functions return an ``ArrayGeometry`` dataclass:

.. code-block:: python

   from dataclasses import dataclass

   @dataclass
   class ArrayGeometry:
       x: np.ndarray      # X positions (wavelengths or meters)
       y: np.ndarray      # Y positions
       z: np.ndarray      # Z positions (for 3D arrays)
       nx: np.ndarray     # Normal vector X components
       ny: np.ndarray     # Normal vector Y components
       nz: np.ndarray     # Normal vector Z components
       element_indices: np.ndarray  # Original indices

   # Properties
   geom.n_elements   # Number of elements
   geom.is_planar    # True if all z values are equal
   geom.is_conformal # True if element normals vary

Visualizing Array Geometries
----------------------------

.. code-block:: python

   # 2D plot (matplotlib)
   pa.plot_array_geometry(geom.x, geom.y)

   # 3D interactive plot (plotly)
   fig = pa.plot_array_geometry_3d_plotly(geom)
   fig.show()
