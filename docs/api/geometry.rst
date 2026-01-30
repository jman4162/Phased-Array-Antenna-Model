Geometry Module
===============

.. module:: phased_array.geometry
   :synopsis: Array geometry creation and manipulation

The geometry module provides classes and functions for creating various array
geometries, from simple rectangular grids to complex conformal and sparse arrays.

Classes
-------

.. autoclass:: phased_array.ArrayGeometry
   :members: n_elements, is_planar, is_conformal, copy
   :exclude-members: x, y, z, nx, ny, nz, element_indices
   :show-inheritance:

.. autoclass:: phased_array.SubarrayArchitecture
   :members: get_subarray_elements
   :exclude-members: geometry, subarray_assignments, n_subarrays, subarray_centers
   :show-inheritance:

Planar Arrays
-------------

.. autofunction:: phased_array.create_rectangular_array

.. autofunction:: phased_array.create_triangular_array

.. autofunction:: phased_array.create_elliptical_array

Circular and Ring Arrays
------------------------

.. autofunction:: phased_array.create_circular_array

.. autofunction:: phased_array.create_concentric_rings_array

Conformal Arrays
----------------

.. autofunction:: phased_array.create_cylindrical_array

.. autofunction:: phased_array.create_spherical_array

.. autofunction:: phased_array.array_factor_conformal

Sparse/Thinned Arrays
---------------------

.. autofunction:: phased_array.thin_array_random

.. autofunction:: phased_array.thin_array_density_tapered

.. autofunction:: phased_array.thin_array_genetic_algorithm

Subarrays
---------

.. autofunction:: phased_array.create_rectangular_subarrays

.. autofunction:: phased_array.compute_subarray_weights
