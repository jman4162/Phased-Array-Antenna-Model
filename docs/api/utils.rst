Utilities Module
================

.. module:: phased_array.utils
   :synopsis: Utility functions for coordinate transforms and conversions

The utils module provides helper functions for coordinate system conversions,
unit transformations, and grid generation.

Coordinate Transforms
---------------------

.. autofunction:: phased_array.theta_phi_to_uv

.. autofunction:: phased_array.uv_to_theta_phi

.. autofunction:: phased_array.azel_to_thetaphi

.. autofunction:: phased_array.thetaphi_to_azel

.. autofunction:: phased_array.is_visible_region

Angle Conversions
-----------------

.. autofunction:: phased_array.deg2rad

.. autofunction:: phased_array.rad2deg

Wavenumber and Wavelength
-------------------------

.. autofunction:: phased_array.wavelength_to_k

.. autofunction:: phased_array.frequency_to_wavelength

.. autofunction:: phased_array.frequency_to_k

Decibel Conversions
-------------------

.. autofunction:: phased_array.db_to_linear

.. autofunction:: phased_array.linear_to_db

.. autofunction:: phased_array.normalize_pattern

Grid Generation
---------------

.. autofunction:: phased_array.create_theta_phi_grid

.. autofunction:: phased_array.create_uv_grid
