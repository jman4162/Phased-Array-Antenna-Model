Polarization Module
===================

.. module:: phased_array.polarization
   :synopsis: Polarization analysis and manipulation for phased arrays

The polarization module provides functions for analyzing and manipulating
antenna polarization states, including Jones vectors, Stokes parameters,
axial ratio calculations, and Ludwig-3 co/cross-pol decomposition.

Jones Vectors
-------------

Jones vectors provide a compact representation of polarization state,
describing the amplitude and phase of orthogonal electric field components.

.. autofunction:: phased_array.jones_vector

Stokes Parameters
-----------------

Stokes parameters provide a complete description of polarization state,
including partially polarized light.

.. autofunction:: phased_array.stokes_parameters

Polarization Ellipse
--------------------

Functions for computing polarization ellipse parameters from Jones vectors.

.. autofunction:: phased_array.axial_ratio

.. autofunction:: phased_array.tilt_angle

Polarization Loss and Discrimination
------------------------------------

Functions for computing polarization mismatch and cross-polarization performance.

.. autofunction:: phased_array.cross_pol_discrimination

.. autofunction:: phased_array.polarization_loss_factor

Ludwig-3 Decomposition
----------------------

The Ludwig-3 definition is the most common convention for separating
antenna patterns into co-polar and cross-polar components.

.. autofunction:: phased_array.ludwig3_decomposition

.. autofunction:: phased_array.co_pol_pattern

.. autofunction:: phased_array.cross_pol_pattern
