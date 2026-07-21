Vector Patterns Module
======================

.. module:: phased_array.vector_patterns
   :synopsis: Polarized (vector) pattern computation for phased arrays

The vector patterns module connects the scalar array-factor engine with
the polarization math: polarized element models produce complex
(E_theta, E_phi) field components, which multiply the array factor to
give full vector patterns, co/cross-polar decompositions, axial-ratio
maps, and polarization-correct conformal array patterns.

A polarized element pattern is any callable
``f(theta, phi, **kwargs) -> (E_theta, E_phi)`` returning the complex
field components in the spherical basis of its evaluation frame, with
boresight along +z.

Pattern Container
-----------------

.. autoclass:: phased_array.VectorPattern
   :members:
   :exclude-members: theta, phi, E_theta, E_phi

Polarized Element Models
------------------------

.. autofunction:: phased_array.dipole_element

.. autofunction:: phased_array.ideal_patch_element

.. autofunction:: phased_array.cos_q_polarized_element

.. autofunction:: phased_array.crossed_dipole_element

.. autoclass:: phased_array.GriddedElementPattern
   :members:

Vector Pattern Computation
--------------------------

.. autofunction:: phased_array.vector_total_pattern

.. autofunction:: phased_array.compute_full_vector_pattern

.. autofunction:: phased_array.compute_co_cross_pattern_cuts

.. autofunction:: phased_array.dual_pol_weights

Conformal Arrays
----------------

.. autofunction:: phased_array.element_rotation_matrices

.. autofunction:: phased_array.global_to_local_angles

.. autofunction:: phased_array.vector_array_factor_conformal
