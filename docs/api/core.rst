Core Module
===========

.. module:: phased_array.core
   :synopsis: Core computation functions for phased array patterns

The core module provides fundamental functions for computing phased array
radiation patterns, including steering vectors, array factors, and pattern cuts.

Steering and Array Factor
-------------------------

.. autofunction:: phased_array.steering_vector

.. autofunction:: phased_array.array_factor_vectorized

.. autofunction:: phased_array.array_factor_uv

.. autofunction:: phased_array.array_factor_fft

Element Patterns
----------------

.. autofunction:: phased_array.element_pattern

.. autofunction:: phased_array.element_pattern_cosine_tapered

Pattern Computation
-------------------

.. autofunction:: phased_array.compute_full_pattern

.. autofunction:: phased_array.compute_pattern_cuts

.. autofunction:: phased_array.total_pattern

Pattern Analysis
----------------

.. autofunction:: phased_array.compute_directivity

.. autofunction:: phased_array.compute_half_power_beamwidth
