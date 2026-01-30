Wideband Module
===============

.. module:: phased_array.wideband
   :synopsis: Wideband beamforming with true time delay

The wideband module provides functions for analyzing and compensating beam
squint effects in wideband phased arrays using true time delay (TTD) and
hybrid phase/TTD architectures.

Beam Squint Analysis
--------------------

.. autofunction:: phased_array.compute_beam_squint

.. autofunction:: phased_array.analyze_instantaneous_bandwidth

.. autofunction:: phased_array.compute_pattern_vs_frequency

True Time Delay Steering
------------------------

.. autofunction:: phased_array.steering_vector_ttd

.. autofunction:: phased_array.steering_delays_ttd

Hybrid Phase/TTD Steering
-------------------------

.. autofunction:: phased_array.steering_vector_hybrid

.. autofunction:: phased_array.compute_subarray_delays_ttd

.. autofunction:: phased_array.compute_subarray_weights_hybrid

Comparison
----------

.. autofunction:: phased_array.compare_steering_modes
