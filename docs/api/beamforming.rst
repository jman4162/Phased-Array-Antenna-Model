Beamforming Module
==================

.. module:: phased_array.beamforming
   :synopsis: Beamforming techniques including tapering and null steering

The beamforming module provides amplitude tapering functions for sidelobe
control, null steering for interference rejection, and multi-beam synthesis.

Amplitude Tapers (1D)
---------------------

.. autofunction:: phased_array.taylor_taper_1d

.. autofunction:: phased_array.chebyshev_taper_1d

.. autofunction:: phased_array.hamming_taper_1d

.. autofunction:: phased_array.hanning_taper_1d

.. autofunction:: phased_array.cosine_taper_1d

.. autofunction:: phased_array.cosine_on_pedestal_taper_1d

.. autofunction:: phased_array.gaussian_taper_1d

Amplitude Tapers (2D)
---------------------

.. autofunction:: phased_array.taylor_taper_2d

.. autofunction:: phased_array.chebyshev_taper_2d

.. autofunction:: phased_array.hamming_taper_2d

.. autofunction:: phased_array.hanning_taper_2d

.. autofunction:: phased_array.cosine_taper_2d

.. autofunction:: phased_array.cosine_on_pedestal_taper_2d

.. autofunction:: phased_array.gaussian_taper_2d

Taper Analysis
--------------

.. autofunction:: phased_array.compute_taper_efficiency

.. autofunction:: phased_array.compute_taper_directivity_loss

.. autofunction:: phased_array.apply_taper_to_geometry

Null Steering
-------------

.. autofunction:: phased_array.null_steering_projection

.. autofunction:: phased_array.null_steering_lcmv

.. autofunction:: phased_array.compute_null_depth

Multi-Beam Synthesis
--------------------

.. autofunction:: phased_array.multi_beam_weights_superposition

.. autofunction:: phased_array.multi_beam_weights_orthogonal

.. autofunction:: phased_array.compute_beam_isolation

Special Patterns
----------------

.. autofunction:: phased_array.monopulse_weights

Beam Spoiling
-------------

Functions for beam broadening using quadratic phase distributions,
commonly used in search mode to cover larger areas.

.. autofunction:: phased_array.quadratic_phase_spoil

.. autofunction:: phased_array.compute_spoil_factor

.. autofunction:: phased_array.spoiled_beam_gain

.. autofunction:: phased_array.spoiled_beamwidth

Adaptive Beamforming (SMI/GSC)
------------------------------

Adaptive beamforming techniques for interference rejection using
Sample Matrix Inversion (SMI) and Generalized Sidelobe Canceller (GSC).

.. autofunction:: phased_array.adaptive_weights_smi

.. autofunction:: phased_array.adaptive_weights_gsc

.. autofunction:: phased_array.compute_sinr_improvement

.. autofunction:: phased_array.plot_adapted_pattern
