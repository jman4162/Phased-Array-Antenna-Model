User Guide
==========

These guides provide in-depth coverage of the phased array modeling library's
capabilities. Each section includes explanations, practical examples, and
best practices.

.. toctree::
   :maxdepth: 2

   geometry
   beamforming
   impairments
   wideband
   visualization

Overview
--------

The library is organized into modules that follow the typical phased array
design workflow:

1. **Geometry**: Define array element positions and orientations
2. **Beamforming**: Compute steering weights, apply tapers, place nulls
3. **Impairments**: Model real-world effects (coupling, quantization, failures)
4. **Wideband**: Handle frequency-dependent effects and TTD compensation
5. **Visualization**: Plot patterns in 2D, 3D, and UV-space

Typical Workflow
----------------

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # 1. Create array geometry
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)

   # 2. Compute steering weights with sidelobe control
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

   # 3. (Optional) Model impairments
   weights_q = pa.quantize_phase(weights, n_bits=5)

   # 4. Compute pattern
   theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights_q, k)

   # 5. Visualize
   pa.plot_pattern_contour(np.rad2deg(theta), np.rad2deg(phi), pattern_dB)
