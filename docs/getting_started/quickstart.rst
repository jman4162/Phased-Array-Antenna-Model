Quickstart
==========

This guide will get you computing phased array patterns in under 5 minutes.

Import the Library
------------------

.. code-block:: python

   import phased_array as pa
   import numpy as np

Create an Array
---------------

Create a 16x16 rectangular array with half-wavelength element spacing:

.. code-block:: python

   geom = pa.create_rectangular_array(
       Nx=16, Ny=16,  # 16x16 elements
       dx=0.5, dy=0.5  # spacing in wavelengths
   )
   print(f"Array has {geom.n_elements} elements")

The ``geom`` object holds the (x, y, z) positions of all elements in meters.
With the default ``wavelength=1.0``, meters and wavelengths coincide, so the
spacing is 0.5 wavelength.

Compute Steering Weights
------------------------

To steer the beam to theta = 30 degrees (measured from broadside):

.. code-block:: python

   # Wavenumber for wavelength = 1 (normalized)
   k = pa.wavelength_to_k(1.0)

   # Steering weights for theta=30 deg, phi=0 deg
   weights = pa.steering_vector(
       k, geom.x, geom.y,
       theta0_deg=30, phi0_deg=0
   )

Apply Amplitude Tapering
------------------------

Apply a Taylor taper to reduce sidelobe levels:

.. code-block:: python

   taper = pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)
   weights = weights * taper

Compute the Pattern
-------------------

Compute the full 3D radiation pattern:

.. code-block:: python

   theta, phi, pattern_dB = pa.compute_full_pattern(
       geom.x, geom.y, weights, k
   )

   print(f"Pattern shape: {pattern_dB.shape}")  # (n_theta, n_phi)
   print(f"Peak: {pattern_dB.max():.1f} dB (normalized to the beam peak)")

   # theta and phi are 1D axes in radians
   i, j = np.unravel_index(np.argmax(pattern_dB), pattern_dB.shape)
   print(f"Beam at theta={np.rad2deg(theta[i]):.1f} deg, phi={np.rad2deg(phi[j]):.1f} deg")

Visualize the Results
---------------------

**Contour Plot**

.. code-block:: python

   pa.plot_pattern_contour(
       np.rad2deg(theta),
       np.rad2deg(phi),
       pattern_dB,
       title="16x16 Array, 30° Scan, Taylor Taper"
   )

**2D Pattern Cut**

.. code-block:: python

   pa.plot_pattern_2d(
       np.rad2deg(theta),
       pattern_dB[:, 0],  # column 0 is phi = 0
       title="E-plane Cut (phi=0°)"
   )

**Interactive 3D Plot (requires Plotly)**

.. code-block:: python

   fig = pa.plot_pattern_3d_plotly(
       theta, phi, pattern_dB,
       title="3D Radiation Pattern"
   )
   fig.show()

Complete Example
----------------

Here's the full script:

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Create array
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)

   # Steering with taper
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

   # Compute and plot
   theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)
   pa.plot_pattern_contour(np.rad2deg(theta), np.rad2deg(phi), pattern_dB)

Next Steps
----------

- Learn about :doc:`concepts` like coordinate systems and pattern multiplication
- Explore different :doc:`../user_guide/geometry` options
- Apply :doc:`../user_guide/beamforming` techniques like null steering
- Model real-world :doc:`../user_guide/impairments` like mutual coupling
