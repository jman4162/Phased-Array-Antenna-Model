Beamforming
===========

This guide covers beamforming techniques including beam steering, amplitude
tapering for sidelobe control, null steering for interference rejection,
and multi-beam synthesis.

Beam Steering
-------------

Basic steering points the main beam toward a desired direction by applying
appropriate phase shifts to each element.

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Create array
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Steer to theta=30 deg, phi=45 deg
   weights = pa.steering_vector(
       k, geom.x, geom.y,
       theta0_deg=30,
       phi0_deg=45
   )

   # All weights have unit magnitude, varying phase
   print(f"Magnitude range: {np.abs(weights).min():.2f} to {np.abs(weights).max():.2f}")

The steering vector applies phase shifts:

.. math::

   w_n = \exp\left(-jk(x_n u_0 + y_n v_0)\right)

where :math:`u_0 = \sin\theta_0\cos\phi_0` and :math:`v_0 = \sin\theta_0\sin\phi_0`.

Amplitude Tapering
------------------

Amplitude tapering (windowing) reduces sidelobe levels at the cost of
increased beamwidth and reduced aperture efficiency.

Taylor Taper
^^^^^^^^^^^^

Most commonly used for radar arrays. Provides specified sidelobe level with
a controlled number of nearly-equal sidelobes before rolloff.

.. code-block:: python

   # -30 dB sidelobes, 4 nearly-equal sidelobes
   taper = pa.taylor_taper_2d(16, 16, sidelobe_dB=-30, nbar=4)

   # Apply to steering weights
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=20, phi0_deg=0)
   weights_tapered = weights * taper

   # Check efficiency loss
   efficiency = pa.compute_taper_efficiency(taper)
   loss_dB = pa.compute_taper_directivity_loss(taper)
   print(f"Efficiency: {efficiency:.2%}, Loss: {loss_dB:.2f} dB")

Chebyshev Taper
^^^^^^^^^^^^^^^

Provides equi-ripple sidelobes (all sidelobes at the same level). Offers the
narrowest beamwidth for a given sidelobe level.

.. code-block:: python

   taper = pa.chebyshev_taper_2d(16, 16, sidelobe_dB=-30)

Comparison of Tapers
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Taper
     - Characteristics
     - Typical Use
     - Efficiency
   * - Uniform
     - Narrowest beam, -13 dB SLL
     - Maximum directivity needed
     - 100%
   * - Taylor
     - Specified SLL, controlled rolloff
     - Radar, communications
     - ~85-95%
   * - Chebyshev
     - Equi-ripple sidelobes
     - Minimum beamwidth for SLL
     - ~80-90%
   * - Hamming
     - Good SLL, simple
     - General purpose
     - ~73%
   * - Gaussian
     - Very low sidelobes, no nulls
     - Low-intercept radar
     - ~70-85%

Example comparing tapers:

.. code-block:: python

   import matplotlib.pyplot as plt

   tapers = {
       'Uniform': np.ones(256),
       'Taylor -30dB': pa.taylor_taper_2d(16, 16, sidelobe_dB=-30),
       'Chebyshev -30dB': pa.chebyshev_taper_2d(16, 16, sidelobe_dB=-30),
       'Hamming': pa.hamming_taper_2d(16, 16),
   }

   for name, taper in tapers.items():
       weights = pa.steering_vector(k, geom.x, geom.y, 0, 0) * taper
       theta_deg, E_plane, _ = pa.compute_pattern_cuts(geom.x, geom.y, weights, k)
       plt.plot(theta_deg, E_plane, label=name)

   plt.xlabel('Theta (deg)')
   plt.ylabel('Pattern (dB)')
   plt.legend()
   plt.grid(True)
   plt.ylim(-60, 0)

Null Steering
-------------

Null steering places pattern nulls in specific directions to reject
interference while maintaining gain in the desired direction.

Projection Method
^^^^^^^^^^^^^^^^^

Projects the desired steering vector onto the null space of interference
directions. Simple and effective for a few nulls.

.. code-block:: python

   # Main beam at 20 deg, nulls at 35 and 50 deg
   null_directions = [(35, 0), (50, 0)]

   weights = pa.null_steering_projection(
       geom, k,
       theta_main_deg=20,
       phi_main_deg=0,
       null_directions=null_directions
   )

   # Verify null depth
   for theta_null, phi_null in null_directions:
       depth = pa.compute_null_depth(geom, k, weights, (theta_null, phi_null))
       print(f"Null at {theta_null} deg: {depth:.1f} dB")

LCMV Beamformer
^^^^^^^^^^^^^^^

Linearly Constrained Minimum Variance - more flexible, allows specifying
response at multiple directions.

.. code-block:: python

   # Constraints: (theta, phi, desired_response)
   constraints = [
       (20, 0, 1.0+0j),   # Unity gain at 20 deg
       (35, 0, 0.0+0j),   # Null at 35 deg
       (50, 0, 0.0+0j),   # Null at 50 deg
   ]

   weights = pa.null_steering_lcmv(
       geom, k,
       constraints=constraints
   )

Multi-Beam Synthesis
--------------------

Generate multiple simultaneous beams for tracking multiple targets or
providing spatial coverage.

Superposition Method
^^^^^^^^^^^^^^^^^^^^

Simple sum of steering vectors. Beams share the available gain.

.. code-block:: python

   # Beams at 15, 30, and 45 degrees
   beam_directions = [(15, 0), (30, 0), (45, 0)]

   weights = pa.multi_beam_weights_superposition(
       geom, k,
       beam_directions
   )

   # Each beam is ~3 dB below single-beam gain

Orthogonal Beams
^^^^^^^^^^^^^^^^

Minimizes inter-beam coupling using orthogonalization.

.. code-block:: python

   weights_list = pa.multi_beam_weights_orthogonal(
       geom, k,
       beam_directions
   )

   # Returns list of weight vectors, one per beam
   # Beams are designed to be orthogonal to each other

   # Check beam isolation
   isolation = pa.compute_beam_isolation(geom, k, weights_list, beam_directions)
   print(f"Beam isolation: {isolation:.1f} dB")

Monopulse Patterns
------------------

Sum and difference patterns for angle tracking.

.. code-block:: python

   weights_sum, weights_diff = pa.monopulse_weights(
       geom, k,
       theta0_deg=20,
       phi0_deg=0,
       plane='azimuth'  # or 'elevation'
   )

   # Sum pattern: conventional beam
   # Difference pattern: null on axis, used for tracking

Applying Tapers to Arbitrary Geometries
---------------------------------------

For non-rectangular arrays, use ``apply_taper_to_geometry``:

.. code-block:: python

   # Create elliptical array
   geom = pa.create_elliptical_array(a=4, b=3, dx=0.5)

   # Apply taper based on position
   weights = pa.apply_taper_to_geometry(
       geom,
       taper_type='taylor',
       sidelobe_dB=-30
   )

Best Practices
--------------

1. **Start with Taylor taper** for most applications - good balance of
   beamwidth and sidelobe control.

2. **Use nbar >= 4** for Taylor tapers to avoid excessive beamwidth increase.

3. **Verify null depths** after null steering - finite array size limits
   achievable null depth.

4. **Consider efficiency loss** when selecting tapers - aggressive sidelobe
   control can cost 2-3 dB of directivity.

5. **For multi-beam**, check isolation between beams when directions are close.
