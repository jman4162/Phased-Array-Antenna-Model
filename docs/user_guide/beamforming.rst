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

Beam Spoiling
-------------

Beam spoiling broadens the beam by introducing quadratic phase across the
aperture. This is commonly used in search/surveillance modes to cover larger
areas with reduced update rate.

Quadratic Phase Spoiling
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import phased_array as pa
   import numpy as np

   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Create spoiled beam with spoil_factor=2.0
   weights = pa.quadratic_phase_spoil(
       geom, k,
       theta0_deg=0, phi0_deg=0,
       spoil_factor=2.0,  # Higher = broader beam
       axis='both'  # Spoil in both x and y
   )

   # Spoil only in one axis (fan beam)
   weights_fan = pa.quadratic_phase_spoil(
       geom, k,
       theta0_deg=0, phi0_deg=0,
       spoil_factor=3.0,
       axis='x'  # Only spoil in x-direction
   )

Computing Required Spoil Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Calculate spoil factor needed for desired beamwidth
   unspoiled_bw = 6.0  # degrees (natural beamwidth)
   desired_bw = 15.0   # degrees (target beamwidth)

   spoil_factor = pa.compute_spoil_factor(
       geom,
       desired_beamwidth_deg=desired_bw,
       unspoiled_beamwidth_deg=unspoiled_bw
   )
   print(f"Required spoil factor: {spoil_factor:.2f}")

Spoiled Beam Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Estimate spoiled beamwidth
   bw_spoiled = pa.spoiled_beamwidth(6.0, spoil_factor=2.0)
   print(f"Spoiled beamwidth: {bw_spoiled:.1f} deg")  # ~13.4 deg

   # Estimate spoiled beam gain
   gain_spoiled = pa.spoiled_beam_gain(
       n_elements=256,
       element_gain_dBi=5.0,
       spoil_factor=2.0,
       taper_efficiency=0.9
   )
   print(f"Spoiled beam gain: {gain_spoiled:.1f} dBi")

**Spoiling effects:**

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Spoil Factor
     - BW Multiplier
     - Gain Loss (dB)
     - Typical Use
   * - 0
     - 1.0x
     - 0.0
     - Normal operation
   * - 1
     - 1.4x
     - 3.0
     - Moderate broadening
   * - 2
     - 2.2x
     - 7.0
     - Search mode
   * - 3
     - 3.2x
     - 10.0
     - Wide area surveillance

Adaptive Beamforming
--------------------

Adaptive beamforming automatically adjusts weights to suppress interference
while maintaining gain toward the desired signal direction.

Sample Matrix Inversion (SMI/MVDR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SMI directly computes the optimal MVDR weights from sample covariance:

.. code-block:: python

   import phased_array as pa
   import numpy as np

   geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Simulate interference data (in practice, from receiver)
   n_snapshots = 100
   n_elements = geom.n_elements

   # Create interference scenario: jammer at 35 deg + noise
   jammer_sv = pa.steering_vector(k, geom.x, geom.y, 35, 0)
   noise = (np.random.randn(n_snapshots, n_elements) +
            1j * np.random.randn(n_snapshots, n_elements)) / np.sqrt(2)
   jammer = 10 * np.outer(np.random.randn(n_snapshots) +
                          1j * np.random.randn(n_snapshots), jammer_sv)
   interference_data = jammer + noise

   # Compute adaptive weights for signal at 0 deg
   weights_adapted = pa.adaptive_weights_smi(
       geom, k,
       theta_desired_deg=0,
       phi_desired_deg=0,
       interference_data=interference_data,
       diagonal_loading=0.01  # Improves robustness
   )

   # Compare with quiescent (non-adaptive) weights
   weights_quiescent = pa.steering_vector(k, geom.x, geom.y, 0, 0)

**Diagonal loading** adds robustness when:

- Number of snapshots is limited
- Signal of interest is present in training data
- Mismatch exists between assumed and actual steering vectors

Generalized Sidelobe Canceller (GSC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GSC provides a constrained adaptive structure that guarantees distortionless
response in the look direction:

.. code-block:: python

   # Compute GSC weights
   weights_gsc, blocking_matrix = pa.adaptive_weights_gsc(
       geom, k,
       theta_desired_deg=0,
       phi_desired_deg=0,
       interference_data=interference_data,
       n_blocking_vectors=None,  # Auto: n_elements - 1
       mu=0.01  # LMS step size
   )

   # GSC structure:
   # w = w_quiescent - B @ w_adaptive
   # where B is the blocking matrix orthogonal to desired steering vector

SINR Analysis
^^^^^^^^^^^^^

Quantify the improvement from adaptive beamforming:

.. code-block:: python

   # Compute SINR improvement
   sinr_before, sinr_after, improvement = pa.compute_sinr_improvement(
       weights_before=weights_quiescent,
       weights_after=weights_adapted,
       geometry=geom,
       k=k,
       signal_direction=(0, 0),
       interference_directions=[(35, 0)],
       signal_power=1.0,
       interference_powers=[100.0],  # 20 dB INR
       noise_power=0.1
   )

   print(f"SINR before: {sinr_before:.1f} dB")
   print(f"SINR after:  {sinr_after:.1f} dB")
   print(f"Improvement: {improvement:.1f} dB")

Visualizing Adapted Patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Plot comparison of quiescent and adapted patterns
   ax = pa.plot_adapted_pattern(
       geom, k,
       weights_quiescent=weights_quiescent,
       weights_adapted=weights_adapted,
       interference_directions=[(35, 0)],
       title="Adaptive Null at 35 degrees",
       phi_cut_deg=0.0
   )

**Adaptive beamforming guidelines:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Parameter
     - Typical Range
     - Effect
   * - Snapshots
     - 2N to 10N
     - More = better estimate, slower adaptation
   * - Diagonal loading
     - 0.001 to 0.1
     - Higher = more robust, less cancellation
   * - LMS step size (mu)
     - 0.001 to 0.1
     - Higher = faster adaptation, risk of instability

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

6. **For beam spoiling**, ensure gain budget accounts for the spoiling loss.

7. **For adaptive beamforming**, use diagonal loading when snapshots are limited
   or signal is present in training data.
