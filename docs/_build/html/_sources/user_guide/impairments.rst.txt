Impairments
===========

This guide covers realistic impairment models that affect phased array
performance in practice: mutual coupling, phase quantization, element
failures, and scan blindness.

Mutual Coupling
---------------

Electromagnetic coupling between adjacent elements affects both the element
patterns and the input impedance of each element.

Theoretical Model
^^^^^^^^^^^^^^^^^

Approximates coupling based on element spacing using a simple dipole model:

.. code-block:: python

   import phased_array as pa
   import numpy as np

   geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)

   # Create coupling matrix
   coupling_matrix = pa.mutual_coupling_matrix_theoretical(
       geom.x, geom.y,
       coupling_coefficient=0.3,  # Coupling at d=lambda/2
       coupling_exponent=2.0      # Decay rate
   )

   print(f"Matrix shape: {coupling_matrix.shape}")  # (64, 64)
   print(f"Self-coupling: {coupling_matrix[0, 0]:.2f}")  # 1.0
   print(f"Neighbor coupling: {np.abs(coupling_matrix[0, 1]):.2f}")

Measured Coupling
^^^^^^^^^^^^^^^^^

For more accurate modeling, import measured S-parameters:

.. code-block:: python

   # S-parameter data from measurement or EM simulation
   s_params = np.load('measured_s_params.npy')  # (N, N) complex matrix

   coupling_matrix = pa.mutual_coupling_matrix_measured(s_params)

Applying Coupling Effects
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   k = pa.wavelength_to_k(1.0)
   weights_ideal = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)

   # Apply coupling (modifies effective weights)
   weights_coupled = pa.apply_mutual_coupling(weights_ideal, coupling_matrix)

   # Compare patterns
   theta, phi, pattern_ideal = pa.compute_full_pattern(
       geom.x, geom.y, weights_ideal, k
   )
   theta, phi, pattern_coupled = pa.compute_full_pattern(
       geom.x, geom.y, weights_coupled, k
   )

Coupling causes:

- Beam pointing errors
- Increased sidelobes
- Main beam distortion
- Input impedance variations

Active Element Pattern
^^^^^^^^^^^^^^^^^^^^^^

The element pattern when embedded in an array differs from an isolated element:

.. code-block:: python

   # Compute active element pattern including coupling
   theta = np.linspace(0, np.pi/2, 91)
   phi = np.zeros_like(theta)

   aep = pa.active_element_pattern(
       theta, phi, geom, coupling_matrix,
       element_idx=32  # Center element
   )

Phase Quantization
------------------

Digital phase shifters have finite resolution (typically 3-8 bits).
Quantization causes beam pointing errors and increased sidelobes.

Basic Quantization
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Ideal weights
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=20, phi0_deg=0)

   # Quantize to 4-bit (16 levels, 22.5 deg steps)
   weights_q = pa.quantize_phase(weights, n_bits=4)

   # Check quantization levels
   phases_deg = np.rad2deg(np.angle(weights_q))
   unique_phases = np.unique(np.round(phases_deg * 16/360) * 360/16)
   print(f"Number of unique phases: {len(unique_phases)}")

Quantization Error Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # RMS phase error
   for bits in [3, 4, 5, 6]:
       rms_error = pa.quantization_rms_error(bits)
       sll_increase = pa.quantization_sidelobe_increase(bits)
       print(f"{bits}-bit: RMS error = {rms_error:.1f} deg, SLL increase ~ {sll_increase:.1f} dB")

Typical results:

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 35

   * - Bits
     - Levels
     - RMS Error
     - Effect on Pattern
   * - 3
     - 8
     - 13 deg
     - Significant beam errors, high quantization lobes
   * - 4
     - 16
     - 6.5 deg
     - Moderate errors, visible quantization lobes
   * - 5
     - 32
     - 3.3 deg
     - Small errors, acceptable for most applications
   * - 6
     - 64
     - 1.6 deg
     - Minimal impact, near-ideal performance

Full Analysis
^^^^^^^^^^^^^

.. code-block:: python

   results = pa.analyze_quantization_effect(
       weights, geom, k,
       n_bits=4,
       theta_range=(0, np.pi/2),
       n_points=361
   )

   # Plot comparison
   import matplotlib.pyplot as plt
   plt.plot(results['theta_deg'], results['pattern_ideal_dB'], label='Ideal')
   plt.plot(results['theta_deg'], results['pattern_quantized_dB'], label='4-bit')
   plt.xlabel('Theta (deg)')
   plt.ylabel('Pattern (dB)')
   plt.legend()
   plt.grid(True)

Element Failures
----------------

Large arrays can tolerate element failures with graceful degradation.
The library models random failures with different failure modes.

Simulating Failures
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 5% random element failures
   weights_failed, failure_mask = pa.simulate_element_failures(
       weights,
       failure_rate=0.05,
       mode='off',  # Failed elements produce no output
       seed=42
   )

   n_failed = np.sum(failure_mask)
   print(f"Failed elements: {n_failed} / {len(weights)} ({100*n_failed/len(weights):.1f}%)")

Failure Modes
^^^^^^^^^^^^^

- **'off'**: Failed elements have zero output (most common)
- **'stuck'**: Failed elements stuck at random phase, nominal amplitude
- **'full'**: Failed elements at full power, random phase (worst case)

.. code-block:: python

   # Compare failure modes
   for mode in ['off', 'stuck', 'full']:
       weights_f, mask = pa.simulate_element_failures(
           weights, 0.1, mode=mode, seed=42
       )
       theta, phi, pattern = pa.compute_full_pattern(
           geom.x, geom.y, weights_f, k
       )
       # 'full' mode causes highest sidelobe increase

Graceful Degradation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze how pattern degrades with increasing failure rate:

.. code-block:: python

   results = pa.analyze_graceful_degradation(
       geom, k, weights,
       failure_rates=[0.0, 0.02, 0.05, 0.10, 0.20],
       n_trials=10,  # Average over multiple random failures
       seed=42
   )

   # Results include:
   # - Mean sidelobe level vs failure rate
   # - Directivity loss
   # - Beamwidth change

For a 256-element array, typical degradation:

- 5% failures: ~0.5 dB gain loss, 2-3 dB SLL increase
- 10% failures: ~1 dB gain loss, 4-5 dB SLL increase
- 20% failures: ~2 dB gain loss, pattern significantly degraded

Scan Blindness
--------------

At certain scan angles, surface waves can be excited, causing the array
reflection coefficient to approach unity (scan blindness).

Surface Wave Angle
^^^^^^^^^^^^^^^^^^

Estimate the scan angle where blindness occurs:

.. code-block:: python

   blind_angle = pa.surface_wave_scan_angle(
       dx=0.5,  # Element spacing in wavelengths
       substrate_er=2.2,  # Substrate dielectric constant
       thickness=0.05  # Substrate thickness in wavelengths
   )
   print(f"Expected blindness near {blind_angle:.1f} degrees")

Scan Blindness Model
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   theta = np.linspace(0, 90, 181)

   # Get scan loss including blindness
   loss = pa.scan_blindness_model(
       theta_deg=theta,
       dx=0.55,
       substrate_er=3.0,
       thickness=0.03,
       bandwidth=0.1  # Blindness bandwidth in sin(theta)
   )

   # Apply to pattern
   scan_loss = pa.compute_scan_loss(theta, geom, k)

Applying to Patterns
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compute pattern at various scan angles
   for scan_angle in [0, 30, 45, 60]:
       weights = pa.steering_vector(k, geom.x, geom.y, scan_angle, 0)

       # Apply scan blindness model
       weights_blind = pa.apply_scan_blindness(
           weights, geom, k,
           scan_angle_deg=scan_angle,
           substrate_er=2.5,
           blind_angle_deg=65
       )

       # Compare patterns with and without blindness

Combined Impairments
--------------------

In practice, multiple impairments occur simultaneously:

.. code-block:: python

   # Start with ideal steering
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

   # Apply impairments in order
   # 1. Mutual coupling
   coupling = pa.mutual_coupling_matrix_theoretical(geom.x, geom.y)
   weights = pa.apply_mutual_coupling(weights, coupling)

   # 2. Phase quantization
   weights = pa.quantize_phase(weights, n_bits=5)

   # 3. Element failures
   weights, _ = pa.simulate_element_failures(weights, failure_rate=0.03, mode='off')

   # Compute final pattern
   theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)

Best Practices
--------------

1. **Budget for coupling** in beam pointing accuracy requirements.

2. **Use at least 5-bit phase shifters** for most applications.

3. **Design for 5-10% failure tolerance** in critical arrays.

4. **Avoid element spacings** that place scan blindness in operational scan range.

5. **Combine impairment models** for realistic performance prediction.
