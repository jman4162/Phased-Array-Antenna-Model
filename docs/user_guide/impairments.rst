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

Active Impedance and VSWR
-------------------------

In a phased array, mutual coupling causes each element to see a different
impedance depending on the scan angle and the excitations of neighboring
elements. This "active impedance" can vary significantly from the isolated
element impedance.

Active Reflection Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The active reflection coefficient accounts for coupling from all other
elements when the array is excited with given weights:

.. code-block:: python

   import phased_array as pa
   import numpy as np

   geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Create coupling matrix
   C = pa.mutual_coupling_matrix_theoretical(geom, k, coupling_coeff=0.2)

   # Compute steering weights
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)

   # Active reflection coefficient for center element
   gamma = pa.active_reflection_coefficient(C, weights, element_idx=32)
   print(f"Active reflection coeff: {np.abs(gamma):.3f} at {np.rad2deg(np.angle(gamma)):.1f} deg")

Active Impedance
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Active impedance seen at element port
   Z_active = pa.active_impedance(C, weights, element_idx=32, Z0=50.0)
   print(f"Active impedance: {Z_active.real:.1f} + j{Z_active.imag:.1f} ohms")

   # Compare with nominal 50 ohms
   # Active impedance varies with scan angle and position in array

   # Get active impedance for all elements at a scan angle
   Z_all = pa.active_scan_impedance_matrix(
       geom, C, k,
       theta_deg=30, phi_deg=0,
       Z0=50.0
   )
   print(f"Impedance range: {Z_all.real.min():.1f} to {Z_all.real.max():.1f} ohms (real)")

VSWR vs Scan Angle
^^^^^^^^^^^^^^^^^^

VSWR is a critical metric that indicates how well elements remain matched
as the array scans. High VSWR indicates potential scan blindness or poor
matching conditions.

.. code-block:: python

   # Compute VSWR for all elements vs scan angle
   theta_deg, vswr_all, vswr_max = pa.vswr_vs_scan(
       geom, C, k,
       theta_range=(0, 60),
       n_angles=31,
       phi_deg=0.0
   )

   # Find maximum VSWR at each scan angle
   for i, theta in enumerate(theta_deg[::5]):  # Every 5th angle
       print(f"Scan {theta:.0f} deg: Max VSWR = {vswr_max[i*5]:.2f}:1")

   # Plot VSWR vs scan angle
   import matplotlib.pyplot as plt
   plt.plot(theta_deg, vswr_max)
   plt.xlabel('Scan Angle (deg)')
   plt.ylabel('Maximum VSWR')
   plt.title('VSWR vs Scan Angle')
   plt.grid(True)
   plt.axhline(y=2.0, color='r', linestyle='--', label='2:1 VSWR spec')
   plt.legend()

**VSWR guidelines:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - VSWR
     - Reflection Loss
     - Interpretation
   * - 1.0:1
     - 0.0 dB
     - Perfect match
   * - 1.5:1
     - 0.2 dB
     - Excellent
   * - 2.0:1
     - 0.5 dB
     - Good (typical spec)
   * - 3.0:1
     - 1.2 dB
     - Marginal
   * - >5:1
     - >2.5 dB
     - Potential scan blindness

Mismatch Loss
^^^^^^^^^^^^^

.. code-block:: python

   # Compute mismatch loss from reflection coefficient
   gamma = 0.333  # Corresponds to 2:1 VSWR
   loss = pa.mismatch_loss(gamma)
   print(f"Mismatch loss at 2:1 VSWR: {loss:.2f} dB")  # ~-0.5 dB

   # Array of reflection coefficients
   gamma_array = np.array([0.0, 0.1, 0.2, 0.333, 0.5])
   loss_array = pa.mismatch_loss(gamma_array)
   for g, l in zip(gamma_array, loss_array):
       vswr = (1 + g) / (1 - g)
       print(f"Gamma={g:.2f}, VSWR={vswr:.1f}:1, Loss={l:.2f} dB")

Edge Effects
^^^^^^^^^^^^

Elements near the array edges typically have different active impedance than
interior elements due to the asymmetric coupling environment:

.. code-block:: python

   # Compare edge and center element active impedance
   Z_center = pa.active_impedance(C, weights, element_idx=27, Z0=50.0)  # Center
   Z_corner = pa.active_impedance(C, weights, element_idx=0, Z0=50.0)   # Corner
   Z_edge = pa.active_impedance(C, weights, element_idx=3, Z0=50.0)     # Edge

   print(f"Center element Z: {Z_center.real:.1f} + j{Z_center.imag:.1f}")
   print(f"Edge element Z: {Z_edge.real:.1f} + j{Z_edge.imag:.1f}")
   print(f"Corner element Z: {Z_corner.real:.1f} + j{Z_corner.imag:.1f}")

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

6. **Check VSWR across the full scan range** to identify potential blind spots.

7. **Account for edge effects** when specifying element matching requirements.
