Research Recipes
================

Recipes for researchers exploring advanced array concepts.

Custom Optimization Objectives
------------------------------

**Problem**: Optimize array thinning for minimum peak sidelobe.

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Full array to thin
   full_geom = pa.create_rectangular_array(24, 24, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   def peak_sidelobe_objective(geom):
       """Objective function: minimize peak sidelobe level."""
       weights = np.ones(geom.n_elements)
       theta_deg, E_plane, _ = pa.compute_pattern_cuts(
           geom.x, geom.y, weights, k, n_points=721
       )

       # Find main beam region
       peak_idx = np.argmax(E_plane)
       half_bw = 10  # samples to exclude

       # Peak sidelobe outside main beam
       sidelobes = np.concatenate([
           E_plane[:max(0, peak_idx-half_bw)],
           E_plane[min(len(E_plane), peak_idx+half_bw):]
       ])

       return np.max(sidelobes) if len(sidelobes) > 0 else 0

   # Optimize using genetic algorithm
   optimized_geom = pa.thin_array_genetic_algorithm(
       full_geom,
       n_target=300,  # Keep 300 of 576 elements
       objective_func=peak_sidelobe_objective,
       population_size=50,
       n_generations=100,
       mutation_rate=0.1,
       seed=42
   )

   print(f"Original elements: {full_geom.n_elements}")
   print(f"Optimized elements: {optimized_geom.n_elements}")

   # Compare results
   w_full = np.ones(full_geom.n_elements)
   w_opt = np.ones(optimized_geom.n_elements)

   _, E_full, _ = pa.compute_pattern_cuts(full_geom.x, full_geom.y, w_full, k)
   _, E_opt, _ = pa.compute_pattern_cuts(optimized_geom.x, optimized_geom.y, w_opt, k)

   print(f"Full array SLL: {np.max(E_full[100:]):.1f} dB")
   print(f"Optimized SLL: {np.max(E_opt[100:]):.1f} dB")

Adaptive Beamforming Simulation
-------------------------------

**Problem**: Simulate MVDR (Capon) beamformer with interference.

.. code-block:: python

   def mvdr_beamformer(geom, k, theta_desired, phi_desired, interference_dirs,
                       snr_dB=20, inr_dB=30):
       """
       Minimum Variance Distortionless Response beamformer.

       Parameters
       ----------
       interference_dirs : list of (theta, phi) tuples
           Interference directions
       snr_dB : float
           Signal-to-noise ratio
       inr_dB : float
           Interference-to-noise ratio
       """
       n = geom.n_elements

       # Steering vector for desired signal
       a_d = pa.steering_vector(k, geom.x, geom.y, theta_desired, phi_desired)

       # Build interference-plus-noise covariance matrix
       sigma_n = 1.0  # Noise power (normalized)
       sigma_s = sigma_n * 10 ** (snr_dB / 10)
       sigma_i = sigma_n * 10 ** (inr_dB / 10)

       R = sigma_n * np.eye(n, dtype=complex)

       for theta_i, phi_i in interference_dirs:
           a_i = pa.steering_vector(k, geom.x, geom.y, theta_i, phi_i)
           R += sigma_i * np.outer(a_i, a_i.conj())

       # MVDR weights: w = R^-1 * a / (a^H * R^-1 * a)
       R_inv = np.linalg.inv(R)
       w = R_inv @ a_d
       w = w / (a_d.conj() @ R_inv @ a_d)

       return w

   # Example
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Signal at 20 deg, interference at 45 and 60 deg
   w_mvdr = mvdr_beamformer(
       geom, k,
       theta_desired=20, phi_desired=0,
       interference_dirs=[(45, 0), (60, 0)],
       snr_dB=20, inr_dB=40
   )

   # Compare with conventional
   w_conv = pa.steering_vector(k, geom.x, geom.y, 20, 0)

   theta_deg, E_conv, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_conv, k)
   _, E_mvdr, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_mvdr, k)

   # MVDR will have deep nulls at interference locations

Custom Array Geometries
-----------------------

**Problem**: Create a non-standard array geometry.

.. code-block:: python

   from phased_array.geometry import ArrayGeometry

   def create_log_periodic_array(n_elements, min_spacing, ratio):
       """
       Create a log-periodic linear array.

       Spacing increases geometrically: d_n = min_spacing * ratio^n
       """
       x = [0]
       for i in range(n_elements - 1):
           spacing = min_spacing * (ratio ** i)
           x.append(x[-1] + spacing)

       x = np.array(x)
       x -= x.mean()  # Center at origin

       return ArrayGeometry(
           x=x,
           y=np.zeros_like(x),
           z=np.zeros_like(x),
           nx=np.zeros_like(x),
           ny=np.zeros_like(x),
           nz=np.ones_like(x),
           element_indices=np.arange(len(x))
       )

   # Create and test
   log_geom = create_log_periodic_array(20, min_spacing=0.3, ratio=1.1)
   print(f"Elements: {log_geom.n_elements}")
   print(f"Total aperture: {log_geom.x.max() - log_geom.x.min():.2f} wavelengths")

Mutual Coupling Compensation
----------------------------

**Problem**: Pre-compensate weights for known mutual coupling.

.. code-block:: python

   def compensate_mutual_coupling(weights, coupling_matrix):
       """
       Pre-distort weights to compensate for mutual coupling.

       The coupled weights are: w_coupled = C @ w
       We want: C @ w_predistorted = w_desired
       So: w_predistorted = C^-1 @ w_desired
       """
       C_inv = np.linalg.inv(coupling_matrix)
       return C_inv @ weights

   # Create coupling matrix
   geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
   coupling = pa.mutual_coupling_matrix_theoretical(
       geom.x, geom.y,
       coupling_coefficient=0.3,
       coupling_exponent=2.0
   )

   # Desired steering
   k = pa.wavelength_to_k(1.0)
   w_desired = pa.steering_vector(k, geom.x, geom.y, 30, 0)
   w_desired *= pa.taylor_taper_2d(8, 8, sidelobe_dB=-30)

   # Compensated weights
   w_comp = compensate_mutual_coupling(w_desired, coupling)

   # Verify: coupling @ w_comp should give pattern close to desired
   w_result = coupling @ w_comp

   _, E_desired, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_desired, k)
   _, E_result, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_result, k)

   # E_result should be close to E_desired

Statistical Analysis with Monte Carlo
-------------------------------------

**Problem**: Characterize performance statistics with manufacturing tolerances.

.. code-block:: python

   def monte_carlo_analysis(geom, k, weights, n_trials=1000,
                            pos_error_std=0.01,  # wavelengths
                            amp_error_std=0.5,   # dB
                            phase_error_std=3.0  # degrees
                            ):
       """
       Monte Carlo analysis of array performance with random errors.
       """
       results = {
           'sll': [],
           'hpbw': [],
           'beam_pointing': [],
           'directivity_loss': []
       }

       for _ in range(n_trials):
           # Position errors
           x_err = geom.x + pos_error_std * np.random.randn(geom.n_elements)
           y_err = geom.y + pos_error_std * np.random.randn(geom.n_elements)

           # Amplitude and phase errors
           amp_err = 10 ** (amp_error_std * np.random.randn(geom.n_elements) / 20)
           phase_err = np.deg2rad(phase_error_std * np.random.randn(geom.n_elements))
           w_err = weights * amp_err * np.exp(1j * phase_err)

           # Compute pattern
           theta_deg, E_plane, _ = pa.compute_pattern_cuts(
               x_err, y_err, w_err, k, n_points=721
           )

           # Extract metrics
           peak_idx = np.argmax(E_plane)
           results['beam_pointing'].append(theta_deg[peak_idx])
           results['hpbw'].append(pa.compute_half_power_beamwidth(theta_deg, E_plane))

           sidelobes = E_plane[np.abs(theta_deg) > 15]
           results['sll'].append(np.max(sidelobes))

       return {k: np.array(v) for k, v in results.items()}

   # Run analysis
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, 20, 0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-35)

   stats = monte_carlo_analysis(geom, k, weights, n_trials=500)

   print("Performance Statistics (500 trials)")
   print("-" * 40)
   print(f"SLL: {np.mean(stats['sll']):.1f} dB (std: {np.std(stats['sll']):.1f})")
   print(f"HPBW: {np.mean(stats['hpbw']):.2f} deg (std: {np.std(stats['hpbw']):.2f})")
   print(f"Pointing: {np.mean(stats['beam_pointing']):.2f} deg (std: {np.std(stats['beam_pointing']):.2f})")

Frequency-Selective Surface Integration
---------------------------------------

**Problem**: Model pattern with frequency-selective surface (FSS) radome.

.. code-block:: python

   def apply_fss_radome(pattern_dB, theta, fss_transmission_dB_func):
       """
       Apply FSS radome transmission function to pattern.

       Parameters
       ----------
       fss_transmission_dB_func : callable
           Function(theta_deg) -> transmission in dB
       """
       theta_deg = np.rad2deg(theta)
       transmission = fss_transmission_dB_func(theta_deg)
       return pattern_dB + transmission

   # Example FSS model: bandpass with angle-dependent cutoff
   def fss_model(theta_deg):
       """Simple FSS model: transmission degrades with scan angle."""
       # Normal incidence: 0 dB loss
       # Grazing incidence: -10 dB loss
       return -0.002 * theta_deg ** 2

   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, 45, 0)

   theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)

   # Apply FSS
   for i, t in enumerate(theta):
       pattern_dB[i, :] = apply_fss_radome(
           pattern_dB[i, :], np.array([t] * len(phi)), fss_model
       )
