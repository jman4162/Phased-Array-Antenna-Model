Systems Engineer Recipes
========================

Recipes for systems engineers designing phased array architectures.

Subarray Architecture Design
----------------------------

**Problem**: Determine optimal subarray size for cost vs. performance.

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # System requirements
   total_elements_x = 64
   total_elements_y = 64
   max_scan_angle = 60  # degrees
   max_beam_squint = 2  # degrees at band edge

   # Analyze different subarray sizes
   subarray_sizes = [4, 8, 16, 32]

   print("Subarray | Phase Shifters | TTD Units | Est. Squint")
   print("-" * 55)

   for sub_size in subarray_sizes:
       n_subarrays = (total_elements_x // sub_size) ** 2
       n_phase_shifters = total_elements_x * total_elements_y
       n_ttd = n_subarrays

       # Estimate squint (proportional to subarray aperture)
       subarray_aperture = sub_size * 0.5  # wavelengths
       est_squint = 0.1 * subarray_aperture * np.tan(np.deg2rad(max_scan_angle))

       print(f"  {sub_size}x{sub_size}   |     {n_phase_shifters:5d}      |   {n_ttd:4d}    | {est_squint:.1f} deg")

Scan Coverage Analysis
----------------------

**Problem**: Verify pattern performance across the full scan volume.

.. code-block:: python

   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Define scan grid
   theta_scan = np.arange(0, 61, 10)
   phi_scan = np.arange(0, 361, 45)

   # Store metrics for each scan position
   results = []

   for theta in theta_scan:
       for phi in phi_scan:
           weights = pa.steering_vector(k, geom.x, geom.y, theta, phi)
           weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

           # Peak gain relative to broadside, with a cos(theta) element pattern
           scan_loss = pa.compute_scan_loss(geom, weights, k, theta, phi,
                                            element_pattern_func=pa.element_pattern)

           # Beamwidths in two orthogonal cuts through the beam
           theta_deg, E_plane, H_plane = pa.compute_pattern_cuts(
               geom.x, geom.y, weights, k,
               theta0_deg=theta, phi0_deg=phi
           )
           hpbw_e = pa.compute_half_power_beamwidth(theta_deg, E_plane)
           hpbw_h = pa.compute_half_power_beamwidth(theta_deg, H_plane)

           results.append({
               'theta': theta, 'phi': phi,
               'scan_loss_dB': scan_loss,
               'hpbw_e': hpbw_e, 'hpbw_h': hpbw_h
           })

   # Report worst case
   worst = min(results, key=lambda r: r['scan_loss_dB'])
   print(f"Worst scan loss: {worst['scan_loss_dB']:.1f} dB at "
         f"theta={worst['theta']}, phi={worst['phi']}")
   print(f"Beamwidth at that angle: {worst['hpbw_e']:.1f} x {worst['hpbw_h']:.1f} deg")

Array Failure Analysis
----------------------

**Problem**: Determine minimum operational element count.

.. code-block:: python

   def analyze_failure_threshold(geom, k, weights, max_gain_loss_dB=3.0,
                                 max_sll_increase_dB=6.0, n_trials=20):
       """Find the failure rate where performance drops below threshold."""
       failure_rates = np.arange(0.0, 0.31, 0.02)

       r = pa.analyze_graceful_degradation(
           weights, geom, k, list(failure_rates), n_trials=n_trials, seed=0
       )
       too_much = ((r['gain_loss_mean_dB'] > max_gain_loss_dB) |
                   (r['sidelobe_increase_mean_dB'] > max_sll_increase_dB))
       if not too_much.any():
           return failure_rates[-1], None
       i = int(np.argmax(too_much))
       return (failure_rates[i - 1] if i > 0 else 0.0), failure_rates[i]

   weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

   safe_rate, fail_rate = analyze_failure_threshold(geom, k, weights)
   print(f"Safe failure rate: {safe_rate*100:.0f}%")
   print(f"Min operational elements: {int(geom.n_elements * (1-safe_rate))}")

Link Budget Integration
-----------------------

**Problem**: Compute array gain for link budget.

.. code-block:: python

   def compute_array_eirp(geom, weights, k, element_gain_dBi=5.0,
                          pa_power_dBm=20.0, losses_dB=2.0):
       """
       Compute array EIRP for link budget.

       Parameters
       ----------
       element_gain_dBi : float
           Single element gain
       pa_power_dBm : float
           PA output power per element
       losses_dB : float
           Feed network and other losses
       """
       # Array factor gain (coherent combining)
       n_elements = geom.n_elements
       taper_efficiency = pa.compute_taper_efficiency(np.abs(weights))

       # Array gain = N * element_gain * taper_efficiency
       array_gain_dB = (10 * np.log10(n_elements) +
                        element_gain_dBi +
                        10 * np.log10(taper_efficiency))

       # Total power
       total_power_dBm = pa_power_dBm + 10 * np.log10(n_elements)

       # EIRP
       eirp_dBm = total_power_dBm + array_gain_dB - losses_dB

       return {
           'array_gain_dBi': array_gain_dB,
           'total_power_dBm': total_power_dBm,
           'eirp_dBm': eirp_dBm,
           'taper_efficiency': taper_efficiency
       }

   result = compute_array_eirp(geom, weights, k)
   print(f"Array gain: {result['array_gain_dBi']:.1f} dBi")
   print(f"Total power: {result['total_power_dBm']:.1f} dBm")
   print(f"EIRP: {result['eirp_dBm']:.1f} dBm")

Multi-Function Array Scheduling
-------------------------------

**Problem**: Verify beam switching between functions doesn't cause conflicts.

.. code-block:: python

   def compute_beam_transition_time(geom, k, theta1, phi1, theta2, phi2,
                                     max_phase_rate=360e6):  # deg/sec
       """
       Estimate beam transition time based on phase shifter slew rate.
       """
       weights1 = pa.steering_vector(k, geom.x, geom.y, theta1, phi1)
       weights2 = pa.steering_vector(k, geom.x, geom.y, theta2, phi2)

       # Maximum phase change
       phase_diff = np.abs(np.angle(weights2) - np.angle(weights1))
       phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
       max_phase_change = np.rad2deg(np.max(phase_diff))

       transition_time = max_phase_change / max_phase_rate

       return transition_time * 1e6  # microseconds

   # Example: radar search to track transition
   t_us = compute_beam_transition_time(geom, k, 30, 0, 45, 90)
   print(f"Beam transition time: {t_us:.1f} μs")

Interference Analysis
---------------------

**Problem**: Assess vulnerability to jammers at specific angles.

.. code-block:: python

   def analyze_jammer_vulnerability(geom, k, weights, jammer_directions,
                                     jammer_power_dB=40):
       """
       Analyze pattern response to jammers.

       Returns required null depth to achieve 20 dB J/S improvement.
       """
       results = []

       for theta_j, phi_j in jammer_directions:
           # Pattern response at jammer direction
           theta_rad = np.deg2rad(theta_j)
           phi_rad = np.deg2rad(phi_j)
           theta_arr = np.array([[theta_rad]])
           phi_arr = np.array([[phi_rad]])

           AF = pa.array_factor_vectorized(
               theta_arr, phi_arr, geom.x, geom.y, weights, k
           )
           response_dB = 20 * np.log10(np.abs(AF[0, 0]) + 1e-10)

           # Required null depth for 20 dB J/S improvement
           required_null = response_dB - jammer_power_dB - 20

           results.append({
               'direction': (theta_j, phi_j),
               'response_dB': response_dB,
               'required_null_dB': required_null
           })

       return results

   jammer_dirs = [(45, 0), (60, 45), (30, 180)]
   vuln = analyze_jammer_vulnerability(geom, k, weights, jammer_dirs)

   for v in vuln:
       print(f"Jammer at {v['direction']}: Response {v['response_dB']:.1f} dB, "
             f"need {v['required_null_dB']:.1f} dB null")
