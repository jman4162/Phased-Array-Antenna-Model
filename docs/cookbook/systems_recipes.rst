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

           # Compute metrics
           theta_arr, phi_arr, pattern = pa.compute_full_pattern(
               geom.x, geom.y, weights, k
           )

           # Peak gain (relative to broadside)
           peak_gain = pattern.max()

           # Approximate directivity
           theta_deg, E_plane, H_plane = pa.compute_pattern_cuts(
               geom.x, geom.y, weights, k
           )
           hpbw_e = pa.compute_half_power_beamwidth(theta_deg, E_plane)
           hpbw_h = pa.compute_half_power_beamwidth(theta_deg, H_plane)

           results.append({
               'theta': theta, 'phi': phi,
               'gain_dB': peak_gain,
               'hpbw_e': hpbw_e, 'hpbw_h': hpbw_h
           })

   # Analyze scan loss
   broadside_gain = [r['gain_dB'] for r in results if r['theta'] == 0][0]
   for r in results:
       r['scan_loss'] = broadside_gain - r['gain_dB']

   # Report worst case
   worst = max(results, key=lambda x: x['scan_loss'])
   print(f"Worst scan loss: {worst['scan_loss']:.1f} dB at theta={worst['theta']}, phi={worst['phi']}")

Array Failure Analysis
----------------------

**Problem**: Determine minimum operational element count.

.. code-block:: python

   def analyze_failure_threshold(geom, k, weights, min_gain_loss_dB=3.0,
                                  max_sll_increase_dB=6.0):
       """Find failure rate where performance drops below threshold."""

       failure_rates = np.arange(0.0, 0.31, 0.02)

       for rate in failure_rates:
           # Average over multiple trials
           gain_losses = []
           sll_increases = []

           for trial in range(10):
               w_failed, _ = pa.simulate_element_failures(
                   weights, rate, mode='off', seed=trial
               )

               _, E_nom, _ = pa.compute_pattern_cuts(geom.x, geom.y, weights, k)
               _, E_fail, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_failed, k)

               gain_loss = E_nom.max() - E_fail.max()
               sll_nom = np.max(E_nom[90:])  # Outside main beam
               sll_fail = np.max(E_fail[90:])
               sll_increase = sll_fail - sll_nom

               gain_losses.append(gain_loss)
               sll_increases.append(sll_increase)

           mean_gain_loss = np.mean(gain_losses)
           mean_sll_increase = np.mean(sll_increases)

           if mean_gain_loss > min_gain_loss_dB or mean_sll_increase > max_sll_increase_dB:
               return rate - 0.02, rate

       return failure_rates[-1], None

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
