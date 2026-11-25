Wideband Beamforming
====================

This guide covers wideband effects in phased arrays and techniques for
mitigation, including true time delay (TTD) and hybrid architectures.

Beam Squint Problem
-------------------

Phase shifters provide a frequency-independent phase shift. When used for
beam steering, the actual time delay varies with frequency, causing the
beam to point at different angles across the bandwidth. This is called
**beam squint**.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import phased_array as pa

   # Array at 10 GHz center frequency
   wavelength = 0.03  # 10 GHz
   geom = pa.create_rectangular_array(32, 32, dx=0.5, dy=0.5, wavelength=wavelength)

   # Analyze beam squint for 20% bandwidth
   center_freq = 10e9
   frequencies = np.linspace(9e9, 11e9, 21)

   results = pa.compute_beam_squint(
       geom.x, geom.y,
       theta0_deg=45, phi0_deg=0,
       center_frequency=center_freq,
       frequencies=frequencies,
       steering_mode='phase'
   )

   # sin(theta) = sin(45 deg) * f0 / f: +6.8 deg at 9 GHz, -5.0 deg at 11 GHz
   print(f"Squint range: {results['squint'].min():.2f} to {results['squint'].max():.2f} deg")

Squint increases with:

- Larger scan angles
- Greater frequency offset from center

The squint angle does not depend on aperture size. A larger aperture has a
narrower beam, so the same squint is a larger fraction of the beamwidth.

Beam Squint Formula
^^^^^^^^^^^^^^^^^^^

Approximate squint for a linear array:

.. math::

   \Delta\theta \approx -\frac{\Delta f}{f_0} \tan\theta_0

where :math:`\Delta f = f - f_0` is the frequency offset, :math:`f_0` is center
frequency, :math:`\theta_0` is scan angle, and :math:`\Delta\theta` is in
radians.

Instantaneous Bandwidth
-----------------------

The instantaneous bandwidth (IBW) is the frequency range over which the
beam remains within an acceptable pointing error, set by
``squint_tolerance_deg`` (a common choice is half the 3 dB beamwidth).

.. code-block:: python

   # Analyze IBW for different scan angles
   scan_angles = [0, 15, 30, 45, 60]

   for angle in scan_angles:
       ibw = pa.analyze_instantaneous_bandwidth(
           geom.x, geom.y,
           theta0_deg=angle,
           phi0_deg=0,
           center_frequency=10e9,
           squint_tolerance_deg=2.0,  # Acceptable pointing error
           steering_mode='phase'
       )
       if 'note' in ibw:
           # Broadside does not squint, so the search reaches its 100% limit
           print(f"Scan {angle} deg: {ibw['note']}")
       else:
           print(f"Scan {angle} deg: IBW = {ibw['ibw_hz']/1e6:.0f} MHz "
                 f"({ibw['ibw_percent']:.1f}%)")

True Time Delay (TTD) Steering
------------------------------

TTD uses actual time delays instead of phase shifts. Since a time delay
produces frequency-proportional phase, the beam points correctly at all
frequencies.

.. code-block:: python

   # TTD steering vector
   weights_ttd = pa.steering_vector_ttd(
       geom.x, geom.y,
       theta0_deg=45, phi0_deg=0,
       frequency=10e9  # Design frequency (any frequency works)
   )

   # Analyze squint (should be nearly zero)
   results_ttd = pa.compute_beam_squint(
       geom.x, geom.y, 45, 0, 10e9, frequencies,
       steering_mode='ttd'
   )
   print(f"TTD squint: {np.max(np.abs(results_ttd['squint'])):.3f} deg")

Time Delay Values
^^^^^^^^^^^^^^^^^

Get the actual time delays required:

.. code-block:: python

   delays = pa.steering_delays_ttd(
       geom.x, geom.y,
       theta0_deg=45, phi0_deg=0
   )

   # Delays in seconds
   print(f"Max delay: {np.max(delays)*1e12:.1f} ps")
   print(f"Delay range: {(np.max(delays) - np.min(delays))*1e12:.1f} ps")

TTD vs Phase Steering Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Phase steering: phase shifters set at the center frequency
   weights_phase = pa.steering_vector(
       pa.frequency_to_k(10e9), geom.x, geom.y, 45, 0
   )

   # Compare peak directions at the band edges
   for f, label in [(9e9, 'Low'), (10e9, 'Center'), (11e9, 'High')]:
       k = pa.frequency_to_k(f)

       # TTD steering: fixed delays, phase evaluated at f
       weights_ttd = pa.steering_vector_ttd(geom.x, geom.y, 45, 0, f)

       theta_deg, cut_phase, _ = pa.compute_pattern_cuts(
           geom.x, geom.y, weights_phase, k,
           theta0_deg=45, phi0_deg=0, n_points=1801
       )
       _, cut_ttd, _ = pa.compute_pattern_cuts(
           geom.x, geom.y, weights_ttd, k,
           theta0_deg=45, phi0_deg=0, n_points=1801
       )

       # Phase steering: peak moves with frequency
       # TTD: peak stays at 45 deg
       print(f"{label} ({f/1e9:.0f} GHz): "
             f"phase peak {theta_deg[np.argmax(cut_phase)]:.1f} deg, "
             f"TTD peak {theta_deg[np.argmax(cut_ttd)]:.1f} deg")

Hybrid Phase/TTD Architecture
-----------------------------

Full TTD per element is expensive. A common compromise uses TTD at the
subarray level and phase shifters within subarrays.

Creating Hybrid Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 64x64 array with 8x8 element subarrays
   architecture = pa.create_rectangular_subarrays(
       Nx_total=64, Ny_total=64,
       Nx_sub=8, Ny_sub=8,
       dx=0.5, dy=0.5,
       wavelength=wavelength
   )

   print(f"Total elements: {architecture.geometry.n_elements}")  # 4096
   print(f"Subarrays: {architecture.n_subarrays}")  # 64 (one TTD each)

Hybrid Steering
^^^^^^^^^^^^^^^

.. code-block:: python

   # Hybrid weights at 11 GHz with the phase shifters set at 10 GHz
   weights_hybrid = pa.steering_vector_hybrid(
       architecture.geometry, architecture,
       theta0_deg=45, phi0_deg=0,
       frequency=11e9,
       center_frequency=10e9
   )

   # Subarrays use TTD, elements within use phase. compute_beam_squint
   # builds these weights at each frequency, phase shifters held at 10 GHz.
   results_hybrid = pa.compute_beam_squint(
       architecture.geometry.x, architecture.geometry.y,
       45, 0, 10e9, frequencies,
       steering_mode='hybrid',
       architecture=architecture
   )

   # Squint is reduced but not eliminated; the band edges also lose gain
   print(f"Hybrid squint: {np.max(np.abs(results_hybrid['squint'])):.3f} deg")
   print(f"Hybrid gain change at band edges: {results_hybrid['relative_gain'][0]:.2f} dB")

Subarray Delay Values
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   subarray_delays = pa.compute_subarray_delays_ttd(
       architecture,
       theta0_deg=45, phi0_deg=0
   )

   print(f"Number of TTD units needed: {len(subarray_delays)}")
   print(f"Max subarray delay: {np.max(subarray_delays)*1e12:.1f} ps")

Pattern vs Frequency
--------------------

Compute and visualize how the pattern changes across the band:

.. code-block:: python

   patterns = pa.compute_pattern_vs_frequency(
       geom.x, geom.y,
       theta0_deg=30, phi0_deg=0,
       center_frequency=10e9,
       frequencies=frequencies,
       steering_mode='phase',
       n_points=361
   )

   # patterns['frequencies'] - frequency array (Hz)
   # patterns['angles'] - theta array (deg)
   # patterns['patterns'] - 2D array (n_freq x n_angles), dB relative to
   #                        the peak at each frequency

Visualization
-------------

Plot beam squint:

.. code-block:: python

   # Compare all steering modes on the 64x64 hybrid array
   squint_data = {}
   for mode in ['phase', 'ttd', 'hybrid']:
       result = pa.compute_beam_squint(
           architecture.geometry.x, architecture.geometry.y,
           theta0_deg=45, phi0_deg=0,
           center_frequency=10e9,
           frequencies=frequencies,
           steering_mode=mode,
           architecture=architecture  # used only by 'hybrid'
       )
       squint_data[mode] = result['squint']

   ax = pa.plot_beam_squint(frequencies, squint_data, center_frequency=10e9)
   plt.show()

Plot pattern vs frequency:

.. code-block:: python

   # Heatmap of the patterns computed above
   fig = pa.plot_pattern_vs_frequency_plotly(
       patterns['angles'],
       patterns['frequencies'],
       patterns['patterns'],
       center_frequency=10e9
   )
   fig.show()

Subarray delay visualization:

.. code-block:: python

   ax = pa.plot_subarray_delays(architecture, subarray_delays)
   plt.show()

Design Guidelines
-----------------

Choosing a Steering Approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Approach
     - Use When
     - Considerations
   * - Phase-only
     - Narrowband, small apertures, cost-sensitive
     - IBW limits operational bandwidth
   * - Full TTD
     - Wideband radar, EW, 5G FR2
     - Expensive, high complexity
   * - Hybrid
     - Moderate bandwidth, large arrays
     - Balance of cost and performance

IBW Requirements
^^^^^^^^^^^^^^^^

For a 256-element (16x16) planar array with a 1 deg squint tolerance:

- **Phase-only at 45 deg scan**: IBW ~ 2-5% of center frequency
- **Hybrid (8x8 subarrays)**: IBW ~ 10-20%
- **Full TTD**: IBW ~ 100% (limited by elements, not steering)

Subarray Sizing
^^^^^^^^^^^^^^^

Larger subarrays = fewer TTD units = more squint remaining

Rule of thumb: Subarray aperture should be small enough that internal squint
is less than half the subarray beamwidth:

.. math::

   D_{sub} < \frac{\lambda_0 \cdot f_0}{2 \cdot \Delta f \cdot \sin\theta_{max}}

where :math:`D_{sub}` is the subarray width in the scan plane and
:math:`\Delta f` is the largest frequency offset from :math:`f_0`.
