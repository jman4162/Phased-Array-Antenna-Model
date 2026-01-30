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

   import phased_array as pa
   import numpy as np

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

   # At 45 deg scan, expect ~1-2 deg squint at band edges
   print(f"Squint range: {results['squint'].min():.2f} to {results['squint'].max():.2f} deg")

Squint increases with:

- Larger scan angles
- Larger array aperture
- Greater frequency offset from center

Beam Squint Formula
^^^^^^^^^^^^^^^^^^^

Approximate squint for a linear array:

.. math::

   \Delta\theta \approx \frac{\Delta f}{f_0} \tan\theta_0

where :math:`\Delta f` is frequency offset, :math:`f_0` is center frequency,
and :math:`\theta_0` is scan angle.

Instantaneous Bandwidth
-----------------------

The instantaneous bandwidth (IBW) is the frequency range over which the
beam remains within acceptable pointing error (typically 3 dB beamwidth).

.. code-block:: python

   # Analyze IBW for different scan angles
   scan_angles = [0, 15, 30, 45, 60]

   for angle in scan_angles:
       ibw = pa.analyze_instantaneous_bandwidth(
           geom.x, geom.y,
           theta0_deg=angle,
           phi0_deg=0,
           center_frequency=10e9,
           max_squint_deg=2.0,  # Acceptable pointing error
           steering_mode='phase'
       )
       print(f"Scan {angle} deg: IBW = {ibw/1e6:.0f} MHz")

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

   # Compare patterns at band edges
   for f, label in [(9e9, 'Low'), (10e9, 'Center'), (11e9, 'High')]:
       k = pa.frequency_to_k(f)

       # Phase steering (designed at center frequency)
       weights_phase = pa.steering_vector(
           pa.frequency_to_k(10e9), geom.x, geom.y, 45, 0
       )

       # TTD steering
       weights_ttd = pa.steering_vector_ttd(geom.x, geom.y, 45, 0, f)

       # Compute patterns and compare peak directions
       # Phase steering: peak moves with frequency
       # TTD: peak stays at 45 deg

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

   # Compute hybrid steering weights
   weights_hybrid = pa.steering_vector_hybrid(
       architecture,
       theta0_deg=45, phi0_deg=0,
       frequency=10e9
   )

   # Subarrays use TTD, elements within use phase
   results_hybrid = pa.compute_beam_squint(
       architecture.geometry.x, architecture.geometry.y,
       45, 0, 10e9, frequencies,
       steering_mode='hybrid',
       architecture=architecture
   )

   # Squint is reduced but not eliminated
   print(f"Hybrid squint: {np.max(np.abs(results_hybrid['squint'])):.3f} deg")

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

   # patterns['frequencies'] - frequency array
   # patterns['theta_deg'] - angle array
   # patterns['patterns_dB'] - 2D array (n_freq x n_theta)

Visualization
-------------

Plot beam squint:

.. code-block:: python

   # Compare all steering modes
   fig = pa.plot_beam_squint(
       geom.x, geom.y,
       theta0_deg=45, phi0_deg=0,
       center_frequency=10e9,
       bandwidth=2e9,
       steering_modes=['phase', 'ttd', 'hybrid'],
       architecture=architecture
   )
   fig.show()

Plot pattern vs frequency:

.. code-block:: python

   # Waterfall plot of pattern vs frequency
   fig = pa.plot_pattern_vs_frequency_plotly(
       geom.x, geom.y,
       theta0_deg=30, phi0_deg=0,
       center_frequency=10e9,
       bandwidth=2e9,
       n_freqs=11,
       steering_mode='phase'
   )
   fig.show()

Subarray delay visualization:

.. code-block:: python

   fig = pa.plot_subarray_delays(
       architecture,
       theta0_deg=45, phi0_deg=0
   )
   fig.show()

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

For a typical 256-element planar array:

- **Phase-only at 45 deg scan**: IBW ~ 2-5% of center frequency
- **Hybrid (8x8 subarrays)**: IBW ~ 10-20%
- **Full TTD**: IBW ~ 100% (limited by elements, not steering)

Subarray Sizing
^^^^^^^^^^^^^^^

Larger subarrays = fewer TTD units = more squint remaining

Rule of thumb: Subarray aperture should be small enough that internal squint
is less than half the beamwidth:

.. math::

   D_{sub} < \frac{\lambda_0^2 \cdot f_0}{4 \cdot \Delta f \cdot \sin\theta_{max}}
