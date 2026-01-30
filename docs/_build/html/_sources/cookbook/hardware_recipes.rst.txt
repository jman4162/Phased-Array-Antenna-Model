Hardware Engineer Recipes
=========================

Practical recipes for hardware engineers designing phased array systems.

Selecting Phase Shifter Resolution
----------------------------------

**Problem**: Determine the minimum phase shifter bits needed for your
sidelobe requirements.

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Your array configuration
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Target sidelobe level
   target_sll_dB = -30

   # Design weights with taper
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=target_sll_dB)

   # Test different bit depths
   print("Bits | RMS Error | Achieved SLL")
   print("-" * 35)

   for bits in range(3, 8):
       weights_q = pa.quantize_phase(weights, n_bits=bits)
       _, E_plane, _ = pa.compute_pattern_cuts(geom.x, geom.y, weights_q, k)

       # Find peak sidelobe (exclude main beam region)
       peak_idx = np.argmax(E_plane)
       sidelobes = np.concatenate([E_plane[:peak_idx-10], E_plane[peak_idx+10:]])
       achieved_sll = np.max(sidelobes) if len(sidelobes) > 0 else -99

       rms = pa.quantization_rms_error(bits)
       print(f"  {bits}  |  {rms:5.1f} deg | {achieved_sll:6.1f} dB")

**Rule of thumb**: For -30 dB sidelobes, use at least 5-bit phase shifters.
For -40 dB, use 6-bit.

Element Spacing vs Scan Range
-----------------------------

**Problem**: Determine maximum element spacing for your scan requirements.

.. code-block:: python

   import matplotlib.pyplot as plt

   def max_spacing_for_scan(theta_max_deg):
       """Maximum spacing to avoid grating lobes."""
       theta_max = np.deg2rad(theta_max_deg)
       return 1.0 / (1.0 + np.sin(theta_max))

   scan_angles = np.arange(0, 91, 5)
   max_spacings = [max_spacing_for_scan(a) for a in scan_angles]

   plt.figure(figsize=(8, 5))
   plt.plot(scan_angles, max_spacings, 'b-', linewidth=2)
   plt.axhline(y=0.5, color='r', linestyle='--', label='λ/2 spacing')
   plt.xlabel('Maximum Scan Angle (deg)')
   plt.ylabel('Maximum Element Spacing (λ)')
   plt.title('Element Spacing vs. Grating Lobe-Free Scan Range')
   plt.grid(True)
   plt.legend()
   plt.xlim(0, 90)
   plt.ylim(0.4, 1.0)

**Key values**:

- λ/2 spacing: scan to ±90° (full hemisphere)
- 0.6λ spacing: scan to ±56°
- 0.7λ spacing: scan to ±46°

Verifying Grating Lobe Locations
--------------------------------

**Problem**: Check if grating lobes appear in the visible region.

.. code-block:: python

   # Array with larger spacing (potential grating lobes)
   dx = 0.7  # Larger than λ/2
   geom = pa.create_rectangular_array(16, 16, dx=dx, dy=dx)
   k = pa.wavelength_to_k(1.0)

   # Scan to 30 degrees
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)

   # Compute UV-space pattern
   u, v, pattern_uv = pa.compute_pattern_uv_space(
       geom.x, geom.y, weights, k,
       u_range=(-1.5, 1.5), v_range=(-1.5, 1.5)  # Include invisible region
   )

   # Plot with visible region marked
   pa.plot_pattern_uv_space(u, v, pattern_uv, show_visible_region=True)

   # Grating lobe locations (for 1D):
   # u_gl = u_0 + n*λ/d where n = ±1, ±2, ...
   u0 = np.sin(np.deg2rad(30))
   print(f"Main beam: u = {u0:.3f}")
   print(f"First grating lobe: u = {u0 + 1/dx:.3f}")

Estimating TTD Requirements
---------------------------

**Problem**: Calculate true-time delay values for your array.

.. code-block:: python

   # Physical array
   frequency = 10e9  # 10 GHz
   wavelength = 3e8 / frequency
   geom = pa.create_rectangular_array(32, 32, dx=0.5, dy=0.5, wavelength=wavelength)

   # Get required delays for 45 degree scan
   delays = pa.steering_delays_ttd(
       geom.x, geom.y,
       theta0_deg=45, phi0_deg=0
   )

   # Statistics
   delay_range = np.max(delays) - np.min(delays)
   print(f"Aperture size: {np.max(geom.x) - np.min(geom.x):.3f} m")
   print(f"Maximum delay: {np.max(delays)*1e12:.1f} ps")
   print(f"Delay range: {delay_range*1e12:.1f} ps")
   print(f"Delay bits needed (10 ps LSB): {int(np.ceil(np.log2(delay_range/10e-12)))}")

Thermal Effects on Beam Pointing
--------------------------------

**Problem**: Estimate beam pointing error from phase shifter temperature drift.

.. code-block:: python

   # Phase shifter temperature coefficient (typical: 0.1-0.5 deg/°C)
   temp_coeff = 0.2  # deg per °C

   # Temperature gradient across array
   temp_gradient = 5  # °C edge-to-edge

   # Create phase errors (linear gradient)
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Ideal weights
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=20, phi0_deg=0)

   # Add temperature-induced phase errors
   x_norm = (geom.x - geom.x.min()) / (geom.x.max() - geom.x.min())
   phase_error = np.deg2rad(temp_coeff * temp_gradient * x_norm)
   weights_thermal = weights * np.exp(1j * phase_error)

   # Compare beam directions
   _, E_ideal, _ = pa.compute_pattern_cuts(geom.x, geom.y, weights, k)
   _, E_thermal, _ = pa.compute_pattern_cuts(geom.x, geom.y, weights_thermal, k)

   # Find peak locations
   theta_deg = np.linspace(-90, 90, 361)
   peak_ideal = theta_deg[np.argmax(E_ideal)]
   peak_thermal = theta_deg[np.argmax(E_thermal)]
   print(f"Beam pointing error: {peak_thermal - peak_ideal:.3f} deg")

Power Amplifier Saturation Effects
----------------------------------

**Problem**: Model amplitude compression in power amplifiers.

.. code-block:: python

   def apply_pa_compression(weights, p1dB_backoff=3.0):
       """
       Model PA compression using soft limiter.

       Parameters
       ----------
       weights : ndarray
           Complex weights
       p1dB_backoff : float
           Backoff from P1dB in dB
       """
       amplitude = np.abs(weights)
       phase = np.angle(weights)

       # Normalize to peak
       amp_norm = amplitude / np.max(amplitude)

       # Soft compression (approximate Rapp model)
       p = 2  # Smoothness factor
       a_sat = 10 ** (-p1dB_backoff / 20)
       amp_compressed = amp_norm / (1 + (amp_norm / a_sat) ** (2*p)) ** (1/(2*p))

       return amp_compressed * np.max(amplitude) * np.exp(1j * phase)

   # Test effect on tapered weights
   weights = pa.steering_vector(k, geom.x, geom.y, 0, 0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

   weights_compressed = apply_pa_compression(weights, p1dB_backoff=2.0)

   # Compression reduces taper effectiveness (higher sidelobes)

Calibration Error Budget
------------------------

**Problem**: Understand how calibration errors affect pattern performance.

.. code-block:: python

   def add_calibration_errors(weights, amp_error_dB, phase_error_deg, seed=None):
       """Add random amplitude and phase calibration errors."""
       if seed is not None:
           np.random.seed(seed)

       n = len(weights)

       # Random errors (Gaussian)
       amp_error = 10 ** (amp_error_dB * np.random.randn(n) / 20)
       phase_error = np.deg2rad(phase_error_deg * np.random.randn(n))

       return weights * amp_error * np.exp(1j * phase_error)

   # Monte Carlo analysis
   n_trials = 100
   sll_results = []

   for _ in range(n_trials):
       w_err = add_calibration_errors(
           weights,
           amp_error_dB=0.5,   # ±0.5 dB amplitude (1-sigma)
           phase_error_deg=3.0  # ±3 deg phase (1-sigma)
       )
       _, E_plane, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_err, k)
       peak_idx = np.argmax(E_plane)
       sll = np.max(E_plane[peak_idx+10:])
       sll_results.append(sll)

   print(f"Mean SLL: {np.mean(sll_results):.1f} dB")
   print(f"Worst SLL (95%): {np.percentile(sll_results, 95):.1f} dB")
