Troubleshooting & FAQ
=====================

Common issues and their solutions.

Installation Issues
-------------------

**ImportError: No module named 'phased_array'**

Ensure the package is installed in your active environment:

.. code-block:: bash

   pip install phased-array-modeling

Verify installation:

.. code-block:: bash

   pip show phased-array-modeling

**Plotly plots not displaying in Jupyter**

Install the plotting dependencies:

.. code-block:: bash

   pip install phased-array-modeling[plotting]

For JupyterLab, you may need:

.. code-block:: bash

   pip install jupyterlab "ipywidgets>=7.6"

**ImportError: cannot import name 'X' from 'phased_array'**

This usually means you have an outdated version. Upgrade:

.. code-block:: bash

   pip install --upgrade phased-array-modeling

Pattern Computation Issues
--------------------------

The Python examples on this page assume this setup: a 16 x 16 array at
half-wavelength spacing, steered to theta = 30 deg with a -30 dB Taylor taper.

.. code-block:: python

   import numpy as np
   import phased_array as pa

   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)
   theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)

**Pattern has unexpected spikes or artifacts**

- Check that positions (meters) and ``k`` come from the same wavelength
  (see :doc:`user_guide/conventions`)
- Verify weights array has the correct length
- Ensure theta/phi arrays have compatible shapes

.. code-block:: python

   # Verify dimensions
   print(f"Elements: {geom.n_elements}")
   print(f"Weights: {len(weights)}")
   assert len(weights) == geom.n_elements

**Pattern peak is not at expected scan angle**

- Steering angles (``theta0_deg``, ``phi0_deg``) are in degrees; pattern
  grids are in radians
- Build the geometry and ``k`` from the same wavelength

.. code-block:: python

   # Locate the beam peak
   i, j = np.unravel_index(np.argmax(pattern_dB), pattern_dB.shape)
   print(f"Peak: theta={np.rad2deg(theta[i]):.1f} deg, phi={np.rad2deg(phi[j]):.1f} deg")

   # Physical units: one wavelength for both the geometry and k
   wavelength = 0.03  # meters (10 GHz)
   geom_10ghz = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5,
                                            wavelength=wavelength)
   k_10ghz = pa.wavelength_to_k(wavelength)

   # Normalized units: default wavelength=1.0 and k for wavelength 1.0
   k = pa.wavelength_to_k(1.0)

**Grating lobes appear unexpectedly**

- A grating lobe enters visible space when the spacing exceeds
  λ/(1 + sin θ0) at scan angle θ0; λ/2 spacing avoids them at all scan angles
- Positions are in meters, so divide the spacing by the wavelength

.. code-block:: python

   wavelength = 2 * np.pi / k
   dx = np.min(np.diff(np.unique(geom.x)))
   print(f"Spacing: {dx / wavelength:.3f} wavelengths")
   print(f"Limit for a 30 deg scan: {1 / (1 + np.sin(np.deg2rad(30))):.3f} wavelengths")

**Pattern is all zeros or constant**

- Weights may have zero magnitude
- Check for NaN values in positions or weights

.. code-block:: python

   print(f"Weight magnitude range: {np.abs(weights).min():.4f} to {np.abs(weights).max():.4f}")
   print(f"Any NaN in weights: {np.any(np.isnan(weights))}")
   print(f"Any NaN in positions: {np.any(np.isnan(geom.x))}")

Beamforming Issues
------------------

**Null steering doesn't produce deep nulls**

- With ideal weights, projection nulls are limited only by floating-point
  precision (about -300 dB). In hardware, phase quantization and amplitude
  and phase errors set the depth.
- An N-element array can place at most N - 1 independent nulls
- Verify null directions are within the visible region

.. code-block:: python

   null_directions = [(20.0, 180.0), (45.0, 0.0)]  # (theta_deg, phi_deg)
   weights_null = pa.null_steering_projection(
       geom, k, theta_main_deg=30, phi_main_deg=0,
       null_directions=null_directions, initial_weights=weights
   )

   # Null depth with ideal weights and with 6-bit phase shifters
   weights_6bit = pa.quantize_phase(weights_null, n_bits=6)
   for theta_null, phi_null in null_directions:
       ideal = pa.compute_null_depth(weights_null, geom, k, theta_null, phi_null, 30, 0)
       quant = pa.compute_null_depth(weights_6bit, geom, k, theta_null, phi_null, 30, 0)
       print(f"Null at ({theta_null}, {phi_null}): {ideal:.0f} dB ideal, "
             f"{quant:.1f} dB with 6-bit phase")

**Taylor taper not achieving specified sidelobe level**

- Measure on cuts through the beam: pass the steering angles to
  ``compute_pattern_cuts``. A cut that misses the beam is not referenced to
  the beam peak.
- Use ``compute_peak_sidelobe_level``, which steps past the main lobe to its
  first nulls, rather than the maximum over a fixed angle window
- Sample finely enough (``n_points``) to resolve the sidelobe peaks

.. code-block:: python

   angles_deg, cut_dB, cut_orth_dB = pa.compute_pattern_cuts(
       geom.x, geom.y, weights, k,
       theta0_deg=30, phi0_deg=0,  # cuts through the steered beam
       n_points=1801               # 0.1 deg steps
   )
   psl = pa.compute_peak_sidelobe_level(angles_deg, cut_dB)
   psl_orth = pa.compute_peak_sidelobe_level(angles_deg, cut_orth_dB)
   print(f"Peak sidelobe: {psl:.1f} dB (phi = 0 cut), {psl_orth:.1f} dB (orthogonal cut)")

Visualization Issues
--------------------

**Plotly figure not showing**

In Jupyter:

.. code-block:: python

   fig = pa.plot_pattern_3d_plotly(theta, phi, pattern_dB)
   fig.show()  # Should work
   # OR
   fig.show(renderer="notebook")

In scripts:

.. code-block:: python

   fig.write_html("pattern.html")  # Save and open in browser

**Contour plot looks blocky**

Increase resolution in pattern computation (the defaults are
``n_theta=181``, ``n_phi=361``):

.. code-block:: python

   theta, phi, pattern_dB = pa.compute_full_pattern(
       geom.x, geom.y, weights, k,
       n_theta=361,
       n_phi=721
   )
   pa.plot_pattern_contour(np.rad2deg(theta), np.rad2deg(phi), pattern_dB)

**UV-space plot shows artifacts outside visible region**

``compute_pattern_uv_space`` evaluates the whole (u, v) rectangle, including
the corners where u² + v² > 1, which correspond to no real direction. Mask
them:

.. code-block:: python

   u, v, pattern_uv = pa.compute_pattern_uv_space(geom, weights, k)
   u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
   pattern_uv = np.where(pa.is_visible_region(u_grid, v_grid), pattern_uv, np.nan)
   pa.plot_pattern_uv_space(u, v, pattern_uv)

Performance Issues
------------------

**Pattern computation is slow**

- Use vectorized computation (default)
- Reduce angular resolution if acceptable
- For uniform rectangular arrays, use FFT method

.. code-block:: python

   # FFT method for uniform arrays (fastest); weights reshaped to (Nx, Ny)
   weights_2d = weights.reshape(16, 16)
   u, v, AF = pa.array_factor_fft(weights_2d, dx=0.5, dy=0.5)
   iu, iv = np.unravel_index(np.argmax(np.abs(AF)), AF.shape)
   print(f"Beam peak: u={u[iu]:.3f}, v={v[iv]:.3f}")  # u = sin(30 deg) = 0.5

**Out of memory for large arrays**

- ``array_factor_vectorized`` and ``compute_full_pattern`` process angles in
  chunks, so intermediate memory stays near 64 MiB; the output grid takes
  16 bytes per complex sample
- Functions that build N x N matrices (mutual coupling, null steering) hold
  a 16 N² byte complex matrix plus working arrays; building the coupling
  matrix peaks near 34 N² bytes (3.4 GB at N = 10,000)
- Reduce pattern resolution, or compute the pattern in phi slices and keep
  only what you need

.. code-block:: python

   # Compute the pattern in phi slices, storing float32 dB
   theta_s = np.linspace(0, np.pi / 2, 181)
   phi_s = np.linspace(0, 2 * np.pi, 361)
   chunk_size = 90
   pattern_s = np.empty((theta_s.size, phi_s.size), dtype=np.float32)
   for start in range(0, phi_s.size, chunk_size):
       T, P = np.meshgrid(theta_s, phi_s[start:start + chunk_size], indexing='ij')
       AF = pa.array_factor_vectorized(T, P, geom.x, geom.y, weights, k)
       pattern_s[:, start:start + chunk_size] = pa.linear_to_db(np.abs(AF)**2)
   pattern_s -= pattern_s.max()

Numerical Issues
----------------

**Null steering with repeated or closely spaced null directions**

- Two null directions much closer than a beamwidth act almost as one
  constraint and make the constraint matrix nearly singular
- ``null_steering_projection`` and ``null_steering_lcmv`` fall back to a
  pseudoinverse when the matrix inverse fails, so they return weights instead
  of raising ``LinAlgError``
- Use LCMV when the main-beam response must stay fixed: it constrains the
  response in every listed direction, including the beam

.. code-block:: python

   # (theta_deg, phi_deg, response): unity at the beam, nulls elsewhere
   constraints = [(30.0, 0.0, 1.0), (20.0, 180.0, 0.0), (45.0, 0.0, 0.0)]
   weights_lcmv = pa.null_steering_lcmv(geom, k, constraints)

   beam = weights_lcmv @ pa.array_manifold(k, geom.x, geom.y, 30, 0)
   print(f"Beam response: {abs(beam):.3f}")

**Warning: divide by zero**

Usually occurs with zero weights or at exact nulls. The library handles most
cases, but check for:

- Zero-magnitude weights
- Evaluation exactly at a null direction

FAQ
---

**Q: What units should element positions use?**

A: Meters. Geometry builders take spacing in wavelengths plus a
``wavelength`` argument (default 1.0) that converts it to meters; pass the
same wavelength to ``pa.wavelength_to_k``. With the defaults, meters and
wavelengths coincide numerically. See :doc:`user_guide/conventions`.

**Q: How many elements can the library handle?**

A: Tested with arrays up to 100,000 elements. Computation time scales with
N_elements × N_angles.

**Q: Can I use measured element patterns?**

A: Yes. Pass a callable as ``element_pattern_func`` to the scalar pattern
functions, or a ``pa.GriddedElementPattern`` built from sampled
(E_theta, E_phi) data as ``element_func`` to the vector-pattern functions
(see :doc:`user_guide/polarization`).

**Q: Does the library support polarization?**

A: Yes. Since v1.4.0, polarized element models produce (E_theta, E_phi)
patterns, with co/cross-pol, XPD and axial-ratio maps. See
:doc:`user_guide/polarization`.

**Q: How do I cite this library?**

A: See the README for citation information. Include the version number used.

Getting Help
------------

- **GitHub Issues**: https://github.com/jman4162/Phased-Array-Antenna-Model/issues
- **Documentation**: https://phased-array-antenna-model.readthedocs.io
