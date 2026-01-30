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

**Pattern has unexpected spikes or artifacts**

- Check that element positions are in consistent units (wavelengths or meters)
- Verify weights array has the correct length
- Ensure theta/phi arrays have compatible shapes

.. code-block:: python

   # Verify dimensions
   print(f"Elements: {geom.n_elements}")
   print(f"Weights: {len(weights)}")
   assert len(weights) == geom.n_elements

**Pattern peak is not at expected scan angle**

- Verify steering angle units are in degrees (not radians)
- Check wavenumber calculation matches position units

.. code-block:: python

   # If positions are in meters:
   wavelength = 0.03  # meters (10 GHz)
   k = pa.wavelength_to_k(wavelength)

   # If positions are in wavelengths:
   k = pa.wavelength_to_k(1.0)  # normalized

**Grating lobes appear unexpectedly**

- Element spacing likely exceeds λ/2
- Check spacing in wavelengths, not meters

.. code-block:: python

   # Check effective spacing
   dx = np.min(np.diff(np.unique(geom.x)))
   print(f"Spacing: {dx} (should be <= 0.5 wavelengths)")

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

- Null depth is limited by array size
- More elements = deeper achievable nulls
- Verify null directions are within the visible region

.. code-block:: python

   # Check null depth
   for theta_null, phi_null in null_directions:
       depth = pa.compute_null_depth(geom, k, weights, (theta_null, phi_null))
       print(f"Null at ({theta_null}, {phi_null}): {depth:.1f} dB")

**Taylor taper not achieving specified sidelobe level**

- Pattern needs sufficient angular resolution to see true sidelobes
- Increase n_theta in compute_full_pattern

.. code-block:: python

   # Use higher resolution
   theta, phi, pattern = pa.compute_full_pattern(
       geom.x, geom.y, weights, k,
       n_theta=361,  # More points
       n_phi=721
   )

Visualization Issues
--------------------

**Plotly figure not showing**

In Jupyter:

.. code-block:: python

   fig.show()  # Should work
   # OR
   fig.show(renderer="notebook")

In scripts:

.. code-block:: python

   fig.write_html("pattern.html")  # Save and open in browser

**Contour plot looks blocky**

Increase resolution in pattern computation:

.. code-block:: python

   theta, phi, pattern = pa.compute_full_pattern(
       geom.x, geom.y, weights, k,
       n_theta=181,  # Increase from default
       n_phi=361
   )

**UV-space plot shows artifacts outside visible region**

Set appropriate limits or mask:

.. code-block:: python

   u, v, pattern = pa.compute_pattern_uv_space(
       geom.x, geom.y, weights, k,
       u_range=(-1, 1),  # Stay within visible region
       v_range=(-1, 1)
   )

Performance Issues
------------------

**Pattern computation is slow**

- Use vectorized computation (default)
- Reduce angular resolution if acceptable
- For uniform rectangular arrays, use FFT method

.. code-block:: python

   # FFT method for uniform arrays (fastest)
   u, v, AF = pa.array_factor_fft(weights_2d, dx, dy)

**Out of memory for large arrays**

- Reduce pattern resolution
- Compute in chunks

.. code-block:: python

   # Compute pattern in phi slices
   for phi_idx in range(0, n_phi, chunk_size):
       # Process subset

Numerical Issues
----------------

**LinAlgError: Singular matrix in null steering**

- Null directions may be too close together
- Try reducing the number of nulls
- Use LCMV instead of projection method

.. code-block:: python

   # LCMV handles degenerate cases better
   weights = pa.null_steering_lcmv(geom, k, constraints)

**Warning: divide by zero**

Usually occurs with zero weights or at exact nulls. The library handles most
cases, but check for:

- Zero-magnitude weights
- Evaluation exactly at a null direction

FAQ
---

**Q: What units should element positions use?**

A: Either wavelengths (recommended for frequency-independent analysis) or
meters. Be consistent and match your wavenumber calculation.

**Q: How many elements can the library handle?**

A: Tested with arrays up to 100,000 elements. Computation time scales with
N_elements × N_angles.

**Q: Can I use measured element patterns?**

A: Yes, pass a custom function to ``element_pattern_func`` parameter in
pattern computation functions.

**Q: Does the library support polarization?**

A: Currently models scalar patterns only. Polarization support may be added
in future versions.

**Q: How do I cite this library?**

A: See the README for citation information. Include the version number used.

Getting Help
------------

- **GitHub Issues**: https://github.com/jman4162/Phased-Array-Antenna-Model/issues
- **Documentation**: https://phased-array-modeling.readthedocs.io
