Visualization
=============

This guide covers the visualization capabilities including 2D/3D plots,
UV-space representation, and interactive Plotly figures.

2D Matplotlib Plots
-------------------

Basic Pattern Cut
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import phased_array as pa
   import numpy as np
   import matplotlib.pyplot as plt

   # Create array and compute pattern
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
   weights *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

   # Compute principal plane cuts
   theta_deg, E_plane, H_plane = pa.compute_pattern_cuts(
       geom.x, geom.y, weights, k,
       theta0_deg=30, phi0_deg=0
   )

   # Plot
   pa.plot_pattern_2d(
       theta_deg, E_plane,
       title='E-plane Pattern Cut',
       xlabel='Theta (deg)',
       ylabel='Normalized Gain (dB)',
       xlim=(-90, 90),
       ylim=(-50, 0)
   )

Polar Plot
^^^^^^^^^^

.. code-block:: python

   pa.plot_pattern_polar(
       theta_deg, E_plane,
       title='E-plane (Polar)'
   )

Contour Plot
^^^^^^^^^^^^

Full 2D pattern as a contour plot:

.. code-block:: python

   theta, phi, pattern_dB = pa.compute_full_pattern(
       geom.x, geom.y, weights, k
   )

   pa.plot_pattern_contour(
       np.rad2deg(theta),
       np.rad2deg(phi),
       pattern_dB,
       title='2D Radiation Pattern',
       levels=20,
       cmap='jet',
       min_dB=-40
   )

Comparison Plots
^^^^^^^^^^^^^^^^

Compare multiple patterns on the same axes:

.. code-block:: python

   patterns = {}

   # Uniform illumination
   w_uniform = pa.steering_vector(k, geom.x, geom.y, 30, 0)
   _, E_uniform, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_uniform, k)
   patterns['Uniform'] = E_uniform

   # Taylor taper
   w_taylor = w_uniform * pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)
   _, E_taylor, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_taylor, k)
   patterns['Taylor -30 dB'] = E_taylor

   # Chebyshev taper
   w_cheb = w_uniform * pa.chebyshev_taper_2d(16, 16, sidelobe_dB=-30)
   _, E_cheb, _ = pa.compute_pattern_cuts(geom.x, geom.y, w_cheb, k)
   patterns['Chebyshev -30 dB'] = E_cheb

   pa.plot_comparison_patterns(
       theta_deg, patterns,
       title='Taper Comparison',
       xlim=(-60, 60),
       ylim=(-50, 0)
   )

Array Geometry Plot
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   pa.plot_array_geometry(
       geom.x, geom.y,
       weights=weights,  # Color by weight magnitude
       title='16x16 Array Element Positions'
   )

UV-Space Visualization
----------------------

Direction cosine (UV) space is often more intuitive for array analysis as
it shows the visible region and grating lobe locations clearly.

Computing UV-Space Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   u, v, pattern_uv = pa.compute_pattern_uv_space(
       geom.x, geom.y, weights, k,
       n_u=201, n_v=201,
       u_range=(-1, 1),
       v_range=(-1, 1)
   )

   pa.plot_pattern_uv_space(
       u, v, pattern_uv,
       title='UV-Space Pattern',
       show_visible_region=True  # Circle at u^2+v^2=1
   )

The visible region (u² + v² ≤ 1) corresponds to real angles. Points outside
are evanescent.

3D Plotly (Interactive)
-----------------------

Plotly provides interactive 3D visualizations that can be rotated, zoomed,
and explored in a web browser.

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install plotly

3D Radiation Pattern
^^^^^^^^^^^^^^^^^^^^

Spherical surface where radius represents gain:

.. code-block:: python

   fig = pa.plot_pattern_3d_plotly(
       theta, phi, pattern_dB,
       title='3D Radiation Pattern',
       min_dB=-40,
       colorscale='Jet',
       surface_type='spherical'
   )
   fig.show()

Cartesian 3D Surface
^^^^^^^^^^^^^^^^^^^^

Theta/phi/gain as a 3D surface:

.. code-block:: python

   fig = pa.plot_pattern_3d_cartesian_plotly(
       theta, phi, pattern_dB,
       title='Pattern Surface'
   )
   fig.show()

UV-Space (Plotly)
^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = pa.plot_pattern_uv_plotly(
       u, v, pattern_uv,
       title='UV-Space Pattern',
       min_dB=-40
   )
   fig.show()

3D Array Geometry
^^^^^^^^^^^^^^^^^

For conformal arrays:

.. code-block:: python

   # Create cylindrical array
   geom_cyl = pa.create_cylindrical_array(16, 4, radius=3.0, height=2.0)

   fig = pa.plot_array_geometry_3d_plotly(
       geom_cyl,
       show_normals=True,  # Display element normal vectors
       title='Cylindrical Array'
   )
   fig.show()

Animated Patterns
^^^^^^^^^^^^^^^^^

Create animations showing pattern changes (e.g., scanning beam):

.. code-block:: python

   # List of patterns at different scan angles
   scan_angles = np.arange(0, 61, 5)
   patterns_list = []
   titles = []

   for angle in scan_angles:
       w = pa.steering_vector(k, geom.x, geom.y, angle, 0)
       w *= pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)
       _, _, pat = pa.compute_full_pattern(geom.x, geom.y, w, k)
       patterns_list.append(pat)
       titles.append(f'Scan: {angle} deg')

   fig = pa.create_pattern_animation_plotly(
       theta, phi,
       patterns_list,
       frame_titles=titles,
       animation_speed=500  # ms per frame
   )
   fig.show()

Wideband Visualization
----------------------

Beam Squint Plot
^^^^^^^^^^^^^^^^

.. code-block:: python

   wavelength = 0.03  # 10 GHz
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5, wavelength=wavelength)

   fig = pa.plot_beam_squint(
       geom.x, geom.y,
       theta0_deg=45, phi0_deg=0,
       center_frequency=10e9,
       bandwidth=2e9,
       steering_modes=['phase', 'ttd']
   )
   fig.show()

Pattern vs Frequency
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = pa.plot_pattern_vs_frequency_plotly(
       geom.x, geom.y,
       theta0_deg=30, phi0_deg=0,
       center_frequency=10e9,
       bandwidth=2e9,
       n_freqs=11,
       steering_mode='phase'
   )
   fig.show()

Subarray Delays
^^^^^^^^^^^^^^^

.. code-block:: python

   architecture = pa.create_rectangular_subarrays(
       Nx_total=32, Ny_total=32,
       Nx_sub=8, Ny_sub=8,
       dx=0.5, dy=0.5, wavelength=wavelength
   )

   fig = pa.plot_subarray_delays(
       architecture,
       theta0_deg=45, phi0_deg=0
   )
   fig.show()

Saving Figures
--------------

Matplotlib
^^^^^^^^^^

.. code-block:: python

   fig, ax = plt.subplots()
   pa.plot_pattern_2d(theta_deg, E_plane, ax=ax)
   fig.savefig('pattern.png', dpi=300, bbox_inches='tight')
   fig.savefig('pattern.pdf', bbox_inches='tight')

Plotly
^^^^^^

.. code-block:: python

   # HTML (interactive)
   fig.write_html('pattern_3d.html')

   # Static image (requires kaleido)
   fig.write_image('pattern_3d.png', scale=2)
   fig.write_image('pattern_3d.pdf')

   # Install kaleido for static export:
   # pip install kaleido

Customization
-------------

Matplotlib Style
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use a built-in style
   plt.style.use('seaborn-v0_8-whitegrid')

   # Or customize
   plt.rcParams.update({
       'font.size': 12,
       'axes.labelsize': 14,
       'axes.titlesize': 16,
       'lines.linewidth': 2,
       'figure.figsize': (10, 6)
   })

Plotly Theme
^^^^^^^^^^^^

.. code-block:: python

   import plotly.io as pio
   pio.templates.default = "plotly_white"

   # Or use dark theme
   pio.templates.default = "plotly_dark"

Best Practices
--------------

1. **Use UV-space** for understanding grating lobes and visible region.

2. **Use 3D Plotly** for presentations and interactive exploration.

3. **Use contour plots** for comparing patterns across the full hemisphere.

4. **Use pattern cuts** for precise numerical comparisons.

5. **Set appropriate min_dB** to focus on the sidelobe region of interest.

6. **Export high-resolution** figures for publications (300+ DPI).
