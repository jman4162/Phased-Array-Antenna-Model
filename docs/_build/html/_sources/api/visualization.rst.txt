Visualization Module
====================

.. module:: phased_array.visualization
   :synopsis: Plotting and visualization functions

The visualization module provides functions for creating 2D and 3D plots of
radiation patterns, array geometries, and UV-space representations using
Matplotlib and Plotly.

2D Matplotlib Plots
-------------------

.. autofunction:: phased_array.plot_pattern_2d

.. autofunction:: phased_array.plot_pattern_polar

.. autofunction:: phased_array.plot_pattern_contour

.. autofunction:: phased_array.plot_array_geometry

.. autofunction:: phased_array.plot_comparison_patterns

UV-Space
--------

.. autofunction:: phased_array.compute_pattern_uv_space

.. autofunction:: phased_array.plot_pattern_uv_space

3D Plotly (Interactive)
-----------------------

.. autofunction:: phased_array.plot_pattern_3d_plotly

.. autofunction:: phased_array.plot_pattern_3d_cartesian_plotly

.. autofunction:: phased_array.plot_array_geometry_3d_plotly

.. autofunction:: phased_array.plot_pattern_uv_plotly

.. autofunction:: phased_array.create_pattern_animation_plotly

Wideband Visualization
----------------------

.. autofunction:: phased_array.plot_beam_squint

.. autofunction:: phased_array.plot_pattern_vs_frequency

.. autofunction:: phased_array.plot_pattern_vs_frequency_plotly

.. autofunction:: phased_array.plot_subarray_delays
