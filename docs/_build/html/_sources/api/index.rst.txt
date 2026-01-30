API Reference
=============

This section provides comprehensive API documentation for all modules in the
phased_array package. Each function includes parameter descriptions, return
values, and usage examples.

Modules
-------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`core`
     - Core computation functions: steering vectors, array factor, pattern cuts
   * - :doc:`geometry`
     - Array geometry creation: rectangular, triangular, circular, conformal, sparse
   * - :doc:`beamforming`
     - Beamforming techniques: amplitude tapers, null steering, multi-beam
   * - :doc:`impairments`
     - Realistic impairments: mutual coupling, quantization, failures, scan blindness
   * - :doc:`visualization`
     - Plotting functions: 2D, 3D, UV-space, Plotly interactive
   * - :doc:`wideband`
     - Wideband analysis: true time delay, hybrid steering, beam squint
   * - :doc:`utils`
     - Utility functions: coordinate transforms, unit conversions
   * - :doc:`export`
     - Data export: CSV, JSON, NPZ formats

Quick Reference
---------------

Most Commonly Used Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Array Creation**

- :func:`~phased_array.create_rectangular_array` - Create rectangular grid
- :func:`~phased_array.create_triangular_array` - Create triangular grid
- :func:`~phased_array.create_circular_array` - Create circular array

**Pattern Computation**

- :func:`~phased_array.steering_vector` - Compute steering weights
- :func:`~phased_array.array_factor_vectorized` - Compute array factor
- :func:`~phased_array.compute_full_pattern` - Compute full 3D pattern

**Beamforming**

- :func:`~phased_array.taylor_taper_2d` - Taylor amplitude taper
- :func:`~phased_array.null_steering_projection` - Null steering

**Visualization**

- :func:`~phased_array.plot_pattern_contour` - 2D contour plot
- :func:`~phased_array.plot_pattern_3d_plotly` - Interactive 3D plot

.. toctree::
   :maxdepth: 2
   :hidden:

   core
   geometry
   beamforming
   impairments
   visualization
   wideband
   utils
   export
