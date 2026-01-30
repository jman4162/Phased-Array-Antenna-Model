Phased Array Modeling
=====================

A comprehensive Python library for computing and visualizing phased array antenna
radiation patterns.

**Getting Started**
   New to phased array modeling? Start with :doc:`getting_started/installation`
   and the :doc:`getting_started/quickstart` tutorial.

**User Guide**
   In-depth guides covering :doc:`user_guide/geometry`, :doc:`user_guide/beamforming`,
   :doc:`user_guide/impairments`, :doc:`user_guide/wideband`, and :doc:`user_guide/visualization`.

**API Reference**
   Complete :doc:`api/index` for all functions, classes, and modules with detailed examples.

**Cookbook**
   Practical recipes for :doc:`cookbook/hardware_recipes`, :doc:`cookbook/systems_recipes`,
   and :doc:`cookbook/research_recipes`.

Features
--------

- **Vectorized computation** - 50-100x faster than loop-based implementations
- **Multiple geometries** - Rectangular, triangular, circular, cylindrical, spherical, and sparse arrays
- **Beamforming** - Amplitude tapering, null steering, multi-beam synthesis
- **Realistic impairments** - Mutual coupling, phase quantization, element failures, scan blindness
- **Wideband support** - True time delay (TTD), hybrid phase/TTD, beam squint analysis
- **Interactive visualization** - 2D/3D plots with Plotly, UV-space, pattern animations
- **Data export** - CSV, JSON, NPZ formats for integration with other tools

Quick Example
-------------

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Create a 16x16 rectangular array with half-wavelength spacing
   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)

   # Compute steering weights for 30 degree scan
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)

   # Apply Taylor taper for -30 dB sidelobes
   taper = pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)
   weights = weights * taper

   # Compute full 3D pattern
   theta, phi, pattern_dB = pa.compute_full_pattern(geom.x, geom.y, weights, k)

   # Visualize
   pa.plot_pattern_contour(np.rad2deg(theta), np.rad2deg(phi), pattern_dB)

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install phased-array-modeling

For interactive Plotly visualizations:

.. code-block:: bash

   pip install phased-array-modeling[plotting]

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   user_guide/index
   user_guide/geometry
   user_guide/beamforming
   user_guide/impairments
   user_guide/wideband
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index
   api/core
   api/geometry
   api/beamforming
   api/impairments
   api/visualization
   api/wideband
   api/utils
   api/export

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Cookbook

   cookbook/index
   cookbook/hardware_recipes
   cookbook/systems_recipes
   cookbook/research_recipes

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Theory

   theory/index
   theory/fundamentals
   theory/coordinate_systems
   theory/tapering_theory

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   troubleshooting
   glossary
