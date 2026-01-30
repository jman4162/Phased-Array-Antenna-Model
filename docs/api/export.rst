Export Module
=============

.. module:: phased_array.export
   :synopsis: Data export functions for integration with other tools

The export module provides functions to save array configurations, patterns,
and analysis results in various formats for use with other software tools.

Pattern Export
--------------

.. autofunction:: phased_array.export_pattern_csv

.. autofunction:: phased_array.export_pattern_2d_csv

.. autofunction:: phased_array.export_uv_pattern_csv

.. autofunction:: phased_array.export_pattern_npz

.. autofunction:: phased_array.load_pattern_npz

Array Configuration
-------------------

.. autofunction:: phased_array.export_weights_csv

.. autofunction:: phased_array.export_geometry_csv

.. autofunction:: phased_array.export_array_config_json

.. autofunction:: phased_array.export_coupling_matrix_csv

Reports
-------

.. autofunction:: phased_array.export_summary_report
