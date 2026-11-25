Data Import Module
==================

.. module:: phased_array.io
   :synopsis: Readers for Touchstone S-parameter files and tabulated element patterns

The io module reads data produced outside the package: Touchstone version
1.x S-parameter files from network analyzers and EM solvers, and far-field
tables of embedded element patterns. See :doc:`/user_guide/data_import` for
worked examples.

Touchstone Files
----------------

.. autofunction:: phased_array.read_touchstone

.. autofunction:: phased_array.write_touchstone

Tabulated Element Patterns
--------------------------

.. autofunction:: phased_array.load_element_pattern_csv

:meth:`GriddedElementPattern.from_table
<phased_array.GriddedElementPattern.from_table>` builds the same pattern
from arrays already in memory; it is documented with
:class:`~phased_array.GriddedElementPattern` in :doc:`vector_patterns`.
