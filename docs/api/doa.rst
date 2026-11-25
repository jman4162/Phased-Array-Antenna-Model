DOA Module
==========

.. module:: phased_array.doa
   :synopsis: Direction-of-arrival estimation

The doa module estimates plane-wave arrival directions from array
snapshots. Snapshots are rows of an ``(n_snapshots, n_elements)`` array and
covariances are ``R = E[x x^H]``; see :doc:`../user_guide/doa` for worked
examples and :doc:`../user_guide/conventions` for the sign convention.

Snapshots and Covariance
------------------------

.. autofunction:: phased_array.generate_snapshots

.. autofunction:: phased_array.sample_covariance

Spectral Estimators
-------------------

These work for any array geometry. Scan angles broadcast against each
other, and the spectrum has their shape.

.. autofunction:: phased_array.bartlett_spectrum

.. autofunction:: phased_array.capon_spectrum

.. autofunction:: phased_array.music_spectrum

.. autofunction:: phased_array.find_spectrum_peaks

Uniform Linear Array Estimators
-------------------------------

These assume a ULA along x with the covariance rows ordered by increasing x.

.. autofunction:: phased_array.root_music

.. autofunction:: phased_array.esprit

Model Order and Bounds
----------------------

.. autofunction:: phased_array.estimate_num_sources

.. autofunction:: phased_array.crb_ula_single_source
