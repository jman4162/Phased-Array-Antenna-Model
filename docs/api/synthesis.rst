Synthesis Module
================

.. module:: phased_array.synthesis
   :synopsis: Array factor synthesis for linear arrays

The synthesis module returns element weights that produce a specified
array factor for a linear array along x, with
``AF(u) = sum_n w_n exp(+j k x_n u)``. The uniform-array functions
return weights for elements centered at the origin, in the order of
``create_rectangular_array(N, 1, d, d)``. A separable planar design is
``np.outer(wx, wy).ravel()``. See :doc:`../user_guide/synthesis` for
worked examples.

Analytical Tapers
-----------------

.. autofunction:: phased_array.binomial_taper_1d

Pattern Matching
----------------

.. autofunction:: phased_array.woodward_lawson_weights

.. autofunction:: phased_array.least_squares_weights
