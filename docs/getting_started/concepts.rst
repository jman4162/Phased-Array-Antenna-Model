Core Concepts
=============

This page explains the fundamental concepts used throughout the library.

Coordinate Systems
------------------

Spherical Coordinates (Theta/Phi)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library uses a standard spherical coordinate system:

- **Theta (θ)**: Angle from the z-axis (zenith), 0° to 180°
- **Phi (φ)**: Azimuth angle in the x-y plane, 0° to 360°
- **Broadside**: θ = 0° (normal to array face)

.. math::

   x = r \sin\theta \cos\phi \\
   y = r \sin\theta \sin\phi \\
   z = r \cos\theta

Direction Cosines (UV-Space)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

UV-space uses direction cosines, which are more convenient for array analysis:

.. math::

   u = \sin\theta \cos\phi \\
   v = \sin\theta \sin\phi \\
   w = \cos\theta

The **visible region** is defined by :math:`u^2 + v^2 \leq 1`. Points outside
this circle correspond to evanescent waves.

Convert between coordinate systems:

.. code-block:: python

   import numpy as np
   import phased_array as pa

   theta = np.deg2rad(30.0)   # angles are in radians
   phi = np.deg2rad(45.0)
   u, v = pa.theta_phi_to_uv(theta, phi)
   theta, phi = pa.uv_to_theta_phi(u, v)

Array Factor
------------

The array factor describes the interference pattern of an array of isotropic
point sources:

.. math::

   AF(\theta, \phi) = \sum_{n=1}^{N} w_n \exp\left(jk(x_n u + y_n v)\right)

Where:

- :math:`w_n` = complex weight (amplitude and phase) for element n
- :math:`k = 2\pi/\lambda` = wavenumber
- :math:`x_n, y_n` = element positions in meters (the same length unit as
  :math:`\lambda`)

The :func:`~phased_array.array_factor_vectorized` function computes this
efficiently using NumPy broadcasting.

Pattern Multiplication
----------------------

The total radiation pattern of an array equals the product of:

1. **Element pattern**: Directivity of a single element
2. **Array factor**: Interference pattern from element spacing and phasing

.. math::

   E_{total}(\theta, \phi) = E_{element}(\theta, \phi) \times AF(\theta, \phi)

The element pattern and the array factor are both **field** quantities,
so they multiply directly; square the magnitude to get power.

.. code-block:: python

   geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=20, phi0_deg=0)

   theta = np.deg2rad(np.linspace(0, 90, 181))
   phi = np.zeros_like(theta)

   # Element field pattern: cos_exp_theta is the exponent of the power pattern
   element = pa.element_pattern(theta, phi, cos_exp_theta=1.3)

   # Array factor (field)
   af = pa.array_factor_vectorized(theta, phi, geom.x, geom.y, weights, k)

   # Total field pattern; same as pa.total_pattern(..., element_pattern_func=pa.element_pattern)
   total = element * af
   power = np.abs(total) ** 2

Steering Vector
---------------

To point the main beam toward direction :math:`(\theta_0, \phi_0)`, apply
phase shifts that cancel the path differences:

.. math::

   w_n = \exp\left(-jk(x_n u_0 + y_n v_0)\right)

Where :math:`u_0 = \sin\theta_0 \cos\phi_0` and :math:`v_0 = \sin\theta_0 \sin\phi_0`.

.. code-block:: python

   weights = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=45)

The weights are the conjugate of the signal a plane wave from
:math:`(\theta_0, \phi_0)` induces at the elements; see
:doc:`../user_guide/conventions` for the sign convention.

Element Spacing and Grating Lobes
---------------------------------

**Grating lobes** are unwanted secondary main beams that appear when element
spacing exceeds half a wavelength. To avoid grating lobes when scanning to
angle :math:`\theta_{max}`:

.. math::

   d < \frac{\lambda}{1 + \sin\theta_{max}}

For λ/2 spacing (d=0.5), the array can scan to ±90° without grating lobes.
For larger spacing, the maximum scan angle is limited.

.. list-table:: Maximum Scan Angle vs. Spacing
   :header-rows: 1

   * - Spacing (d/λ)
     - Max Scan Angle
   * - 0.50
     - 90°
   * - 0.55
     - 54.9°
   * - 0.60
     - 41.8°
   * - 0.70
     - 25.4°

Amplitude Tapering
------------------

Uniform amplitude weighting produces the narrowest beamwidth but has high
sidelobes (~-13 dB for a rectangular array). Amplitude tapering reduces
sidelobes at the cost of:

1. Increased beamwidth
2. Reduced aperture efficiency (directivity loss)

Common tapers include:

- **Taylor**: Specified sidelobe level with controlled rolloff
- **Chebyshev**: Equiripple sidelobes, narrowest beamwidth for given SLL
- **Hamming/Hanning**: Simple, good for general use

.. code-block:: python

   # Taylor taper for -30 dB sidelobes (same 8 x 8 size as geom)
   taper = pa.taylor_taper_2d(8, 8, sidelobe_dB=-30)

   # Apply to steering weights
   weights = weights * taper

Units and Normalization
-----------------------

Positions
^^^^^^^^^

Element positions are in **meters**, and every pattern function takes the
wavenumber ``k = 2*pi/wavelength`` in rad/m. Geometry builders take spacing
in wavelengths plus a ``wavelength`` argument that converts to meters. With
the default ``wavelength=1.0`` and ``k = pa.wavelength_to_k(1.0)``, meters and
wavelengths coincide numerically, which keeps examples frequency-independent.
For a physical design, use the same wavelength in both places:

.. code-block:: python

   wavelength = 0.03                          # 10 GHz
   geom_10g = pa.create_rectangular_array(16, 16, 0.5, 0.5, wavelength=wavelength)
   k_10g = pa.wavelength_to_k(wavelength)     # or pa.frequency_to_k(10e9)

Wavenumber
^^^^^^^^^^

The wavenumber relates frequency to spatial phase:

.. math::

   k = \frac{2\pi}{\lambda} = \frac{2\pi f}{c}

.. code-block:: python

   k = pa.wavelength_to_k(1.0)  # For normalized wavelength=1
   k = pa.frequency_to_k(10e9)  # For 10 GHz

Pattern Normalization
^^^^^^^^^^^^^^^^^^^^^

Patterns are typically shown in dB, normalized to the peak.
:func:`~phased_array.linear_to_db` expects **power**, so square the field
magnitude first, then subtract the peak in dB:

.. code-block:: python

   pattern_dB = pa.linear_to_db(np.abs(af) ** 2)
   pattern_dB -= pattern_dB.max()  # Peak at 0 dB

Passing ``np.abs(af)`` without squaring halves every dB value (a -13.3 dB
sidelobe would read -6.6 dB). :func:`~phased_array.normalize_pattern` works on
linear field values, not on dB.

ArrayGeometry Class
-------------------

The :class:`~phased_array.ArrayGeometry` dataclass stores array element
positions and optional metadata:

Its fields (all per-element arrays, positions in meters):

- ``x``, ``y``, ``z``: element positions (``z`` optional)
- ``nx``, ``ny``, ``nz``: element normal vectors (optional)
- ``tx``, ``ty``, ``tz``: element tangent (polarization reference) vectors
  (optional)
- ``element_indices``: indices into the original array, kept through thinning

``geom.n_elements``, ``geom.is_planar`` and ``geom.is_conformal`` are derived
properties. Element normals and tangents are used for conformal arrays where
elements point in different directions.

.. code-block:: python

   geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
   print(geom.n_elements, geom.is_planar)
