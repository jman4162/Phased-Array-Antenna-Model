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
- :math:`x_n, y_n` = element positions in wavelengths

The :func:`~phased_array.array_factor_vectorized` function computes this
efficiently using NumPy broadcasting.

Pattern Multiplication
----------------------

The total radiation pattern of an array equals the product of:

1. **Element pattern**: Directivity of a single element
2. **Array factor**: Interference pattern from element spacing and phasing

.. math::

   E_{total}(\theta, \phi) = E_{element}(\theta, \phi) \times AF(\theta, \phi)

.. code-block:: python

   # Compute element pattern (cosine model)
   element = pa.element_pattern(theta, exponent=1.3)

   # Compute array factor
   af = pa.array_factor_vectorized(k, geom.x, geom.y, weights, theta, phi)

   # Total pattern
   total = element * af

Steering Vector
---------------

To point the main beam toward direction :math:`(\theta_0, \phi_0)`, apply
phase shifts that cancel the path differences:

.. math::

   w_n = \exp\left(-jk(x_n u_0 + y_n v_0)\right)

Where :math:`u_0 = \sin\theta_0 \cos\phi_0` and :math:`v_0 = \sin\theta_0 \sin\phi_0`.

.. code-block:: python

   weights = pa.steering_vector(k, x, y, theta0_deg=30, phi0_deg=45)

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
     - 65°
   * - 0.60
     - 56°
   * - 0.70
     - 46°

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

   # Taylor taper for -30 dB sidelobes
   taper = pa.taylor_taper_2d(16, 16, sidelobe_dB=-30)

   # Apply to steering weights
   weights = weights * taper

Units and Normalization
-----------------------

Positions
^^^^^^^^^

Element positions are in **wavelengths** by default. This makes patterns
frequency-independent. For physical units:

.. code-block:: python

   # If positions are in meters
   wavelength = 0.03  # 10 GHz
   x_wavelengths = x_meters / wavelength

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

Patterns are typically shown in dB, normalized to the peak:

.. code-block:: python

   pattern_dB = pa.linear_to_db(np.abs(af))
   pattern_dB = pa.normalize_pattern(pattern_dB)  # Peak at 0 dB

ArrayGeometry Class
-------------------

The :class:`~phased_array.ArrayGeometry` dataclass stores array element
positions and optional metadata:

.. code-block:: python

   from dataclasses import dataclass

   @dataclass
   class ArrayGeometry:
       x: np.ndarray      # X positions (wavelengths)
       y: np.ndarray      # Y positions (wavelengths)
       z: np.ndarray      # Z positions (wavelengths)
       nx: np.ndarray     # Normal vector x components (optional)
       ny: np.ndarray     # Normal vector y components (optional)
       nz: np.ndarray     # Normal vector z components (optional)
       n_elements: int    # Number of elements

Element normals are used for conformal arrays where elements point in
different directions.
