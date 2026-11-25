Conventions
===========

This page collects the units, sign conventions and field-versus-power rules
that every module follows. Most wrong results in array modeling come from
mixing two of these, so check here first when a number looks off.

Units
-----

- **Positions** (``geom.x``, ``geom.y``, ``geom.z``) are in meters.
- **Wavenumber** ``k = 2*pi/wavelength`` is in rad/m and is passed to every
  pattern function. ``pa.wavelength_to_k(wavelength)`` and
  ``pa.frequency_to_k(f)`` build it.
- **Geometry builders** take spacing in wavelengths plus a ``wavelength``
  argument (meters) that converts to meters. Use the same wavelength for the
  geometry and for ``k``. With the default ``wavelength=1.0`` and
  ``k = pa.wavelength_to_k(1.0)``, meters and wavelengths coincide
  numerically.
- **Angles**: pattern grids (``theta``, ``phi``) are in radians. Beam-direction
  arguments carry a ``_deg`` suffix (``theta0_deg``, ``phi0_deg``) and are in
  degrees. Cut functions return angles in degrees.

Coordinates
-----------

Theta is measured from +z (array boresight for a planar array in the x-y
plane) and phi from +x in the x-y plane. Direction cosines are
``u = sin(theta) cos(phi)``, ``v = sin(theta) sin(phi)``. Two az/el
conventions exist: :func:`~phased_array.thetaphi_to_azel` puts boresight at
az = el = 0, while :func:`~phased_array.antenna_to_radar` treats the array
face as pointing at zenith, so boresight is at el = 90 deg.

Sign convention
---------------

The array factor is

.. math::

   AF(\mathbf{u}) = \sum_n w_n \exp\left(+jk\,\mathbf{r}_n \cdot \mathbf{u}\right)

(:func:`~phased_array.array_factor_vectorized`,
:func:`~phased_array.array_factor_uv`, :func:`~phased_array.array_factor_fft`).
Three vectors follow from it:

- :func:`~phased_array.steering_vector` returns
  ``exp(-jk r_n . u0)``, the weights that point the beam at ``u0``.
- :func:`~phased_array.array_manifold` returns ``exp(+jk r_n . u)``, the
  signal a plane wave from ``u`` induces at the elements. It is the
  conjugate of the steering vector.
- The array output for received snapshots ``x`` is ``w @ x`` with no
  conjugation, which equals the array factor in the arrival direction.

Build received data from :func:`~phased_array.array_manifold`, not from
:func:`~phased_array.steering_vector`. Using the steering vector conjugates
the data and makes an adaptive beamformer null the mirror image of the
interferer. :func:`~phased_array.adaptive_weights_smi` and
:func:`~phased_array.adaptive_weights_gsc` both return weights with
``w @ array_manifold(look) == 1``.

.. code-block:: python

   import numpy as np
   import phased_array as pa

   geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   w = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)
   a = pa.array_manifold(k, geom.x, geom.y, 30, 0)
   print(abs(w @ a))        # 8.0: full array gain toward the beam

Field versus power
------------------

Array factors, element patterns and total patterns are **field** (complex
amplitude) quantities. Power is ``abs(field)**2``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Expects / returns
   * - :func:`~phased_array.linear_to_db`
     - Expects **power**: ``linear_to_db(np.abs(af)**2)``. Passing
       ``np.abs(af)`` halves every dB value.
   * - :func:`~phased_array.compute_directivity`
     - Expects field amplitude (complex or magnitude), not power or dB.
   * - :func:`~phased_array.compute_full_pattern`,
       :func:`~phased_array.compute_pattern_cuts`
     - Return power in dB, normalized to the peak.
   * - :func:`~phased_array.element_pattern`
     - Returns field; ``cos_exp_theta`` is the exponent of the **power**
       pattern (field ``cos(theta)**(cos_exp_theta/2)``).
   * - :func:`~phased_array.ideal_patch_element`,
       :func:`~phased_array.cos_q_polarized_element`
     - Return field; ``cos_exp`` / ``q_theta`` is the exponent of the
       **field**. ``q_theta = cos_exp_theta / 2`` gives the same pattern as
       ``element_pattern``.
   * - :func:`~phased_array.normalize_pattern`
     - Works on linear field values. To normalize a dB pattern, subtract its
       maximum.

Directivity: hemisphere or full sphere
--------------------------------------

:func:`~phased_array.compute_directivity` integrates only over the grid it is
given and assumes no radiation elsewhere. A ground-plane-backed array
radiates into the front hemisphere, so integrate over
``theta_range=(0, pi/2)``. A free-standing array of isotropic elements
radiates equally to both sides, so integrate over the full sphere; the
result is 3 dB lower. Haupt's 6 x 10 example (dx = 0.5, dy = 0.7) gives
23.8 dB over the hemisphere and 20.8 dB over the full sphere.

Pattern cuts
------------

:func:`~phased_array.compute_pattern_cuts` returns two orthogonal cuts through
the beam direction ``(theta0_deg, phi0_deg)``: the ``phi = phi0`` plane, and
the great circle through the beam perpendicular to it. Pass the steering
angles so the second cut passes through the beam peak. Both cuts share one
normalization, so a cut that misses the beam shows its true level. The
``E_plane``/``H_plane`` names are historical; they are E and H planes only when
the element polarization lines up with ``phi0``.

Coupling matrices
-----------------

:func:`~phased_array.mutual_coupling_matrix_measured` returns the normalized
impedance matrix ``C = (I + S)(I - S)^-1`` by default, or ``C = I + S`` with
``formulation='voltage'``. The active-impedance functions
(:func:`~phased_array.active_reflection_coefficient`,
:func:`~phased_array.active_impedance`, :func:`~phased_array.vswr_vs_scan`,
:func:`~phased_array.active_scan_impedance_matrix`) take the same
``formulation`` argument, default ``'impedance'``, and compute the active
reflection coefficient ``(S w)_n / w_n`` with the weights as incident waves.
