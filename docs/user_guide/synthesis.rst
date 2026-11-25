Pattern Synthesis
=================

The tapers in :doc:`beamforming` are fixed windows chosen for a sidelobe
level. The functions on this page start from a desired array factor and
return the element weights that produce it, for a linear array along x.
They follow the package convention (see :doc:`conventions`):

.. math::

   AF(u) = \sum_n w_n \exp\left(+jk\,x_n u\right), \qquad u = \sin\theta

in the x-z plane. The uniform-array functions take the spacing ``d`` in
wavelengths and return weights for elements centered at the origin,
``x_n = (n - (N - 1)/2) d``, in the same order as
``create_rectangular_array(N, 1, d, d)``. The methods and examples follow
Haupt, *Antenna Arrays: A Computational Approach* (2010), Sections 3.1
and 3.2.1.

.. code-block:: python

   import numpy as np
   import phased_array as pa

   k = pa.wavelength_to_k(1.0)

   def cut_uv(u, geom, w):
       """Array factor of a linear array along x at direction cosines u."""
       return pa.array_factor_uv(u, np.zeros_like(u), geom.x, geom.y, w, k)

Binomial Taper
--------------

The binomial taper uses the coefficients ``C(N-1, i)`` as amplitude
weights. Its array factor is :math:`(1 + z)^{N-1}` with
:math:`z = e^{jkdu}`, so all ``N - 1`` zeros sit at :math:`kdu = \pi`. At
``d = 0.5`` wavelength the pattern falls monotonically from broadside to
the null at ``u = +/-1`` and has no sidelobes.

.. code-block:: python

   w = pa.binomial_taper_1d(5)
   print(w * 6)                                       # [1. 4. 6. 4. 1.]
   print(round(pa.compute_taper_efficiency(w), 3))    # 0.731

   geom = pa.create_rectangular_array(9, 1, 0.5, 0.5)
   angles, cut, _ = pa.compute_pattern_cuts(
       geom.x, geom.y, pa.binomial_taper_1d(9), k, n_points=3601)
   print(pa.compute_peak_sidelobe_level(angles, cut))      # -inf
   print(round(pa.compute_half_power_beamwidth(angles, cut), 1))  # 21.4 deg

   angles, cut, _ = pa.compute_pattern_cuts(geom.x, geom.y, np.ones(9), k,
                                            n_points=3601)
   print(round(pa.compute_half_power_beamwidth(angles, cut), 1))  # 11.4 deg

The price is efficiency and beamwidth: the 9-element binomial array has a
taper efficiency of 0.566 (Haupt Table 3.2) and a 21.4 deg half-power
beamwidth against 11.4 deg for uniform weights. Element errors create
sidelobes anyway, so tapers with a finite sidelobe level near the error
floor (Taylor, Chebyshev) are usually the better choice. Table 3.2 lists
the efficiency of the 6-element array as 0.667; the coefficients
1, 5, 10, 10, 5, 1 give :math:`32^2 / (6 \cdot 252) = 0.677`.

Woodward-Lawson Synthesis
-------------------------

A uniform array steered to :math:`u_m` has the array factor
:math:`\sin[Nkd(u - u_m)/2] / (N \sin[kd(u - u_m)/2])`, which peaks at
:math:`u_m` and is zero at :math:`u_m + p\lambda/(Nd)` for every integer
:math:`p` that is not a multiple of N. Beams at

.. math::

   u_m = \frac{m\lambda}{Nd}

are therefore orthogonal. :func:`~phased_array.woodward_lawson_weights`
samples the desired pattern at these points, :math:`b_m = D(u_m)`, and
superposes the beams:

.. math::

   w_n = \frac{1}{N}\sum_m b_m \exp(-jk\,x_n u_m).

The synthesized pattern equals ``desired`` exactly at every sample point
and interpolates between them. The samples are the :math:`u_m` with
:math:`|u_m| \le 1`. For ``d < 0.5`` there are
:math:`2\lfloor Nd/\lambda \rfloor + 1 \le N` of them. For
``d >= 0.5`` visible space holds a full period of the array factor, so
one period is sampled, :math:`-\lambda/(2d) \le u_m < \lambda/(2d)`,
which is exactly N samples.

Haupt's example is a 16-element array with :math:`|AF| = 1` for
:math:`|u| \le 0.5`. The weights match Table 3.1:

.. code-block:: python

   def sector(half_width):
       return lambda u: (np.abs(u) <= half_width + 1e-12).astype(float)

   w = pa.woodward_lawson_weights(sector(0.5), 16, spacing=0.5)
   print(np.round(w.real[8:], 4))   # center element to edge element
   # [ 0.4929  0.1015 -0.1269 -0.0097  0.0805 -0.0206 -0.0576  0.0398]

A 20-element flat-top sector beam over :math:`|u| \le 0.3` has 7 samples
inside the sector, beam spacing :math:`\lambda/(Nd) = 0.1`, and its first
zero-valued sample at :math:`|u| = 0.4`:

.. code-block:: python

   import matplotlib.pyplot as plt

   geom = pa.create_rectangular_array(20, 1, 0.5, 0.5)
   w, u_m = pa.woodward_lawson_weights(sector(0.3), 20, return_samples=True)

   u = np.linspace(-1, 1, 4001)
   level_dB = 20 * np.log10(np.abs(cut_uv(u, geom, w)) + 1e-12)
   inside = level_dB[np.abs(u) <= 0.3]
   print(f"passband ripple {inside.max() - inside.min():.2f} dB")       # 1.31 dB
   print(f"peak for |u| >= 0.4: {level_dB[np.abs(u) >= 0.4].max():.1f} dB")  # -15.5 dB

   samples_dB = 20 * np.log10(np.abs(cut_uv(u_m, geom, w)) + 1e-12)
   plt.plot(u, level_dB, label="synthesized")
   plt.plot(u_m, samples_dB, "o", label="samples")
   plt.xlabel("u")
   plt.ylabel("|AF| (dB)")
   plt.ylim(-40, 5)
   plt.legend()
   plt.grid(True)

The ripple comes from the step at the sector edge (Gibbs ripple). Doubling
the array to 40 elements narrows the transition and lowers the
out-of-sector level for :math:`|u| \ge 0.4` to -20.9 dB, while the ripple
stays near 1.6 dB.

Least-Squares Synthesis
-----------------------

:func:`~phased_array.least_squares_weights` writes the array factor at M
sample points as a linear system :math:`A w = D` with
:math:`A_{in} = \exp(jk\,x_n u_i)` (Haupt eq. 3.9) and minimizes

.. math::

   \sum_i c_i \left|AF(u_i) - D_i\right|^2 + \mu \lVert w \rVert^2

with optional sample weights :math:`c_i` and Tikhonov regularization
:math:`\mu`. It solves the problem with :func:`numpy.linalg.lstsq`. The
target is complex: for a magnitude-only target, pick a phase (zero phase
below). Repeating Haupt's 16-element example reproduces the least-squares
column of Table 3.1 with 21 samples at :math:`u = 2m/21`,
:math:`m = -10 \ldots 10` (the text says 20 samples):

.. code-block:: python

   u21 = 2 * np.arange(-10, 11) / 21
   w = pa.least_squares_weights(u21, sector(0.5)(u21), n_elements=16)
   print(np.round(w.real[8:], 4))
   # [ 0.4671  0.1334 -0.1077 -0.0476  0.0688  0.0237 -0.0551 -0.0118]

With N samples at the Woodward-Lawson points the system is square and
the result equals :func:`~phased_array.woodward_lawson_weights`.

Shaped beam
^^^^^^^^^^^

A cosecant-squared pattern (Skolnik, *Introduction to Radar Systems*)
has power gain proportional to :math:`\csc^2\theta` between
:math:`\theta_1` and :math:`\theta_2`, which returns constant power from
targets at constant altitude. The field target below is
:math:`\sin\theta_1 / \sin\theta = u_1/u` on that interval, flat at the
peak from broadside to :math:`u_1`, and zero elsewhere:

.. code-block:: python

   u1, u2 = np.sin(np.deg2rad([10.0, 45.0]))

   def csc2_field(u):
       d = np.zeros_like(u)
       d[(u >= 0) & (u <= u1)] = 1.0
       shaped = (u > u1) & (u <= u2)
       d[shaped] = u1 / u[shaped]
       return d

   u = np.linspace(-1, 1, 401)
   target = csc2_field(u)
   for n in (8, 16, 32):
       geom = pa.create_rectangular_array(n, 1, 0.5, 0.5)
       w = pa.least_squares_weights(u, target, n_elements=n)
       err = np.linalg.norm(cut_uv(u, geom, w) - target) / np.linalg.norm(target)
       print(f"N = {n:2d}: relative RMS error {err:.3f}")
   # N =  8: relative RMS error 0.323
   # N = 16: relative RMS error 0.201
   # N = 32: relative RMS error 0.146

The jumps at ``u = 0`` and ``u = u2`` account for most of the error. The
``weights`` argument sets the relative importance of each sample, for
example to hold down the region below the horizon.

Regularization
^^^^^^^^^^^^^^

When the samples do not constrain every degree of freedom, the
unregularized solution can have very large weights. At ``d = 0.25``
wavelength, visible space covers half the period of the array factor and
the fit uses the unconstrained invisible region (a superdirective
solution). A small :math:`\mu` trades a little pattern error for weights
of reasonable size:

.. code-block:: python

   geom = pa.create_rectangular_array(16, 1, 0.25, 0.25)
   for mu in (0.0, 1e-3, 1e-1):
       w = pa.least_squares_weights(u, target, n_elements=16, spacing=0.25,
                                    regularization=mu)
       err = np.linalg.norm(cut_uv(u, geom, w) - target) / np.linalg.norm(target)
       print(f"mu = {mu:g}: |w| = {np.linalg.norm(w):.3g}, error {err:.3f}")
   # mu = 0: |w| = 4.19e+03, error 0.243
   # mu = 0.001: |w| = 7.76, error 0.261
   # mu = 0.1: |w| = 1.36, error 0.269

Arbitrary element positions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass ``positions`` (meters) and ``k`` instead of ``n_elements`` for a
nonuniform linear array. With uniform positions the result equals the
``n_elements`` path.

.. code-block:: python

   wavelength = 0.03                                  # 10 GHz
   k10 = pa.wavelength_to_k(wavelength)
   x = np.array([-2.6, -1.9, -1.3, -0.8, -0.3, 0.3, 0.8, 1.3, 1.9, 2.6]) * wavelength

   w = pa.least_squares_weights(u, target, positions=x, k=k10)
   af = pa.array_factor_uv(u, np.zeros_like(u), x, np.zeros_like(x), w, k10)
   print(f"relative RMS error {np.linalg.norm(af - target) / np.linalg.norm(target):.3f}")  # 0.249

Planar Arrays
-------------

All three functions are one-dimensional. A separable planar design is the
outer product of two linear designs; ``np.outer(wx, wy).ravel()`` matches
the element order of ``create_rectangular_array(Nx, Ny, ...)``:

.. code-block:: python

   wx = pa.woodward_lawson_weights(sector(0.3), 20)   # sector beam in x
   wy = pa.binomial_taper_1d(8)                       # no sidelobes in y
   geom = pa.create_rectangular_array(20, 8, 0.5, 0.5)
   weights = np.outer(wx, wy).ravel()

   theta_deg, cut_x, cut_y = pa.compute_pattern_cuts(geom.x, geom.y, weights, k)

Choosing a Method
-----------------

- **Binomial**: no sidelobes at half-wavelength spacing, low efficiency
  and a wide beam. Mostly of reference value.
- **Woodward-Lawson**: closed form, exact at N sample points, uniform
  spacing only. Suited to sector and other shaped beams.
- **Least squares**: any sample set, sample weighting, regularization and
  arbitrary element positions. The weights can vary strongly from
  element to element; check them before building hardware, and add
  regularization when they grow large.
