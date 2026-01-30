Array Fundamentals
==================

This section derives the key equations for phased array analysis.

Array Factor Derivation
-----------------------

Consider an array of N identical elements located at positions
:math:`\vec{r}_n = (x_n, y_n, z_n)`. For a plane wave arriving from direction
:math:`(\theta, \phi)`, the array factor is:

.. math::

   AF(\theta, \phi) = \sum_{n=1}^{N} w_n \exp\left(jk \vec{r}_n \cdot \hat{u}\right)

where:

- :math:`w_n` is the complex weight for element n
- :math:`k = 2\pi/\lambda` is the wavenumber
- :math:`\hat{u} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)` is the unit vector toward the observation direction

For a planar array in the xy-plane (:math:`z_n = 0`):

.. math::

   AF(\theta, \phi) = \sum_{n=1}^{N} w_n \exp\left(jk(x_n u + y_n v)\right)

where :math:`u = \sin\theta\cos\phi` and :math:`v = \sin\theta\sin\phi` are
direction cosines.

Uniform Linear Array
--------------------

For a uniform linear array (ULA) along the x-axis with spacing d:

.. math::

   x_n = (n - 1) d, \quad n = 1, 2, \ldots, N

With uniform weights (:math:`w_n = 1`):

.. math::

   AF(u) = \sum_{n=0}^{N-1} \exp(jknd \cdot u) = \frac{\sin(Nkdu/2)}{\sin(kdu/2)}

This is the classical array factor with:

- Main beam at :math:`u = 0` (broadside)
- First null at :math:`u = \pm \lambda/(Nd)`
- Half-power beamwidth :math:`\approx 0.886\lambda/(Nd)`

Beam Steering
-------------

To steer the main beam to direction :math:`(u_0, v_0)`, apply progressive
phase shifts:

.. math::

   w_n = \exp\left(-jk(x_n u_0 + y_n v_0)\right)

This cancels the phase variation at the desired direction, making all elements
add coherently. The resulting array factor becomes:

.. math::

   AF(u, v) = \sum_{n=1}^{N} \exp\left(jk[x_n(u-u_0) + y_n(v-v_0)]\right)

The pattern shifts in (u, v) space to center on :math:`(u_0, v_0)`.

Pattern Multiplication
----------------------

The total radiation pattern of an array is the product of:

1. **Element pattern** :math:`E_e(\theta, \phi)`: Individual element's directivity
2. **Array factor** :math:`AF(\theta, \phi)`: Interference pattern from element positions

.. math::

   E_{total}(\theta, \phi) = E_e(\theta, \phi) \cdot AF(\theta, \phi)

This separability assumes:

- Identical elements
- Negligible mutual coupling
- Elements small compared to wavelength

Grating Lobes
-------------

Grating lobes are additional main beams that appear when the array factor
is periodic in :math:`(u, v)` space. For a rectangular grid with spacing
:math:`(d_x, d_y)`, grating lobes appear at:

.. math::

   (u_{gl}, v_{gl}) = \left(u_0 + \frac{m\lambda}{d_x}, v_0 + \frac{n\lambda}{d_y}\right)

for integers m, n.

A grating lobe enters the visible region (:math:`u^2 + v^2 \leq 1`) when:

.. math::

   d > \frac{\lambda}{1 + |\sin\theta_{max}|}

For :math:`d = \lambda/2`, grating lobes stay outside the visible region
for all scan angles.

Directivity
-----------

Array directivity is the ratio of peak radiation intensity to average
intensity:

.. math::

   D = \frac{4\pi |AF_{max}|^2}{\int_0^{2\pi}\int_0^{\pi} |AF(\theta,\phi)|^2 \sin\theta \, d\theta \, d\phi}

For a uniform rectangular array with N elements at :math:`\lambda/2` spacing:

.. math::

   D_{max} \approx \frac{4\pi A}{\lambda^2} = \pi N

where A is the physical aperture area.

Beamwidth
---------

The half-power beamwidth (HPBW) for a uniform rectangular aperture of
dimensions :math:`L_x \times L_y`:

.. math::

   \theta_{HPBW} \approx \frac{0.886\lambda}{L} \text{ (radians)}

For an N-element array with spacing d:

.. math::

   \theta_{HPBW} \approx \frac{0.886\lambda}{Nd} = \frac{0.886}{N} \cdot \frac{\lambda}{d}

At :math:`\lambda/2` spacing:

.. math::

   \theta_{HPBW} \approx \frac{1.77}{N} \text{ radians} = \frac{101°}{N}

Scan Loss
---------

When the beam is steered away from broadside, two effects reduce gain:

1. **Projected aperture**: Effective area decreases as :math:`\cos\theta`
2. **Element pattern**: Element gain typically decreases off-axis

Combined scan loss:

.. math::

   L_{scan}(\theta) \approx \cos^n\theta

where n depends on the element pattern (typically 1.3-1.5 for patch elements).

FFT-Based Computation
---------------------

For a uniform rectangular array, the array factor can be computed efficiently
using the 2D FFT. If weights are arranged on a grid :math:`W[m, n]`:

.. math::

   AF(u, v) = \text{FFT2}\{W[m, n]\}

This is :math:`O(N \log N)` instead of :math:`O(N \cdot M)` for direct
computation over M observation angles.

The FFT output corresponds to direction cosines:

.. math::

   u_k = \frac{k \lambda}{N_x d_x}, \quad v_l = \frac{l \lambda}{N_y d_y}
