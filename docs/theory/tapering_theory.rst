Tapering Theory
===============

Amplitude tapering (windowing) is a fundamental technique for sidelobe control
in phased arrays. This section covers the theory behind common taper functions.

Why Taper?
----------

A uniform amplitude distribution has the narrowest main beam but relatively
high sidelobes (~-13 dB for a rectangular aperture). The sidelobes arise
from the sharp discontinuity at the aperture edges.

Tapering smooths the amplitude distribution, reducing sidelobes at the cost of:

1. **Increased beamwidth**: Main lobe becomes wider
2. **Reduced directivity**: Less aperture efficiency
3. **Reduced gain**: Peak gain decreases

The Fourier Relationship
------------------------

The array factor is the Fourier transform of the aperture distribution:

.. math::

   AF(u) = \int_{-L/2}^{L/2} a(x) e^{jkux} \, dx

where :math:`a(x)` is the amplitude distribution.

For discrete arrays:

.. math::

   AF(u) = \sum_{n=1}^{N} a_n e^{jkx_n u}

Sharp edges in :math:`a(x)` produce high-frequency components (sidelobes).
Smooth tapers reduce these high frequencies.

Aperture Efficiency
-------------------

The aperture efficiency (taper efficiency) is:

.. math::

   \eta = \frac{\left|\sum_n a_n\right|^2}{N \sum_n |a_n|^2}

For uniform weighting (:math:`a_n = 1`): :math:`\eta = 1`

For any taper: :math:`\eta < 1`

The directivity loss in dB:

.. math::

   \text{Loss} = 10 \log_{10}(\eta)

Taylor Taper
------------

The Taylor distribution is designed for a specified sidelobe level with
controlled sidelobe decay. It produces :math:`\bar{n}` nearly-equal sidelobes
before rolling off.

The continuous distribution:

.. math::

   a(x) = 1 + 2\sum_{m=1}^{\bar{n}-1} F(m, A, \bar{n}) \cos\left(\frac{2\pi m x}{L}\right)

where A is determined by the desired sidelobe level:

.. math::

   \text{SLL} = -20\log_{10}(\cosh(\pi A))

**Properties:**

- Provides best efficiency for a given sidelobe level
- Smooth rolloff (no discontinuities)
- :math:`\bar{n}` controls sidelobe behavior (higher = more equal-level sidelobes)

**Typical values:**

Computed with ``pa.taylor_taper_1d`` for a 1000-element line array. The
beamwidth factor is the half-power beamwidth relative to uniform weighting.

.. list-table::
   :header-rows: 1

   * - SLL (dB)
     - nbar
     - Efficiency
     - Beamwidth Factor
   * - -25
     - 3
     - 90%
     - 1.20
   * - -30
     - 4
     - 85%
     - 1.27
   * - -35
     - 5
     - 81%
     - 1.34
   * - -40
     - 6
     - 77%
     - 1.41

Chebyshev (Dolph-Chebyshev) Taper
---------------------------------

The Chebyshev distribution produces equi-ripple sidelobes (all at the same
level). This provides the minimum beamwidth for a given sidelobe level.

The distribution is related to Chebyshev polynomials:

.. math::

   a_n = \frac{1}{N} \sum_{k=0}^{N-1} T_{N-1}(x_0 \cos(\pi k/N)) e^{j2\pi nk/N}

where :math:`x_0` is determined by the sidelobe level:

.. math::

   x_0 = \cosh\left(\frac{1}{N-1}\cosh^{-1}(10^{\text{SLL}/20})\right)

**Properties:**

- All sidelobes equal (equi-ripple)
- Narrowest beamwidth for given SLL
- Has discontinuities at aperture edges
- Less efficient than Taylor for same SLL

Hamming and Hanning Windows
---------------------------

Classic signal processing windows, simple to implement:

**Hanning (Hann):**

.. math::

   a(x) = 0.5 - 0.5\cos\left(\frac{2\pi x}{L}\right)

- First sidelobe: -31 dB
- Sidelobe rolloff: -18 dB/octave

**Hamming:**

.. math::

   a(x) = 0.54 - 0.46\cos\left(\frac{2\pi x}{L}\right)

- First sidelobe: -42 dB
- Sidelobe rolloff: -6 dB/octave (slower than Hanning)

Cosine Tapers
-------------

**Cosine (Sine) Taper:**

.. math::

   a(x) = \sin\left(\frac{\pi x}{L}\right)

**Cosine-on-Pedestal:**

.. math::

   a(x) = p + (1-p)\sin\left(\frac{\pi x}{L}\right)

where p is the pedestal level (edge amplitude).

Gaussian Taper
--------------

.. math::

   a(x) = \exp\left(-\frac{x^2}{2\sigma^2}\right)

**Properties:**

- Very low sidelobes (theoretically none for infinite Gaussian)
- Smooth decay in all regions
- Practical truncation introduces sidelobes

2D Tapers
---------

For 2D arrays, tapers can be applied as:

**Separable (product):**

.. math::

   a(x, y) = a_x(x) \cdot a_y(y)

**Circularly symmetric:**

.. math::

   a(r) = f\left(\sqrt{x^2 + y^2}\right)

Separable tapers are simpler. Their diagonal-plane sidelobes are roughly the
product of the two principal-plane levels (near -60 dB for a -30 dB taper),
and their efficiency is the product of the two 1D efficiencies. The library
uses separable tapers:

.. code-block:: python

   import numpy as np
   import phased_array as pa

   Nx, Ny = 16, 16

   # 2D Taylor = product of 1D Taylors
   taper_2d = pa.taylor_taper_2d(Nx, Ny, sidelobe_dB=-30)

   def taper_efficiency(a):
       return np.abs(a.sum())**2 / (a.size * np.sum(np.abs(a)**2))

   eta_1d = taper_efficiency(pa.taylor_taper_1d(Nx, sidelobe_dB=-30))
   print(f"1D efficiency: {eta_1d:.3f}, squared: {eta_1d**2:.3f}")
   print(f"2D efficiency: {taper_efficiency(taper_2d):.3f}")

Taper Selection Guidelines
--------------------------

1. **Taylor**: Best general-purpose choice for radar and communications

2. **Chebyshev**: When minimum beamwidth is critical and equi-ripple sidelobes
   are acceptable

3. **Hamming**: Good balance, simple implementation

4. **Gaussian**: When very low far-out sidelobes are needed (low probability
   of intercept)

5. **Cosine-on-pedestal**: When moderate sidelobe control with high efficiency
   is needed

Combining Tapers with Steering
------------------------------

Amplitude taper and phase steering are multiplicative:

.. code-block:: python

   geom = pa.create_rectangular_array(Nx, Ny, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Steering weights (complex, unit magnitude)
   steer = pa.steering_vector(k, geom.x, geom.y, theta0_deg=30, phi0_deg=0)

   # Amplitude taper (real, varying magnitude)
   taper = pa.taylor_taper_2d(Nx, Ny, sidelobe_dB=-30)

   # Combined weights
   weights = steer * taper
