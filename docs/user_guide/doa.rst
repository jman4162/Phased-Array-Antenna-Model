Direction-of-Arrival Estimation
===============================

The :mod:`phased_array.doa` module estimates the arrival directions of
plane waves from array snapshots. It covers three spectral estimators
(Bartlett, Capon, MUSIC) for any geometry, two search-free estimators
(root-MUSIC, ESPRIT) for uniform linear arrays, source counting with MDL
and AIC, forward-backward averaging for coherent sources, and the
Cramer-Rao bound for one source on a ULA.

All estimators assume narrowband signals, far-field sources, a known array
geometry and spatially white noise.

Data model
----------

Snapshot ``t`` is the vector of complex element outputs

.. math::

   \mathbf{x}(t) = \sum_i s_i(t)\,\mathbf{a}(\theta_i, \phi_i) + \mathbf{n}(t),
   \qquad a_n = \exp\left(+jk\,\mathbf{r}_n \cdot \mathbf{u}\right)

where :math:`\mathbf{a}` is :func:`~phased_array.array_manifold` (see
:doc:`conventions`). Snapshots are stored as the rows of an
``(n_snapshots, n_elements)`` array, the same layout that
:func:`~phased_array.adaptive_weights_smi` takes. The covariance is
:math:`\mathbf{R} = E[\mathbf{x}\mathbf{x}^H]`, estimated as
``X.T @ X.conj() / K`` by :func:`~phased_array.sample_covariance`.

:func:`~phased_array.generate_snapshots` draws independent unit-power
complex Gaussian sources, scales each by its per-element SNR, and adds unit
complex Gaussian noise. The example below puts two sources 10 deg apart in
front of an 8-element half-wavelength ULA. Its half-power beamwidth is
12.8 deg, so the two sources fall inside one beam.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import phased_array as pa

   geom = pa.create_rectangular_array(8, 1, dx=0.5, dy=0.5)  # ULA along x
   k = pa.wavelength_to_k(1.0)

   X = pa.generate_snapshots(geom, k, sources=[(0, 0), (10, 0)], snr_db=10,
                             n_snapshots=200, seed=0)
   R = pa.sample_covariance(X)
   print(X.shape, R.shape)   # (200, 8) (8, 8)

Angles for a linear array
^^^^^^^^^^^^^^^^^^^^^^^^^

A ULA along x measures only ``u = sin(theta) cos(phi)``. To scan the x-z
plane, pass ``theta`` from -90 to 90 deg with ``phi = 0``: a negative
``theta`` at ``phi = 0`` is the direction ``|theta|`` at ``phi = 180``.
:func:`~phased_array.root_music` and :func:`~phased_array.esprit` report
angles with the same sign.

Spectral estimators
-------------------

Each spectrum takes the covariance, the geometry, ``k`` and scan angles in
degrees. ``theta_deg`` and ``phi_deg`` are broadcast against each other, so
a 1-D ``theta`` with scalar ``phi`` gives a cut, and two meshgrid arrays give
a 2-D map. The returned values are linear and not normalized.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Function
     - Spectrum
     - Behavior
   * - :func:`~phased_array.bartlett_spectrum`
     - :math:`\mathbf{a}^H \mathbf{R}\, \mathbf{a}`
     - Output power of a steered uniform beam. Resolution is one beamwidth.
   * - :func:`~phased_array.capon_spectrum`
     - :math:`1 / (\mathbf{a}^H \mathbf{R}^{-1} \mathbf{a})`
     - MVDR output power. Resolves inside the beamwidth at moderate SNR;
       ``diagonal_loading`` handles a singular R.
   * - :func:`~phased_array.music_spectrum`
     - :math:`1 / \lVert \mathbf{E}_n^H \mathbf{a} \rVert^2`
     - Pseudo-spectrum from the noise subspace :math:`\mathbf{E}_n`. Needs
       the number of sources; peak heights carry no power information.

:func:`~phased_array.find_spectrum_peaks` returns the angles of the
``n_sources`` highest local maxima of a 1-D spectrum, sorted by angle.

.. code-block:: python

   theta = np.arange(-90, 90.25, 0.25)
   spectra = {
       "Bartlett": pa.bartlett_spectrum(R, geom, k, theta, 0),
       "Capon": pa.capon_spectrum(R, geom, k, theta, 0),
       "MUSIC": pa.music_spectrum(R, geom, k, theta, 0, n_sources=2),
   }
   for name, P in spectra.items():
       print(f"{name:8s} peaks: {pa.find_spectrum_peaks(P, theta, 2)}")
   # Bartlett peaks: [-22.25   5.25]   one merged lobe plus a sidelobe
   # Capon    peaks: [0.25 9.75]
   # MUSIC    peaks: [ 0. 10.]

   fig, ax = plt.subplots(figsize=(7, 4))
   for name, P in spectra.items():
       ax.plot(theta, 10 * np.log10(P / P.max()), label=name)
   for t in (0, 10):
       ax.axvline(t, color="k", ls=":", lw=0.8)
   ax.set(xlim=(-60, 60), ylim=(-40, 1), xlabel="Angle from broadside (deg)",
          ylabel="Normalized spectrum (dB)")
   ax.legend()
   plt.show()

Bartlett shows a single lobe centered at 5 deg. Capon splits it with a dip
of about 5 dB, and MUSIC with a dip of about 21 dB.

Root-MUSIC and ESPRIT
---------------------

For a uniform linear array along x, :func:`~phased_array.root_music` and
:func:`~phased_array.esprit` return the angles directly, without a scan grid,
so their accuracy is not limited by a grid step.

- Root-MUSIC writes the MUSIC denominator as a polynomial in
  :math:`z = e^{jkd\sin\theta}` and takes the roots nearest the unit circle.
- ESPRIT (least-squares form) uses the shift between the first and last
  ``N - 1`` elements of the signal subspace.

Both require the rows and columns of R in order of increasing x, the order
``create_rectangular_array(N, 1, ...)`` produces. Reversing the order
mirrors every angle. Spacing above half a wavelength makes
:math:`\sin\theta` ambiguous.

.. code-block:: python

   print("root-MUSIC:", pa.root_music(R, n_sources=2, spacing=0.5))
   print("ESPRIT:    ", pa.esprit(R, n_sources=2, spacing=0.5))
   # root-MUSIC: [0.02457712 9.93107442]
   # ESPRIT:     [-0.04427926  9.91930297]

``spacing`` and ``wavelength`` are in meters. With the default
``wavelength=1.0``, spacing is in wavelengths.

Number of sources
-----------------

MUSIC, root-MUSIC and ESPRIT need the number of sources.
:func:`~phased_array.estimate_num_sources` applies the Wax-Kailath
information criteria to the eigenvalues of R. MDL is consistent as the
snapshot count grows; AIC tends to overestimate.

.. code-block:: python

   print(pa.estimate_num_sources(R, n_snapshots=200, method="mdl"))   # 2

   # At -10 dB per-element SNR the two sources are no longer both detected
   X_weak = pa.generate_snapshots(geom, k, [(0, 0), (10, 0)], -10, 200, seed=0)
   print(pa.estimate_num_sources(pa.sample_covariance(X_weak), 200))  # 1

Coherent sources and forward-backward averaging
-----------------------------------------------

A multipath reflection carries the same waveform as the direct path. Two
such coherent sources produce a rank-one signal covariance, so the
subspace methods see one source. Forward-backward averaging,
``sample_covariance(X, forward_backward=True)``, forms
:math:`(\mathbf{R} + \mathbf{J}\mathbf{R}^*\mathbf{J})/2` with
:math:`\mathbf{J}` the exchange matrix. It is valid for arrays that are
centro-symmetric in the element order (ULAs, rectangular grids). For two
sources it restores rank two unless their relative phase at the array
center is 0 or 180 deg. It decorrelates at most two coherent sources; more
need spatial smoothing, which this module does not provide.

.. code-block:: python

   rng = np.random.default_rng(1)
   s = np.sqrt(10) * (rng.standard_normal(200) + 1j * rng.standard_normal(200)) / np.sqrt(2)
   a1 = pa.array_manifold(k, geom.x, geom.y, 0, 0)
   a2 = pa.array_manifold(k, geom.x, geom.y, 10, 0)
   noise = (rng.standard_normal((200, 8)) + 1j * rng.standard_normal((200, 8))) / np.sqrt(2)
   X_coh = np.outer(s, a1 + 1j * a2) + noise   # same waveform, 90 deg apart

   for fb in (False, True):
       R_c = pa.sample_covariance(X_coh, forward_backward=fb)
       eigs = np.linalg.eigvalsh(R_c)[::-1][:3]
       peaks = pa.find_spectrum_peaks(
           pa.music_spectrum(R_c, geom, k, theta, 0, n_sources=2), theta, 2)
       print(f"FB={fb}: eigenvalues {np.round(eigs, 1)}, MUSIC peaks {peaks}")
   # FB=False: eigenvalues [133.1   1.3   1.1], MUSIC peaks [ 4.75 31.5 ]
   # FB=True: eigenvalues [92.9 41.3  1.1], MUSIC peaks [ 0. 10.]

Without averaging the second eigenvalue sits at the noise floor and MUSIC
finds one merged peak at 4.75 deg. With averaging the second eigenvalue
rises to 41 and MUSIC recovers both angles.

Accuracy and the Cramer-Rao bound
---------------------------------

:func:`~phased_array.crb_ula_single_source` gives the standard deviation
allowed by the Cramer-Rao bound for one source on a ULA,

.. math::

   \mathrm{var}(\psi) \ge \frac{6}{K\,\mathrm{SNR}\,N(N^2-1)}, \qquad
   \psi = kd\sin\theta,

converted to theta through :math:`\partial\psi/\partial\theta = kd\cos\theta`.
SNR is per element and per snapshot, as in
:func:`~phased_array.generate_snapshots`. The bound falls as
:math:`1/\sqrt{K}` and :math:`1/\sqrt{\mathrm{SNR}}`.

.. code-block:: python

   theta0, snr_db, n_snap = 10.0, 20.0, 100
   rng = np.random.default_rng(2)
   err_rm, err_es = [], []
   for _ in range(100):
       X1 = pa.generate_snapshots(geom, k, [(theta0, 0)], snr_db, n_snap, seed=rng)
       R1 = pa.sample_covariance(X1)
       err_rm.append(pa.root_music(R1, 1, 0.5)[0] - theta0)
       err_es.append(pa.esprit(R1, 1, 0.5)[0] - theta0)

   crb = pa.crb_ula_single_source(8, 0.5, snr_db, n_snap, theta0)
   print(f"CRB:        {crb:.4f} deg")                                # 0.0202
   print(f"root-MUSIC: {np.sqrt(np.mean(np.square(err_rm))):.4f} deg")  # 0.0201
   print(f"ESPRIT:     {np.sqrt(np.mean(np.square(err_es))):.4f} deg")  # 0.0229

Root-MUSIC reaches the bound for a single source at high SNR. LS-ESPRIT
has a standard deviation about :math:`\sqrt{N(N+1)/(6(N-1))}` times the
bound, 1.31 for N = 8 (4000 trials at 30 dB give 1.33). With 100 trials
the RMSE estimate has a relative uncertainty of about 7 %, so the ratios
above change with the seed.

Planar arrays
-------------

The spectral estimators work for any geometry. For a planar array, scan a
theta-phi grid and pick the 2-D local maxima. A planar array in the x-y
plane cannot tell ``theta`` from ``180 - theta``, so scan the front
hemisphere only.

.. code-block:: python

   from scipy.ndimage import maximum_filter

   geom2 = pa.create_rectangular_array(6, 6, dx=0.5, dy=0.5)
   X2 = pa.generate_snapshots(geom2, k, [(20, 30), (40, 200)], snr_db=5,
                              n_snapshots=100, seed=3)
   R2 = pa.sample_covariance(X2)
   n_src = pa.estimate_num_sources(R2, 100)        # 2

   th, ph = np.meshgrid(np.arange(0, 90.5, 0.5), np.arange(0, 360, 1.0),
                        indexing="ij")
   P2 = pa.music_spectrum(R2, geom2, k, th, ph, n_sources=n_src)

   is_peak = P2 == maximum_filter(P2, size=5, mode=("nearest", "wrap"))
   top = np.argsort(P2[is_peak])[::-1][:n_src]
   for t, p in zip(th[is_peak][top], ph[is_peak][top]):
       print(f"theta = {t:.1f} deg, phi = {p:.0f} deg")
   # theta = 40.0 deg, phi = 200 deg
   # theta = 20.0 deg, phi = 30 deg

Choosing a method
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Method
     - Geometry
     - Notes
   * - Bartlett
     - Any
     - No model order needed; resolution limited to one beamwidth
   * - Capon
     - Any
     - No model order needed; resolves closer sources than Bartlett as SNR
       rises; use diagonal loading when K is small
   * - MUSIC
     - Any
     - Needs the source count and non-coherent sources (or FB averaging);
       resolution limited by the scan grid
   * - Root-MUSIC
     - ULA
     - Needs the source count; no grid; reaches the CRB for one source at
       high SNR
   * - ESPRIT
     - ULA
     - Needs the source count; no grid or polynomial rooting; standard
       deviation 1.31 times the CRB for one source on 8 elements
