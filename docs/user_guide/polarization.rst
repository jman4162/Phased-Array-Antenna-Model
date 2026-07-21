Polarization
============

This guide covers polarization analysis for phased array antennas, including
Jones vectors, Stokes parameters, axial ratio calculations, and Ludwig-3
co/cross-pol decomposition.

Introduction to Polarization
----------------------------

The polarization of an electromagnetic wave describes the orientation and
behavior of the electric field vector as the wave propagates. For antenna
applications, understanding polarization is critical for:

- Maximizing power transfer between transmit and receive antennas
- Minimizing cross-polarization interference
- Designing dual-polarized arrays for MIMO and polarimetric radar

Jones Vectors
-------------

Jones vectors provide a compact mathematical representation of polarization
state, describing the amplitude and phase of two orthogonal electric field
components.

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Linear horizontal polarization
   j_h = pa.jones_vector(Ex=1.0, Ey=0.0)
   print(f"Horizontal: {j_h}")

   # Linear vertical polarization
   j_v = pa.jones_vector(Ex=0.0, Ey=1.0)
   print(f"Vertical: {j_v}")

   # Right-hand circular polarization (RHCP)
   j_rhcp = pa.jones_vector(Ex=1.0, Ey=1.0, phase_diff=-np.pi/2)
   print(f"RHCP: {j_rhcp}")

   # Left-hand circular polarization (LHCP)
   j_lhcp = pa.jones_vector(Ex=1.0, Ey=1.0, phase_diff=np.pi/2)
   print(f"LHCP: {j_lhcp}")

   # 45-degree linear polarization
   j_45 = pa.jones_vector(Ex=1.0, Ey=1.0, phase_diff=0.0)
   print(f"45-degree linear: {j_45}")

Common polarization states:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Polarization
     - Ex
     - Ey
     - Phase Diff
   * - Horizontal
     - 1
     - 0
     - -
   * - Vertical
     - 0
     - 1
     - -
   * - +45 Linear
     - 1
     - 1
     - 0
   * - RHCP
     - 1
     - 1
     - -90 deg
   * - LHCP
     - 1
     - 1
     - +90 deg

Stokes Parameters
-----------------

Stokes parameters provide a complete description of polarization state,
including partially polarized light. They are especially useful for
incoherent measurements.

.. code-block:: python

   # Compute Stokes parameters
   j = pa.jones_vector(1.0, 1.0, phase_diff=-np.pi/2)  # RHCP
   S0, S1, S2, S3 = pa.stokes_parameters(j)

   print(f"S0 (total intensity): {S0:.3f}")
   print(f"S1 (H-V preference): {S1:.3f}")
   print(f"S2 (+45/-45 preference): {S2:.3f}")
   print(f"S3 (R-L preference): {S3:.3f}")

   # For RHCP: S0=2, S1=0, S2=0, S3=-2

**Stokes parameter interpretation:**

- **S0**: Total intensity
- **S1**: Preference for horizontal (>0) vs vertical (<0)
- **S2**: Preference for +45 deg (>0) vs -45 deg (<0)
- **S3**: Preference for RHCP (<0) vs LHCP (>0)

Axial Ratio
-----------

The axial ratio (AR) is the ratio of major to minor axes of the polarization
ellipse:

- AR = 1: Circular polarization
- AR = infinity: Linear polarization
- 1 < AR < infinity: Elliptical polarization

.. code-block:: python

   # Circular polarization: AR = 1
   j_circ = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
   ar_circ = pa.axial_ratio(j_circ)
   print(f"Circular AR: {ar_circ:.2f}")  # 1.00

   # Linear polarization: AR = infinity
   j_lin = pa.jones_vector(1.0, 0.0)
   ar_lin = pa.axial_ratio(j_lin)
   print(f"Linear AR: {ar_lin}")  # inf

   # Elliptical (3 dB axial ratio)
   j_ellip = pa.jones_vector(1.0, 0.5, phase_diff=np.pi/2)
   ar_ellip = pa.axial_ratio(j_ellip)
   print(f"Elliptical AR: {ar_ellip:.2f}")

   # Convert to dB
   ar_dB = 20 * np.log10(ar_ellip)
   print(f"AR in dB: {ar_dB:.2f} dB")

Tilt Angle
----------

The tilt angle is the orientation of the polarization ellipse major axis
relative to the horizontal.

.. code-block:: python

   # Horizontal: tilt = 0
   j_h = pa.jones_vector(1.0, 0.0)
   tilt_h = pa.tilt_angle(j_h)
   print(f"Horizontal tilt: {np.rad2deg(tilt_h):.1f} deg")  # 0.0

   # 45-degree linear: tilt = 45 deg
   j_45 = pa.jones_vector(1.0, 1.0, phase_diff=0.0)
   tilt_45 = pa.tilt_angle(j_45)
   print(f"45-deg linear tilt: {np.rad2deg(tilt_45):.1f} deg")  # 45.0

Polarization Loss Factor
------------------------

When the antenna polarization doesn't match the incident wave polarization,
power is lost. The polarization loss factor (PLF) quantifies this loss.

.. code-block:: python

   # Matched polarizations: PLF = 1 (no loss)
   j_ant = pa.jones_vector(1.0, 0.0)  # H-pol antenna
   j_inc = pa.jones_vector(1.0, 0.0)  # H-pol wave
   plf = pa.polarization_loss_factor(j_ant, j_inc)
   print(f"Matched PLF: {plf:.2f}")  # 1.00

   # Orthogonal polarizations: PLF = 0 (total loss)
   j_ant = pa.jones_vector(1.0, 0.0)  # H-pol antenna
   j_inc = pa.jones_vector(0.0, 1.0)  # V-pol wave
   plf = pa.polarization_loss_factor(j_ant, j_inc)
   print(f"Orthogonal PLF: {plf:.2f}")  # 0.00

   # Circular antenna receiving linear: PLF = 0.5 (3 dB loss)
   j_circ = pa.jones_vector(1.0, 1.0, phase_diff=np.pi/2)
   j_lin = pa.jones_vector(1.0, 0.0)
   plf = pa.polarization_loss_factor(j_circ, j_lin)
   print(f"Circular/Linear PLF: {plf:.2f}")  # 0.50
   print(f"Loss in dB: {10*np.log10(plf):.2f} dB")  # -3.01 dB

Cross-Polarization Discrimination
---------------------------------

Cross-polarization discrimination (XPD) measures how well an antenna
distinguishes between co-polarized and cross-polarized signals.

.. code-block:: python

   # Perfect match: high XPD
   j_ref = pa.jones_vector(1.0, 0.0)
   j_act = pa.jones_vector(1.0, 0.0)
   xpd = pa.cross_pol_discrimination(j_ref, j_act)
   print(f"Perfect match XPD: {xpd:.1f} dB")  # Very high

   # Small cross-pol component
   j_ref = pa.jones_vector(1.0, 0.0)  # Desired: H-pol
   j_act = pa.jones_vector(1.0, 0.1)  # Actual: H-pol with 10% V-pol
   xpd = pa.cross_pol_discrimination(j_ref, j_act)
   print(f"10% cross-pol XPD: {xpd:.1f} dB")  # ~20 dB

Ludwig-3 Co/Cross-Pol Decomposition
-----------------------------------

The Ludwig-3 definition is the standard for separating antenna far-field
patterns into co-polar and cross-polar components. It's based on aligning
the reference polarization with the principal planes.

.. code-block:: python

   # Simulate pattern data (theta/phi field components)
   theta = np.linspace(0, np.pi/2, 91)
   phi = np.linspace(0, 2*np.pi, 181)
   theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

   # Example: dipole-like pattern
   E_theta = np.cos(theta_grid) * np.cos(phi_grid)
   E_phi = -np.sin(phi_grid)

   # Decompose into Ludwig-3 co/cross-pol
   E_co, E_cross = pa.ludwig3_decomposition(theta_grid, phi_grid, E_theta, E_phi)

   # At phi=0, E_theta is co-pol
   print(f"At phi=0: E_co = E_theta, E_cross = E_phi")

   # Compute co-pol and cross-pol patterns
   co_pattern_dB = 20 * np.log10(np.abs(E_co) + 1e-10)
   cross_pattern_dB = 20 * np.log10(np.abs(E_cross) + 1e-10)

Using the convenience functions:

.. code-block:: python

   # Get just co-pol component
   E_co = pa.co_pol_pattern(theta_grid, phi_grid, E_theta, E_phi,
                            reference_pol='ludwig3')

   # Get just cross-pol component
   E_cross = pa.cross_pol_pattern(theta_grid, phi_grid, E_theta, E_phi,
                                  reference_pol='ludwig3')

   # Alternative reference polarizations
   E_co_theta = pa.co_pol_pattern(theta_grid, phi_grid, E_theta, E_phi,
                                  reference_pol='theta')  # E_theta as co-pol

Practical Applications
----------------------

Dual-Polarized Array Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Create array geometry
   geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)

   # Simulate H-pol and V-pol element responses
   weights_h = pa.steering_vector(k, geom.x, geom.y, theta0_deg=20, phi0_deg=0)
   weights_v = pa.steering_vector(k, geom.x, geom.y, theta0_deg=20, phi0_deg=0)

   # Apply different tapers to each polarization if needed
   taper = pa.taylor_taper_2d(8, 8, sidelobe_dB=-25)
   weights_h *= taper
   weights_v *= taper

   # Analyze cross-pol isolation between ports

Circular Polarization Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Verify axial ratio across scan angles
   scan_angles = np.linspace(0, 60, 13)
   ar_values = []

   for scan_theta in scan_angles:
       # Get polarization state at this scan angle
       # (simplified example - actual implementation depends on element type)
       j = pa.jones_vector(1.0, 1.0, phase_diff=-np.pi/2 + 0.1*np.deg2rad(scan_theta))
       ar = pa.axial_ratio(j)
       ar_values.append(ar)

   # Check if AR stays within spec (e.g., < 3 dB)
   ar_dB = 20 * np.log10(ar_values)
   print(f"Max AR: {max(ar_dB):.2f} dB")

Full Array Vector Patterns
--------------------------

As of v1.4.0 the polarization math is wired into the pattern engine:
polarized element models produce complex (E_theta, E_phi) components,
so co/cross-polar patterns, XPD maps, and axial-ratio maps can be
computed directly from an array definition.

.. code-block:: python

   import numpy as np
   import phased_array as pa

   geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)
   weights = pa.steering_vector(k, geom.x, geom.y, 20, 0)

   # x-polarized ideal patch elements, full vector pattern
   element = pa.ideal_patch_element('x', cos_exp=1.3)
   pattern = pa.compute_full_vector_pattern(
       geom.x, geom.y, weights, k, element_func=element
   )

   # Total power in dB, co/cross decomposition, XPD map
   power_dB = pattern.power_dB()
   E_co, E_cross = pattern.co_cross()
   xpd_dB = pattern.xpd_map()

Co/cross-polar pattern cuts through any phi plane:

.. code-block:: python

   # Crossed dipoles radiate finite cross-pol away from boresight
   theta_deg, co_dB, cross_dB = pa.compute_co_cross_pattern_cuts(
       geom.x, geom.y, weights, k,
       element_func=pa.dipole_element('x'),
       phi_cut_deg=45.0,
   )

Circularly polarized arrays with crossed-dipole elements:

.. code-block:: python

   element = pa.crossed_dipole_element(phase_diff=-np.pi/2)  # RHCP
   pattern = pa.compute_full_vector_pattern(
       geom.x, geom.y, weights, k, element_func=element
   )
   ar_map = pattern.axial_ratio_map()   # 1.0 at boresight

Conformal arrays evaluate each element in its own local frame (local z
along the element normal) and rotate the fields back to the global
basis, so polarization behavior versus scan is captured correctly:

.. code-block:: python

   cyl = pa.create_cylindrical_array(16, 8, radius=2.0, height=4.0)
   w = np.ones(cyl.n_elements, dtype=complex)
   E_theta, E_phi = pa.vector_array_factor_conformal(
       np.deg2rad(45), np.deg2rad(10), cyl, w, k,
       element_func=pa.ideal_patch_element('x'),
   )

Measured or simulated element patterns (e.g. exported from a full-wave
solver) can be used anywhere an element function is accepted:

.. code-block:: python

   pattern_data = pa.GriddedElementPattern(
       theta_grid, phi_grid, E_theta_data, E_phi_data
   )
   result = pa.compute_full_vector_pattern(
       geom.x, geom.y, weights, k, element_func=pattern_data
   )

Best Practices
--------------

1. **Use Ludwig-3** for co/cross-pol decomposition in standard antenna
   measurements.

2. **Specify axial ratio requirements** in dB for circular polarization
   systems (typical: < 3 dB).

3. **Account for polarization loss** in link budget calculations when
   antenna polarizations may not be perfectly matched.

4. **Verify XPD** at scan limits where cross-pol typically degrades.

5. **Consider polarization diversity** for fading mitigation in
   communications systems.
