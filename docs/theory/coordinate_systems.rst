Coordinate Systems
==================

Phased array analysis uses several coordinate systems. Understanding the
transformations between them is essential.

Spherical Coordinates (Theta/Phi)
---------------------------------

The standard physics convention used in this library:

- **Theta (θ)**: Polar angle from the z-axis, :math:`0 \leq \theta \leq \pi`
- **Phi (φ)**: Azimuth angle in the xy-plane from x-axis, :math:`0 \leq \phi < 2\pi`

.. math::

   x &= r \sin\theta \cos\phi \\
   y &= r \sin\theta \sin\phi \\
   z &= r \cos\theta

**Key directions:**

- Broadside (normal to array): :math:`\theta = 0`
- x-axis: :math:`\theta = 90°, \phi = 0°`
- y-axis: :math:`\theta = 90°, \phi = 90°`

Direction Cosines (UV-Space)
----------------------------

Direction cosines project the observation direction onto the coordinate axes:

.. math::

   u &= \sin\theta \cos\phi \\
   v &= \sin\theta \sin\phi \\
   w &= \cos\theta

Properties:

- :math:`u^2 + v^2 + w^2 = 1`
- **Visible region**: :math:`u^2 + v^2 \leq 1`
- Points outside the visible region represent evanescent waves

**Advantages of UV-space:**

- Array factor is a Fourier transform of element positions
- Grating lobes appear at regular intervals in (u, v)
- Beam steering is a simple translation

Conversion Functions
--------------------

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Theta/phi to UV
   theta = np.deg2rad(30)
   phi = np.deg2rad(45)
   u, v = pa.theta_phi_to_uv(theta, phi)
   print(f"theta={30}, phi={45} -> u={u:.3f}, v={v:.3f}")

   # UV to theta/phi
   theta, phi = pa.uv_to_theta_phi(u, v)
   print(f"u={u:.3f}, v={v:.3f} -> theta={np.rad2deg(theta):.1f}, phi={np.rad2deg(phi):.1f}")

Azimuth/Elevation
-----------------

Engineering convention often uses azimuth (Az) and elevation (El):

- **Azimuth (Az)**: Angle in horizontal plane, typically from north or boresight
- **Elevation (El)**: Angle above the horizon

The relationship depends on the mounting convention. For an array facing
the +x direction:

.. math::

   \text{Az} &= \phi \\
   \text{El} &= 90° - \theta

.. code-block:: python

   # Convert between conventions
   theta, phi = pa.azel_to_thetaphi(az_deg=45, el_deg=30)
   az, el = pa.thetaphi_to_azel(theta, phi)

Sine-Space
----------

Sometimes called "k-space" or "direction cosine space", this is similar to
UV-space but normalized differently:

.. math::

   k_x &= \frac{2\pi}{\lambda} u = k \sin\theta \cos\phi \\
   k_y &= \frac{2\pi}{\lambda} v = k \sin\theta \sin\phi

This represents the transverse components of the wavevector.

Array-Centered vs. Global Coordinates
-------------------------------------

For conformal arrays on curved surfaces, each element has a local coordinate
system defined by its normal vector. The element pattern is evaluated in
local coordinates, then transformed to global.

If element n has normal direction :math:`\hat{n}_n`, the local observation
angle is:

.. math::

   \cos\theta_{local} = \hat{n}_n \cdot \hat{u}_{observation}

.. code-block:: python

   # Conformal array example
   geom = pa.create_cylindrical_array(16, 4, radius=3.0, height=2.0)

   # Element normals
   print(f"Element 0 normal: ({geom.nx[0]:.2f}, {geom.ny[0]:.2f}, {geom.nz[0]:.2f})")

   # Use array_factor_conformal to account for element orientations
   AF = pa.array_factor_conformal(theta_grid, phi_grid, geom, weights, k)

Coordinate Grid Generation
--------------------------

.. code-block:: python

   # Theta/phi grid
   theta_1d, phi_1d, theta_grid, phi_grid = pa.create_theta_phi_grid(
       theta_range=(0, np.pi/2),
       phi_range=(0, 2*np.pi),
       n_theta=91,
       n_phi=181
   )

   # UV grid
   u_1d, v_1d, u_grid, v_grid = pa.create_uv_grid(
       u_range=(-1, 1),
       v_range=(-1, 1),
       n_u=201,
       n_v=201
   )

   # Check visible region
   is_visible = pa.is_visible_region(u_grid, v_grid)

Antenna vs Radar Coordinate Systems
-----------------------------------

Different communities use different coordinate conventions. The library
provides conversion functions between the two most common systems.

**Antenna Coordinates (theta/phi):**

- Standard physics spherical coordinates
- :math:`\theta` measured from boresight (z-axis)
- :math:`\phi` measured in the xy-plane from x-axis

**Radar Coordinates (az/el):**

- Engineering convention for tracking systems
- Azimuth in horizontal plane
- Elevation above horizon

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Convert antenna coords to radar coords
   theta_ant = np.deg2rad(30)  # 30 deg from boresight
   phi_ant = np.deg2rad(45)    # 45 deg azimuthal

   az, el = pa.antenna_to_radar(theta_ant, phi_ant)
   print(f"Antenna: theta={np.rad2deg(theta_ant):.1f}, phi={np.rad2deg(phi_ant):.1f}")
   print(f"Radar: az={np.rad2deg(az):.1f}, el={np.rad2deg(el):.1f}")

   # Convert back
   theta_back, phi_back = pa.radar_to_antenna(az, el)
   print(f"Round-trip: theta={np.rad2deg(theta_back):.1f}, phi={np.rad2deg(phi_back):.1f}")

Cone/Clock Coordinates
----------------------

Cone/clock coordinates are useful for describing patterns on aircraft
radomes or for visualizing scan limits in a polar format.

- **Cone angle**: Distance from boresight (same as theta)
- **Clock angle**: Azimuthal position (same as phi)

.. code-block:: python

   # Convert antenna to cone/clock
   theta = np.deg2rad(30)
   phi = np.deg2rad(60)

   cone, clock = pa.antenna_to_cone(theta, phi)
   print(f"Cone angle: {np.rad2deg(cone):.1f} deg")
   print(f"Clock angle: {np.rad2deg(clock):.1f} deg")

   # Convert back
   theta_back, phi_back = pa.cone_to_antenna(cone, clock)

This representation is intuitive for displaying scan coverage:

- The cone angle represents "how far" from boresight
- The clock angle represents "which direction" from boresight (like a clock face)

Rotation Matrices
-----------------

For transforming between coordinate systems or rotating patterns, the library
provides standard rotation matrices using aircraft/aerospace conventions.

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Roll: rotation about x-axis (right wing down is positive)
   R_roll = pa.rotation_matrix_roll(np.deg2rad(10))

   # Pitch: rotation about y-axis (nose up is positive)
   R_pitch = pa.rotation_matrix_pitch(np.deg2rad(5))

   # Yaw: rotation about z-axis (nose left is positive)
   R_yaw = pa.rotation_matrix_yaw(np.deg2rad(-3))

   # Combined rotation (yaw-pitch-roll order)
   R_combined = R_yaw @ R_pitch @ R_roll

   # Apply to a direction vector
   v = np.array([0, 0, 1])  # Boresight direction
   v_rotated = R_combined @ v
   print(f"Original: {v}")
   print(f"Rotated: {v_rotated}")

Pattern Rotation
----------------

Rotate an entire radiation pattern by specified Euler angles:

.. code-block:: python

   import phased_array as pa
   import numpy as np

   # Create a pattern grid
   theta = np.linspace(0, np.pi/2, 46)
   phi = np.linspace(0, 2*np.pi, 73)
   theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

   # Simple cosine pattern
   pattern = np.cos(theta_grid)

   # Rotate pattern by 30 degrees in yaw
   theta_rot, phi_rot, pattern_rot = pa.rotate_pattern(
       theta_grid, phi_grid, pattern,
       roll_deg=0,
       pitch_deg=0,
       yaw_deg=30
   )

   print(f"Original pattern shape: {pattern.shape}")
   print(f"Rotated pattern shape: {pattern_rot.shape}")

**Rotation order (intrinsic):**

1. Yaw (about z-axis)
2. Pitch (about y-axis)
3. Roll (about x-axis)

This follows the standard aerospace convention for describing aircraft attitude.

**Use cases:**

- Mounting the array at an angle on a platform
- Compensating for platform attitude in tracking systems
- Transforming measured patterns to different reference frames
- Visualizing patterns from different observation perspectives
