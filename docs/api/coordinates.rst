Coordinate Transforms Module
============================

.. module:: phased_array.coordinates
   :synopsis: Coordinate system transformations for phased arrays

The coordinates module provides functions for converting between various
coordinate systems used in antenna and radar applications, including
antenna coordinates, radar coordinates, cone/clock coordinates, and
rotation matrices for pattern rotation.

Antenna and Radar Coordinates
-----------------------------

Functions for converting between antenna (theta/phi) and radar (az/el)
coordinate systems.

.. autofunction:: phased_array.antenna_to_radar

.. autofunction:: phased_array.radar_to_antenna

Cone/Clock Coordinates
----------------------

Cone/clock coordinates are useful for describing patterns on aircraft
radomes or for visualizing scan limits.

.. autofunction:: phased_array.antenna_to_cone

.. autofunction:: phased_array.cone_to_antenna

Rotation Matrices
-----------------

3x3 rotation matrices for transforming coordinate systems using
aircraft-style roll, pitch, and yaw angles.

.. autofunction:: phased_array.rotation_matrix_roll

.. autofunction:: phased_array.rotation_matrix_pitch

.. autofunction:: phased_array.rotation_matrix_yaw

Pattern Rotation
----------------

Functions for rotating radiation patterns in 3D space.

.. autofunction:: phased_array.rotate_pattern
