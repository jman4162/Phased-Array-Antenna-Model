Glossary
========

Technical terms used in phased array antenna analysis.

.. glossary::
   :sorted:

   Amplitude Taper
      A weighting applied to element excitations to control sidelobe levels.
      Common tapers include Taylor, Chebyshev, and Hamming windows. Tapering
      reduces sidelobes at the cost of increased beamwidth and reduced
      aperture efficiency.

   Aperture
      The physical area of an antenna that captures or radiates electromagnetic
      energy. For arrays, this is typically the area encompassing all elements.

   Aperture Efficiency
      The ratio of the effective aperture area to the physical aperture area.
      Reduced by amplitude tapering and other non-uniform illumination.

   Array Factor
      The radiation pattern of an array of isotropic elements. The total
      pattern equals the element pattern multiplied by the array factor
      (pattern multiplication principle).

   Azimuth
      The angle in the horizontal plane, measured from a reference direction
      (typically the x-axis). Also called the phi angle.

   Beam Steering
      The process of electronically pointing the main beam by adjusting
      element phase shifts, without physically moving the antenna.

   Beam Squint
      The phenomenon where a phase-steered beam points at different angles
      for different frequencies. Caused by frequency-dependent phase shifts.

   Beamwidth
      The angular width of the main beam, typically measured at the half-power
      (-3 dB) points. Also called Half-Power Beamwidth (HPBW).

   Broadside
      The direction perpendicular to the array face (θ = 0°). An array with
      uniform phase radiates broadside.

   Chebyshev Taper
      An amplitude taper that produces equi-ripple sidelobes, providing the
      narrowest beamwidth for a given sidelobe level.

   Conformal Array
      An array mounted on a curved surface where elements have different
      orientations. Requires accounting for element normal directions in
      pattern calculations.

   Direction Cosines
      The projections of a unit vector onto coordinate axes. For antenna
      patterns: u = sin(θ)cos(φ), v = sin(θ)sin(φ), w = cos(θ).

   Directivity
      The ratio of radiation intensity in a given direction to the average
      intensity over all directions. Expressed in dBi (decibels relative
      to isotropic).

   E-Plane
      The plane containing the electric field vector and the direction of
      maximum radiation.

   Element Pattern
      The radiation pattern of a single antenna element. Combined with the
      array factor via pattern multiplication to get the total pattern.

   Element Spacing
      The distance between adjacent array elements, typically expressed in
      wavelengths. Spacing > λ/2 can cause grating lobes.

   Elevation
      The angle measured from the zenith (z-axis) or from the horizon,
      depending on convention. Also called the theta angle.

   Grating Lobes
      Unwanted secondary main beams that appear when element spacing exceeds
      λ/2. Located at angles where the path difference between elements
      equals a multiple of wavelengths.

   Graceful Degradation
      The ability of an array to maintain acceptable performance when some
      elements fail. Large arrays degrade gradually rather than catastrophically.

   H-Plane
      The plane containing the magnetic field vector and the direction of
      maximum radiation. Perpendicular to the E-plane.

   Half-Power Beamwidth (HPBW)
      The angular width between the -3 dB points of the main beam. A key
      performance metric for antenna resolution.

   Hybrid Steering
      A beamforming architecture using TTD at the subarray level and phase
      shifters within subarrays. Balances cost and wideband performance.

   LCMV
      Linearly Constrained Minimum Variance - an adaptive beamforming algorithm
      that minimizes output power while satisfying linear constraints on the
      response in specified directions.

   Main Beam
      The lobe containing the direction of maximum radiation. Also called
      the main lobe.

   Monopulse
      A tracking technique using sum and difference patterns to determine
      target angle. Provides angle information from a single pulse.

   Mutual Coupling
      Electromagnetic interaction between array elements that affects element
      patterns and input impedance. Can cause beam pointing errors if not
      compensated.

   Null
      A direction where the radiation pattern has zero (or very low) response.
      Can be intentionally placed to reject interference.

   Null Steering
      The process of placing pattern nulls in specific directions, typically
      to reject interference sources while maintaining the main beam.

   Pattern Multiplication
      The principle that an array's total pattern equals the product of the
      element pattern and the array factor.

   Phase Quantization
      The discretization of phase shifter settings to a finite number of
      levels. Causes beam pointing errors and increased sidelobes.

   Phase Shifter
      A device that adjusts the phase of the signal to/from each element,
      enabling beam steering.

   Planar Array
      A 2D array of elements arranged in a flat plane, typically rectangular
      or triangular grid.

   Scan Blindness
      A phenomenon where the array reflection coefficient approaches unity
      at specific scan angles, causing severe gain loss. Related to surface
      wave excitation.

   Scan Loss
      The reduction in gain when steering away from broadside, primarily due
      to the element pattern and projected aperture reduction.

   Sidelobe
      Any lobe other than the main beam. Sidelobe Level (SLL) is typically
      specified as dB below the main beam peak.

   Sparse Array
      An array with some elements removed (thinned) to reduce cost while
      maintaining aperture size. Results in higher sidelobes.

   Steering Vector
      The complex weights that produce a beam in a specified direction.
      For direction (θ₀, φ₀): w = exp(-jk·r·û) where û is the unit vector
      toward the beam direction.

   Subarray
      A group of elements that share a common phase shifter. Reduces hardware
      cost but limits scan range due to quantization lobes.

   Taylor Taper
      An amplitude distribution designed to produce a specified number of
      nearly-equal sidelobes. Provides good tradeoff between beamwidth and
      sidelobe control.

   True Time Delay (TTD)
      A beamforming technique using actual time delays instead of phase shifts.
      Provides frequency-independent beam pointing, eliminating beam squint.

   UV-Space
      A coordinate system using direction cosines u and v, where the visible
      region is defined by u² + v² ≤ 1.

   Visible Region
      The range of direction cosines (u, v) corresponding to real angles:
      u² + v² ≤ 1. Grating lobes outside this region are evanescent.

   Wavenumber
      The spatial frequency of electromagnetic waves: k = 2π/λ. Used to
      calculate phase shifts for steering.

   Weights
      The complex excitation coefficients applied to each element, determining
      amplitude and phase. Control beam direction, sidelobes, and nulls.
