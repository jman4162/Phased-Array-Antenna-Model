# Glossary of Phased Array Antenna Terms

This glossary provides definitions for technical terms used in phased array antenna design and analysis.

## A

### Amplitude Taper
A weighting applied to element excitations to control sidelobe levels. Common tapers include Taylor, Chebyshev, and Hamming windows. Tapering reduces sidelobes at the cost of increased beamwidth and reduced aperture efficiency.

### Aperture
The physical area of an antenna that captures or radiates electromagnetic energy. For arrays, this is typically the area encompassing all elements.

### Aperture Efficiency
The ratio of the effective aperture area to the physical aperture area. Reduced by amplitude tapering and other non-uniform illumination.

### Array Factor (AF)
The radiation pattern of an array of isotropic elements. The total pattern equals the element pattern multiplied by the array factor (pattern multiplication principle).

### Azimuth (φ)
The angle in the horizontal plane, measured from a reference direction (typically the x-axis). Also called the phi angle.

## B

### Beam Steering
The process of electronically pointing the main beam by adjusting element phase shifts, without physically moving the antenna.

### Beamwidth
The angular width of the main beam, typically measured at the half-power (-3 dB) points. Also called Half-Power Beamwidth (HPBW).

### Broadside
The direction perpendicular to the array face (θ = 0°). An array with uniform phase radiates broadside.

## C

### Chebyshev Taper
An amplitude taper that produces equi-ripple sidelobes, providing the narrowest beamwidth for a given sidelobe level.

### Conformal Array
An array mounted on a curved surface where elements have different orientations. Requires accounting for element normal directions in pattern calculations.

### Cosine Element Pattern
A simplified model for element directivity where gain decreases as cos(θ) from broadside.

## D

### Direction Cosines
The projections of a unit vector onto coordinate axes. For antenna patterns: u = sin(θ)cos(φ), v = sin(θ)sin(φ), w = cos(θ).

### Directivity
The ratio of radiation intensity in a given direction to the average intensity over all directions. Expressed in dBi (decibels relative to isotropic).

## E

### E-Plane
The plane containing the electric field vector and the direction of maximum radiation. For a vertical dipole, this is typically the elevation plane.

### Element Pattern
The radiation pattern of a single antenna element. Combined with the array factor via pattern multiplication to get the total pattern.

### Element Spacing
The distance between adjacent array elements, typically expressed in wavelengths. Spacing > λ/2 can cause grating lobes.

### Elevation (θ)
The angle measured from the zenith (z-axis) or from the horizon, depending on convention. Also called the theta angle.

### Endfire
The direction along the array axis (θ = 90°). Requires close element spacing (< λ/4) to avoid grating lobes.

## G

### Grating Lobes
Unwanted secondary main beams that appear when element spacing exceeds λ/2. Located at angles where the path difference between elements equals a multiple of wavelengths.

### Graceful Degradation
The ability of an array to maintain acceptable performance when some elements fail. Large arrays degrade gradually rather than catastrophically.

## H

### H-Plane
The plane containing the magnetic field vector and the direction of maximum radiation. Perpendicular to the E-plane.

### Half-Power Beamwidth (HPBW)
The angular width between the -3 dB points of the main beam. A key performance metric for antenna resolution.

### Hamming Window
An amplitude taper that provides good sidelobe suppression with moderate beamwidth increase. Defined as 0.54 - 0.46·cos(2πn/N).

## I

### Isotropic Radiator
A theoretical antenna that radiates equally in all directions. Used as a reference for gain measurements (dBi).

## L

### LCMV (Linearly Constrained Minimum Variance)
An adaptive beamforming algorithm that minimizes output power while satisfying linear constraints on the response in specified directions.

## M

### Main Beam
The lobe containing the direction of maximum radiation. Also called the main lobe.

### Monopulse
A tracking technique using sum and difference patterns to determine target angle. Provides angle information from a single pulse.

### Mutual Coupling
Electromagnetic interaction between array elements that affects element patterns and input impedance. Can cause beam pointing errors if not compensated.

## N

### Null
A direction where the radiation pattern has zero (or very low) response. Can be intentionally placed to reject interference.

### Null Steering
The process of placing pattern nulls in specific directions, typically to reject interference sources while maintaining the main beam.

## P

### Pattern Multiplication
The principle that an array's total pattern equals the product of the element pattern and the array factor.

### Phase Quantization
The discretization of phase shifter settings to a finite number of levels. Causes beam pointing errors and increased sidelobes.

### Phase Shifter
A device that adjusts the phase of the signal to/from each element, enabling beam steering.

### Planar Array
A 2D array of elements arranged in a flat plane, typically rectangular or triangular grid.

## R

### Radiation Pattern
The spatial distribution of radiated power as a function of angle. Can be expressed in terms of field strength or power density.

## S

### Scan Blindness
A phenomenon where the array reflection coefficient approaches unity at specific scan angles, causing severe gain loss. Related to surface wave excitation.

### Scan Loss
The reduction in gain when steering away from broadside, primarily due to the element pattern and projected aperture reduction. Approximately cos(θ) for planar arrays.

### Sidelobe
Any lobe other than the main beam. Sidelobe Level (SLL) is typically specified as dB below the main beam peak.

### Sparse Array
An array with some elements removed (thinned) to reduce cost while maintaining aperture size. Results in higher sidelobes.

### Steering Vector
The complex weights that produce a beam in a specified direction. For direction (θ₀, φ₀): w = exp(-jk·r·û) where û is the unit vector toward the beam direction.

### Subarray
A group of elements that share a common phase shifter. Reduces hardware cost but limits scan range due to quantization lobes.

## T

### Taylor Taper
An amplitude distribution designed to produce a specified number of nearly-equal sidelobes. Provides good tradeoff between beamwidth and sidelobe control.

### Theta (θ)
The polar angle measured from the z-axis (zenith). θ = 0° is broadside for a planar array in the xy-plane.

### Thinned Array
An array with randomly or systematically removed elements to reduce cost. See Sparse Array.

## U

### Uniform Linear Array (ULA)
A 1D array with equally-spaced elements and uniform amplitude weighting.

### UV-Space
A coordinate system using direction cosines u and v, where the visible region is defined by u² + v² ≤ 1.

## V

### Visible Region
The range of direction cosines (u, v) corresponding to real angles: u² + v² ≤ 1. Grating lobes outside this region are evanescent.

## W

### Wavenumber (k)
The spatial frequency of electromagnetic waves: k = 2π/λ. Used to calculate phase shifts for steering.

### Weights
The complex excitation coefficients applied to each element, determining amplitude and phase. Control beam direction, sidelobes, and nulls.

---

## Common Equations

### Wavenumber
```
k = 2π/λ = 2πf/c
```

### Direction Cosines
```
u = sin(θ)cos(φ)
v = sin(θ)sin(φ)
w = cos(θ)
```

### Array Factor (1D)
```
AF(θ) = Σ wₙ exp(jknd·sin(θ))
```

### Steering Vector
```
wₙ = exp(-jkxₙu₀ - jkyₙv₀)
```

### Grating Lobe Condition
```
d > λ/(1 + sin(θₘₐₓ))
```

### Half-Power Beamwidth (approximate)
```
HPBW ≈ 0.886λ/L (radians)
```
where L is the aperture length.

### Directivity (uniform rectangular aperture)
```
D = 4πA/λ² = 4π(Lₓ·Lᵧ)/λ²
```

---

## References

1. Balanis, C.A. (2016). *Antenna Theory: Analysis and Design*, 4th Edition. Wiley.
2. Mailloux, R.J. (2017). *Phased Array Antenna Handbook*, 3rd Edition. Artech House.
3. Hansen, R.C. (2009). *Phased Array Antennas*, 2nd Edition. Wiley.
4. Skolnik, M.I. (2008). *Radar Handbook*, 3rd Edition. McGraw-Hill.
