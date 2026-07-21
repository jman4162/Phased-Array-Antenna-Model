---
title: 'phased-array-modeling: A Python package for phased array antenna analysis'
tags:
  - Python
  - phased arrays
  - antennas
  - beamforming
  - electromagnetics
  - radar
authors:
  - name: John Hodge
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 20 July 2026
bibliography: paper.bib
---

# Summary

Phased array antennas steer radio beams electronically by controlling the
phase and amplitude of many radiating elements. They are core technology in
radar, 5G/6G communications, satellite terminals, and radio astronomy.
`phased-array-modeling` is a Python package for computing and visualizing
phased array radiation patterns. It provides vectorized array factor
computation; rectangular, triangular, elliptical, circular, cylindrical,
spherical, and sparse/thinned array geometries; amplitude tapering, null
steering, multi-beam synthesis, and adaptive beamforming (SMI/MVDR, GSC,
LCMV); wideband true-time-delay steering with beam squint analysis;
subarray architectures including overlapped subarrays and hybrid
TTD-plus-phase steering; impairment models for mutual coupling, phase
quantization, element failure, scan blindness, and active impedance; and a
vector (polarized) pattern engine with dipole, patch, and circularly
polarized element models, Ludwig-3 co/cross-polar decomposition
[@ludwig1973], axial ratio and cross-polar discrimination maps, and
polarization-correct conformal array patterns. Interactive visualization is
provided through matplotlib and Plotly, including UV-space and 3D pattern
plots, and results export to CSV, JSON, and NumPy formats.

# Statement of need

Array analysis at the system-engineering level - pattern synthesis, taper
trade studies, impairment budgets, scan-loss analysis - sits between two
kinds of existing tools. Full-wave electromagnetic solvers (HFSS, CST,
openEMS) capture element-level physics but are impractical for rapid
whole-array trade studies. Commercial system toolboxes, principally the
MATLAB Phased Array System Toolbox [@matlabphased], cover this level well
but require proprietary licenses. In the open-source Python ecosystem the
niche is underserved: `arraytool` [@arraytool] has been unmaintained since
2017, `pyArgus` [@pyargus] covers only the signal-processing side and is
dormant, and link-level simulators such as Sionna [@sionna] model arrays
internally but do not expose pattern-analysis utilities.

`phased-array-modeling` targets practicing antenna engineers, students, and
researchers who need textbook-consistent [@mailloux2017; @balanis2016]
array analysis with reproducible scripts and notebooks. The package is
pure Python on the scientific stack (NumPy, SciPy, matplotlib), installs
from PyPI, and is validated by a test suite of more than 250 unit tests
including doctests of every documented example. Features such as
overlapped-subarray synthesis, quantization and failure Monte Carlo
analysis, active reflection coefficient versus scan, and measured
element-pattern import address workflows that practitioners otherwise
reimplement ad hoc. A hosted documentation site with theory background,
cookbook recipes, and executable Google Colab notebooks, plus an
interactive Streamlit application, lower the barrier for teaching and
exploration.

# Acknowledgements

The package builds on NumPy [@harris2020], SciPy [@virtanen2020], and
matplotlib [@hunter2007].

# References
