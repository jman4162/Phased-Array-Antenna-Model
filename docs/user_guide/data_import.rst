Measured and Simulated Data
===========================

Two kinds of external data feed the models in this package: S-parameters,
which give the mutual coupling, and embedded element patterns, which
replace the analytic element models. This page covers reading both.

- :func:`~phased_array.read_touchstone` and
  :func:`~phased_array.write_touchstone`: Touchstone version 1.x
  (``.sNp``) S-parameter files.
- :meth:`~phased_array.GriddedElementPattern.from_table` and
  :func:`~phased_array.load_element_pattern_csv`: far-field tables with one
  row per (theta, phi) sample.

The examples write small files to a temporary directory and read them back,
so they run as written.

Touchstone S-parameter files
----------------------------

A Touchstone file starts with an option line such as ``# GHz S MA R 50``
(frequency unit, parameter type, number format, reference impedance),
followed by one record per frequency. ``!`` starts a comment anywhere on a
line. The port count comes from the extension (``.s4p`` has 4 ports); for
other extensions, pass ``n_ports``.

.. code-block:: python

   import os
   import tempfile

   import numpy as np
   import phased_array as pa

   workdir = tempfile.mkdtemp()

   # A 4-port network at three frequencies: 0.2 reflection at every port,
   # -20 dB coupling between neighboring ports
   freqs = np.array([9.5e9, 10.0e9, 10.5e9])
   s = np.zeros((3, 4, 4), dtype=complex)
   for i in range(4):
       s[:, i, i] = 0.2 * np.exp(-1j * np.pi / 4)
       if i < 3:
           s[:, i, i + 1] = s[:, i + 1, i] = 0.1 * np.exp(-1j * np.pi / 2)

   path = os.path.join(workdir, "array.s4p")
   pa.write_touchstone(path, freqs, s, fmt="MA", unit="GHz")

   data = pa.read_touchstone(path)
   print(data["frequencies"])       # [9.50e+09 1.00e+10 1.05e+10], in Hz
   print(data["s"].shape)           # (3, 4, 4): (frequency, row, column)
   print(data["z0"], data["n_ports"])  # 50.0 4

The option line may list its fields in any order and case. Omitted fields
take the defaults ``GHz``, ``S``, ``MA`` and ``R 50``. The supported number
formats are:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Format
     - Pair of numbers
   * - ``RI``
     - Real part, imaginary part
   * - ``MA``
     - Magnitude, angle in degrees
   * - ``DB``
     - ``20*log10`` of the magnitude, angle in degrees

Record layout
^^^^^^^^^^^^^

Each frequency starts a new line. A 1-port record is ``f S11``. A 2-port
record holds all four parameters on one line in the order
``S11 S21 S12 S22``: S21 comes before S12. Networks with three or more
ports list the matrix row by row, each row on a new line and wrapped after
four pairs. The reader returns ``s[f, i, j]`` = S_(i+1)(j+1) for every
port count, so the 2-port column order is handled for you:

.. code-block:: python

   text = """! frequency  S11      S21      S12      S22
   # MHz S RI R 50
   1000         0.5 0    0.9 0    0.01 0   0.3 0
   """
   path = os.path.join(workdir, "amplifier.s2p")
   with open(path, "w") as f:
       f.write(text)

   amp = pa.read_touchstone(path)
   print(amp["s"][0, 1, 0].real)    # 0.9, S21 (the second pair on the line)
   print(amp["s"][0, 0, 1].real)    # 0.01, S12 (the third pair)

If a 2-port file ends with a noise-parameter block, it is returned under
``data["noise"]``; otherwise ``data["noise"]`` is None.

What the reader rejects
^^^^^^^^^^^^^^^^^^^^^^^

The reader raises ``ValueError`` instead of guessing when:

- the parameter type is Y, Z, H or G (only S is supported);
- the file uses Touchstone 2.0 keywords such as ``[Version] 2.0``;
- the option line has an unknown token or more than one reference
  impedance (Touchstone 1.x has a single ``R`` for all ports);
- a record does not hold exactly ``1 + 2*N**2`` numbers, which is what a
  wrong port count produces;
- frequencies are negative or not strictly increasing.

From S-parameters to active reflection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take one frequency slice of ``data["s"]`` and pass it to
:func:`~phased_array.mutual_coupling_matrix_measured`. The active
reflection coefficient of element n under weights ``w`` is
``(S w)_n / w_n``:

.. code-block:: python

   S = data["s"][1]                 # 10 GHz
   C = pa.mutual_coupling_matrix_measured(S)

   # Progressive phase for a scan along the 4-element row
   w = np.exp(-1j * np.pi * np.sin(np.deg2rad(30)) * np.arange(4))

   gamma = [pa.active_reflection_coefficient(C, w, n) for n in range(4)]
   print(np.allclose(gamma, (S @ w) / w))   # True
   print(np.round(np.abs(gamma), 3))        # [0.147 0.2   0.2   0.28 ]

At this scan the contributions of the two neighbors cancel for the interior
elements, which see only their own reflection of 0.2; the edge elements
have one neighbor and differ. ``C`` is the normalized impedance matrix
``(I + S)(I - S)^-1``; see
:doc:`impairments` and :doc:`conventions` for the ``formulation`` argument.

Tabulated element patterns
--------------------------

EM solvers and antenna ranges export far fields as a table with one row per
(theta, phi) direction. :meth:`~phased_array.GriddedElementPattern.from_table`
takes those columns as arrays, in any row order, checks that they cover
every combination of the distinct theta and phi values exactly once, and
returns a :class:`~phased_array.GriddedElementPattern`. That object works
anywhere an element function does, for example in
:func:`~phased_array.vector_total_pattern`.

.. code-block:: python

   # Stand-in for simulated data: samples of an analytic patch element,
   # with the rows shuffled the way an export might order them
   element = pa.ideal_patch_element("x", cos_exp=1.5)
   T, P = np.meshgrid(np.arange(0, 91, 2.0), np.arange(-180, 181, 5.0),
                      indexing="ij")
   rows = np.random.default_rng(0).permutation(T.size)
   theta_deg, phi_deg = T.ravel()[rows], P.ravel()[rows]
   e_theta, e_phi = element(np.deg2rad(theta_deg), np.deg2rad(phi_deg))

   pattern = pa.GriddedElementPattern.from_table(theta_deg, phi_deg, e_theta, e_phi)

   theta_q, phi_q = np.array([0.4]), np.array([1.0])   # radians
   print(abs(pattern(theta_q, phi_q)[0][0] - element(theta_q, phi_q)[0][0]) < 2e-3)

The grid rules:

- Theta must lie in [0, 180] deg. Tables with negative theta (theta in
  [-180, 180], phi in [0, 180)) must be folded first: the sample at
  ``(-theta, phi)`` is the direction ``(theta, phi + 180 deg)`` with both
  field components negated.
- Phi may cover any range up to one turn, such as [-180, 180] or [0, 360).
  A [0, 360) grid interpolates across the 360 deg seam.
- The spacing need not be uniform, but every theta value must appear with
  every phi value. A missing or repeated sample raises ``ValueError``.
- Directions outside the theta range of the table (behind a hemisphere-only
  table, for example) return zero field.

Loading a CSV or whitespace table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~phased_array.load_element_pattern_csv` reads such a table from a
file with a header line. The ``columns`` argument maps each field role to a
header name or a 0-based column index, so the loader does not depend on any
one tool's naming. For example, a far-field export with the header
``Theta [deg],Phi [deg],re(rETheta) [V],im(rETheta) [V],re(rEPhi) [V],im(rEPhi) [V]``
maps as follows:

.. code-block:: python

   columns = {
       "theta": "Theta [deg]",
       "phi": "Phi [deg]",
       "e_theta_re": "re(rETheta) [V]",
       "e_theta_im": "im(rETheta) [V]",
       "e_phi_re": "re(rEPhi) [V]",
       "e_phi_im": "im(rEPhi) [V]",
   }

   # Write the shuffled samples from above in that layout
   path = os.path.join(workdir, "element.csv")
   table = np.column_stack([theta_deg, phi_deg, e_theta.real, e_theta.imag,
                            e_phi.real, e_phi.imag])
   np.savetxt(path, table, delimiter=",", header=",".join(columns.values()),
              comments="")

   embedded = pa.load_element_pattern_csv(path, columns)

   geom = pa.create_rectangular_array(8, 8, dx=0.5, dy=0.5)
   k = pa.wavelength_to_k(1.0)
   w = pa.steering_vector(k, geom.x, geom.y, theta0_deg=20, phi0_deg=0)
   E_theta, E_phi = pa.vector_total_pattern(
       np.deg2rad([20.0]), np.array([0.0]), geom.x, geom.y, w, k, embedded
   )
   print(abs(E_theta[0]))   # about 64 * cos(20 deg)**1.5 = 58.3

Tables that give magnitude in dB and phase in degrees use
``field_format="db_phase"`` with the roles ``e_theta_db``,
``e_theta_phase_deg``, ``e_phi_db`` and ``e_phi_phase_deg``. The dB values
are ``20*log10|E|``, so a gain or directivity column in dBi also works;
``-inf`` reads as zero field. Whitespace-separated tables use
``delimiter=None``:

.. code-block:: python

   db = lambda e: 20 * np.log10(np.maximum(np.abs(e), 1e-15))
   path = os.path.join(workdir, "element_db.txt")
   with open(path, "w") as f:
       f.write("theta phi Et_dB Et_phase Ep_dB Ep_phase\n")
       f.write("-" * 44 + "\n")   # lines without digits are skipped
       for row in zip(theta_deg, phi_deg, db(e_theta), np.angle(e_theta, deg=True),
                      db(e_phi), np.angle(e_phi, deg=True)):
           f.write(" ".join(f"{v:.12g}" for v in row) + "\n")

   from_db = pa.load_element_pattern_csv(
       path,
       {"theta": "theta", "phi": "phi",
        "e_theta_db": "Et_dB", "e_theta_phase_deg": "Et_phase",
        "e_phi_db": "Ep_dB", "e_phi_phase_deg": "Ep_phase"},
       delimiter=None,
       field_format="db_phase",
   )
   print(np.allclose(from_db(theta_q, phi_q)[0], embedded(theta_q, phi_q)[0],
                     atol=1e-9))   # True

When header names contain spaces in a whitespace-separated file, map the
roles to column indices instead (``{"theta": 0, "phi": 1, ...}``).
``skip_rows`` skips preamble lines before the header, and lines starting
with ``comments`` (default ``'#'``) are ignored, with one exception: if the
first uncommented line holds only numbers, the comment line just above it
is the header. That is the layout ``np.savetxt(..., header=...)`` writes.
A table with no header at all needs column indices; its first row is read
as data.
