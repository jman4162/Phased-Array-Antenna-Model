"""
Data import: Touchstone S-parameter files and tabulated element patterns.

Functions for bringing measured or simulated data into the package:

- :func:`read_touchstone` and :func:`write_touchstone` handle Touchstone
  version 1.x (``.sNp``) S-parameter files, as exported by network
  analyzers and circuit or EM solvers.
- :func:`load_element_pattern_csv` reads a delimited table of far-field
  samples, one row per (theta, phi) direction, into a
  :class:`~phased_array.GriddedElementPattern`.
"""

import csv
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from .vector_patterns import GriddedElementPattern

PathLike = Union[str, "os.PathLike[str]"]

# ============== Touchstone ==============

_FREQ_SCALE = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
_UNIT_NAMES = {"HZ": "Hz", "KHZ": "kHz", "MHZ": "MHz", "GHZ": "GHz"}
_FORMATS = ("RI", "MA", "DB")
_PARAMETERS = ("S", "Y", "Z", "H", "G")
_OPTION_NAMES = {
    "unit": "frequency unit",
    "parameter": "parameter type",
    "format": "data format",
    "z0": "reference impedance",
}
_EXTENSION = re.compile(r"\.s(\d+)p$", re.IGNORECASE)
# DB format cannot represent zero; magnitudes below this are written as
# -6000 dB (10**-300), which reads back as 1e-300.
_DB_FLOOR = 1e-300


def _ports_from_extension(path: str) -> Optional[int]:
    """Port count encoded in a ``.sNp`` file name, or None."""
    match = _EXTENSION.search(path)
    return int(match.group(1)) if match else None


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _parse_option_line(text: str, where: str) -> Dict[str, Any]:
    """Parse ``# <unit> <parameter> <format> R <z0>`` (any order, any case)."""
    options: Dict[str, Any] = {
        "unit": "GHZ", "parameter": "S", "format": "MA", "z0": 50.0,
    }
    seen = set()
    tokens = text[1:].split()
    i = 0
    while i < len(tokens):
        token = tokens[i].upper()
        value: Any
        if token in _FREQ_SCALE:
            key, value = "unit", token
        elif token in _PARAMETERS:
            key, value = "parameter", token
        elif token in _FORMATS:
            key, value = "format", token
        elif token == "R":
            if i + 1 >= len(tokens):
                raise ValueError(
                    f"{where}: 'R' must be followed by the reference impedance"
                )
            if not _is_number(tokens[i + 1]):
                raise ValueError(
                    f"{where}: reference impedance {tokens[i + 1]!r} is not a number"
                )
            value = float(tokens[i + 1])
            if not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"{where}: reference impedance must be positive, got {value}"
                )
            key = "z0"
            i += 1
        else:
            hint = ""
            if _is_number(tokens[i]):
                hint = ("; Touchstone 1.x allows a single reference impedance "
                        "for all ports")
            raise ValueError(
                f"{where}: unrecognized option-line token {tokens[i]!r}{hint}"
            )
        if key in seen:
            raise ValueError(
                f"{where}: the option line sets the {_OPTION_NAMES[key]} twice"
            )
        seen.add(key)
        options[key] = value
        i += 1

    if options["parameter"] != "S":
        raise ValueError(
            f"{where}: {options['parameter']}-parameter files are not "
            "supported; only S-parameters can be read (convert the data to "
            "S-parameters first)"
        )
    return options


def _pairs_to_complex(a: np.ndarray, b: np.ndarray, fmt: str) -> np.ndarray:
    """Complex values from Touchstone number pairs."""
    if fmt == "RI":
        return a + 1j * b
    magnitude = a if fmt == "MA" else 10.0 ** (a / 20.0)
    return magnitude * np.exp(1j * np.deg2rad(b))


def read_touchstone(
    path: PathLike,
    n_ports: Optional[int] = None
) -> Dict[str, Any]:
    """
    Read a Touchstone version 1.x S-parameter file (``.sNp``).

    Parameters
    ----------
    path : str or path-like
        File to read.
    n_ports : int, optional
        Number of ports. Taken from the extension when it has the form
        ``.sNp`` (``.s2p`` -> 2, case-insensitive); required otherwise.
        If both are present they must agree.

    Returns
    -------
    data : dict
        ``'frequencies'``
            Frequencies in Hz, shape (F,).
        ``'s'``
            Complex S-parameters, shape (F, N, N), with ``s[f, i, j]`` =
            S_(i+1)(j+1) (row = receiving port, column = driven port).
        ``'z0'``
            Reference impedance in ohms (float).
        ``'parameter'``
            ``'S'``.
        ``'n_ports'``
            Number of ports N (int).
        ``'noise'``
            Two-port noise parameters, or None if the file has none. A
            dict with ``'frequencies'`` (Hz), ``'nf_min_db'``,
            ``'gamma_opt'`` (complex source reflection coefficient for
            minimum noise figure) and ``'rn'`` (effective noise resistance,
            normalized to ``z0``), each of shape (K,).

    Raises
    ------
    ValueError
        If the port count cannot be determined or contradicts the
        extension; the file holds Y, Z, H or G parameters; it uses
        Touchstone 2.0 keywords (``[Version] 2.0`` or any other ``[...]``
        line); the option line has an unknown token, more than one
        reference impedance, or a second option line that differs from
        the first; a record does not hold exactly ``1 + 2*N**2`` numbers or
        two records share a line; a value is not a finite number;
        frequencies are negative or not strictly increasing.

    Notes
    -----
    The option line ``# <unit> <parameter> <format> R <z0>`` may list its
    fields in any order and case; omitted fields (or a missing option line)
    take the defaults ``# GHz S MA R 50``. Units are Hz, kHz, MHz or GHz.
    Formats are ``RI`` (real, imaginary), ``MA`` (magnitude, angle in
    degrees) and ``DB`` (``20*log10`` magnitude, angle in degrees). Text
    after ``!`` on any line is a comment.

    Each frequency starts a new line and holds the frequency followed by
    N**2 number pairs:

    - 1-port: ``f S11``, one frequency per line.
    - 2-port: ``f S11 S21 S12 S22`` on one line. S21 comes **before** S12;
      this column-major order applies only to 2-port files.
    - 3 or more ports: row by row (``S11 S12 ... S1N``, then
      ``S21 ... S2N``, ...), each row on a new line and wrapped after four
      pairs.

    The reader accumulates numbers across line breaks within a record, so
    files that wrap rows differently but keep this order also load. In a
    2-port file, a line of five numbers whose frequency is not above the
    last S-parameter frequency starts the noise-parameter block. In that
    block the reflection coefficient is always magnitude and angle,
    whatever the data format.

    Examples
    --------
    Write a 2-port file and read it back:

    >>> import os, tempfile
    >>> import numpy as np
    >>> import phased_array as pa
    >>> s = np.array([[[0.1, 0.02j], [0.02j, 0.1]]])
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = os.path.join(tmp, 'pair.s2p')
    ...     pa.write_touchstone(path, [10e9], s, fmt='MA')
    ...     data = pa.read_touchstone(path)
    >>> data['s'].shape
    (1, 2, 2)
    >>> bool(np.allclose(data['s'], s))
    True

    A hand-written 2-port record lists S21 before S12:

    >>> text = '''! frequency S11 S21 S12 S22
    ... # MHz S RI R 50
    ... 1000  0.5 0  0.1 0  0.2 0  0.4 0
    ... '''
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = os.path.join(tmp, 'amp.s2p')
    ...     with open(path, 'w') as f:
    ...         _ = f.write(text)
    ...     data = pa.read_touchstone(path)
    >>> data['frequencies']
    array([1.e+09])
    >>> data['s'][0].real
    array([[0.5, 0.2],
           [0.1, 0.4]])
    """
    path_str = os.fspath(path)
    from_extension = _ports_from_extension(path_str)
    if n_ports is None:
        if from_extension is None:
            raise ValueError(
                f"{path_str}: cannot infer the port count from the file "
                "extension (expected .sNp, e.g. .s2p); pass n_ports"
            )
        n_ports = from_extension
    else:
        n_ports = int(n_ports)
        if from_extension is not None and from_extension != n_ports:
            raise ValueError(
                f"{path_str}: n_ports={n_ports} contradicts the extension, "
                f"which implies {from_extension} ports"
            )
    if n_ports < 1:
        raise ValueError(f"n_ports must be at least 1, got {n_ports}")

    with open(path_str, encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    options: Optional[Dict[str, Any]] = None
    data_lines: List[Tuple[int, List[float]]] = []
    for lineno, raw in enumerate(lines, start=1):
        text = raw.split("!", 1)[0].strip()
        if not text:
            continue
        where = f"{path_str}, line {lineno}"
        if text.startswith("["):
            keyword = text.split("]", 1)[0] + "]"
            raise ValueError(
                f"{where}: keyword {keyword} marks a Touchstone 2.0 file; "
                "only Touchstone 1.x files are supported"
            )
        if text.startswith("#"):
            parsed = _parse_option_line(text, where)
            if options is None:
                if data_lines:
                    raise ValueError(
                        f"{where}: the option line must come before the data"
                    )
                options = parsed
            elif parsed != options:
                raise ValueError(
                    f"{where}: a second option line differs from the first"
                )
            continue
        try:
            values = [float(token) for token in text.split()]
        except ValueError:
            raise ValueError(
                f"{where}: cannot parse {text!r} as numbers"
            ) from None
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{where}: non-finite value in {text!r}")
        data_lines.append((lineno, values))

    if options is None:
        options = _parse_option_line("#", path_str)

    # Group lines into records of one frequency plus N**2 number pairs.
    n_values = 1 + 2 * n_ports ** 2
    records: List[List[float]] = []
    noise_rows: List[List[float]] = []
    current: List[float] = []
    first_line = 0
    for lineno, values in data_lines:
        starts_noise = bool(
            n_ports == 2 and not current and records and len(values) == 5
            and values[0] <= records[-1][0]
        )
        if noise_rows or starts_noise:
            if len(values) != 5:
                raise ValueError(
                    f"{path_str}, line {lineno}: expected 5 noise-parameter "
                    f"values, got {len(values)}"
                )
            noise_rows.append(values)
            continue
        if not current:
            first_line = lineno
        current.extend(values)
        if len(current) == n_values:
            records.append(current)
            current = []
        elif len(current) > n_values:
            raise ValueError(
                f"{path_str}, lines {first_line}-{lineno}: {len(current)} "
                f"values where a {n_ports}-port record has {n_values} "
                f"(frequency plus {n_ports ** 2} pairs, each frequency on a "
                "new line); check the port count"
            )
    if current:
        raise ValueError(
            f"{path_str}, line {first_line}: incomplete record at end of "
            f"file ({len(current)} of {n_values} values); check the port count"
        )
    if not records:
        raise ValueError(f"{path_str}: no network data found")

    scale = _FREQ_SCALE[options["unit"]]
    table = np.array(records)
    frequencies = table[:, 0] * scale
    if np.any(frequencies < 0):
        raise ValueError(f"{path_str}: negative frequency")
    if np.any(np.diff(frequencies) <= 0):
        bad = int(np.argmax(np.diff(frequencies) <= 0)) + 1
        raise ValueError(
            f"{path_str}: frequencies must be strictly increasing "
            f"(record {bad + 1} at {table[bad, 0]} {_UNIT_NAMES[options['unit']]})"
        )

    entries = _pairs_to_complex(table[:, 1::2], table[:, 2::2], options["format"])
    s = entries.reshape(len(records), n_ports, n_ports)
    if n_ports == 2:
        s = s.transpose(0, 2, 1)  # file order is S11 S21 S12 S22
    s = np.ascontiguousarray(s)

    noise = None
    if noise_rows:
        nt = np.array(noise_rows)
        noise_freq = nt[:, 0] * scale
        if np.any(np.diff(noise_freq) <= 0):
            raise ValueError(
                f"{path_str}: noise-parameter frequencies must be strictly "
                "increasing"
            )
        noise = {
            "frequencies": noise_freq,
            "nf_min_db": nt[:, 1],
            "gamma_opt": _pairs_to_complex(nt[:, 2], nt[:, 3], "MA"),
            "rn": nt[:, 4],
        }

    return {
        "frequencies": frequencies,
        "s": s,
        "z0": float(options["z0"]),
        "parameter": "S",
        "n_ports": n_ports,
        "noise": noise,
    }


def write_touchstone(
    path: PathLike,
    frequencies: np.ndarray,
    s: np.ndarray,
    z0: float = 50.0,
    fmt: str = "RI",
    unit: str = "GHz"
) -> None:
    """
    Write S-parameters to a Touchstone version 1.1 file (``.sNp``).

    The layout follows the rules described in :func:`read_touchstone`:
    2-port records in ``S11 S21 S12 S22`` order on one line, larger
    networks row by row with four pairs per line. Numbers are written as
    the shortest decimal strings that convert back to the same float
    (Python's ``repr``), so :func:`read_touchstone` recovers the RI values
    exactly and MA/DB values to rounding error.

    Parameters
    ----------
    path : str or path-like
        Output file. If its extension has the form ``.sNp``, N must equal
        the number of ports in ``s``.
    frequencies : array_like
        Frequencies in Hz, shape (F,), nonnegative and strictly
        increasing.
    s : array_like
        Complex S-parameters, shape (F, N, N), ``s[f, i, j]`` =
        S_(i+1)(j+1).
    z0 : float
        Reference impedance in ohms, the same for all ports.
    fmt : str
        ``'RI'`` (default), ``'MA'`` or ``'DB'`` (case-insensitive).
        Angles are written in degrees. In ``'DB'``, zero magnitudes are
        written as -6000 dB.
    unit : str
        Frequency unit of the file: ``'Hz'``, ``'kHz'``, ``'MHz'`` or
        ``'GHz'`` (default; case-insensitive).

    Raises
    ------
    ValueError
        If ``s`` is not (F, N, N), ``frequencies`` is not (F,), the
        extension implies a different port count, frequencies are negative
        or not strictly increasing, any value is not finite, ``z0`` is not
        positive, or ``fmt`` or ``unit`` is unknown.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> import phased_array as pa
    >>> freqs = np.array([9e9, 10e9])
    >>> s = np.zeros((2, 3, 3), dtype=complex)
    >>> s[:, 0, 1] = s[:, 1, 0] = 0.05
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = os.path.join(tmp, 'three.s3p')
    ...     pa.write_touchstone(path, freqs, s, fmt='DB', unit='MHz')
    ...     with open(path) as f:
    ...         option_line = [ln for ln in f if ln.startswith('#')][0]
    ...     data = pa.read_touchstone(path)
    >>> option_line.strip()
    '# MHz S DB R 50.0'
    >>> bool(np.allclose(data['s'], s, rtol=0, atol=1e-12))
    True
    """
    from . import __version__

    path_str = os.fspath(path)
    s = np.asarray(s, dtype=complex)
    frequencies = np.asarray(frequencies, dtype=float)
    if s.ndim != 3 or s.shape[1] != s.shape[2] or s.shape[1] < 1:
        raise ValueError(f"s must have shape (F, N, N), got {s.shape}")
    n_freq, n_ports = s.shape[0], s.shape[1]
    if n_freq == 0:
        raise ValueError("s must hold at least one frequency")
    if frequencies.shape != (n_freq,):
        raise ValueError(
            f"frequencies must have shape ({n_freq},) to match s, "
            f"got {frequencies.shape}"
        )
    from_extension = _ports_from_extension(path_str)
    if from_extension is not None and from_extension != n_ports:
        raise ValueError(
            f"{path_str}: extension implies {from_extension} ports but s "
            f"has {n_ports}"
        )

    fmt_key = str(fmt).upper()
    if fmt_key not in _FORMATS:
        raise ValueError(f"fmt must be 'RI', 'MA' or 'DB', got {fmt!r}")
    unit_key = str(unit).upper()
    if unit_key not in _FREQ_SCALE:
        raise ValueError(
            f"unit must be 'Hz', 'kHz', 'MHz' or 'GHz', got {unit!r}"
        )
    z0 = float(z0)
    if not (np.isfinite(z0) and z0 > 0):
        raise ValueError(f"z0 must be a positive number, got {z0}")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("frequencies must be finite and nonnegative")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    if not np.all(np.isfinite(s)):
        raise ValueError("s contains non-finite values")

    ordered = s.transpose(0, 2, 1) if n_ports == 2 else s
    flat = ordered.reshape(n_freq, n_ports * n_ports)
    if fmt_key == "RI":
        first, second = flat.real, flat.imag
    else:
        magnitude = np.abs(flat)
        first = (magnitude if fmt_key == "MA"
                 else 20.0 * np.log10(np.maximum(magnitude, _DB_FLOOR)))
        second = np.rad2deg(np.angle(flat))

    lines = [
        f"! Touchstone 1.1 file written by phased_array {__version__}",
        f"! {n_ports}-port S-parameters, {n_freq} "
        f"frequenc{'y' if n_freq == 1 else 'ies'}",
        f"# {_UNIT_NAMES[unit_key]} S {fmt_key} R {z0!r}",
    ]
    scale = _FREQ_SCALE[unit_key]
    for fi in range(n_freq):
        f_text = repr(float(frequencies[fi] / scale))
        pairs = [f"{float(a)!r} {float(b)!r}"
                 for a, b in zip(first[fi], second[fi])]
        if n_ports <= 2:
            lines.append("  ".join([f_text] + pairs))
            continue
        indent = " " * len(f_text)
        for row in range(n_ports):
            row_pairs = pairs[row * n_ports:(row + 1) * n_ports]
            for start in range(0, n_ports, 4):
                lead = f_text if (row == 0 and start == 0) else indent
                lines.append("  ".join([lead] + row_pairs[start:start + 4]))

    with open(path_str, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ============== Tabulated element patterns ==============

_PATTERN_ROLES = {
    "re_im": ("theta", "phi", "e_theta_re", "e_theta_im",
              "e_phi_re", "e_phi_im"),
    "db_phase": ("theta", "phi", "e_theta_db", "e_theta_phase_deg",
                 "e_phi_db", "e_phi_phase_deg"),
}


def _split_rows(lines: List[str], delimiter: Optional[str]) -> List[List[str]]:
    """Split text lines into fields: whitespace if delimiter is None."""
    if delimiter is None:
        return [line.split() for line in lines]
    return list(csv.reader(lines, delimiter=delimiter))


def load_element_pattern_csv(
    path: PathLike,
    columns: Mapping[str, Union[str, int]],
    angle_unit: str = "deg",
    delimiter: Optional[str] = ",",
    field_format: str = "re_im",
    comments: Optional[str] = "#",
    skip_rows: int = 0,
    method: str = "linear"
) -> GriddedElementPattern:
    """
    Load a tabulated polarized element pattern from a delimited text file.

    The file holds a header line followed by one row per (theta, phi)
    sample, in any row order, as exported by full-wave solvers and
    measurement ranges. The samples must cover a complete theta/phi grid;
    see :meth:`~phased_array.GriddedElementPattern.from_table`.

    Parameters
    ----------
    path : str or path-like
        File to read.
    columns : mapping
        Maps each field role to a header name (str, matched exactly after
        stripping whitespace and quotes) or a 0-based column index (int).
        For ``field_format='re_im'`` the roles are ``'theta'``, ``'phi'``,
        ``'e_theta_re'``, ``'e_theta_im'``, ``'e_phi_re'``,
        ``'e_phi_im'``. For ``field_format='db_phase'`` they are
        ``'theta'``, ``'phi'``, ``'e_theta_db'``, ``'e_theta_phase_deg'``,
        ``'e_phi_db'``, ``'e_phi_phase_deg'``.
    angle_unit : str
        Unit of the theta and phi columns, ``'deg'`` (default) or
        ``'rad'``. Phase columns are always in degrees.
    delimiter : str or None
        Field separator, default ``','``. ``None`` splits on runs of
        whitespace; header names containing spaces then cannot be matched,
        so use column indices.
    field_format : str
        ``'re_im'`` (default): real and imaginary parts of E_theta and
        E_phi. ``'db_phase'``: ``20*log10|E|`` in dB and phase in degrees
        (a gain or directivity column in dBi works, since
        ``10*log10(G) = 20*log10(sqrt(G))``); ``-inf`` dB reads as zero
        field.
    comments : str or None
        Lines starting with this prefix (after leading whitespace) are
        skipped, default ``'#'``. Pass None if the header line itself
        starts with ``#``.
    skip_rows : int
        Number of lines to skip at the top of the file before looking for
        the header.
    method : str
        Interpolation method passed to
        :class:`~phased_array.GriddedElementPattern`.

    Returns
    -------
    pattern : GriddedElementPattern
        Callable ``f(theta, phi) -> (E_theta, E_phi)`` with angles in
        radians.

    Raises
    ------
    ValueError
        If ``columns`` has unknown or missing roles, a header name is not
        found or appears twice, ``field_format`` or ``angle_unit`` is
        unknown, a data row is too short or holds a non-number, or the
        samples do not form a complete grid.

    Notes
    -----
    The first line after ``skip_rows`` that is neither blank nor a comment
    is the header. After it, blank lines, comment lines and lines without
    any digit (such as a ``-----`` rule under the header) are skipped.
    Theta is measured from +z and phi from +x; E_theta and E_phi are the
    components along the spherical unit vectors, as everywhere in the
    package (see :doc:`/user_guide/conventions`).

    Examples
    --------
    A far-field export with the header
    ``Theta [deg],Phi [deg],re(rETheta) [V],im(rETheta) [V],re(rEPhi) [V],im(rEPhi) [V]``
    maps as

    >>> columns = {
    ...     'theta': 'Theta [deg]', 'phi': 'Phi [deg]',
    ...     'e_theta_re': 're(rETheta) [V]', 'e_theta_im': 'im(rETheta) [V]',
    ...     'e_phi_re': 're(rEPhi) [V]', 'e_phi_im': 'im(rEPhi) [V]',
    ... }

    Write such a file from an analytic element and load it back:

    >>> import os, tempfile
    >>> import numpy as np
    >>> import phased_array as pa
    >>> f = pa.ideal_patch_element('x')
    >>> T, P = np.meshgrid(np.arange(0, 91, 2.0), np.arange(0, 360, 5.0),
    ...                    indexing='ij')
    >>> Et, Ep = f(np.deg2rad(T), np.deg2rad(P))
    >>> header = ','.join(columns.values())
    >>> rows = np.column_stack([T.ravel(), P.ravel(), Et.real.ravel(),
    ...                         Et.imag.ravel(), Ep.real.ravel(),
    ...                         Ep.imag.ravel()])
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = os.path.join(tmp, 'element.csv')
    ...     np.savetxt(path, rows, delimiter=',', header=header, comments='')
    ...     pattern = pa.load_element_pattern_csv(path, columns)
    >>> E_theta, E_phi = pattern(np.array([0.4]), np.array([1.0]))
    >>> E_ref, _ = f(np.array([0.4]), np.array([1.0]))
    >>> bool(abs(E_theta[0] - E_ref[0]) < 2e-3)
    True
    """
    if field_format not in _PATTERN_ROLES:
        raise ValueError(
            f"field_format must be 're_im' or 'db_phase', got {field_format!r}"
        )
    if angle_unit not in ("deg", "rad"):
        raise ValueError(f"angle_unit must be 'deg' or 'rad', got {angle_unit!r}")
    roles = _PATTERN_ROLES[field_format]
    unknown = sorted(set(columns) - set(roles))
    if unknown:
        raise ValueError(
            f"unknown column role(s) {unknown} for field_format="
            f"{field_format!r}; expected {list(roles)}"
        )
    missing = [role for role in roles if role not in columns]
    if missing:
        raise ValueError(
            f"columns is missing role(s) {missing} required for "
            f"field_format={field_format!r}"
        )

    path_str = os.fspath(path)
    with open(path_str, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()

    header: Optional[List[str]] = None
    data_text: List[str] = []
    data_lineno: List[int] = []
    for lineno, line in enumerate(lines[skip_rows:], start=skip_rows + 1):
        stripped = line.strip()
        if not stripped or (comments and stripped.startswith(comments)):
            continue
        if header is None:
            header = [field.strip().strip("\"'").strip()
                      for field in _split_rows([line], delimiter)[0]]
            continue
        if not any(ch.isdigit() for ch in stripped):
            continue
        data_text.append(line)
        data_lineno.append(lineno)
    if header is None:
        raise ValueError(f"{path_str}: no header line found")
    if not data_text:
        raise ValueError(f"{path_str}: no data rows after the header")

    indices = []
    for role in roles:
        spec = columns[role]
        if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
            if spec < 0:
                raise ValueError(f"column index for {role!r} must be >= 0, got {spec}")
            indices.append(int(spec))
        elif isinstance(spec, str):
            name = spec.strip().strip("\"'").strip()
            matches = [i for i, field in enumerate(header) if field == name]
            if not matches:
                raise ValueError(
                    f"{path_str}: column {spec!r} (for {role!r}) is not in "
                    f"the header {header}; check the names and the delimiter"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"{path_str}: column {spec!r} appears {len(matches)} "
                    "times in the header; use a column index"
                )
            indices.append(matches[0])
        else:
            raise TypeError(
                f"columns[{role!r}] must be a header name or column index, "
                f"got {type(spec).__name__}"
            )

    rows = _split_rows(data_text, delimiter)
    table = np.empty((len(rows), len(roles)))
    for r, row in enumerate(rows):
        try:
            table[r] = [float(row[c]) for c in indices]
        except IndexError:
            raise ValueError(
                f"{path_str}, line {data_lineno[r]}: {len(row)} fields, but "
                f"column index {max(indices)} is needed"
            ) from None
        except ValueError:
            bad = next(row[c] for c in indices if not _is_number(row[c]))
            raise ValueError(
                f"{path_str}, line {data_lineno[r]}: cannot parse {bad!r} "
                "as a number"
            ) from None

    theta, phi = table[:, 0], table[:, 1]
    if field_format == "re_im":
        e_theta = table[:, 2] + 1j * table[:, 3]
        e_phi = table[:, 4] + 1j * table[:, 5]
    else:
        e_theta = _pairs_to_complex(table[:, 2], table[:, 3], "DB")
        e_phi = _pairs_to_complex(table[:, 4], table[:, 5], "DB")

    try:
        return GriddedElementPattern.from_table(
            theta, phi, e_theta, e_phi, angle_unit=angle_unit, method=method
        )
    except ValueError as exc:
        raise ValueError(f"{path_str}: {exc}") from None
