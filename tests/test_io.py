"""
Tests for phased_array.io: Touchstone files and tabulated element patterns.
"""

import numpy as np
import pytest

import phased_array as pa


def write_text(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def random_s(rng, n_freq, n_ports, scale=0.9):
    """Random complex S-parameters with spectral norm below ``scale``."""
    s = rng.normal(size=(n_freq, n_ports, n_ports)) \
        + 1j * rng.normal(size=(n_freq, n_ports, n_ports))
    norms = np.linalg.norm(s, ord=2, axis=(1, 2))
    return s * (scale / norms)[:, None, None]


# ============== read_touchstone: hand-written files ==============

class TestReadTouchstoneFormats:

    def test_one_port_ri_hz_with_comments(self, tmp_path):
        path = write_text(tmp_path, "load.s1p", (
            "! measured load\n"
            "! second comment line\n"
            "# Hz S RI R 50 ! option line comment\n"
            "1.0e9  0.5 -0.25\n"
            "! comment between records\n"
            "\n"
            "2.0e9  0.125 0.75 ! trailing comment\n"
        ))
        data = pa.read_touchstone(path)
        np.testing.assert_array_equal(data["frequencies"], [1e9, 2e9])
        assert data["s"].shape == (2, 1, 1)
        np.testing.assert_allclose(data["s"][:, 0, 0], [0.5 - 0.25j, 0.125 + 0.75j])
        assert data["z0"] == 50.0
        assert data["parameter"] == "S"
        assert data["n_ports"] == 1
        assert data["noise"] is None

    def test_defaults_without_option_line(self, tmp_path):
        # Default option line is "# GHz S MA R 50"
        path = write_text(tmp_path, "x.s1p", "1.5  0.5 90\n")
        data = pa.read_touchstone(path)
        assert data["frequencies"][0] == pytest.approx(1.5e9)
        assert data["s"][0, 0, 0] == pytest.approx(0.5j, abs=1e-15)
        assert data["z0"] == 50.0

    def test_partial_option_line_uses_defaults(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# MHz\n100  0.5 180\n")
        data = pa.read_touchstone(path)
        assert data["frequencies"][0] == pytest.approx(100e6)
        assert data["s"][0, 0, 0] == pytest.approx(-0.5, abs=1e-15)  # MA

    def test_option_line_any_order_and_case(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# r 75 ri khz s\n3  0.1 0.2\n")
        data = pa.read_touchstone(path)
        assert data["frequencies"][0] == pytest.approx(3e3)
        assert data["s"][0, 0, 0] == pytest.approx(0.1 + 0.2j)
        assert data["z0"] == 75.0

    @pytest.mark.parametrize("fmt, a, b", [
        ("RI", 0.3535533905932738, 0.3535533905932738),
        ("MA", 0.5, 45.0),
        ("DB", float(20 * np.log10(0.5)), 45.0),
    ])
    def test_formats_give_same_value(self, tmp_path, fmt, a, b):
        path = write_text(tmp_path, "x.s1p", f"# GHz S {fmt} R 50\n10 {a!r} {b!r}\n")
        data = pa.read_touchstone(path)
        assert data["s"][0, 0, 0] == pytest.approx(0.5 * np.exp(1j * np.pi / 4), abs=1e-15)

    def test_two_port_column_major_order(self, tmp_path):
        # Values tagged by port pair: 0.ij means S_ij
        path = write_text(tmp_path, "amp.s2p", (
            "# GHz S RI R 50\n"
            "! freq  S11       S21       S12       S22\n"
            "1.0  0.11 -1.1  0.21 -2.1  0.12 -1.2  0.22 -2.2\n"
            "2.0  0.11 -1.1  0.21 -2.1  0.12 -1.2  0.22 -2.2\n"
        ))
        s = pa.read_touchstone(path)["s"]
        assert s.shape == (2, 2, 2)
        assert s[0, 0, 0] == 0.11 - 1.1j   # S11
        assert s[0, 1, 0] == 0.21 - 2.1j   # S21: second pair on the line
        assert s[0, 0, 1] == 0.12 - 1.2j   # S12: third pair on the line
        assert s[0, 1, 1] == 0.22 - 2.2j   # S22

    def test_three_port_row_per_line(self, tmp_path):
        path = write_text(tmp_path, "x.s3p", (
            "# GHz S RI R 50\n"
            "1.0  11 0  12 0  13 0\n"
            "     21 0  22 0  23 0\n"
            "     31 0  32 0  33 0\n"
        ))
        s = pa.read_touchstone(path)["s"]
        expected = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        np.testing.assert_array_equal(s[0].real, expected)

    def test_four_port_wrapped_rows(self, tmp_path):
        # Each row of a 4-port record fills one line (4 pairs); two
        # frequencies, with a comment inside the first record
        rows = []
        for f in (1.0, 2.0):
            for i in range(1, 5):
                pairs = "  ".join(f"{10 * i + j} {-f * (10 * i + j)}" for j in range(1, 5))
                lead = f"{f}" if i == 1 else "   "
                rows.append(f"{lead}  {pairs}")
        rows.insert(2, "! comment inside a record")
        path = write_text(tmp_path, "quad.s4p", "# GHz S RI R 50\n" + "\n".join(rows) + "\n")
        data = pa.read_touchstone(path)
        ij = 10 * np.arange(1, 5)[:, None] + np.arange(1, 5)[None, :]
        assert data["s"].shape == (2, 4, 4)
        np.testing.assert_array_equal(data["s"][0], ij - 1j * ij)
        np.testing.assert_array_equal(data["s"][1], ij - 2j * ij)

    def test_five_port_wraps_after_four_pairs(self, tmp_path):
        # Rows of 5 pairs: 4 pairs on one line, the fifth on the next
        lines = ["# MHz S MA R 50"]
        for i in range(1, 6):
            pairs = [f"{10 * i + j} 0" for j in range(1, 6)]
            lead = "500" if i == 1 else "   "
            lines.append(lead + "  " + "  ".join(pairs[:4]))
            lines.append("     " + pairs[4])
        path = write_text(tmp_path, "x.s5p", "\n".join(lines) + "\n")
        data = pa.read_touchstone(path)
        ij = 10 * np.arange(1, 6)[:, None] + np.arange(1, 6)[None, :]
        np.testing.assert_allclose(data["s"][0], ij)
        assert data["frequencies"][0] == pytest.approx(500e6)

    def test_port_count_from_argument(self, tmp_path):
        path = write_text(tmp_path, "data.txt", "# GHz S RI\n1 0.1 0 0.2 0 0.3 0 0.4 0\n")
        data = pa.read_touchstone(path, n_ports=2)
        assert data["s"][0, 1, 0] == 0.2
        with pytest.raises(ValueError, match="pass n_ports"):
            pa.read_touchstone(path)

    def test_uppercase_extension(self, tmp_path):
        path = write_text(tmp_path, "X.S1P", "# GHz S RI\n1 0.1 0\n")
        assert pa.read_touchstone(path)["n_ports"] == 1

    def test_n_ports_contradicting_extension(self, tmp_path):
        path = write_text(tmp_path, "x.s2p", "# GHz S RI\n1 0.1 0 0.2 0 0.3 0 0.4 0\n")
        with pytest.raises(ValueError, match="contradicts"):
            pa.read_touchstone(path, n_ports=3)

    def test_two_port_noise_parameters(self, tmp_path):
        path = write_text(tmp_path, "lna.s2p", (
            "# GHz S MA R 50\n"
            "1.0  0.5 -30  4.0 150  0.01 60  0.4 -20\n"
            "2.0  0.4 -60  3.5 120  0.02 50  0.3 -40\n"
            "! noise parameters: f NFmin |Gopt| ang(Gopt) Rn\n"
            "1.0  0.8  0.6  45  0.2\n"
            "2.0  1.0  0.5  90  0.25\n"
        ))
        data = pa.read_touchstone(path)
        assert data["s"].shape == (2, 2, 2)
        assert data["s"][1, 1, 0] == pytest.approx(3.5 * np.exp(1j * np.deg2rad(120)))
        noise = data["noise"]
        np.testing.assert_allclose(noise["frequencies"], [1e9, 2e9])
        np.testing.assert_allclose(noise["nf_min_db"], [0.8, 1.0])
        np.testing.assert_allclose(noise["gamma_opt"],
                                   [0.6 * np.exp(1j * np.pi / 4), 0.5j], atol=1e-15)
        np.testing.assert_allclose(noise["rn"], [0.2, 0.25])

    def test_utf8_bom_and_crlf(self, tmp_path):
        path = tmp_path / "x.s1p"
        path.write_bytes(b"\xef\xbb\xbf! caf\xc3\xa9 \xb0\r\n# GHz S RI\r\n1 0.1 0.2\r\n")
        assert pa.read_touchstone(path)["s"][0, 0, 0] == 0.1 + 0.2j


class TestReadTouchstoneRejects:

    @pytest.mark.parametrize("param", ["Y", "Z", "H", "G"])
    def test_non_s_parameters(self, tmp_path, param):
        path = write_text(tmp_path, "x.s2p", f"# GHz {param} RI R 50\n1 1 0 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match=f"{param}-parameter files are not supported"):
            pa.read_touchstone(path)

    def test_touchstone_2(self, tmp_path):
        path = write_text(tmp_path, "x.s2p", (
            "! v2 file\n[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 2\n"
            "[Network Data]\n1 0.1 0 0.2 0 0.3 0 0.4 0\n[End]\n"
        ))
        with pytest.raises(ValueError, match=r"\[Version\].*Touchstone 2\.0"):
            pa.read_touchstone(path)

    def test_unknown_option_token(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S XY R 50\n1 0.1 0\n")
        with pytest.raises(ValueError, match="unrecognized option-line token 'XY'"):
            pa.read_touchstone(path)

    def test_per_port_reference_impedances(self, tmp_path):
        path = write_text(tmp_path, "x.s2p", "# GHz S RI R 50 75\n1 0.1 0 0.2 0 0.3 0 0.4 0\n")
        with pytest.raises(ValueError, match="single reference impedance"):
            pa.read_touchstone(path)

    def test_option_set_twice(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz MHz S RI\n1 0.1 0\n")
        with pytest.raises(ValueError, match="frequency unit twice"):
            pa.read_touchstone(path)

    def test_missing_impedance_value(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S RI R\n1 0.1 0\n")
        with pytest.raises(ValueError, match="followed by the reference impedance"):
            pa.read_touchstone(path)

    def test_conflicting_second_option_line(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S RI\n1 0.1 0\n# MHz S RI\n2 0.1 0\n")
        with pytest.raises(ValueError, match="second option line differs"):
            pa.read_touchstone(path)

    def test_identical_second_option_line_is_ignored(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S RI\n# ghz s ri r 50\n1 0.1 0\n")
        assert pa.read_touchstone(path)["s"][0, 0, 0] == 0.1

    def test_option_line_after_data(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "1 0.1 0\n# MHz S RI\n")
        with pytest.raises(ValueError, match="before the data"):
            pa.read_touchstone(path)

    def test_short_two_port_record(self, tmp_path):
        path = write_text(tmp_path, "x.s2p", "# GHz S RI\n1 0.1 0 0.2 0 0.3 0 0.4\n")
        with pytest.raises(ValueError, match="incomplete record"):
            pa.read_touchstone(path)

    def test_two_records_on_one_line(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S RI\n1 0.1 0 2 0.2 0\n")
        with pytest.raises(ValueError, match="each frequency on a new line"):
            pa.read_touchstone(path)

    def test_wrong_port_count_from_extension(self, tmp_path):
        # A 2-port record read as a 3-port file
        path = write_text(tmp_path, "x.s3p", "# GHz S RI\n1 0.1 0 0.2 0 0.3 0 0.4 0\n")
        with pytest.raises(ValueError, match="incomplete record"):
            pa.read_touchstone(path)

    def test_non_increasing_frequencies(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S RI\n2 0.1 0\n1 0.2 0\n")
        with pytest.raises(ValueError, match="strictly increasing"):
            pa.read_touchstone(path)

    def test_non_numeric_value(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S RI\n1 0.1 abc\n")
        with pytest.raises(ValueError, match="line 2: cannot parse"):
            pa.read_touchstone(path)

    def test_non_finite_value(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "# GHz S RI\n1 nan 0\n")
        with pytest.raises(ValueError, match="non-finite"):
            pa.read_touchstone(path)

    def test_no_data(self, tmp_path):
        path = write_text(tmp_path, "x.s1p", "! empty\n# GHz S RI\n")
        with pytest.raises(ValueError, match="no network data"):
            pa.read_touchstone(path)

    def test_malformed_noise_block(self, tmp_path):
        path = write_text(tmp_path, "x.s2p", (
            "# GHz S RI\n1 0.1 0 0.2 0 0.3 0 0.4 0\n2 0.1 0 0.2 0 0.3 0 0.4 0\n"
            "1 0.8 0.6 45 0.2\n2 0.1 0 0.2 0 0.3 0 0.4 0\n"
        ))
        with pytest.raises(ValueError, match="expected 5 noise-parameter values"):
            pa.read_touchstone(path)


# ============== write_touchstone ==============

class TestWriteTouchstone:

    @pytest.mark.parametrize("n_ports", [1, 2, 3, 4])
    @pytest.mark.parametrize("fmt", ["RI", "MA", "DB", "ri"])
    @pytest.mark.parametrize("unit", ["Hz", "kHz", "MHz", "GHz"])
    def test_round_trip(self, tmp_path, n_ports, fmt, unit):
        rng = np.random.default_rng(n_ports)
        freqs = np.linspace(8e9, 12e9, 5)
        s = random_s(rng, 5, n_ports)
        path = tmp_path / f"net.s{n_ports}p"
        pa.write_touchstone(path, freqs, s, z0=75.0, fmt=fmt, unit=unit)
        data = pa.read_touchstone(path)
        np.testing.assert_allclose(data["s"], s, rtol=0, atol=1e-12)
        np.testing.assert_allclose(data["frequencies"], freqs, rtol=1e-12)
        assert data["z0"] == 75.0
        assert data["n_ports"] == n_ports

    def test_round_trip_large_network(self, tmp_path):
        rng = np.random.default_rng(9)
        s = random_s(rng, 3, 7)
        path = tmp_path / "x.s7p"
        pa.write_touchstone(path, [1e9, 2e9, 3e9], s, fmt="MA")
        np.testing.assert_allclose(pa.read_touchstone(path)["s"], s, rtol=0, atol=1e-12)

    def test_two_port_written_s21_before_s12(self, tmp_path):
        s = np.array([[[0.11, 0.12], [0.21, 0.22]]])
        path = tmp_path / "x.s2p"
        pa.write_touchstone(path, [1e9], s)
        data_line = [ln for ln in path.read_text().splitlines()
                     if ln and ln[0] not in "!#"][0]
        values = [float(t) for t in data_line.split()]
        assert values[0] == 1.0  # GHz
        assert values[1::2] == [0.11, 0.21, 0.12, 0.22]

    def test_layout_wraps_rows_at_four_pairs(self, tmp_path):
        s = np.zeros((2, 5, 5), dtype=complex)
        path = tmp_path / "x.s5p"
        pa.write_touchstone(path, [1e9, 2e9], s)
        data_lines = [ln for ln in path.read_text().splitlines()
                      if ln and ln[0] not in "!#"]
        counts = [len(ln.split()) for ln in data_lines]
        # Per frequency: 5 rows, each split 4 + 1 pairs; frequency leads row 1
        assert counts == ([9, 2] + [8, 2] * 4) * 2

    def test_three_port_one_row_per_line(self, tmp_path):
        path = tmp_path / "x.s3p"
        pa.write_touchstone(path, [1e9], np.zeros((1, 3, 3)))
        data_lines = [ln for ln in path.read_text().splitlines()
                      if ln and ln[0] not in "!#"]
        assert [len(ln.split()) for ln in data_lines] == [7, 6, 6]

    def test_option_line(self, tmp_path):
        path = tmp_path / "x.s1p"
        pa.write_touchstone(path, [1e6], np.zeros((1, 1, 1)), z0=50, fmt="db", unit="mhz")
        assert "# MHz S DB R 50.0" in path.read_text().splitlines()

    def test_db_zero_magnitude(self, tmp_path):
        s = np.array([[[0.0, 0.5], [0.5, 0.0]]])
        path = tmp_path / "x.s2p"
        pa.write_touchstone(path, [1e9], s, fmt="DB")
        np.testing.assert_allclose(pa.read_touchstone(path)["s"], s, atol=1e-12)

    def test_non_sNp_extension(self, tmp_path):
        path = tmp_path / "net.txt"
        pa.write_touchstone(path, [1e9], np.full((1, 2, 2), 0.1))
        assert pa.read_touchstone(path, n_ports=2)["s"].shape == (1, 2, 2)

    @pytest.mark.parametrize("kwargs, match", [
        (dict(s=np.zeros((2, 2))), "shape"),
        (dict(s=np.zeros((2, 2, 3))), "shape"),
        (dict(frequencies=[1e9]), "frequencies must have shape"),
        (dict(frequencies=[2e9, 1e9]), "strictly increasing"),
        (dict(frequencies=[-1.0, 1e9]), "nonnegative"),
        (dict(s=np.full((2, 2, 2), np.nan)), "non-finite"),
        (dict(z0=0.0), "z0"),
        (dict(fmt="XY"), "fmt"),
        (dict(unit="THz"), "unit"),
    ])
    def test_invalid_input(self, tmp_path, kwargs, match):
        args = dict(frequencies=[1e9, 2e9], s=np.zeros((2, 2, 2)))
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            pa.write_touchstone(tmp_path / "x.s2p", **args)

    def test_extension_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="extension implies 3 ports"):
            pa.write_touchstone(tmp_path / "x.s3p", [1e9], np.zeros((1, 2, 2)))


# ============== Touchstone -> coupling and active reflection ==============

class TestTouchstoneCoupling:

    @pytest.mark.parametrize("formulation", ["impedance", "voltage"])
    def test_active_reflection_from_file(self, tmp_path, formulation):
        rng = np.random.default_rng(3)
        n = 4
        freqs = np.array([9e9, 10e9, 11e9])
        s_true = random_s(rng, len(freqs), n, scale=0.6)
        path = tmp_path / "array.s4p"
        pa.write_touchstone(path, freqs, s_true, fmt="MA")
        data = pa.read_touchstone(path)

        w = np.exp(1j * rng.uniform(0, 2 * np.pi, n)) * rng.uniform(0.5, 1.0, n)
        for i in range(len(freqs)):
            S = data["s"][i]
            C = pa.mutual_coupling_matrix_measured(S, formulation=formulation)
            expected = (S @ w) / w
            gammas = [pa.active_reflection_coefficient(C, w, m, formulation=formulation)
                      for m in range(n)]
            np.testing.assert_allclose(gammas, expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(expected, (s_true[i] @ w) / w, atol=1e-12)


# ============== GriddedElementPattern.from_table ==============

def patch_samples(theta_deg, phi_deg, cos_exp=1.0):
    """Flat (theta, phi, E_theta, E_phi) columns of the ideal patch element."""
    f = pa.ideal_patch_element("x", cos_exp=cos_exp)
    T, P = np.meshgrid(theta_deg, phi_deg, indexing="ij")
    Et, Ep = f(np.deg2rad(T), np.deg2rad(P))
    return T.ravel(), P.ravel(), Et.ravel(), Ep.ravel()


class TestFromTable:

    @pytest.mark.parametrize("phi_deg", [
        np.arange(-180, 181, 5.0),   # [-180, 180]
        np.arange(0, 360, 5.0),      # [0, 360)
    ])
    def test_shuffled_rows_match_sorted_grid(self, phi_deg):
        theta_deg = np.arange(0, 91, 3.0)
        t, p, et, ep = patch_samples(theta_deg, phi_deg)
        perm = np.random.default_rng(1).permutation(t.size)
        table = pa.GriddedElementPattern.from_table(t[perm], p[perm], et[perm], ep[perm])
        direct = pa.GriddedElementPattern(
            np.deg2rad(theta_deg), np.deg2rad(phi_deg),
            et.reshape(theta_deg.size, -1), ep.reshape(theta_deg.size, -1)
        )
        rng = np.random.default_rng(2)
        qt = rng.uniform(0, np.pi / 2, 200)
        qp = rng.uniform(-2 * np.pi, 2 * np.pi, 200)
        for a, b in zip(table(qt, qp), direct(qt, qp)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_radians(self):
        t, p, et, ep = patch_samples(np.arange(0, 91, 3.0), np.arange(0, 360, 5.0))
        pat_deg = pa.GriddedElementPattern.from_table(t, p, et, ep)
        pat_rad = pa.GriddedElementPattern.from_table(
            np.deg2rad(t), np.deg2rad(p), et, ep, angle_unit="rad"
        )
        q = (np.array([0.2, 1.0]), np.array([0.5, 4.0]))
        np.testing.assert_allclose(pat_deg(*q)[0], pat_rad(*q)[0], atol=1e-12)

    def test_angle_noise_is_merged(self):
        t, p, et, ep = patch_samples(np.arange(0, 91, 10.0), np.arange(0, 360, 30.0))
        noise = np.random.default_rng(0).uniform(-1e-9, 1e-9, t.size)
        pat = pa.GriddedElementPattern.from_table(t + noise, p + noise, et, ep)
        assert pat(np.array([0.0]), np.array([0.0]))[0][0] == pytest.approx(1.0, abs=1e-9)

    def test_incomplete_grid(self):
        t, p, et, ep = patch_samples(np.arange(0, 91, 10.0), np.arange(0, 360, 30.0))
        with pytest.raises(ValueError, match="incomplete grid: 1 of 120"):
            pa.GriddedElementPattern.from_table(t[1:], p[1:], et[1:], ep[1:])

    def test_duplicate_sample(self):
        t, p, et, ep = patch_samples(np.arange(0, 91, 10.0), np.arange(0, 360, 30.0))
        t, p = np.append(t, t[5]), np.append(p, p[5])
        et, ep = np.append(et, et[5]), np.append(ep, ep[5])
        with pytest.raises(ValueError, match="more than once"):
            pa.GriddedElementPattern.from_table(t, p, et, ep)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            pa.GriddedElementPattern.from_table([0, 1], [0, 1], [1, 1], [1])

    def test_negative_theta(self):
        t, p, et, ep = patch_samples(np.arange(-90, 91, 10.0), np.arange(0, 180, 30.0))
        with pytest.raises(ValueError, match="theta must lie in"):
            pa.GriddedElementPattern.from_table(t, p, et, ep)

    def test_phi_span_too_large(self):
        t, p, et, ep = patch_samples(np.arange(0, 91, 10.0), np.arange(-180, 361, 30.0))
        with pytest.raises(ValueError, match="more than 360"):
            pa.GriddedElementPattern.from_table(t, p, et, ep)

    def test_single_phi_value(self):
        t, p, et, ep = patch_samples(np.arange(0, 91, 10.0), np.array([0.0]))
        with pytest.raises(ValueError, match="at least two"):
            pa.GriddedElementPattern.from_table(t, p, et, ep)

    def test_non_finite(self):
        t, p, et, ep = patch_samples(np.arange(0, 91, 10.0), np.arange(0, 360, 30.0))
        et[3] = np.nan
        with pytest.raises(ValueError, match="e_theta contains non-finite"):
            pa.GriddedElementPattern.from_table(t, p, et, ep)

    def test_bad_angle_unit(self):
        with pytest.raises(ValueError, match="angle_unit"):
            pa.GriddedElementPattern.from_table([0], [0], [1], [1], angle_unit="grad")


# ============== load_element_pattern_csv ==============

RE_IM_COLUMNS = {
    "theta": "Theta[deg]", "phi": "Phi[deg]",
    "e_theta_re": "Re(Etheta)", "e_theta_im": "Im(Etheta)",
    "e_phi_re": "Re(Ephi)", "e_phi_im": "Im(Ephi)",
}
DB_COLUMNS = {
    "theta": "Theta[deg]", "phi": "Phi[deg]",
    "e_theta_db": "dB(Etheta)", "e_theta_phase_deg": "Phase(Etheta)[deg]",
    "e_phi_db": "dB(Ephi)", "e_phi_phase_deg": "Phase(Ephi)[deg]",
}


def write_pattern_csv(path, theta_deg, phi_deg, field_format="re_im",
                      delimiter=",", shuffle_seed=0, cos_exp=1.0):
    t, p, et, ep = patch_samples(theta_deg, phi_deg, cos_exp)
    if field_format == "re_im":
        names = list(RE_IM_COLUMNS.values())
        cols = [t, p, et.real, et.imag, ep.real, ep.imag]
    else:
        def db(e):
            return 20 * np.log10(np.maximum(np.abs(e), 1e-15))
        names = list(DB_COLUMNS.values())
        cols = [t, p, db(et), np.rad2deg(np.angle(et)), db(ep), np.rad2deg(np.angle(ep))]
    table = np.column_stack(cols)
    if shuffle_seed is not None:
        table = table[np.random.default_rng(shuffle_seed).permutation(len(table))]
    lines = [delimiter.join(names)]
    lines += [delimiter.join(repr(float(v)) for v in row) for row in table]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestLoadElementPatternCSV:

    @pytest.mark.parametrize("phi_deg", [
        np.arange(-180, 181, 5.0),
        np.arange(0, 360, 5.0),
    ])
    def test_matches_analytic_element(self, tmp_path, phi_deg):
        theta_deg = np.arange(0, 91, 2.0)
        path = write_pattern_csv(tmp_path / "patch.csv", theta_deg, phi_deg, cos_exp=1.5)
        pattern = pa.load_element_pattern_csv(path, RE_IM_COLUMNS)
        f = pa.ideal_patch_element("x", cos_exp=1.5)

        # Exact at the samples
        T, P = np.meshgrid(np.deg2rad(theta_deg), np.deg2rad(phi_deg), indexing="ij")
        for a, b in zip(pattern(T, P), f(T, P)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

        # Within bilinear-interpolation error between samples
        rng = np.random.default_rng(4)
        qt = rng.uniform(0, np.deg2rad(88), 500)
        qp = rng.uniform(0, 2 * np.pi, 500)
        for a, b in zip(pattern(qt, qp), f(qt, qp)):
            assert np.max(np.abs(a - b)) < 2e-3

    def test_db_phase_matches_re_im(self, tmp_path):
        theta_deg, phi_deg = np.arange(0, 91, 5.0), np.arange(-180, 181, 10.0)
        ri = pa.load_element_pattern_csv(
            write_pattern_csv(tmp_path / "ri.csv", theta_deg, phi_deg), RE_IM_COLUMNS)
        dbp = pa.load_element_pattern_csv(
            write_pattern_csv(tmp_path / "db.csv", theta_deg, phi_deg, "db_phase",
                              shuffle_seed=7),
            DB_COLUMNS, field_format="db_phase")
        rng = np.random.default_rng(5)
        qt, qp = rng.uniform(0, np.pi / 2, 300), rng.uniform(-np.pi, np.pi, 300)
        for a, b in zip(ri(qt, qp), dbp(qt, qp)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_db_minus_inf_is_zero_field(self, tmp_path):
        path = tmp_path / "x.csv"
        rows = ["Theta[deg],Phi[deg],dB(Etheta),Phase(Etheta)[deg],dB(Ephi),Phase(Ephi)[deg]"]
        for t in (0, 90):
            for p in (0, 90, 180, 270):
                rows.append(f"{t},{p},0,0,-inf,0")
        path.write_text("\n".join(rows) + "\n")
        pattern = pa.load_element_pattern_csv(path, DB_COLUMNS, field_format="db_phase")
        Et, Ep = pattern(np.array([0.5]), np.array([1.0]))
        assert Et[0] == pytest.approx(1.0) and Ep[0] == 0

    def test_whitespace_table_with_indices_and_rule(self, tmp_path):
        # Header names contain spaces, a rule line follows the header, and
        # a preamble line precedes it
        theta_deg, phi_deg = np.arange(0, 91, 10.0), np.arange(0, 360, 30.0)
        t, p, et, ep = patch_samples(theta_deg, phi_deg)
        lines = ["Exported far field, 10 GHz",
                 "# comment",
                 "Theta [deg]   Phi [deg]   Re(Et)   Im(Et)   Re(Ep)   Im(Ep)",
                 "-" * 60]
        lines += [f"{a:8.3f} {b:8.3f} {c.real: .17e} {c.imag: .17e} "
                  f"{d.real: .17e} {d.imag: .17e}" for a, b, c, d in zip(t, p, et, ep)]
        path = tmp_path / "farfield.txt"
        path.write_text("\n".join(lines) + "\n")
        columns = dict(theta=0, phi=1, e_theta_re=2, e_theta_im=3, e_phi_re=4, e_phi_im=5)
        pattern = pa.load_element_pattern_csv(path, columns, delimiter=None, skip_rows=1)
        T, P = np.meshgrid(np.deg2rad(theta_deg), np.deg2rad(phi_deg), indexing="ij")
        np.testing.assert_allclose(pattern(T, P)[0], et.reshape(T.shape), atol=1e-12)

    def test_quoted_header_semicolon_radians(self, tmp_path):
        theta, phi = np.linspace(0, np.pi / 2, 7), np.linspace(-np.pi, np.pi, 9)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        et, ep = pa.ideal_patch_element("y")(T, P)
        lines = ['"theta";"phi";"etr";"eti";"epr";"epi"']
        lines += [";".join(repr(float(v)) for v in row) for row in np.column_stack(
            [T.ravel(), P.ravel(), et.real.ravel(), et.imag.ravel(),
             ep.real.ravel(), ep.imag.ravel()])]
        path = tmp_path / "x.csv"
        path.write_text("\n".join(lines) + "\n")
        columns = dict(theta="theta", phi="phi", e_theta_re="etr", e_theta_im="eti",
                       e_phi_re="epr", e_phi_im="epi")
        pattern = pa.load_element_pattern_csv(path, columns, angle_unit="rad", delimiter=";")
        np.testing.assert_allclose(pattern(T, P)[1], ep, atol=1e-12)

    def test_incomplete_grid_file(self, tmp_path):
        path = write_pattern_csv(tmp_path / "x.csv", np.arange(0, 91, 10.0),
                                 np.arange(0, 360, 30.0))
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        with pytest.raises(ValueError, match="incomplete grid"):
            pa.load_element_pattern_csv(path, RE_IM_COLUMNS)

    def test_missing_header_name(self, tmp_path):
        path = write_pattern_csv(tmp_path / "x.csv", np.arange(0, 91, 30.0),
                                 np.arange(0, 360, 90.0))
        columns = dict(RE_IM_COLUMNS, theta="Theta")
        with pytest.raises(ValueError, match="'Theta'.*not in the header"):
            pa.load_element_pattern_csv(path, columns)

    def test_wrong_delimiter(self, tmp_path):
        path = write_pattern_csv(tmp_path / "x.csv", np.arange(0, 91, 30.0),
                                 np.arange(0, 360, 90.0))
        with pytest.raises(ValueError, match="delimiter"):
            pa.load_element_pattern_csv(path, RE_IM_COLUMNS, delimiter="\t")

    def test_role_errors(self, tmp_path):
        path = write_pattern_csv(tmp_path / "x.csv", np.arange(0, 91, 30.0),
                                 np.arange(0, 360, 90.0))
        with pytest.raises(ValueError, match="unknown column role"):
            pa.load_element_pattern_csv(path, dict(RE_IM_COLUMNS, e_theta_db="x"))
        incomplete = {k: v for k, v in RE_IM_COLUMNS.items() if k != "e_phi_im"}
        with pytest.raises(ValueError, match="missing role"):
            pa.load_element_pattern_csv(path, incomplete)
        with pytest.raises(ValueError, match="field_format"):
            pa.load_element_pattern_csv(path, RE_IM_COLUMNS, field_format="polar")

    def test_bad_data_row(self, tmp_path):
        path = write_pattern_csv(tmp_path / "x.csv", np.arange(0, 91, 30.0),
                                 np.arange(0, 360, 90.0), shuffle_seed=None)
        lines = path.read_text().splitlines()
        lines[3] = lines[3].replace(lines[3].split(",")[2], "n/a")
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="line 4: cannot parse 'n/a'"):
            pa.load_element_pattern_csv(path, RE_IM_COLUMNS)

    def test_short_data_row(self, tmp_path):
        path = write_pattern_csv(tmp_path / "x.csv", np.arange(0, 91, 30.0),
                                 np.arange(0, 360, 90.0))
        with open(path, "a") as f:
            f.write("1.0,2.0\n")
        with pytest.raises(ValueError, match="2 fields"):
            pa.load_element_pattern_csv(path, RE_IM_COLUMNS)

    def test_usable_as_array_element(self, tmp_path):
        path = write_pattern_csv(tmp_path / "x.csv", np.arange(0, 91, 2.0),
                                 np.arange(0, 360, 5.0))
        pattern = pa.load_element_pattern_csv(path, RE_IM_COLUMNS)
        geom = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5)
        k = pa.wavelength_to_k(1.0)
        Et, _ = pa.vector_total_pattern(np.array([0.0]), np.array([0.0]),
                                        geom.x, geom.y, np.ones(16), k, pattern)
        assert abs(Et[0]) == pytest.approx(16.0)
