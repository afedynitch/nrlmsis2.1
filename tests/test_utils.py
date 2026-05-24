"""Unit tests for utility functions."""

import math

import numpy as np
import pytest

from nrlmsis.utils import alt2gph, bspline, bspline_vec, dilog, gph2alt
import nrlmsis.constants as _C
from nrlmsis.parameters import load_model


class TestAlt2Gph:
    """Tests for geodetic altitude to geopotential height conversion."""

    def test_zero_altitude(self):
        """At sea level, geopotential height should be ~0."""
        assert alt2gph(0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_positive_altitude(self):
        """Check conversion at moderate altitude."""
        gph = alt2gph(45.0, 100.0)
        assert 98.0 < gph < 99.0  # GPH is ~1.5 km less than geodetic alt at 100 km

    def test_high_altitude(self):
        """At 500 km, GPH should be noticeably less than geodetic altitude."""
        gph = alt2gph(45.0, 500.0)
        assert gph < 500.0
        assert gph > 400.0

    def test_latitude_dependence(self):
        """GPH should vary with latitude due to gravity variation."""
        gph_eq = alt2gph(0.0, 100.0)
        gph_pole = alt2gph(90.0, 100.0)
        # Gravity is stronger at poles, so GPH should be slightly higher
        assert gph_pole > gph_eq


class TestGph2Alt:
    """Tests for geopotential height to geodetic altitude (inverse)."""

    def test_roundtrip(self):
        """alt2gph and gph2alt should be inverses."""
        for lat in [0.0, 45.0, 90.0]:
            for alt in [0.0, 50.0, 200.0, 500.0]:
                gph = alt2gph(lat, alt)
                alt_back = gph2alt(lat, gph)
                assert alt_back == pytest.approx(alt, abs=0.01)


class TestDilog:
    """Tests for the dilogarithm function."""

    def test_zero(self):
        """Li2(0) = 0."""
        assert dilog(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_small_value(self):
        """Li2(x) ≈ x for small x."""
        x = 0.01
        assert dilog(x) == pytest.approx(x, rel=0.02)

    def test_half(self):
        """Li2(0.5) = pi²/12 - ln(2)²/2."""
        expected = math.pi**2 / 12.0 - math.log(2.0)**2 / 2.0
        assert dilog(0.5) == pytest.approx(expected, rel=1e-4)

    def test_near_one(self):
        """Li2(x) approaches pi²/6 as x→1."""
        x = 0.999
        result = dilog(x)
        assert result == pytest.approx(math.pi**2 / 6.0, rel=0.01)


class TestBsplineVec:
    """Vectorized B-spline (bspline_vec) must match scalar bspline for any kmax."""

    @pytest.fixture(scope='class')
    def fixtures(self):
        params = load_model()
        return _C.NODES_TN, _C.ND + 2, params.eta_tn

    @pytest.mark.parametrize('kmax', [2, 3, 4, 5, 6])
    def test_matches_scalar_on_sweep(self, fixtures, kmax):
        nodes, nd, eta = fixtures
        x_sweep = np.concatenate([
            np.array([-20.0, nodes[0], nodes[0] + 1e-9]),
            np.linspace(nodes[0] + 0.1, nodes[nd] - 0.1, 200),
            nodes[3:8].copy(),
            nodes[15:22].copy(),
            np.array([nodes[nd] - 1e-9, nodes[nd], nodes[nd] + 5.0]),
        ])

        S_vec, iz_vec = bspline_vec(x_sweep, nodes, nd, kmax, eta)

        S_ref = np.zeros_like(S_vec)
        iz_ref = np.zeros_like(iz_vec)
        for n, x in enumerate(x_sweep):
            s, i = bspline(float(x), nodes, nd, kmax, eta)
            S_ref[n] = s
            iz_ref[n] = i

        # The scalar bspline writes to columns kmax-2 .. 4 regardless of kmax
        # (kmax only short-circuits between k=5 and k=6). Only cols [0, kmax-2]
        # are meaningfully populated by either path; callers only read those.
        cols = kmax - 1
        np.testing.assert_array_equal(iz_vec, iz_ref)
        np.testing.assert_array_equal(S_vec[:, :, :cols], S_ref[:, :, :cols])

    def test_out_of_bounds_returns_zero(self, fixtures):
        nodes, nd, eta = fixtures
        x = np.array([-100.0, nodes[0] - 1e-3, nodes[nd], nodes[nd] + 50.0])
        S_vec, iz_vec = bspline_vec(x, nodes, nd, 6, eta)
        # All-zero S for OOB; iz = -1 (low) or nd (high)
        np.testing.assert_array_equal(S_vec, 0.0)
        assert iz_vec[0] == -1
        assert iz_vec[1] == -1
        assert iz_vec[2] == nd
        assert iz_vec[3] == nd

    def test_empty_input(self, fixtures):
        nodes, nd, eta = fixtures
        S_vec, iz_vec = bspline_vec(np.array([]), nodes, nd, 6, eta)
        assert S_vec.shape == (0, 6, 5)
        assert iz_vec.shape == (0,)
