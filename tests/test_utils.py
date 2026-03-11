"""Unit tests for utility functions."""

import math

import numpy as np
import pytest

from nrlmsis.utils import alt2gph, gph2alt, dilog


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
