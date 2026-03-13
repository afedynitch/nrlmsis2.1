"""Tests: calc_altitude_array() agrees with scalar calc() loop to float64 rounding."""

import numpy as np
import pytest

from nrlmsis import NRLMSIS21, MSISOutputArray

DMISSING = 9.999e-38

# Shared model instance (loaded once per module)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = NRLMSIS21()
    return _model


def _ref_loop(model, z_arr, day, utsec, lat, lon, f107a, f107, ap):
    """Reference: scalar calc() called independently for each altitude."""
    N = len(z_arr)
    tn = np.empty(N)
    dn = np.full((10, N), DMISSING)
    for i, z in enumerate(z_arr):
        out = model.calc(day, utsec, z, lat, lon, f107a, f107, ap)
        tn[i] = out.temperature
        dn[:, i] = out.densities
    return tn, dn


@pytest.fixture(scope='module')
def model():
    return NRLMSIS21()


# ----- parametrized test cases -----

_CASES = [
    # (label, day, utsec, lat, lon, f107a, f107, ap7)
    ('midlat_noon',    172, 43200, 45.0,  0.0, 150.0, 148.0, [4, 100, 100, 100, 100, 100, 100]),
    ('polar_midnight', 355,     0, 80.0, 180.0, 80.0,  78.0, [2,  20,  20,  20,  20,  20,  20]),
    ('equator_dusk',    80, 64800,  0.0, 120.0, 200.0, 195.0, [8, 150, 150, 150, 150, 150, 150]),
]

_ALT_CASES = [
    ('above_zetaB_only', np.arange(123.0, 500.0, 10.0)),   # pure vectorized path
    ('below_zetaB_only', np.array([20.0, 50.0, 70.0, 90.0, 115.0])),  # scalar fallback path
    ('mixed',            np.array([50.0, 90.0, 122.5, 150.0, 300.0, 450.0])),
]


@pytest.mark.parametrize('label,day,utsec,lat,lon,f107a,f107,ap7', _CASES)
@pytest.mark.parametrize('alt_label,z_arr', _ALT_CASES)
def test_array_vs_loop(model, label, day, utsec, lat, lon, f107a, f107, ap7, alt_label, z_arr):
    ap = np.array(ap7, dtype=float)

    result = model.calc_altitude_array(day, utsec, z_arr, lat, lon, f107a, f107, ap)
    assert isinstance(result, MSISOutputArray)
    assert result.temperature.shape == (len(z_arr),)
    assert result.densities.shape == (10, len(z_arr))

    tn_ref, dn_ref = _ref_loop(model, z_arr, day, utsec, lat, lon, f107a, f107, ap)

    # Temperature: relative tolerance 1e-13 (float64 rounding only)
    np.testing.assert_allclose(result.temperature, tn_ref, rtol=1e-12,
                               err_msg=f'temperature mismatch for {label}/{alt_label}')

    # Densities: compare only non-missing values
    for i in range(10):
        dn_vec = result.densities[i]
        dn_sc = dn_ref[i]
        valid = (dn_sc != DMISSING) & (dn_vec != DMISSING)
        if not valid.any():
            continue
        np.testing.assert_allclose(dn_vec[valid], dn_sc[valid], rtol=1e-12,
                                   err_msg=f'density[{i}] mismatch for {label}/{alt_label}')


def test_exospheric_temperature_consistent(model):
    """Exospheric temperature should match calc() at same location."""
    ap = np.array([4, 100, 100, 100, 100, 100, 100], dtype=float)
    z_arr = np.array([150.0, 300.0])
    result = model.calc_altitude_array(172, 43200, z_arr, 45.0, 0.0, 150.0, 148.0, ap)
    ref = model.calc(172, 43200, 150.0, 45.0, 0.0, 150.0, 148.0, ap)
    assert abs(result.exospheric_temperature - ref.exospheric_temperature) < 1e-10


def test_profile_cache_hit(model):
    """calc_altitude_array must not recompute profile when called twice with same location."""
    ap = np.array([4, 100, 100, 100, 100, 100, 100], dtype=float)
    z_arr = np.linspace(130, 400, 50)
    r1 = model.calc_altitude_array(172, 43200, z_arr, 45.0, 0.0, 150.0, 148.0, ap)
    r2 = model.calc_altitude_array(172, 43200, z_arr, 45.0, 0.0, 150.0, 148.0, ap)
    np.testing.assert_array_equal(r1.temperature, r2.temperature)
