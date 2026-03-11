"""Validation tests: compare Python NRLMSIS 2.1 against Fortran reference output.

Reads 200 test cases from msis2.1_test_in.txt and compares against
msis2.1_test_ref_dp.txt (legacy gtd8d output in CGS units).
"""

from pathlib import Path

import numpy as np
import pytest

from nrlmsis import NRLMSIS21

DATA_DIR = Path(__file__).parent / "data"
DMISSING = 9.999e-38
DMISSING_CGS = DMISSING * 1e-6  # After CGS conversion


def _load_test_cases():
    """Load input and reference output test cases."""
    inputs = []
    with open(DATA_DIR / "msis2.1_test_in.txt") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            iyd = int(parts[0])
            sec = float(parts[1])
            alt = float(parts[2])
            glat = float(parts[3])
            glong = float(parts[4])
            stl = float(parts[5])
            f107a = float(parts[6])
            f107 = float(parts[7])
            ap_val = float(parts[8])
            inputs.append((iyd, sec, alt, glat, glong, stl, f107a, f107, ap_val))

    references = []
    with open(DATA_DIR / "msis2.1_test_ref_dp.txt") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.split()
            if len(parts) < 20:
                continue
            # Columns after 9 inputs: He, O, N2, O2, Ar, rho, H, N, O*, NO, T
            densities = [float(parts[i]) for i in range(9, 19)]
            temp = float(parts[19])
            references.append((densities, temp))

    return inputs, references


@pytest.fixture(scope="module")
def model():
    """Create a single model instance for all test cases."""
    return NRLMSIS21()


@pytest.fixture(scope="module")
def test_data():
    """Load test input/reference data once."""
    return _load_test_cases()


def test_number_of_cases(test_data):
    """Verify we loaded the expected number of test cases."""
    inputs, references = test_data
    assert len(inputs) == 200
    assert len(references) == 200


@pytest.mark.parametrize("case_idx", range(200))
def test_case(model, test_data, case_idx):
    """Run a single validation test case against Fortran reference."""
    inputs, references = test_data
    iyd, sec, alt, glat, glong, stl, f107a, f107, ap_val = inputs[case_idx]
    ref_densities, ref_temp = references[case_idx]

    ap = np.array([ap_val] * 7)
    d, t = model.gtd8d(iyd, sec, alt, glat, glong, stl, f107a, f107, ap)

    # Check temperature
    if ref_temp != 0.0:
        assert t[1] == pytest.approx(ref_temp, rel=1e-4), (
            f"Case {case_idx}: T mismatch: got {t[1]}, expected {ref_temp}"
        )

    # Check densities (legacy order: He, O, N2, O2, Ar, rho, H, N, O*, NO)
    for i in range(10):
        ref = ref_densities[i]
        got = float(d[i])
        if abs(ref) < 1e-35:
            # Missing value -- just check it's also missing/tiny
            assert abs(got) < 1e-30, (
                f"Case {case_idx}, species {i}: expected missing, got {got}"
            )
        else:
            assert got == pytest.approx(ref, rel=1e-4), (
                f"Case {case_idx}, species {i}: got {got}, expected {ref}"
            )
