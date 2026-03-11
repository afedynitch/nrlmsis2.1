# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package in development mode
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_utils.py

# Run a single test by name
pytest tests/test_validation.py::test_gtd8d_case[0]

# Run validation tests only (200 cases against Fortran reference)
pytest tests/test_validation.py -v
```

## Architecture

This is a Python translation of the NRLMSIS 2.1 empirical atmospheric model, which computes temperature and species densities at a given location/time/solar condition.

### Data Flow

```
User Input (alt, lat, lon, time, solar flux, Ap)
  → globe.py: compute 512 horizontal/temporal basis functions
  → temperature.py: compute vertical temperature profile parameters
  → density.py: compute each species' density profile parameters
  → Evaluate at requested altitude using B-spline interpolation
  → model.py: assemble MSISOutput
```

### Module Responsibilities

- **`model.py`** — Public API entry point. `NRLMSIS21.calc()` for modern use; `gtd8d()` for legacy NRLMSISE-00-compatible interface (CGS units). Contains three-level caching: `ProfileCache` (same location/time/solar skips globe+profile computation), `AltitudeCache` (same altitude skips B-spline weight recomputation).
- **`globe.py`** — Computes all 512 basis functions (Legendre polynomials, Fourier harmonics of day/LST/longitude, tides, waves, solar flux terms, geomagnetic terms). These are dot-producted with per-species spline coefficients in temperature.py and density.py.
- **`temperature.py`** — Builds the vertical temperature profile as a B-spline between ~70–122.5 km, with a Bates exponential profile above. Returns `TnParm` used by `eval_temperature()`.
- **`density.py`** — Builds species-specific density profiles via hydrostatic integration. Returns `DnParm` used by `eval_density()`. Handles 10 species: N2, O2, O, He, H, Ar, N, Anomalous O, NO plus total mass density.
- **`parameters.py`** — Loads the binary `msis21.parm` file (little-endian float64). `ModelParameters` holds 11 `BasisSubset` objects (TN, PR, N2, O2, O1, HE, H1, AR, N1, OA, NO). `tselec()` maps 25 legacy switches to 512 modern switches.
- **`utils.py`** — Geodetic↔geopotential altitude conversion (WGS84), B-spline evaluation (Cox-de Boor, orders 1–6), dilogarithm approximation.
- **`constants.py`** — Physical constants, B-spline node arrays, continuity matrices, and precomputed B-spline weights at key reference altitudes.

### Key Design Decisions

- The binary parameter file `src/nrlmsis/data/msis21.parm` is loaded once at module import via `load_model()` and stored as a module-level singleton.
- All vertical profiles are parameterized in geopotential height (km), not geometric altitude — `alt2gph()` is called at the start of every evaluation.
- The caching in `model.py` is critical for performance when vectorizing over altitude at a fixed location/time. The `LocationKey` dataclass determines cache validity.
- Validation tests compare against Fortran double-precision reference output (`tests/data/msis2.1_test_ref_dp.txt`) using the `gtd8d()` legacy interface. The reference file uses 4-significant-figure Fortran E-format printing (e.g. `0.1255E+15`), giving ~5e-4 relative precision; the test tolerance is set accordingly.

## Fortran→Python Translation Pitfalls

These bugs were found and fixed during an initial validation pass. Keep them in mind for any future modifications:

### 1. Fortran array memory layout: `reshape` with `order='F'`

When Fortran stores a 2D array using a nested loop (outer loop = columns), the flat layout is column-major. When that data ends up in a Python 1D slice and needs reshaping back to 2D, use `order='F'`:

```python
# Fortran fills: for m in range(2): for n in range(7): bf[c] = plg[n, m]
# → data is [plg[0,0]...plg[6,0], plg[0,1]...plg[6,1]]
gf[C.CMAG + 13:C.CMAG + 27].reshape(7, 2, order='F')  # correct
gf[C.CMAG + 13:C.CMAG + 27].reshape(7, 2)              # wrong: C order swaps rows/cols
```

This applies everywhere `geomag()` is called (temperature.py and density.py).

### 2. Fortran `matmul(vec, matrix)` ≠ `vec @ matrix.T`

Fortran stores 2D arrays column-major. A hardcoded Fortran matrix defined via `reshape((/...values.../), (/m,n/))` fills columns first. Its Python equivalent (without any transpose) is:

```python
# Fortran: matmul(bc, c2tn)   where c2tn is stored column-major
# Python:  bc @ C2TN           where C2TN rows = Fortran columns (already correct)
```

The continuity matrices `C2TN`, `C1O1`, `C1NO` in `constants.py` had an erroneous `.T` applied on construction. This caused the temperature B-spline to produce negative temperatures above ~100 km, making all species densities go wrong. When transcribing Fortran matrix constants, **do not apply `.T`** unless you are certain the Fortran code calls `matmul(matrix, vec)` (not `matmul(vec, matrix)`).

### 3. Reference data precision vs. test tolerance

The Fortran reference file (`msis2.1_test_ref_dp.txt`) prints values in `0.XXXXE+YY` format (4 significant figures). This limits meaningful comparison to ~5e-4 relative tolerance. Testing at 1e-4 will produce spurious failures even for a correct implementation.

### 4. The `geomag()` function `plg` argument

`geomag()` in `globe.py` expects `plg` as a `(7, 2)` ndarray indexed as `plg[n, m]`. The Fortran reference was structured the same way. All callers must pass `gf[C.CMAG+13:C.CMAG+27].reshape(7, 2, order='F')`.

### 5. Fortran source location

The original Fortran source files are at `../nrlmsis2.1/*.F90`. When in doubt about a matrix layout, constant value, or algorithm detail, consult the corresponding `.F90` file directly (e.g. `msis_constants.F90` for matrix definitions, `msis_tfn.F90` for temperature, `msis_dfn.F90` for density).
