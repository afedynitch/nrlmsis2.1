# NRLMSIS 2.1 — Python atmospheric model

Python translation of the NRL Mass Spectrometer Incoherent Scatter radar empirical atmospheric model (NRLMSIS 2.1). Computes temperature and number densities of 10 species (N2, O2, O, He, H, Ar, N, anomalous O, NO, and total mass density) from the surface to ~2000 km altitude. The model is driven by solar flux (F10.7) and geomagnetic activity (Ap) indices.

## Installation

```bash
pip install -e .             # editable install
pip install -e ".[test]"     # with test dependencies
```

## Quick Start

```python
import numpy as np
from nrlmsis import NRLMSIS21

model = NRLMSIS21()

ap = np.array([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0])  # geomagnetic indices
result = model.calc(
    day=172,        # day of year (summer solstice)
    utsec=43200.0,  # universal time (seconds), noon
    z=300.0,        # altitude (km)
    lat=45.0,       # geodetic latitude (degrees)
    lon=-75.0,      # geodetic longitude (degrees)
    sfluxavg=150.0, # 81-day average F10.7
    sflux=150.0,    # daily F10.7 (previous day)
    ap=ap,
)

print(f"Temperature:            {result.temperature:.1f} K")
print(f"Exospheric temperature: {result.exospheric_temperature:.1f} K")
print(f"O density:              {result.densities[3]:.3e} m⁻³")
print(f"Mass density:           {result.densities[0]:.3e} kg/m³")
```

## API Reference

### `NRLMSIS21(...)` constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parm_file` | path or None | bundled | Path to binary parameter file (`msis21.parm`) |
| `alt_type` | str | `'geodetic'` | Altitude convention: `'geodetic'` or `'geopotential'` |
| `switch_legacy` | array(25) or None | None | Legacy F10.7/Ap switch array (see Fortran docs) |
| `switch_gfn` | array(512) bool or None | None | Per-basis-function on/off switches |
| `species_select` | array(10) bool or None | None | Which species to compute |
| `mass_include` | array(10) bool or None | None | Which species contribute to mass density |
| `n2_msis00` | bool | False | Use NRLMSISE-00 thermospheric N2 variation |

### `calc(day, utsec, z, lat, lon, sfluxavg, sflux, ap) → MSISOutput`

| Parameter | Units | Range | Description |
|-----------|-------|-------|-------------|
| `day` | days | 1–366 | Day of year |
| `utsec` | seconds | 0–86400 | Universal time |
| `z` | km | 0–2000 | Altitude (geodetic or geopotential per `alt_type`) |
| `lat` | degrees | -90 to +90 | Geodetic latitude |
| `lon` | degrees | -180 to +360 | Geodetic longitude |
| `sfluxavg` | sfu | ~65–300 | 81-day average F10.7 solar flux |
| `sflux` | sfu | ~65–300 | Daily F10.7 for previous day |
| `ap` | nT | 0–400 | 7-element geomagnetic activity array |

The `ap` array follows the standard MSIS convention:
- `ap[0]`: daily Ap
- `ap[1]`: 3-hour Ap for current interval
- `ap[2]`: 3-hour Ap for 3 hours before
- `ap[3]`: 3-hour Ap for 6 hours before
- `ap[4]`: 3-hour Ap for 9 hours before
- `ap[5]`: average of 8 3-hour Ap values (12–33 hours before)
- `ap[6]`: average of 8 3-hour Ap values (36–57 hours before)

### `MSISOutput` fields

| Field | Units | Description |
|-------|-------|-------------|
| `temperature` | K | Temperature at requested altitude |
| `exospheric_temperature` | K | Exospheric (∞) temperature |
| `densities[0]` | kg/m³ | Total mass density |
| `densities[1]` | m⁻³ | N2 number density |
| `densities[2]` | m⁻³ | O2 number density |
| `densities[3]` | m⁻³ | O number density |
| `densities[4]` | m⁻³ | He number density |
| `densities[5]` | m⁻³ | H number density |
| `densities[6]` | m⁻³ | Ar number density |
| `densities[7]` | m⁻³ | N number density |
| `densities[8]` | m⁻³ | Anomalous O number density |
| `densities[9]` | m⁻³ | NO number density |

## Legacy (NRLMSISE-00-compatible) Interface

The `gtd8d()` method provides the original Fortran-compatible interface for drop-in replacement of NRLMSISE-00 code. It uses CGS units and YYDDD date format.

```python
ap = np.array([7.0] * 7)
d, t = model.gtd8d(
    iyd=100172,    # year-day: 2010, day 172 (YYDDD format)
    sec=43200.0,
    alt=300.0,
    glat=45.0,
    glong=-75.0,
    stl=0.0,       # local solar time (ignored — computed from sec/glong)
    f107a=150.0,
    f107=150.0,
    ap=ap,
)

# d[0..9] in cm⁻³ (or g/cm³ for mass density):
# [He, O, N2, O2, Ar, rho, H, N, Anomalous-O, NO]
# t[0]: exospheric temperature (K)
# t[1]: temperature at alt (K)
print(f"T = {t[1]:.1f} K,  O = {d[1]:.3e} cm⁻³")

# Convert O to SI:
o_per_m3 = float(d[1]) * 1e6
```

## Running Tests

```bash
pytest tests/                          # all tests
pytest tests/test_validation.py -v    # 200 validation cases vs Fortran reference
```

## References

- Emmert, J. T., et al. (2021). NRLMSIS 2.0: A whole-atmosphere empirical model of temperature and neutral species densities. *Earth and Space Science*, 8, e2020EA001321. https://doi.org/10.1029/2020EA001321
- Fortran source: https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.1
