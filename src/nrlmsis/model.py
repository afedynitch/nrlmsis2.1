"""NRLMSIS 2.1 main model class with smart caching."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import constants as C
from .parameters import ModelParameters, load_model
from .globe import GlobeFunction
from .temperature import TnParm, compute_temperature, eval_temperature
from .density import DnParm, compute_density, eval_density, _pwmp_vec
from .utils import alt2gph, alt2gph_vec, bspline, dilog, dilog_vec


@dataclass(frozen=True)
class LocationKey:
    """Immutable key for location/time/solar conditions."""

    day: float
    utsec: float
    lat: float
    lon: float
    sfluxavg: float
    sflux: float
    ap: tuple


@dataclass
class ProfileCache:
    """Cached profile computations, replaced when LocationKey changes."""

    key: LocationKey
    gf: np.ndarray
    tpro: TnParm
    dpro: dict[int, DnParm]


@dataclass
class AltitudeCache:
    """Cached B-spline weights, replaced when altitude changes."""

    zeta: float
    sz: np.ndarray
    iz: int


@dataclass
class MSISOutput:
    """Output from NRLMSIS 2.1 calculation."""

    temperature: float
    densities: np.ndarray  # 10 elements: [mass_density, N2, O2, O, He, H, Ar, N, O*, NO]
    exospheric_temperature: float


@dataclass
class MSISOutputArray:
    """Vectorized output from NRLMSIS 2.1 calc_altitude_array()."""

    temperature: np.ndarray           # shape (N,) in K
    densities: np.ndarray             # shape (10, N): [mass_density, N2, O2, O, He, H, Ar, N, O*, NO]
    exospheric_temperature: float


class NRLMSIS21:
    """NRLMSIS 2.1 empirical atmospheric model.

    Args:
        parm_file: Path to binary parameter file. Defaults to bundled msis21.parm.
        switch_legacy: 25-element float array of legacy switches.
        switch_gfn: 512-element boolean array of individual switches.
        alt_type: 'geodetic' (default) or 'geopotential'.
        species_select: 10-element boolean array for species selection.
        mass_include: 10-element boolean array for mass density inclusion.
        n2_msis00: Flag for NRLMSISE-00 thermospheric N2 variations.
        profile_cache_size: Number of location/time profiles to cache (default 1).
    """

    def __init__(self, parm_file: str | Path | None = None,
                 switch_legacy: np.ndarray | None = None,
                 switch_gfn: np.ndarray | None = None,
                 alt_type: str = 'geodetic',
                 species_select: np.ndarray | None = None,
                 mass_include: np.ndarray | None = None,
                 n2_msis00: bool = False,
                 profile_cache_size: int = 1) -> None:
        self._params = load_model(
            parm_file=parm_file,
            switch_legacy=switch_legacy,
            switch_gfn=switch_gfn,
            zalt_type=(alt_type == 'geodetic'),
            spec_select=species_select,
            mass_include=mass_include,
            n2_msis00=n2_msis00,
        )
        self._globe = GlobeFunction()
        self._profile_cache: OrderedDict = OrderedDict()
        self._profile_cache_size = profile_cache_size
        self._alt_cache: Optional[AltitudeCache] = None

    def _get_or_compute_profile(self, key: LocationKey) -> ProfileCache:
        """Return cached profile for key, computing it if needed (LRU eviction)."""
        if key in self._profile_cache:
            self._profile_cache.move_to_end(key)
            return self._profile_cache[key]

        params = self._params
        gf = self._globe.evaluate(
            key.day, key.utsec, key.lat, key.lon,
            key.sfluxavg, key.sflux, np.array(key.ap), params.swg)
        tpro = compute_temperature(gf, params)
        dpro = {}
        for ispec in range(2, C.NSPEC):
            if params.spec_flag[ispec - 1]:
                dpro[ispec] = compute_density(ispec, gf, tpro, params)
        pcache = ProfileCache(key=key, gf=gf, tpro=tpro, dpro=dpro)
        self._profile_cache[key] = pcache
        if len(self._profile_cache) > self._profile_cache_size:
            self._profile_cache.popitem(last=False)
        return pcache

    def calc(self, day: float, utsec: float, z: float, lat: float, lon: float,
             sfluxavg: float, sflux: float, ap: np.ndarray) -> MSISOutput:
        """Calculate temperature and densities.

        Args:
            day: Day of year (1-366).
            utsec: Universal time (seconds).
            z: Altitude (km), geodetic or geopotential per alt_type.
            lat: Geodetic latitude (degrees).
            lon: Geodetic longitude (degrees).
            sfluxavg: 81-day average F10.7.
            sflux: Daily F10.7 for previous day.
            ap: 7-element geomagnetic activity array.

        Returns:
            MSISOutput with temperature, densities, and exospheric temperature.
        """
        params = self._params

        # Convert altitude if needed
        if params.zalt_flag:
            zeta = alt2gph(float(lat), float(z))
        else:
            zeta = float(z)

        # Update altitude cache
        if zeta < C.ZETA_B:
            if self._alt_cache is None or zeta != self._alt_cache.zeta:
                kmax = 5 if zeta < C.ZETA_F else 6
                sz, iz = bspline(zeta, C.NODES_TN, C.ND + 2, kmax, params.eta_tn)
                self._alt_cache = AltitudeCache(zeta=zeta, sz=sz, iz=iz)

        # Update profile cache
        key = LocationKey(
            day=float(day), utsec=float(utsec), lat=float(lat), lon=float(lon),
            sfluxavg=float(sfluxavg), sflux=float(sflux),
            ap=tuple(float(x) for x in ap),
        )
        pcache = self._get_or_compute_profile(key)
        tpro = pcache.tpro

        # Get B-spline data
        if self._alt_cache is not None:
            sz = self._alt_cache.sz
            iz = self._alt_cache.iz
        else:
            sz = np.zeros((6, 5))
            iz = 0

        # Temperature at altitude
        # wght maps to S(-3:0, 4) → sz[2:6, 2]
        wght = sz[2:6, 2]
        tn = eval_temperature(zeta, iz, wght, tpro)

        # Temperature integration terms
        delz = zeta - C.ZETA_B
        if zeta < C.ZETA_F:
            i_start = max(iz - 4, 0)
            if iz < 4:
                j_start = -iz
            else:
                j_start = -4
            # S(j:0, 5) → sz[j+5:6, 3]
            n_pts = iz - i_start + 1
            s_start = j_start + 5
            Vz = np.dot(tpro.beta[i_start:iz + 1], sz[s_start:s_start + n_pts, 3]) + tpro.cVs
            Wz = 0.0
            lnPz = C.LNP0 - C.MBARG0DIVKB * (Vz - tpro.Vzeta0)
            lndtotz = lnPz - math.log(C.KB * tn)
        else:
            lndtotz = 0.0  # Not used above zetaF for mixing ratio species
            if zeta < C.ZETA_B:
                # S(-4:0, 5) → sz[1:6, 3]
                Vz = np.dot(tpro.beta[iz - 4:iz + 1], sz[1:6, 3]) + tpro.cVs
                # S(-5:0, 6) → sz[0:6, 4]
                Wz = np.dot(tpro.gamma[iz - 5:iz + 1], sz[0:6, 4]) + tpro.cVs * delz + tpro.cWs
            else:
                Vz = (delz + math.log(tn / tpro.tex) / tpro.sigma) / tpro.tex + tpro.cVb
                Wz = ((0.5 * delz * delz
                       + dilog(tpro.b * math.exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
                      + tpro.cVb * delz + tpro.cWb)

        # Species densities
        HRfact = 0.5 * (1.0 + math.tanh(C.H_GAMMA * (zeta - C.ZETA_GAMMA)))
        dn = np.full(10, C.DMISSING)
        for ispec in range(2, C.NSPEC):
            if params.spec_flag[ispec - 1] and ispec in pcache.dpro:
                dn[ispec - 1] = eval_density(
                    zeta, tn, lndtotz, Vz, Wz, HRfact, tpro, pcache.dpro[ispec], params)
            else:
                dn[ispec - 1] = C.DMISSING

        # Mass density
        if params.spec_flag[0]:
            dn[0] = np.dot(dn, params.mass_wgt)
        else:
            dn[0] = C.DMISSING

        return MSISOutput(
            temperature=tn,
            densities=dn,
            exospheric_temperature=tpro.tex,
        )

    def calc_altitude_array(self, day: float, utsec: float, z: np.ndarray,
                            lat: float, lon: float, sfluxavg: float,
                            sflux: float, ap: np.ndarray) -> MSISOutputArray:
        """Vectorized calculation over an array of altitudes at fixed location/time.

        Altitudes at or above ZETA_B (122.5 km geopotential) are evaluated with
        fully vectorized numpy operations.  Altitudes below ZETA_B fall back to the
        scalar calc() path (B-spline evaluation is not easily vectorized).

        Args:
            day: Day of year (1-366).
            utsec: Universal time (seconds).
            z: Array of altitudes (km), geodetic or geopotential per alt_type.
            lat: Geodetic latitude (degrees).
            lon: Geodetic longitude (degrees).
            sfluxavg: 81-day average F10.7.
            sflux: Daily F10.7 for previous day.
            ap: 7-element geomagnetic activity array.

        Returns:
            MSISOutputArray with temperature (N,), densities (10,N), exospheric_temperature.
        """
        params = self._params
        z_arr = np.asarray(z, dtype=float)
        N = len(z_arr)

        # Convert altitudes to geopotential height
        if params.zalt_flag:
            zeta_arr = alt2gph_vec(float(lat), z_arr)
        else:
            zeta_arr = z_arr.copy()

        # Ensure profile cache is hot
        key = LocationKey(
            day=float(day), utsec=float(utsec), lat=float(lat), lon=float(lon),
            sfluxavg=float(sfluxavg), sflux=float(sflux),
            ap=tuple(float(x) for x in ap),
        )
        pcache = self._get_or_compute_profile(key)
        tpro = pcache.tpro

        tn_arr = np.empty(N)
        dn_arr = np.full((10, N), C.DMISSING)

        above_mask = zeta_arr >= C.ZETA_B
        below_mask = ~above_mask

        # Vectorized path for all altitudes >= ZETA_B
        if above_mask.any():
            zeta_ab = zeta_arr[above_mask]
            tn_ab, dn_ab = self._eval_above_zetaB(zeta_ab, tpro, pcache, params)
            tn_arr[above_mask] = tn_ab
            dn_arr[:, above_mask] = dn_ab

        # Scalar fallback for altitudes < ZETA_B
        for idx in np.where(below_mask)[0]:
            result = self.calc(day, utsec, float(z_arr[idx]), lat, lon,
                               sfluxavg, sflux, ap)
            tn_arr[idx] = result.temperature
            dn_arr[:, idx] = result.densities

        return MSISOutputArray(
            temperature=tn_arr,
            densities=dn_arr,
            exospheric_temperature=tpro.tex,
        )

    def _eval_above_zetaB(self, zeta_arr: np.ndarray, tpro: TnParm,
                          pcache: ProfileCache,
                          params: ModelParameters) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized density/temperature evaluation for zeta >= ZETA_B.

        Returns (tn_arr shape (N,), dn_arr shape (10,N)).
        """
        N = len(zeta_arr)

        # Bates temperature profile
        tn_arr = tpro.tex - (tpro.tex - tpro.tb0) * np.exp(-tpro.sigma * (zeta_arr - C.ZETA_B))

        # Hydrostatic integration terms
        delz_arr = zeta_arr - C.ZETA_B
        Vz_arr = (delz_arr + np.log(tn_arr / tpro.tex) / tpro.sigma) / tpro.tex + tpro.cVb
        b_arr = tpro.b * np.exp(-tpro.sigma * delz_arr)
        Wz_arr = ((0.5 * delz_arr ** 2 + dilog_vec(b_arr) / tpro.sigmasq) / tpro.tex
                  + tpro.cVb * delz_arr + tpro.cWb)

        HRfact_arr = 0.5 * (1.0 + np.tanh(C.H_GAMMA * (zeta_arr - C.ZETA_GAMMA)))

        dn_arr = np.full((10, N), C.DMISSING)

        for ispec in range(2, C.NSPEC):
            if not (params.spec_flag[ispec - 1] and ispec in pcache.dpro):
                continue
            dpro = pcache.dpro[ispec]

            if ispec == 9:  # Anomalous O: simple exponential profile
                result = (dpro.lndref
                          - (zeta_arr - dpro.zref) / C.HOA
                          - dpro.C_coeff * np.exp(-(zeta_arr - dpro.zetaC) / dpro.HC))
                dn_arr[ispec - 1] = np.exp(result)
                continue

            if ispec == 10 and dpro.lndref == 0.0:
                continue  # remains DMISSING

            # Chapman / logistic correction
            tanh_R = np.tanh((zeta_arr - dpro.zetaR) / (HRfact_arr * dpro.HR))
            if ispec in (2, 3, 5, 7):
                ccor = dpro.R * (1.0 + tanh_R)
            else:
                ccor = (-dpro.C_coeff * np.exp(-(zeta_arr - dpro.zetaC) / dpro.HC)
                        + dpro.R * (1.0 + tanh_R))

            # Hydrostatic integral
            Mz_arr = _pwmp_vec(zeta_arr, dpro.zetaMi, dpro.Mi, dpro.aMi)
            Ihyd_arr = Mz_arr * Vz_arr - dpro.Izref

            # XMi corrections (masks are disjoint)
            above4 = zeta_arr >= dpro.zetaMi[4]
            in_range = (zeta_arr > dpro.zetaMi[0]) & ~above4

            Ihyd_arr[above4] -= dpro.XMi[4]
            if in_range.any():
                z_in = zeta_arr[in_range]
                seg = np.searchsorted(dpro.zetaMi[1:4], z_in, side='left')  # 0-3
                Ihyd_arr[in_range] -= dpro.aMi[seg] * Wz_arr[in_range] + dpro.XMi[seg]

            dn_arr[ispec - 1] = (np.exp(dpro.lndref - Ihyd_arr * C.G0DIVKB + ccor)
                                 * dpro.Tref / tn_arr)

        if params.spec_flag[0]:
            dn_arr[0] = params.mass_wgt @ dn_arr

        return tn_arr, dn_arr

    def gtd8d(self, iyd: int, sec: float, alt: float, glat: float, glong: float,
              stl: float, f107a: float, f107: float, ap: np.ndarray,
              mass: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Legacy NRLMSISE-00 compatible interface.

        Args:
            iyd: Year-day (YYDDD format).
            sec: Universal time (seconds).
            alt: Geodetic altitude (km).
            glat: Geodetic latitude (degrees).
            glong: Geodetic longitude (degrees).
            stl: Local solar time (ignored).
            f107a: 81-day average F10.7.
            f107: Daily F10.7.
            ap: 7-element geomagnetic activity array.
            mass: Mass number (ignored).

        Returns:
            Tuple (d, t) where:
            - d: 10-element density array [He, O, N2, O2, Ar, rho, H, N, O*, NO] (cm⁻³/g/cm³)
            - t: 2-element temperature array [Tex, T_alt] (K)
        """
        xday = float(iyd % 1000)
        result = self.calc(xday, float(sec), float(alt), float(glat), float(glong),
                          float(f107a), float(f107), np.asarray(ap, dtype=float))

        xdn = result.densities.copy()

        # Convert from MKS to CGS
        mask = xdn != C.DMISSING
        xdn[mask] *= 1e-6  # m⁻³ → cm⁻³
        if xdn[0] != C.DMISSING:
            xdn[0] *= 1e3  # Already converted to cm⁻³ above, but mass density is kg/m³→g/cm³
            # Actually: dn[0] was in kg/m³, after *1e-6 it's wrong. Fix:
            # Fortran does: xdn = xdn*1d-6, then xdn(1) = xdn(1)*1e3
            # So mass density goes: kg/m³ * 1e-6 * 1e3 = kg/m³ * 1e-3 = g/cm³

        # Reorder to legacy format
        d = np.zeros(10, dtype=np.float32)
        d[0] = xdn[4]   # He
        d[1] = xdn[3]   # O
        d[2] = xdn[1]   # N2
        d[3] = xdn[2]   # O2
        d[4] = xdn[6]   # Ar
        d[5] = xdn[0]   # Mass density
        d[6] = xdn[5]   # H
        d[7] = xdn[7]   # N
        d[8] = xdn[8]   # Anomalous O
        d[9] = xdn[9]   # NO

        t = np.zeros(2, dtype=np.float32)
        t[0] = result.exospheric_temperature
        t[1] = result.temperature

        return d, t
