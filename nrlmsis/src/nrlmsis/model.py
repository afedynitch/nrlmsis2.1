"""NRLMSIS 2.1 main model class with smart caching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import constants as C
from .parameters import ModelParameters, load_model
from .globe import GlobeFunction
from .temperature import TnParm, compute_temperature, eval_temperature
from .density import DnParm, compute_density, eval_density
from .utils import alt2gph, bspline, dilog


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
    """

    def __init__(self, parm_file: str | Path | None = None,
                 switch_legacy: np.ndarray | None = None,
                 switch_gfn: np.ndarray | None = None,
                 alt_type: str = 'geodetic',
                 species_select: np.ndarray | None = None,
                 mass_include: np.ndarray | None = None,
                 n2_msis00: bool = False) -> None:
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
        self._profile_cache: Optional[ProfileCache] = None
        self._alt_cache: Optional[AltitudeCache] = None

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
        if self._profile_cache is None or key != self._profile_cache.key:
            gf = self._globe.evaluate(
                key.day, key.utsec, key.lat, key.lon,
                key.sfluxavg, key.sflux, np.array(key.ap), params.swg)
            tpro = compute_temperature(gf, params)
            dpro = {}
            for ispec in range(2, C.NSPEC):
                if params.spec_flag[ispec - 1]:
                    dpro[ispec] = compute_density(ispec, gf, tpro, params)
            self._profile_cache = ProfileCache(key=key, gf=gf, tpro=tpro, dpro=dpro)

        pcache = self._profile_cache
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
