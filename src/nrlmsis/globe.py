"""Global (horizontal and time-dependent) basis functions for NRLMSIS 2.1."""

from __future__ import annotations

import math

import numpy as np

from . import constants as C
from .parameters import ModelParameters, BasisSubset


class GlobeFunction:
    """Computes horizontal and temporal basis functions.

    Maintains internal caches for Legendre polynomials and Fourier coefficients,
    invalidated when the corresponding inputs change.
    """

    SFLUXAVG_REF = 150.0
    SFLUXAVG_QUAD_CUTOFF = 150.0

    def __init__(self) -> None:
        self._plg = np.zeros((C.MAXN + 1, C.MAXN + 1))
        self._cdoy = np.zeros(2)
        self._sdoy = np.zeros(2)
        self._clst = np.zeros(3)
        self._slst = np.zeros(3)
        self._clon = np.zeros(2)
        self._slon = np.zeros(2)
        self._last_lat = -999.9
        self._last_doy = -999.9
        self._last_lst = -999.9
        self._last_lon = -999.9

    def evaluate(self, doy: float, utsec: float, lat: float, lon: float,
                 sfluxavg: float, sflux: float, ap: np.ndarray,
                 swg: np.ndarray) -> np.ndarray:
        """Calculate basis function array (512 elements)."""
        plg = self._plg

        # Associated Legendre polynomials (cached on latitude)
        if lat != self._last_lat:
            clat = math.sin(lat * C.DEG2RAD)
            slat = math.cos(lat * C.DEG2RAD)
            clat2 = clat * clat
            clat4 = clat2 * clat2
            slat2 = slat * slat

            plg[0, 0] = 1.0
            plg[1, 0] = clat
            plg[2, 0] = 0.5 * (3.0 * clat2 - 1.0)
            plg[3, 0] = 0.5 * (5.0 * clat * clat2 - 3.0 * clat)
            plg[4, 0] = (35.0 * clat4 - 30.0 * clat2 + 3.0) / 8.0
            plg[5, 0] = (63.0 * clat2 * clat2 * clat - 70.0 * clat2 * clat + 15.0 * clat) / 8.0
            plg[6, 0] = (11.0 * clat * plg[5, 0] - 5.0 * plg[4, 0]) / 6.0

            plg[1, 1] = slat
            plg[2, 1] = 3.0 * clat * slat
            plg[3, 1] = 1.5 * (5.0 * clat2 - 1.0) * slat
            plg[4, 1] = 2.5 * (7.0 * clat2 * clat - 3.0 * clat) * slat
            plg[5, 1] = 1.875 * (21.0 * clat4 - 14.0 * clat2 + 1.0) * slat
            plg[6, 1] = (11.0 * clat * plg[5, 1] - 6.0 * plg[4, 1]) / 5.0

            plg[2, 2] = 3.0 * slat2
            plg[3, 2] = 15.0 * slat2 * clat
            plg[4, 2] = 7.5 * (7.0 * clat2 - 1.0) * slat2
            plg[5, 2] = 3.0 * clat * plg[4, 2] - 2.0 * plg[3, 2]
            plg[6, 2] = (11.0 * clat * plg[5, 2] - 7.0 * plg[4, 2]) / 4.0

            plg[3, 3] = 15.0 * slat2 * slat
            plg[4, 3] = 105.0 * slat2 * slat * clat
            plg[5, 3] = (9.0 * clat * plg[4, 3] - 7.0 * plg[3, 3]) / 2.0
            plg[6, 3] = (11.0 * clat * plg[5, 3] - 8.0 * plg[4, 3]) / 3.0

            self._last_lat = lat

        # Fourier harmonics of day of year (cached)
        if doy != self._last_doy:
            self._cdoy[0] = math.cos(C.DOY2RAD * doy)
            self._sdoy[0] = math.sin(C.DOY2RAD * doy)
            self._cdoy[1] = math.cos(C.DOY2RAD * doy * 2.0)
            self._sdoy[1] = math.sin(C.DOY2RAD * doy * 2.0)
            self._last_doy = doy

        # Local solar time
        lst = (utsec / 3600.0 + lon / 15.0 + 24.0) % 24.0
        if lst != self._last_lst:
            self._clst[0] = math.cos(C.LST2RAD * lst)
            self._slst[0] = math.sin(C.LST2RAD * lst)
            self._clst[1] = math.cos(C.LST2RAD * lst * 2.0)
            self._slst[1] = math.sin(C.LST2RAD * lst * 2.0)
            self._clst[2] = math.cos(C.LST2RAD * lst * 3.0)
            self._slst[2] = math.sin(C.LST2RAD * lst * 3.0)
            self._last_lst = lst

        # Fourier harmonics of longitude (cached)
        if lon != self._last_lon:
            self._clon[0] = math.cos(C.DEG2RAD * lon)
            self._slon[0] = math.sin(C.DEG2RAD * lon)
            self._clon[1] = math.cos(C.DEG2RAD * lon * 2.0)
            self._slon[1] = math.sin(C.DEG2RAD * lon * 2.0)
            self._last_lon = lon

        bf = np.zeros(C.MAXNBF)

        # Time-independent terms
        c = C.CTIMEIND
        for n in range(C.AMAXN + 1):
            bf[c] = plg[n, 0]
            c += 1

        # Intra-annual terms
        for s in range(1, C.AMAXS + 1):
            cosdoy = self._cdoy[s - 1]
            sindoy = self._sdoy[s - 1]
            for n in range(C.AMAXN + 1):
                pl = plg[n, 0]
                bf[c] = pl * cosdoy
                bf[c + 1] = pl * sindoy
                c += 2

        # Migrating tides
        for l in range(1, C.TMAXL + 1):
            coslst = self._clst[l - 1]
            sinlst = self._slst[l - 1]
            for n in range(l, C.TMAXN + 1):
                pl = plg[n, l]
                bf[c] = pl * coslst
                bf[c + 1] = pl * sinlst
                c += 2
            for s in range(1, C.TMAXS + 1):
                cosdoy = self._cdoy[s - 1]
                sindoy = self._sdoy[s - 1]
                for n in range(l, C.TMAXN + 1):
                    pl = plg[n, l]
                    bf[c] = pl * coslst * cosdoy
                    bf[c + 1] = pl * sinlst * cosdoy
                    bf[c + 2] = pl * coslst * sindoy
                    bf[c + 3] = pl * sinlst * sindoy
                    c += 4

        # Stationary planetary waves
        for m in range(1, C.PMAXM + 1):
            coslon = self._clon[m - 1]
            sinlon = self._slon[m - 1]
            for n in range(m, C.PMAXN + 1):
                pl = plg[n, m]
                bf[c] = pl * coslon
                bf[c + 1] = pl * sinlon
                c += 2
            for s in range(1, C.PMAXS + 1):
                cosdoy = self._cdoy[s - 1]
                sindoy = self._sdoy[s - 1]
                for n in range(m, C.PMAXN + 1):
                    pl = plg[n, m]
                    bf[c] = pl * coslon * cosdoy
                    bf[c + 1] = pl * sinlon * cosdoy
                    bf[c + 2] = pl * coslon * sindoy
                    bf[c + 3] = pl * sinlon * sindoy
                    c += 4

        # Linear solar flux terms
        dfa = sfluxavg - self.SFLUXAVG_REF
        df = sflux - sfluxavg
        bf[c] = dfa
        bf[c + 1] = dfa * dfa
        bf[c + 2] = df
        bf[c + 3] = df * df
        bf[c + 4] = df * dfa
        c += C.NSFX

        # Additional linear terms
        sza = self.solar_zenith(doy, lst, lat, lon)
        bf[c] = -0.5 * math.tanh((sza - 98.0) / 6.0)
        bf[c + 1] = -0.5 * math.tanh((sza - 101.5) / 20.0)
        bf[c + 2] = dfa * bf[c]
        bf[c + 3] = dfa * bf[c + 1]
        bf[c + 4] = dfa * plg[2, 0]
        bf[c + 5] = dfa * plg[4, 0]
        bf[c + 6] = dfa * plg[0, 0] * self._cdoy[0]
        bf[c + 7] = dfa * plg[0, 0] * self._sdoy[0]
        bf[c + 8] = dfa * plg[0, 0] * self._cdoy[1]
        bf[c + 9] = dfa * plg[0, 0] * self._sdoy[1]
        if sfluxavg <= self.SFLUXAVG_QUAD_CUTOFF:
            bf[c + 10] = dfa * dfa
        else:
            bf[c + 10] = (self.SFLUXAVG_QUAD_CUTOFF - self.SFLUXAVG_REF) * (
                2.0 * dfa - (self.SFLUXAVG_QUAD_CUTOFF - self.SFLUXAVG_REF))
        bf[c + 11] = bf[c + 10] * plg[2, 0]
        bf[c + 12] = bf[c + 10] * plg[4, 0]
        bf[c + 13] = df * plg[2, 0]
        bf[c + 14] = df * plg[4, 0]

        # Nonlinear solar flux modulation terms
        c = C.CNONLIN
        bf[c] = dfa
        bf[c + 1] = dfa * dfa
        bf[c + 2] = df
        bf[c + 3] = df * df
        bf[c + 4] = df * dfa
        c += C.NSFXMOD

        # Legacy geomagnetic activity terms
        bf[c:c + 7] = ap[:7] - 4.0
        bf[c + 8] = C.DOY2RAD * doy
        bf[c + 9] = C.LST2RAD * lst
        bf[c + 10] = C.DEG2RAD * lon
        bf[c + 11] = C.LST2RAD * utsec / 3600.0
        bf[c + 12] = abs(lat)
        c += 13
        for m in range(2):
            for n in range(C.AMAXN + 1):
                bf[c] = plg[n, m]
                c += 1

        # Legacy UT terms
        c = C.CUT
        bf[c] = C.LST2RAD * utsec / 3600.0
        bf[c + 1] = C.DOY2RAD * doy
        bf[c + 2] = dfa
        bf[c + 3] = C.DEG2RAD * lon
        bf[c + 4] = plg[1, 0]
        bf[c + 5] = plg[3, 0]
        bf[c + 6] = plg[5, 0]
        bf[c + 7] = plg[3, 2]
        bf[c + 8] = plg[5, 2]

        # Apply switches
        bf[:C.MBF + 1][~swg[:C.MBF + 1]] = 0.0

        return bf

    @staticmethod
    def solar_zenith(ddd: float, lst: float, lat: float, lon: float) -> float:
        """Calculate solar zenith angle in degrees."""
        humr = C.PI / 12.0
        p = np.array([0.017203534, 0.034407068, 0.051610602, 0.068814136, 0.103221204])

        teqnx = ddd + 0.9369

        # Solar declination
        dec = (23.256 * math.sin(p[0] * (teqnx - 82.242))
               + 0.381 * math.sin(p[1] * (teqnx - 44.855))
               + 0.167 * math.sin(p[2] * (teqnx - 23.355))
               - 0.013 * math.sin(p[3] * (teqnx + 11.97))
               + 0.011 * math.sin(p[4] * (teqnx - 10.410))
               + 0.339137)
        dec *= C.DEG2RAD

        # Equation of time
        tf = teqnx - 0.5
        teqt = (-7.38 * math.sin(p[0] * (tf - 4.0))
                - 9.87 * math.sin(p[1] * (tf + 9.0))
                + 0.27 * math.sin(p[2] * (tf - 53.0))
                - 0.2 * math.cos(p[3] * (tf - 17.0)))

        phi = humr * (lst - 12.0) + teqt * C.DEG2RAD / 4.0
        rlat = lat * C.DEG2RAD

        cosx = math.sin(rlat) * math.sin(dec) + math.cos(rlat) * math.cos(dec) * math.cos(phi)
        if abs(cosx) > 1.0:
            cosx = math.copysign(1.0, cosx)

        return math.acos(cosx) / C.DEG2RAD


def sfluxmod(iz: int, gf: np.ndarray, parmset: BasisSubset,
             dffact: float, swg: np.ndarray,
             zsfx: np.ndarray, tsfx: np.ndarray, psfx: np.ndarray) -> float:
    """Legacy nonlinear modulation of intra-annual, tide, and SPW terms."""
    # Intra-annual modulation factor
    if swg[C.CSFXMOD]:
        f1 = (parmset.beta[C.CSFXMOD, iz] * gf[C.CSFXMOD]
              + (parmset.beta[C.CSFX + 2, iz] * gf[C.CSFXMOD + 2]
                 + parmset.beta[C.CSFX + 3, iz] * gf[C.CSFXMOD + 3]) * dffact)
    else:
        f1 = 0.0

    # Tide modulation factor
    if swg[C.CSFXMOD + 1]:
        f2 = (parmset.beta[C.CSFXMOD + 1, iz] * gf[C.CSFXMOD]
              + (parmset.beta[C.CSFX + 2, iz] * gf[C.CSFXMOD + 2]
                 + parmset.beta[C.CSFX + 3, iz] * gf[C.CSFXMOD + 3]) * dffact)
    else:
        f2 = 0.0

    # SPW modulation factor
    if swg[C.CSFXMOD + 2]:
        f3 = parmset.beta[C.CSFXMOD + 2, iz] * gf[C.CSFXMOD]
    else:
        f3 = 0.0

    total = 0.0
    for j in range(C.MBF + 1):
        if zsfx[j]:
            total += parmset.beta[j, iz] * gf[j] * f1
        elif tsfx[j]:
            total += parmset.beta[j, iz] * gf[j] * f2
        elif psfx[j]:
            total += parmset.beta[j, iz] * gf[j] * f3

    return total


def _g0fn(a: float, k00r: float, k00s: float) -> float:
    """Polynomial saturation function for geomagnetic activity."""
    return a + (k00r - 1.0) * (a + (math.exp(-a * k00s) - 1.0) / k00s)


def geomag(p0: np.ndarray, bf: np.ndarray, plg: np.ndarray,
           swg: np.ndarray) -> float:
    """Legacy nonlinear ap dependence (daily and 3-hr history modes)."""
    if not (swg[C.CMAG] or swg[C.CMAG + 1]):
        return 0.0

    p = p0.copy()
    swg1 = swg[C.CMAG:C.CMAG + C.NMAG].copy()

    if swg1[0] == swg1[1]:
        # Daily Ap mode
        if p[1] == 0:
            return 0.0
        p[2:26][~swg1[2:26]] = 0.0
        p[8] = p0[8]
        delA = _g0fn(bf[0], p[0], p[1])
        result = (
            p[2] * plg[0, 0] + p[3] * plg[2, 0] + p[4] * plg[4, 0]
            + (p[5] * plg[1, 0] + p[6] * plg[3, 0] + p[7] * plg[5, 0]) * math.cos(bf[8] - p[8])
            + (p[9] * plg[1, 1] + p[10] * plg[3, 1] + p[11] * plg[5, 1]) * math.cos(bf[9] - p[12])
            + (1.0 + p[13] * plg[1, 0])
            * (p[14] * plg[2, 1] + p[15] * plg[4, 1] + p[16] * plg[6, 1]) * math.cos(bf[10] - p[17])
            + (p[18] * plg[1, 1] + p[19] * plg[3, 1] + p[20] * plg[5, 1]) * math.cos(bf[10] - p[21])
            * math.cos(bf[8] - p[8])
            + (p[22] * plg[1, 0] + p[23] * plg[3, 0] + p[24] * plg[5, 0]) * math.cos(bf[11] - p[25])
        ) * delA
    else:
        # 3-hour ap history mode
        if p[28] == 0:
            return 0.0
        p[30:][~swg1[30:len(p)]] = 0.0
        p[36] = p0[36]
        gbeta = p[28] / (1 + p[29] * (45.0 - bf[12]))
        ex = math.exp(-10800.0 * gbeta)
        sumex = 1 + (1 - ex**19.0) * ex**0.5 / (1 - ex)
        G = [0.0] * 7
        for ii in range(1, 7):
            G[ii] = _g0fn(bf[ii], p[26], p[27])
        delA = (G[1]
                + (G[2] * ex + G[3] * ex * ex + G[4] * ex**3.0
                   + (G[5] * ex**4.0 + G[6] * ex**12.0) * (1 - ex**8.0) / (1 - ex))
                ) / sumex
        result = (
            p[30] * plg[0, 0] + p[31] * plg[2, 0] + p[32] * plg[4, 0]
            + (p[33] * plg[1, 0] + p[34] * plg[3, 0] + p[35] * plg[5, 0]) * math.cos(bf[8] - p[36])
            + (p[37] * plg[1, 1] + p[38] * plg[3, 1] + p[39] * plg[5, 1]) * math.cos(bf[9] - p[40])
            + (1.0 + p[41] * plg[1, 0])
            * (p[42] * plg[2, 1] + p[43] * plg[4, 1] + p[44] * plg[6, 1]) * math.cos(bf[10] - p[45])
            + (p[46] * plg[1, 1] + p[47] * plg[3, 1] + p[48] * plg[5, 1]) * math.cos(bf[10] - p[49])
            * math.cos(bf[8] - p[36])
            + (p[50] * plg[1, 0] + p[51] * plg[3, 0] + p[52] * plg[5, 0]) * math.cos(bf[11] - p[53])
        ) * delA

    return result


def utdep(p0: np.ndarray, bf: np.ndarray, swg: np.ndarray) -> float:
    """Legacy nonlinear UT dependence."""
    p = p0.copy()
    swg1 = swg[C.CUT:C.CUT + C.NUT]
    p[3:][~swg1[3:len(p)]] = 0.0

    return (
        math.cos(bf[0] - p[0])
        * (1 + p[3] * bf[4] * math.cos(bf[1] - p[1]))
        * (1 + p[4] * bf[2]) * (1 + p[5] * bf[4])
        * (p[6] * bf[4] + p[7] * bf[5] + p[8] * bf[6])
        + math.cos(bf[0] - p[2] + 2 * bf[3]) * (p[9] * bf[7] + p[10] * bf[8]) * (1 + p[11] * bf[2])
    )
