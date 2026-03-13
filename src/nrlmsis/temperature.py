"""Vertical temperature profile parameters and evaluation for NRLMSIS 2.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .parameters import ModelParameters
from .globe import sfluxmod, geomag, utdep
from .utils import dilog


@dataclass
class TnParm:
    """Temperature profile parameters."""

    cf: np.ndarray = field(default_factory=lambda: np.zeros(C.NL + 1))
    tzetaF: float = 0.0
    tzetaA: float = 0.0
    dlntdzA: float = 0.0
    lndtotF: float = 0.0
    tex: float = 0.0
    tgb0: float = 0.0
    tb0: float = 0.0
    sigma: float = 0.0
    sigmasq: float = 0.0
    b: float = 0.0
    beta: np.ndarray = field(default_factory=lambda: np.zeros(C.NL + 1))
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(C.NL + 1))
    cVs: float = 0.0
    cVb: float = 0.0
    cWs: float = 0.0
    cWb: float = 0.0
    VzetaF: float = 0.0
    VzetaA: float = 0.0
    WzetaA: float = 0.0
    Vzeta0: float = 0.0


def compute_temperature(gf: np.ndarray, params: ModelParameters) -> TnParm:
    """Compute vertical temperature and species-independent profile parameters."""
    tpro = TnParm()
    TN = params.TN
    mbf = C.MBF

    # Unconstrained spline coefficients (single BLAS matmul instead of 22 dot calls)
    tpro.cf[:C.ITB0] = params.TN_beta_lin_T @ gf[:mbf + 1]

    for ix in range(C.ITB0):
        if params.smod[ix]:
            tpro.cf[ix] += sfluxmod(ix, gf, TN, 1.0 / TN.beta[0, ix],
                                     params.swg, params.zsfx, params.tsfx, params.psfx)

    # Exospheric temperature
    tpro.tex = np.dot(TN.beta[:mbf + 1, C.ITEX], gf[:mbf + 1])
    tpro.tex += sfluxmod(C.ITEX, gf, TN, 1.0 / TN.beta[0, C.ITEX],
                          params.swg, params.zsfx, params.tsfx, params.psfx)
    tpro.tex += geomag(TN.beta[C.CMAG:C.CMAG + C.NMAG, C.ITEX],
                        gf[C.CMAG:C.CMAG + 13],
                        gf[C.CMAG + 13:C.CMAG + 27].reshape(7, 2, order='F'),
                        params.swg)
    tpro.tex += utdep(TN.beta[C.CUT:C.CUT + C.NUT, C.ITEX],
                       gf[C.CUT:C.CUT + 9], params.swg)

    # Temperature gradient at zetaB
    tpro.tgb0 = np.dot(TN.beta[:mbf + 1, C.ITGB0], gf[:mbf + 1])
    if params.smod[C.ITGB0]:
        tpro.tgb0 += sfluxmod(C.ITGB0, gf, TN, 1.0 / TN.beta[0, C.ITGB0],
                                params.swg, params.zsfx, params.tsfx, params.psfx)
    tpro.tgb0 += geomag(TN.beta[C.CMAG:C.CMAG + C.NMAG, C.ITGB0],
                          gf[C.CMAG:C.CMAG + 13],
                          gf[C.CMAG + 13:C.CMAG + 27].reshape(7, 2, order='F'),
                          params.swg)

    # Temperature at zetaB
    tpro.tb0 = np.dot(TN.beta[:mbf + 1, C.ITB0], gf[:mbf + 1])
    if params.smod[C.ITB0]:
        tpro.tb0 += sfluxmod(C.ITB0, gf, TN, 1.0 / TN.beta[0, C.ITB0],
                               params.swg, params.zsfx, params.tsfx, params.psfx)
    tpro.tb0 += geomag(TN.beta[C.CMAG:C.CMAG + C.NMAG, C.ITB0],
                         gf[C.CMAG:C.CMAG + 13],
                         gf[C.CMAG + 13:C.CMAG + 27].reshape(7, 2, order='F'),
                         params.swg)

    # Shape factor
    tpro.sigma = tpro.tgb0 / (tpro.tex - tpro.tb0)

    # Constrain top three spline coefficients for C2 continuity
    bc = np.array([
        1.0 / tpro.tb0,
        -tpro.tgb0 / (tpro.tb0 * tpro.tb0),
        0.0
    ])
    bc[2] = -bc[1] * (tpro.sigma + 2.0 * tpro.tgb0 / tpro.tb0)
    tpro.cf[C.ITB0:C.ITEX + 1] = bc @ C.C2TN

    # Reference temperature at zetaF
    tpro.tzetaF = 1.0 / np.dot(tpro.cf[C.IZFX:C.IZFX + 3], C.S4ZETAF)

    # Reference temperature and gradient at zetaA
    tpro.tzetaA = 1.0 / np.dot(tpro.cf[C.IZAX:C.IZAX + 3], C.S4ZETAA)
    tpro.dlntdzA = -np.dot(tpro.cf[C.IZAX:C.IZAX + 3], C.WGHTAXDZ) * tpro.tzetaA

    # Integration coefficients (first and second 1/T integrals)
    tpro.beta[0] = tpro.cf[0] * C.WBETA[0]
    for ix in range(1, C.NL + 1):
        tpro.beta[ix] = tpro.beta[ix - 1] + tpro.cf[ix] * C.WBETA[ix]
    tpro.gamma[0] = tpro.beta[0] * C.WGAMMA[0]
    for ix in range(1, C.NL + 1):
        tpro.gamma[ix] = tpro.gamma[ix - 1] + tpro.beta[ix] * C.WGAMMA[ix]

    # Integration terms and constants
    tpro.b = 1 - tpro.tb0 / tpro.tex
    tpro.sigmasq = tpro.sigma * tpro.sigma
    tpro.cVs = -np.dot(tpro.beta[C.ITB0 - 1:C.ITB0 + 3], C.S5ZETAB)
    tpro.cWs = -np.dot(tpro.gamma[C.ITB0 - 2:C.ITB0 + 3], C.S6ZETAB)
    import math
    tpro.cVb = -math.log(1 - tpro.b) / (tpro.sigma * tpro.tex)
    tpro.cWb = -dilog(tpro.b) / (tpro.sigmasq * tpro.tex)
    tpro.VzetaF = np.dot(tpro.beta[C.IZFX - 1:C.IZFX + 3], C.S5ZETAF) + tpro.cVs
    tpro.VzetaA = np.dot(tpro.beta[C.IZAX - 1:C.IZAX + 3], C.S5ZETAA) + tpro.cVs
    tpro.WzetaA = np.dot(tpro.gamma[C.IZAX - 2:C.IZAX + 3], C.S6ZETAA) + tpro.cVs * (C.ZETA_A - C.ZETA_B) + tpro.cWs
    tpro.Vzeta0 = np.dot(tpro.beta[0:3], C.S5ZETA0) + tpro.cVs

    # Total number density at zetaF
    tpro.lndtotF = C.LNP0 - C.MBARG0DIVKB * (tpro.VzetaF - tpro.Vzeta0) - math.log(C.KB * tpro.tzetaF)

    return tpro


def eval_temperature(z: float, iz: int, wght: np.ndarray, tpro: TnParm) -> float:
    """Compute temperature at specified geopotential height.

    Args:
        z: Geopotential height (km).
        iz: B-spline reference index.
        wght: B-spline weights, 4 elements (from S[-3:1, 2] in Fortran indexing).
        tpro: Temperature profile parameters.

    Returns:
        Temperature (K).
    """
    import math
    if z < C.ZETA_B:
        i = max(iz - 3, 0)
        if iz < 3:
            j = -iz
        else:
            j = -3
        # wght has indices j:0 (length 4 max), cf has indices i:iz
        npts = iz - i + 1
        w_start = j + 3  # Map j to 0-based index in wght (j=-3→0, j=-2→1, etc.)
        return 1.0 / np.dot(tpro.cf[i:iz + 1], wght[w_start:w_start + npts])
    else:
        return tpro.tex - (tpro.tex - tpro.tb0) * math.exp(-tpro.sigma * (z - C.ZETA_B))
