"""Vertical species density profiles for NRLMSIS 2.1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .parameters import ModelParameters
from .temperature import TnParm
from .globe import sfluxmod, geomag, utdep
from .utils import bspline, dilog


@dataclass
class DnParm:
    """Density profile parameters for a single species."""

    ispec: int = 0
    lnPhiF: float = 0.0
    lndref: float = 0.0
    zetaM: float = 0.0
    HML: float = 0.0
    HMU: float = 0.0
    C_coeff: float = 0.0  # Chapman term coefficient (renamed from C to avoid builtin conflict)
    zetaC: float = 0.0
    HC: float = 0.0
    R: float = 0.0
    zetaR: float = 0.0
    HR: float = 0.0
    cf: np.ndarray = field(default_factory=lambda: np.zeros(C.NSPLO1 + 2))
    zref: float = 0.0
    Mi: np.ndarray = field(default_factory=lambda: np.zeros(5))
    zetaMi: np.ndarray = field(default_factory=lambda: np.zeros(5))
    aMi: np.ndarray = field(default_factory=lambda: np.zeros(5))
    WMi: np.ndarray = field(default_factory=lambda: np.zeros(5))
    XMi: np.ndarray = field(default_factory=lambda: np.zeros(5))
    Izref: float = 0.0
    Tref: float = 0.0
    zmin: float = 0.0
    zhyd: float = 0.0


def compute_density(ispec: int, gf: np.ndarray, tpro: TnParm,
                    params: ModelParameters) -> DnParm:
    """Compute species density profile parameters.

    Args:
        ispec: Species index (2=N2, 3=O2, 4=O, 5=He, 6=H, 7=Ar, 8=N, 9=AnomalousO, 10=NO).
        gf: Basis function array (512 elements).
        tpro: Temperature profile parameters.
        params: Model parameters.

    Returns:
        DnParm for the species.
    """
    dpro = DnParm(ispec=ispec)
    mbf = C.MBF
    gf_lin = gf[:mbf + 1]

    # Helper for geomag call
    def _geomag(pset, iz):
        return geomag(pset.beta[C.CMAG:C.CMAG + C.NMAG, iz],
                      gf[C.CMAG:C.CMAG + 13],
                      gf[C.CMAG + 13:C.CMAG + 27].reshape(-1),
                      params.swg)

    def _utdep(pset, iz):
        return utdep(pset.beta[C.CUT:C.CUT + C.NUT, iz],
                     gf[C.CUT:C.CUT + 9], params.swg)

    def _sfluxmod(iz, pset, dffact):
        return sfluxmod(iz, gf, pset, dffact,
                        params.swg, params.zsfx, params.tsfx, params.psfx)

    if ispec == 2:  # N2
        dpro.lnPhiF = C.LNVMR[1]
        dpro.lndref = tpro.lndtotF + dpro.lnPhiF
        dpro.zref = C.ZETA_F
        dpro.zmin = -1.0
        dpro.zhyd = C.ZETA_F
        dpro.zetaM = np.dot(params.N2.beta[:mbf + 1, 1], gf_lin)
        dpro.HML = params.N2.beta[0, 2]
        dpro.HMU = params.N2.beta[0, 3]
        dpro.R = 0.0
        if params.n2r_flag:
            dpro.R = np.dot(params.N2.beta[:mbf + 1, 7], gf_lin)
        dpro.zetaR = params.N2.beta[0, 8]
        dpro.HR = params.N2.beta[0, 9]

    elif ispec == 3:  # O2
        dpro.lnPhiF = C.LNVMR[2]
        dpro.lndref = tpro.lndtotF + dpro.lnPhiF
        dpro.zref = C.ZETA_F
        dpro.zmin = -1.0
        dpro.zhyd = C.ZETA_F
        dpro.zetaM = params.O2.beta[0, 1]
        dpro.HML = params.O2.beta[0, 2]
        dpro.HMU = params.O2.beta[0, 3]
        dpro.R = np.dot(params.O2.beta[:mbf + 1, 7], gf_lin)
        dpro.R += _geomag(params.O2, 7)
        dpro.zetaR = params.O2.beta[0, 8]
        dpro.HR = params.O2.beta[0, 9]

    elif ispec == 4:  # O
        dpro.lnPhiF = 0.0
        dpro.lndref = np.dot(params.O1.beta[:mbf + 1, 0], gf_lin)
        dpro.zref = C.ZETAREF_O1
        dpro.zmin = C.NODES_O1[3]
        dpro.zhyd = C.ZETAREF_O1
        dpro.zetaM = params.O1.beta[0, 1]
        dpro.HML = params.O1.beta[0, 2]
        dpro.HMU = params.O1.beta[0, 3]
        dpro.C_coeff = np.dot(params.O1.beta[:mbf + 1, 4], gf_lin)
        dpro.zetaC = params.O1.beta[0, 5]
        dpro.HC = params.O1.beta[0, 6]
        dpro.R = np.dot(params.O1.beta[:mbf + 1, 7], gf_lin)
        dpro.R += _sfluxmod(7, params.O1, 0.0)
        dpro.R += _geomag(params.O1, 7)
        dpro.R += _utdep(params.O1, 7)
        dpro.zetaR = params.O1.beta[0, 8]
        dpro.HR = params.O1.beta[0, 9]
        for izf in range(C.NSPLO1):
            dpro.cf[izf] = np.dot(params.O1.beta[:mbf + 1, izf + 10], gf_lin)

    elif ispec == 5:  # He
        dpro.lnPhiF = C.LNVMR[4]
        dpro.lndref = tpro.lndtotF + dpro.lnPhiF
        dpro.zref = C.ZETA_F
        dpro.zmin = -1.0
        dpro.zhyd = C.ZETA_F
        dpro.zetaM = params.HE.beta[0, 1]
        dpro.HML = params.HE.beta[0, 2]
        dpro.HMU = params.HE.beta[0, 3]
        dpro.R = np.dot(params.HE.beta[:mbf + 1, 7], gf_lin)
        dpro.R += _sfluxmod(7, params.HE, 1.0)
        dpro.R += _geomag(params.HE, 7)
        dpro.R += _utdep(params.HE, 7)
        dpro.zetaR = params.HE.beta[0, 8]
        dpro.HR = params.HE.beta[0, 9]

    elif ispec == 6:  # H
        dpro.lnPhiF = 0.0
        dpro.lndref = np.dot(params.H1.beta[:mbf + 1, 0], gf_lin)
        dpro.zref = C.ZETA_A
        dpro.zmin = 75.0
        dpro.zhyd = C.ZETA_F
        dpro.zetaM = params.H1.beta[0, 1]
        dpro.HML = params.H1.beta[0, 2]
        dpro.HMU = params.H1.beta[0, 3]
        dpro.C_coeff = np.dot(params.H1.beta[:mbf + 1, 4], gf_lin)
        dpro.zetaC = np.dot(params.H1.beta[:mbf + 1, 5], gf_lin)
        dpro.HC = params.H1.beta[0, 6]
        dpro.R = np.dot(params.H1.beta[:mbf + 1, 7], gf_lin)
        dpro.R += _sfluxmod(7, params.H1, 0.0)
        dpro.R += _geomag(params.H1, 7)
        dpro.R += _utdep(params.H1, 7)
        dpro.zetaR = params.H1.beta[0, 8]
        dpro.HR = params.H1.beta[0, 9]

    elif ispec == 7:  # Ar
        dpro.lnPhiF = C.LNVMR[6]
        dpro.lndref = tpro.lndtotF + dpro.lnPhiF
        dpro.zref = C.ZETA_F
        dpro.zmin = -1.0
        dpro.zhyd = C.ZETA_F
        dpro.zetaM = params.AR.beta[0, 1]
        dpro.HML = params.AR.beta[0, 2]
        dpro.HMU = params.AR.beta[0, 3]
        dpro.R = np.dot(params.AR.beta[:mbf + 1, 7], gf_lin)
        dpro.R += _geomag(params.AR, 7)
        dpro.R += _utdep(params.AR, 7)
        dpro.zetaR = params.AR.beta[0, 8]
        dpro.HR = params.AR.beta[0, 9]

    elif ispec == 8:  # N
        dpro.lnPhiF = 0.0
        dpro.lndref = np.dot(params.N1.beta[:mbf + 1, 0], gf_lin)
        dpro.lndref += _sfluxmod(0, params.N1, 0.0)
        dpro.lndref += _geomag(params.N1, 0)
        dpro.lndref += _utdep(params.N1, 0)
        dpro.zref = C.ZETA_B
        dpro.zmin = 90.0
        dpro.zhyd = C.ZETA_F
        dpro.zetaM = params.N1.beta[0, 1]
        dpro.HML = params.N1.beta[0, 2]
        dpro.HMU = params.N1.beta[0, 3]
        dpro.C_coeff = params.N1.beta[0, 4]
        dpro.zetaC = params.N1.beta[0, 5]
        dpro.HC = params.N1.beta[0, 6]
        dpro.R = np.dot(params.N1.beta[:mbf + 1, 7], gf_lin)
        dpro.zetaR = params.N1.beta[0, 8]
        dpro.HR = params.N1.beta[0, 9]

    elif ispec == 9:  # Anomalous O
        dpro.lndref = np.dot(params.OA.beta[:mbf + 1, 0], gf_lin)
        dpro.lndref += _geomag(params.OA, 0)
        dpro.zref = C.ZETAREF_OA
        dpro.zmin = 120.0
        dpro.zhyd = 0.0
        dpro.C_coeff = params.OA.beta[0, 4]
        dpro.zetaC = params.OA.beta[0, 5]
        dpro.HC = params.OA.beta[0, 6]
        return dpro  # No further parameters for anomalous O

    elif ispec == 10:  # NO
        if params.NO.beta[0, 0] == 0.0:
            dpro.lndref = 0.0
            return dpro
        dpro.lnPhiF = 0.0
        dpro.lndref = np.dot(params.NO.beta[:mbf + 1, 0], gf_lin)
        dpro.lndref += _geomag(params.NO, 0)
        dpro.zref = C.ZETAREF_NO
        dpro.zmin = 72.5
        dpro.zhyd = C.ZETAREF_NO
        dpro.zetaM = np.dot(params.NO.beta[:mbf + 1, 1], gf_lin)
        dpro.HML = np.dot(params.NO.beta[:mbf + 1, 2], gf_lin)
        dpro.HMU = np.dot(params.NO.beta[:mbf + 1, 3], gf_lin)
        dpro.C_coeff = np.dot(params.NO.beta[:mbf + 1, 4], gf_lin)
        dpro.C_coeff += _geomag(params.NO, 4)
        dpro.zetaC = np.dot(params.NO.beta[:mbf + 1, 5], gf_lin)
        dpro.HC = np.dot(params.NO.beta[:mbf + 1, 6], gf_lin)
        dpro.R = np.dot(params.NO.beta[:mbf + 1, 7], gf_lin)
        dpro.zetaR = np.dot(params.NO.beta[:mbf + 1, 8], gf_lin)
        dpro.HR = np.dot(params.NO.beta[:mbf + 1, 9], gf_lin)
        for izf in range(C.NSPLNO):
            dpro.cf[izf] = np.dot(params.NO.beta[:mbf + 1, izf + 10], gf_lin)
            dpro.cf[izf] += geomag(
                params.NO.beta[C.CMAG:C.CMAG + C.NMAG, izf + 10],
                gf[C.CMAG:C.CMAG + 13],
                gf[C.CMAG + 13:C.CMAG + 27].reshape(-1),
                params.swg)
    else:
        raise ValueError(f'Species {ispec} not implemented')

    # Compute piecewise mass profile
    dpro.zetaMi[0] = dpro.zetaM - 2.0 * dpro.HML
    dpro.zetaMi[1] = dpro.zetaM - dpro.HML
    dpro.zetaMi[2] = dpro.zetaM
    dpro.zetaMi[3] = dpro.zetaM + dpro.HMU
    dpro.zetaMi[4] = dpro.zetaM + 2.0 * dpro.HMU
    dpro.Mi[0] = C.MBAR
    dpro.Mi[4] = C.SPECMASS[ispec - 1]  # ispec is 2-10, SPECMASS is 0-indexed with 0=dummy
    dpro.Mi[2] = (dpro.Mi[0] + dpro.Mi[4]) / 2.0
    delM = C.TANH1 * (dpro.Mi[4] - dpro.Mi[0]) / 2.0
    dpro.Mi[1] = dpro.Mi[2] - delM
    dpro.Mi[3] = dpro.Mi[2] + delM

    for i in range(4):
        dpro.aMi[i] = (dpro.Mi[i + 1] - dpro.Mi[i]) / (dpro.zetaMi[i + 1] - dpro.zetaMi[i])

    for i in range(5):
        delz = dpro.zetaMi[i] - C.ZETA_B
        if dpro.zetaMi[i] < C.ZETA_B:
            Si, iz = bspline(dpro.zetaMi[i], C.NODES_TN, C.ND + 2, 6, params.eta_tn)
            # Si[offset+5, k-2]: k=6 → col 4, offsets iz-5:iz → rows (iz-5+5):(iz+5+1) = 0:6
            dpro.WMi[i] = np.dot(tpro.gamma[iz - 5:iz + 1], Si[:6, 4]) + tpro.cVs * delz + tpro.cWs
        else:
            dpro.WMi[i] = ((0.5 * delz * delz
                            + dilog(tpro.b * math.exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
                           + tpro.cVb * delz + tpro.cWb)

    dpro.XMi[0] = -dpro.aMi[0] * dpro.WMi[0]
    for i in range(1, 4):
        dpro.XMi[i] = dpro.XMi[i - 1] - dpro.WMi[i] * (dpro.aMi[i] - dpro.aMi[i - 1])
    dpro.XMi[4] = dpro.XMi[3] + dpro.WMi[4] * dpro.aMi[3]

    # Hydrostatic integral at reference height
    if dpro.zref == C.ZETA_F:
        Mzref = C.MBAR
        dpro.Tref = tpro.tzetaF
        dpro.Izref = C.MBAR * tpro.VzetaF
    elif dpro.zref == C.ZETA_B:
        Mzref = _pwmp(dpro.zref, dpro.zetaMi, dpro.Mi, dpro.aMi)
        dpro.Tref = tpro.tb0
        dpro.Izref = 0.0
        if C.ZETA_B > dpro.zetaMi[0] and C.ZETA_B < dpro.zetaMi[4]:
            i = 0
            for i1 in range(1, 4):
                if C.ZETA_B < dpro.zetaMi[i1]:
                    break
                i = i1
            dpro.Izref -= dpro.XMi[i]
        else:
            dpro.Izref -= dpro.XMi[4]
    elif dpro.zref == C.ZETA_A:
        Mzref = _pwmp(dpro.zref, dpro.zetaMi, dpro.Mi, dpro.aMi)
        dpro.Tref = tpro.tzetaA
        dpro.Izref = Mzref * tpro.VzetaA
        if C.ZETA_A > dpro.zetaMi[0] and C.ZETA_A < dpro.zetaMi[4]:
            i = 0
            for i1 in range(1, 4):
                if C.ZETA_A < dpro.zetaMi[i1]:
                    break
                i = i1
            dpro.Izref -= (dpro.aMi[i] * tpro.WzetaA + dpro.XMi[i])
        else:
            dpro.Izref -= dpro.XMi[4]
    else:
        raise ValueError('Integrals at reference height not available')

    # C1 constraint for O1 at 85 km
    if ispec == 4:
        Cterm = dpro.C_coeff * math.exp(-(dpro.zref - dpro.zetaC) / dpro.HC)
        Rterm0 = math.tanh((dpro.zref - dpro.zetaR) / (params.hr_fact_o1_ref * dpro.HR))
        Rterm = dpro.R * (1 + Rterm0)
        bc = np.array([
            dpro.lndref - Cterm + Rterm - dpro.cf[7] * C.C1O1ADJ[0],
            (-Mzref * C.G0DIVKB / tpro.tzetaA
             - tpro.dlntdzA
             + Cterm / dpro.HC
             + Rterm * (1 - Rterm0) / dpro.HR * params.dhr_fact_o1_ref
             - dpro.cf[7] * C.C1O1ADJ[1])
        ])
        dpro.cf[8:10] = bc @ C.C1O1

    # C1 constraint for NO at 122.5 km
    if ispec == 10:
        Cterm = dpro.C_coeff * math.exp(-(dpro.zref - dpro.zetaC) / dpro.HC)
        Rterm0 = math.tanh((dpro.zref - dpro.zetaR) / (params.hr_fact_no_ref * dpro.HR))
        Rterm = dpro.R * (1 + Rterm0)
        bc = np.array([
            dpro.lndref - Cterm + Rterm - dpro.cf[7] * C.C1NOADJ[0],
            (-Mzref * C.G0DIVKB / tpro.tb0
             - tpro.tgb0 / tpro.tb0
             + Cterm / dpro.HC
             + Rterm * (1 - Rterm0) / dpro.HR * params.dhr_fact_no_ref
             - dpro.cf[7] * C.C1NOADJ[1])
        ])
        dpro.cf[8:10] = bc @ C.C1NO

    return dpro


def eval_density(z: float, tnz: float, lndtotz: float, Vz: float, Wz: float,
                 HRfact: float, tpro: TnParm, dpro: DnParm,
                 params: ModelParameters) -> float:
    """Compute species density at specified geopotential height.

    Returns:
        Number density (m⁻³) or mass density (kg/m³).
    """
    if z < dpro.zmin:
        return C.DMISSING

    # Anomalous Oxygen
    if dpro.ispec == 9:
        result = dpro.lndref - (z - dpro.zref) / C.HOA - dpro.C_coeff * math.exp(-(z - dpro.zetaC) / dpro.HC)
        return math.exp(result)

    # NO: skip if not defined
    if dpro.ispec == 10 and dpro.lndref == 0.0:
        return C.DMISSING

    # Chapman and logistic corrections
    if dpro.ispec in (2, 3, 5, 7):
        ccor = dpro.R * (1 + math.tanh((z - dpro.zetaR) / (HRfact * dpro.HR)))
    else:  # 4, 6, 8, 10
        ccor = (-dpro.C_coeff * math.exp(-(z - dpro.zetaC) / dpro.HC)
                + dpro.R * (1 + math.tanh((z - dpro.zetaR) / (HRfact * dpro.HR))))

    # Below hydrostatic height
    if z < dpro.zhyd:
        if dpro.ispec in (2, 3, 5, 7):
            return math.exp(lndtotz + dpro.lnPhiF + ccor)
        elif dpro.ispec == 4:
            Sz, iz = bspline(z, C.NODES_O1, C.NDO1, 4, params.eta_o1)
            # S[-3:1, 4] → S[2:6, 2] (k=4 is col index 2)
            return math.exp(np.dot(dpro.cf[iz - 3:iz + 1], Sz[2:6, 2]))
        elif dpro.ispec == 10:
            Sz, iz = bspline(z, C.NODES_NO, C.NDNO, 4, params.eta_no)
            return math.exp(np.dot(dpro.cf[iz - 3:iz + 1], Sz[2:6, 2]))

    # Hydrostatic term
    Mz = _pwmp(z, dpro.zetaMi, dpro.Mi, dpro.aMi)
    Ihyd = Mz * Vz - dpro.Izref
    if z > dpro.zetaMi[0] and z < dpro.zetaMi[4]:
        i = 0
        for i1 in range(1, 4):
            if z < dpro.zetaMi[i1]:
                break
            i = i1
        Ihyd -= (dpro.aMi[i] * Wz + dpro.XMi[i])
    elif z >= dpro.zetaMi[4]:
        Ihyd -= dpro.XMi[4]

    result = dpro.lndref - Ihyd * C.G0DIVKB + ccor
    return math.exp(result) * dpro.Tref / tnz


def _pwmp(z: float, zm: np.ndarray, m: np.ndarray, dmdz: np.ndarray) -> float:
    """Piecewise effective mass profile interpolation."""
    if z >= zm[4]:
        return m[4]
    if z <= zm[0]:
        return m[0]
    for inode in range(4):
        if z < zm[inode + 1]:
            return m[inode] + dmdz[inode] * (z - zm[inode])
    raise RuntimeError('Error in _pwmp')
