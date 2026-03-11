"""Parameter loading, BasisSubset dataclass, and switch management."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import constants as C


@dataclass
class BasisSubset:
    """Parameter storage structure for a single species or temperature."""

    name: str
    bl: int
    nl: int
    beta: np.ndarray      # shape (MAXNBF, nl-bl+1)
    active: np.ndarray     # boolean, same shape
    fitb: np.ndarray       # int, same shape

    @staticmethod
    def create(name: str, bl: int, nl: int) -> BasisSubset:
        ncols = nl - bl + 1
        return BasisSubset(
            name=name, bl=bl, nl=nl,
            beta=np.zeros((C.MAXNBF, ncols)),
            active=np.full((C.MAXNBF, ncols), False),
            fitb=np.zeros((C.MAXNBF, ncols), dtype=int),
        )


@dataclass
class ModelParameters:
    """All loaded model parameters and configuration."""

    TN: BasisSubset
    PR: BasisSubset
    N2: BasisSubset
    O2: BasisSubset
    O1: BasisSubset
    HE: BasisSubset
    H1: BasisSubset
    AR: BasisSubset
    N1: BasisSubset
    OA: BasisSubset
    NO: BasisSubset

    # Reciprocal node difference arrays for B-spline calculations
    eta_tn: np.ndarray = field(default_factory=lambda: np.zeros((31, 5)))
    eta_o1: np.ndarray = field(default_factory=lambda: np.zeros((31, 5)))
    eta_no: np.ndarray = field(default_factory=lambda: np.zeros((31, 5)))

    # C1 constraint terms for O and NO
    hr_fact_o1_ref: float = 0.0
    dhr_fact_o1_ref: float = 0.0
    hr_fact_no_ref: float = 0.0
    dhr_fact_no_ref: float = 0.0

    # Switch arrays
    swg: np.ndarray = field(default_factory=lambda: np.ones(C.MAXNBF, dtype=bool))

    # Solar flux modulation flags
    zsfx: np.ndarray = field(default_factory=lambda: np.zeros(C.MBF + 1, dtype=bool))
    tsfx: np.ndarray = field(default_factory=lambda: np.zeros(C.MBF + 1, dtype=bool))
    psfx: np.ndarray = field(default_factory=lambda: np.zeros(C.MBF + 1, dtype=bool))
    smod: np.ndarray = field(default_factory=lambda: np.zeros(C.NL + 1, dtype=bool))

    # Species and mass flags
    spec_flag: np.ndarray = field(default_factory=lambda: np.ones(C.NSPEC - 1, dtype=bool))
    mass_flag: np.ndarray = field(default_factory=lambda: np.ones(C.NSPEC - 1, dtype=bool))
    mass_wgt: np.ndarray = field(default_factory=lambda: np.zeros(C.NSPEC - 1))
    zalt_flag: bool = True
    n2r_flag: bool = False


def _compute_eta(nodes: np.ndarray, nd: int, kmax: int) -> np.ndarray:
    """Compute reciprocal node difference arrays for B-spline calculations."""
    eta = np.zeros((31, 5))
    nl = nd - C.P_ORDER
    for k in range(2, kmax + 1):
        for j in range(nd - k + 2):
            diff = nodes[j + k - 1] - nodes[j]
            if diff != 0:
                eta[j, k - 2] = 1.0 / diff
    return eta


def _init_parm_space() -> tuple[ModelParameters, int]:
    """Initialize and allocate model parameter space."""
    subsets = {
        'TN': (0, C.NL),
        'PR': (0, C.NL),
        'N2': (0, C.NLS),
        'O2': (0, C.NLS),
        'O1': (0, C.NLS + C.NSPLO1),
        'HE': (0, C.NLS),
        'H1': (0, C.NLS),
        'AR': (0, C.NLS),
        'N1': (0, C.NLS),
        'OA': (0, C.NLS),
        'NO': (0, C.NLS + C.NSPLNO),
    }

    nvertparm = 0
    created = {}
    for name, (bl, nl) in subsets.items():
        created[name] = BasisSubset.create(name, bl, nl)
        if name != 'PR':
            nvertparm += nl - bl + 1
    nvertparm += 1  # Surface pressure parameter

    params = ModelParameters(**created)

    # Compute reciprocal node difference arrays
    params.eta_tn = _compute_eta(C.NODES_TN, C.ND + 2, 6)
    params.eta_o1 = _compute_eta(C.NODES_O1, C.NDO1, 4)
    params.eta_no = _compute_eta(C.NODES_NO, C.NDNO, 4)

    # Solar flux modulation flags
    params.zsfx[:] = False
    params.tsfx[:] = False
    params.psfx[:] = False
    # F1: solar flux modulation of zonal mean asymmetric annual terms
    params.zsfx[9:11] = True
    params.zsfx[13:15] = True
    params.zsfx[17:19] = True
    # F2: solar flux modulation of tides
    params.tsfx[C.CTIDE:C.CSPW] = True
    # F3: solar flux modulation of SPW1
    params.psfx[C.CSPW:C.CSPW + 60] = True

    # C1 constraint terms for O and NO
    gammaterm0 = math.tanh((C.ZETAREF_O1 - C.ZETA_GAMMA) * C.H_GAMMA)
    params.hr_fact_o1_ref = 0.5 * (1.0 + gammaterm0)
    params.dhr_fact_o1_ref = (1.0 - (C.ZETAREF_O1 - C.ZETA_GAMMA) * (1.0 - gammaterm0) * C.H_GAMMA) / params.hr_fact_o1_ref

    gammaterm0 = math.tanh((C.ZETAREF_NO - C.ZETA_GAMMA) * C.H_GAMMA)
    params.hr_fact_no_ref = 0.5 * (1.0 + gammaterm0)
    params.dhr_fact_no_ref = (1.0 - (C.ZETAREF_NO - C.ZETA_GAMMA) * (1.0 - gammaterm0) * C.H_GAMMA) / params.hr_fact_no_ref

    return params, nvertparm


def _load_parm_file(params: ModelParameters, nvertparm: int, filepath: str | Path) -> None:
    """Load binary parameter file into model parameter structures."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MSIS parameter file {filepath} not found.")

    # Read binary file: little-endian double precision
    raw = np.fromfile(filepath, dtype='<f8')
    parmin = raw.reshape((nvertparm, C.MAXNBF)).T  # Shape: (512, nvertparm)

    # Transfer parameters to structures
    i0 = 0
    for subset in [params.TN, None, params.N2, params.O2, params.O1,
                   params.HE, params.H1, params.AR, params.N1, params.OA, params.NO]:
        if subset is None:
            # PR gets one column
            i1 = i0
            params.PR.beta[:, 0] = parmin[:, i0]
            i0 = i1 + 1
            continue
        ncols = subset.nl - subset.bl + 1
        i1 = i0 + ncols
        subset.beta[:, :] = parmin[:, i0:i1]
        i0 = i1

    # Set solar flux modulation flags
    params.smod[:] = False
    for ix in range(C.NL + 1):
        if (params.TN.beta[C.CSFXMOD, ix] != 0 or
                params.TN.beta[C.CSFXMOD + 1, ix] != 0 or
                params.TN.beta[C.CSFXMOD + 2, ix] != 0):
            params.smod[ix] = True

    # Compute log pressure spline coefficients
    _pressparm(params)


def _pressparm(params: ModelParameters) -> None:
    """Compute log pressure spline coefficients from temperature coefficients."""
    for j in range(C.MBF + 1):
        lnz = 0.0
        for b in range(4):
            lnz += params.TN.beta[j, b] * C.GWHT[b] * C.MBARG0DIVKB
        params.PR.beta[j, 1] = -lnz
        for iz in range(1, C.IZFMX + 1):
            lnz = 0.0
            for b in range(4):
                lnz += params.TN.beta[j, iz + b] * C.GWHT[b] * C.MBARG0DIVKB
            params.PR.beta[j, iz + 1] = params.PR.beta[j, iz] - lnz


def tselec(params: ModelParameters, sv: np.ndarray) -> None:
    """Map 25 legacy NRLMSISE-00 switches to 512 modern switches."""
    swleg = np.zeros(26, dtype=np.float32)
    swc = np.zeros(26, dtype=np.float32)

    for i in range(1, 26):
        swleg[i] = sv[i - 1] % 2.0
        if abs(sv[i - 1]) == 1.0 or abs(sv[i - 1]) == 2.0:
            swc[i] = 1.0
        else:
            swc[i] = 0.0

    swg = params.swg

    # Main effects
    swg[0] = True
    swg[C.CSFX:C.CSFX + C.NSFX] = (swleg[1] == 1.0)
    swg[310] = (swleg[1] == 1.0)
    swg[1:7] = (swleg[2] == 1.0)
    swg[304:306] = (swleg[2] == 1.0)
    swg[311:313] = (swleg[2] == 1.0)
    swg[313:315] = (swleg[2] == 1.0)
    for idx in [7, 8, 11, 12, 15, 16, 19, 20]:
        swg[idx] = (swleg[3] == 1.0)
    swg[306:308] = (swleg[3] == 1.0)
    for idx in [21, 22, 25, 26, 29, 30, 33, 34]:
        swg[idx] = (swleg[4] == 1.0)
    swg[308:310] = (swleg[4] == 1.0)
    for idx in [9, 10, 13, 14, 17, 18]:
        swg[idx] = (swleg[5] == 1.0)
    for idx in [23, 24, 27, 28, 31, 32]:
        swg[idx] = (swleg[6] == 1.0)
    swg[35:95] = (swleg[7] == 1.0)
    swg[300:304] = (swleg[7] == 1.0)
    swg[95:145] = (swleg[8] == 1.0)
    swg[145:185] = (swleg[14] == 1.0)

    # Geomagnetic activity
    swg[C.CMAG:C.CMAG + 2] = False
    if swleg[9] > 0 or swleg[13] == 1:
        swg[C.CMAG:C.CMAG + 2] = True
    if swleg[9] < 0:
        swg[C.CMAG] = False
        swg[C.CMAG + 1] = True
    swg[C.CMAG + 2:C.CMAG + 13] = (swleg[9] == 1.0)
    swg[C.CMAG + 28:C.CMAG + 41] = (swleg[9] == -1.0)
    swg[C.CSPW:C.CSFX] = (swleg[11] == 1.0) and (swleg[10] == 1.0)
    swg[C.CUT:C.CUT + C.NUT] = (swleg[12] == 1.0) and (swleg[10] == 1.0)
    swg[C.CMAG + 13:C.CMAG + 26] = (swleg[13] == 1.0) and (swleg[10] == 1.0)
    swg[C.CMAG + 41:C.CMAG + 54] = (swleg[13] == 1.0) and (swleg[10] == 1.0)

    # Cross terms
    swg[C.CSFXMOD:C.CSFXMOD + C.NSFXMOD] = (swc[1] == 1.0)
    if swc[1] == 0:
        for idx in [302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 447, 454]:
            swg[idx] = False
    if swc[2] == 0:
        swg[9:21] = False
        swg[23:35] = False
        swg[35:185] = False
        swg[185:295] = False
        swg[392:415] = False
        swg[420:443] = False
        swg[449:454] = False
    if swc[3] == 0:
        for idx_range in [(201, 205), (209, 213), (217, 221), (255, 259), (263, 267), (271, 275), (306, 308)]:
            swg[idx_range[0]:idx_range[1]] = False
    if swc[4] == 0:
        for idx_range in [(225, 229), (233, 237), (241, 245), (275, 279), (283, 287), (291, 295), (308, 310)]:
            swg[idx_range[0]:idx_range[1]] = False
    if swc[5] == 0:
        for idx_range in [(47, 71), (105, 125), (153, 169), (197, 201), (205, 209),
                          (213, 217), (259, 263), (267, 271), (394, 398), (407, 411),
                          (422, 426), (435, 439)]:
            swg[idx_range[0]:idx_range[1]] = False
        swg[446] = False
    if swc[6] == 0:
        for idx_range in [(221, 225), (229, 233), (237, 241), (279, 283), (287, 291)]:
            swg[idx_range[0]:idx_range[1]] = False
    if swc[7] == 0:
        swg[398:402] = False
        swg[426:430] = False
    if swc[11] == 0:
        swg[402:411] = False
        swg[430:439] = False
        swg[452:454] = False
    if swc[12] == 0:
        swg[411:415] = False
        swg[439:441] = False


def load_model(parm_file: str | Path | None = None,
               switch_legacy: np.ndarray | None = None,
               switch_gfn: np.ndarray | None = None,
               zalt_type: bool = True,
               spec_select: np.ndarray | None = None,
               mass_include: np.ndarray | None = None,
               n2_msis00: bool = False) -> ModelParameters:
    """Initialize and load model parameters.

    Args:
        parm_file: Path to binary parameter file. Defaults to bundled msis21.parm.
        switch_legacy: 25-element float array of legacy switches.
        switch_gfn: 512-element boolean array of individual switches.
        zalt_type: True for geodetic altitude, False for geopotential height.
        spec_select: 10-element boolean array for species selection.
        mass_include: 10-element boolean array for mass density inclusion.
        n2_msis00: Flag for NRLMSISE-00 thermospheric N2 variations.

    Returns:
        Initialized ModelParameters.
    """
    params, nvertparm = _init_parm_space()

    # Determine parameter file path
    if parm_file is None:
        parm_file = Path(__file__).parent / 'data' / 'msis21.parm'

    _load_parm_file(params, nvertparm, parm_file)

    # Set switches
    params.swg[:] = True
    if switch_gfn is not None:
        params.swg[:] = switch_gfn
    elif switch_legacy is not None:
        tselec(params, switch_legacy)

    # Altitude type
    params.zalt_flag = zalt_type

    # Species flags
    if spec_select is not None:
        params.spec_flag[:] = spec_select
    else:
        params.spec_flag[:] = True

    if params.spec_flag[0]:  # Mass density requested
        if mass_include is not None:
            params.mass_flag[:] = mass_include
        else:
            params.mass_flag[:] = True
    else:
        params.mass_flag[:] = False

    # Ensure species needed for mass density are calculated
    params.spec_flag[params.mass_flag] = True

    # Compute mass weights
    params.mass_wgt[:] = 0.0
    params.mass_wgt[params.mass_flag] = 1.0
    params.mass_wgt[0] = 0.0  # Mass density slot
    params.mass_wgt *= C.SPECMASS
    params.mass_wgt[9] = 0.0  # NO not in mass density

    # N2 flag
    params.n2r_flag = n2_msis00

    return params
