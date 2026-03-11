"""Physical constants, spline nodes, and precomputed weights for NRLMSIS 2.1."""

import math
import numpy as np

# Missing density value
DMISSING = 9.999e-38

# Trigonometric constants
PI = math.pi
DEG2RAD = PI / 180.0
DOY2RAD = 2.0 * PI / 365.0
LST2RAD = PI / 12.0
TANH1 = math.tanh(1.0)

# Thermodynamic constants
KB = 1.380649e-23         # Boltzmann constant (J/K) (CODATA 2018)
NA = 6.02214076e23        # Avogadro constant (CODATA 2018)
G0 = 9.80665              # Reference gravity (m/s^2)

# Species molecular masses (kg/molecule) — index 0 is mass density dummy
SPECMASS = np.array([
    0.0,                        # 0: Mass density (dummy)
    28.0134,                    # 1: N2
    31.9988,                    # 2: O2
    31.9988 / 2.0,              # 3: O
    4.0,                        # 4: He
    1.0,                        # 5: H
    39.948,                     # 6: Ar
    28.0134 / 2.0,              # 7: N
    31.9988 / 2.0,              # 8: Anomalous O
    (28.0134 + 31.9988) / 2.0,  # 9: NO
]) / (1.0e3 * NA)

# Dry air mean mass (kg/molecule) (CIPM 2007)
MBAR = 28.96546 / (1.0e3 * NA)

# Dry air log volume mixing ratios (CIPM 2007) — index 0 is mass density dummy
LNVMR = np.log(np.array([
    1.0,         # 0: Mass density (dummy)
    0.780848,    # 1: N2
    0.209390,    # 2: O2
    1.0,         # 3: O (dummy)
    0.0000052,   # 4: He
    1.0,         # 5: H (dummy)
    0.009332,    # 6: Ar
    1.0,         # 7: N (dummy)
    1.0,         # 8: Anomalous O (dummy)
    1.0,         # 9: NO (dummy)
]))

# Natural log of global average surface pressure (Pa)
LNP0 = 11.515614

# Derived constants
G0DIVKB = G0 / KB * 1.0e3        # K/(kg km)
MBARG0DIVKB = MBAR * G0 / KB * 1.0e3  # K/km

# Vertical profile parameters
NSPEC = 11    # Number of species including temperature
ND = 27       # Number of temperature profile nodes
P_ORDER = 4   # Spline order
NL = ND - P_ORDER  # Last temperature profile level index
NLS = 9       # Last parameter index for each species (excluding O, NO splines)

BWALT = 122.5    # Reference geopotential height for Bates Profile (km)
ZETA_F = 70.0    # Fully mixed below this (km)
ZETA_B = BWALT   # Bates Profile above this altitude (km)
ZETA_A = 85.0    # Default reference height for active minor species (km)
ZETA_GAMMA = 100.0   # Reference height of tanh taper
H_GAMMA = 1.0 / 30.0  # Inverse scale height of tanh taper

# Nodes for temperature profile splines (0:nd+2)
NODES_TN = np.array([
    -15., -10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50.,
    55., 60., 65., 70., 75., 80., 85., 92.5, 102.5, 112.5, 122.5, 132.5, 142.5,
    152.5, 162.5, 172.5
])

IZFMX = 13   # Fully mixed below this spline index
IZFX = 14    # Spline index at zetaF
IZAX = 17    # Spline index at zetaA
ITEX = NL    # Index of Bates exospheric temperature
ITGB0 = NL - 1  # Index of Bates temperature gradient at lower boundary
ITB0 = NL - 2   # Index of Bates temperature at lower boundary

# O1 Spline parameters
NDO1 = 13
NSPLO1 = NDO1 - 5  # Number of unconstrained spline parameters for O1
NODES_O1 = np.array([
    35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 92.5, 102.5, 112.5
])
ZETAREF_O1 = ZETA_A  # Joining height and reference height for O1 density

# NO Spline parameters
NDNO = 13
NSPLNO = NDNO - 5   # Number of unconstrained spline parameters for NO
NODES_NO = np.array([
    47.5, 55., 62.5, 70., 77.5, 85., 92.5, 100., 107.5, 115., 122.5, 130., 137.5, 145.
])
ZETAREF_NO = ZETA_B  # Joining height and reference height for NO density

# C2 Continuity matrix for temperature (3x3, Fortran column-major → row-major)
C2TN = np.array([
    [1.0, 1.0, 1.0],
    [-10.0, 0.0, 10.0],
    [33.333333333333336, -16.666666666666668, 33.333333333333336],
]).T  # Transpose because Fortran reshape fills column-major

# C1 Continuity for O1 (2x2)
C1O1 = np.array([
    [1.75, -1.624999900076852],
    [-2.916666573405061, 21.458332647194382],
]).T
C1O1ADJ = np.array([0.257142857142857, -0.102857142686844])

# C1 Continuity for NO (2x2)
C1NO = np.array([
    [1.5, 0.0],
    [-3.75, 15.0],
]).T
C1NOADJ = np.array([0.166666666666667, -0.066666666666667])

# Anomalous Oxygen parameters
ZETAREF_OA = ZETA_B
TOA = 4000.0  # Temperature of anomalous oxygen density (K)
HOA = (KB * TOA) / ((16.0 / (1.0e3 * NA)) * G0) * 1.0e-3  # Scale height (km)

# Horizontal and time-dependent basis function parameters
MAXNBF = 512  # Number of basis functions
MAXN = 6      # Maximum latitude (Legendre) spectral degree
MAXL = 3      # Maximum local time (tidal) spectral order
MAXM = 2      # Maximum longitude (stationary planetary wave) order
MAXS = 2      # Maximum day of year (intra-annual) Fourier order
AMAXN = 6
AMAXS = 2
TMAXL = 3
TMAXN = 6
TMAXS = 2
PMAXM = 2
PMAXN = 6
PMAXS = 2
NSFX = 5       # Number of linear solar flux terms
NSFXMOD = 5    # Number of nonlinear modulating solar flux terms
NMAG = 54      # Number of geomagnetic terms
NUT = 12       # Number of UT terms

# Basis function index offsets
CTIMEIND = 0
CINTANN = CTIMEIND + (AMAXN + 1)
CTIDE = CINTANN + ((AMAXN + 1) * 2 * AMAXS)
CSPW = CTIDE + (4 * TMAXS + 2) * (TMAXL * (TMAXN + 1) - (TMAXL * (TMAXL + 1)) // 2)
CSFX = CSPW + (4 * PMAXS + 2) * (PMAXM * (PMAXN + 1) - (PMAXM * (PMAXM + 1)) // 2)
CEXTRA = CSFX + NSFX
MBF = 383      # Last index of linear terms
CNONLIN = MBF + 1
CSFXMOD = CNONLIN
CMAG = CSFXMOD + NSFXMOD
CUT = CMAG + NMAG

# Weights for log pressure spline coefficients from temperature
GWHT = np.array([5.0 / 24.0, 55.0 / 24.0, 55.0 / 24.0, 5.0 / 24.0])

# Constants for analytical integration (hydrostatic piecewise effective mass profile)
WBETA = (NODES_TN[4:ND + 1] - NODES_TN[0:NL + 1]) / 4.0
WGAMMA = (NODES_TN[5:ND + 2] - NODES_TN[0:NL + 1]) / 5.0

# Non-zero bspline values at key altitudes (precomputed)
S5ZETAB = np.array([0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667])
S6ZETAB = np.array([0.008771929824561, 0.216228070175439, 0.550000000000000, 0.216666666666667, 0.008333333333333])

WGHTAXDZ = np.array([-0.102857142857, 0.0495238095238, 0.053333333333])

S4ZETAA = np.array([0.257142857142857, 0.653968253968254, 0.088888888888889])
S5ZETAA = np.array([0.085714285714286, 0.587590187590188, 0.313020313020313, 0.013675213675214])
S6ZETAA = np.array([0.023376623376623, 0.378732378732379, 0.500743700743701, 0.095538448479625, 0.001608848667672])

S4ZETAF = np.array([0.166666666666667, 0.666666666666667, 0.166666666666667])
S5ZETAF = np.array([0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667])

S5ZETA0 = np.array([0.458333333333333, 0.458333333333333, 0.041666666666667])
