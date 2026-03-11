"""Utility functions: altitude conversion, B-spline evaluation, dilogarithm."""

import math
import numpy as np

from .constants import PI


def alt2gph(lat: float, alt: float) -> float:
    """Convert geodetic altitude to geopotential height.

    Args:
        lat: Geodetic latitude (degrees).
        alt: Geodetic altitude (km).

    Returns:
        Geopotential height (km).
    """
    deg2rad = 0.017453292519943295

    # WGS84 Defining parameters
    a = 6378.1370e3         # Semi-major axis (m)
    finv = 298.257223563    # Reciprocal of flattening
    w = 7292115e-11         # Angular velocity (rad/s)
    GM = 398600.4418e9      # Gravitational constant × Earth mass (m³/s²)

    # Derived parameters
    asq = a * a
    wsq = w * w
    f = 1.0 / finv
    esq = 2 * f - f * f
    e = math.sqrt(esq)
    Elin = a * e
    Elinsq = Elin * Elin
    epr = e / (1 - f)
    q0 = ((1.0 + 3.0 / (epr * epr)) * math.atan(epr) - 3.0 / epr) / 2.0
    U0 = -GM * math.atan(epr) / Elin - wsq * asq / 3.0
    g0 = 9.80665
    GMdivElin = GM / Elin

    # Centrifugal potential taper parameters
    x0sq = 2e7**2
    Hsq = 1.2e7**2

    # Compute Cartesian and ellipsoidal coordinates
    altm = alt * 1000.0
    sinsqlat = math.sin(lat * deg2rad)**2
    v = a / math.sqrt(1 - esq * sinsqlat)
    xsq = (v + altm)**2 * (1 - sinsqlat)
    zsq = (v * (1 - esq) + altm)**2 * sinsqlat
    rsqminElinsq = xsq + zsq - Elinsq
    usq = rsqminElinsq / 2.0 + math.sqrt(rsqminElinsq**2 / 4.0 + Elinsq * zsq)
    cossqdelta = zsq / usq

    # Gravitational potential
    epru = Elin / math.sqrt(usq)
    atanepru = math.atan(epru)
    q = ((1 + 3.0 / (epru * epru)) * atanepru - 3.0 / epru) / 2.0
    U = -GMdivElin * atanepru - wsq * (asq * q * (cossqdelta - 1 / 3.0) / q0) / 2.0

    # Centrifugal potential with taper
    if xsq <= x0sq:
        Vc = (wsq / 2.0) * xsq
    else:
        Vc = (wsq / 2.0) * (Hsq * math.tanh((xsq - x0sq) / Hsq) + x0sq)
    U = U - Vc

    return (U - U0) / g0 / 1000.0


def gph2alt(theta: float, gph: float) -> float:
    """Convert geopotential height to geodetic altitude via Newton-Raphson.

    Args:
        theta: Geodetic latitude (degrees).
        gph: Geopotential height (km).

    Returns:
        Geodetic altitude (km).
    """
    epsilon = 0.0005
    x = gph
    dx = epsilon + epsilon
    for _ in range(10):
        if abs(dx) <= epsilon:
            break
        y = alt2gph(theta, x)
        dydz = (alt2gph(theta, x + dx) - y) / dx
        dx = (gph - y) / dydz
        x = x + dx
    return x


def bspline(x: float, nodes: np.ndarray, nd: int, kmax: int,
            eta: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute B-splines up to specified order using Cox-de Boor recursion.

    Args:
        x: Location at which splines are evaluated.
        nodes: Spline node locations (0:nd).
        nd: Number of spline nodes minus one.
        kmax: Maximum order (up to 6) of evaluated splines.
        eta: Precomputed reciprocal node differences, shape (N, 5) for orders 2-6.
             eta[j, k-2] = 1/(nodes[j+k-1] - nodes[j]).

    Returns:
        Tuple of (S, i) where:
        - S is shape (6, 5) array. S[offset+5, k-2] gives the spline value
          for offset in {-5,...,0} and order k in {2,...,6}.
          Fortran S(-5:0, 2:6) maps to Python S[0:6, 0:5].
        - i is the index of last nonzero spline.
    """
    # Initialize output
    S = np.zeros((6, 5))
    i = -1

    # Find index of last nonzero spline (binary search)
    if x >= nodes[nd]:
        return S, nd
    if x <= nodes[0]:
        return S, -1

    low, high = 0, nd
    i = (low + high) // 2
    while x < nodes[i] or x >= nodes[i + 1]:
        if x < nodes[i]:
            high = i
        else:
            low = i
        i = (low + high) // 2

    # Linear splines (k=2)
    # S[offset+5, k-2]: offset 0 → row 5, offset -1 → row 4
    S[5, 0] = (x - nodes[i]) * eta[i, 0]  # S(0, 2)
    if i > 0:
        S[4, 0] = 1 - S[5, 0]  # S(-1, 2)
    if i >= nd - 1:
        S[5, 0] = 0.0

    # Quadratic splines (k=3)
    w = np.zeros(5)  # w[offset+4] maps to Fortran w(-4:0)
    w[4] = (x - nodes[i]) * eta[i, 1]  # w(0)
    if i != 0:
        w[3] = (x - nodes[i - 1]) * eta[i - 1, 1]  # w(-1)
    if i < (nd - 2):
        S[5, 1] = w[4] * S[5, 0]  # S(0,3)
    if (i - 1) >= 0 and (i - 1) < (nd - 2):
        S[4, 1] = w[3] * S[4, 0] + (1.0 - w[4]) * S[5, 0]  # S(-1,3)
    if (i - 2) >= 0:
        S[3, 1] = (1.0 - w[3]) * S[4, 0]  # S(-2,3)

    # Cubic splines (k=4)
    for l_off in range(0, -3, -1):  # 0, -1, -2
        j = i + l_off
        if j < 0:
            break
        w[l_off + 4] = (x - nodes[j]) * eta[j, 2]
    if i < (nd - 3):
        S[5, 2] = w[4] * S[5, 1]  # S(0,4)
    for l_off in range(-1, -3, -1):  # -1, -2
        if (i + l_off) >= 0 and (i + l_off) < (nd - 3):
            S[l_off + 5, 2] = w[l_off + 4] * S[l_off + 5, 1] + (1.0 - w[l_off + 5]) * S[l_off + 6, 1]
    if (i - 3) >= 0:
        S[2, 2] = (1.0 - w[2]) * S[3, 1]  # S(-3,4)

    # k=5
    for l_off in range(0, -4, -1):  # 0, -1, -2, -3
        j = i + l_off
        if j < 0:
            break
        w[l_off + 4] = (x - nodes[j]) * eta[j, 3]
    if i < (nd - 4):
        S[5, 3] = w[4] * S[5, 2]
    for l_off in range(-1, -4, -1):  # -1, -2, -3
        if (i + l_off) >= 0 and (i + l_off) < (nd - 4):
            S[l_off + 5, 3] = w[l_off + 4] * S[l_off + 5, 2] + (1.0 - w[l_off + 5]) * S[l_off + 6, 2]
    if (i - 4) >= 0:
        S[1, 3] = (1.0 - w[1]) * S[2, 2]  # S(-4,5)
    if kmax == 5:
        return S, i

    # k=6
    for l_off in range(0, -5, -1):  # 0, -1, -2, -3, -4
        j = i + l_off
        if j < 0:
            break
        w[l_off + 4] = (x - nodes[j]) * eta[j, 4]
    if i < (nd - 5):
        S[5, 4] = w[4] * S[5, 3]
    for l_off in range(-1, -5, -1):  # -1, -2, -3, -4
        if (i + l_off) >= 0 and (i + l_off) < (nd - 5):
            S[l_off + 5, 4] = w[l_off + 4] * S[l_off + 5, 3] + (1.0 - w[l_off + 5]) * S[l_off + 6, 3]
    if (i - 5) >= 0:
        S[0, 4] = (1.0 - w[0]) * S[1, 3]  # S(-5,6)

    return S, i


def dilog(x0: float) -> float:
    """Compute dilogarithm Li₂(x) for domain [0, 1).

    Retains terms up to order 3, relative error < 1E-5.

    Args:
        x0: Input value in [0, 1).

    Returns:
        Dilogarithm value.
    """
    pi2_6 = PI * PI / 6.0

    x = x0
    if x > 0.5:
        lnx = math.log(x)
        x = 1.0 - x
        xx = x * x
        x4 = 4.0 * x
        return pi2_6 - lnx * math.log(x) - (
            4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
            + x4 + 3.0 * (1.0 - xx) * lnx
        ) / (1.0 + x4 + xx)
    else:
        xx = x * x
        x4 = 4.0 * x
        return (
            4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
            + x4 + 3.0 * (1.0 - xx) * math.log(1.0 - x)
        ) / (1.0 + x4 + xx)
