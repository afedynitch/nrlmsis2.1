"""Utility functions: altitude conversion, B-spline evaluation, dilogarithm."""

import math
import numpy as np

from .constants import PI


def alt2gph_vec(lat: float, alt: np.ndarray) -> np.ndarray:
    """Vectorized geodetic altitude to geopotential height conversion.

    Args:
        lat: Geodetic latitude (degrees), scalar.
        alt: Geodetic altitude (km), array.

    Returns:
        Geopotential height (km), same shape as alt.
    """
    deg2rad = 0.017453292519943295

    # WGS84 Defining parameters
    a = 6378.1370e3
    finv = 298.257223563
    w = 7292115e-11
    GM = 398600.4418e9

    # Derived parameters (all scalar — lat is fixed)
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

    x0sq = 2e7**2
    Hsq = 1.2e7**2

    sinsqlat = math.sin(lat * deg2rad)**2
    v = a / math.sqrt(1 - esq * sinsqlat)

    # Altitude-dependent quantities (vectorized)
    altm = np.asarray(alt, dtype=float) * 1000.0
    xsq = (v + altm)**2 * (1 - sinsqlat)
    zsq = (v * (1 - esq) + altm)**2 * sinsqlat
    rsqminElinsq = xsq + zsq - Elinsq
    usq = rsqminElinsq / 2.0 + np.sqrt(rsqminElinsq**2 / 4.0 + Elinsq * zsq)
    cossqdelta = zsq / usq

    epru = Elin / np.sqrt(usq)
    atanepru = np.arctan(epru)
    q = ((1 + 3.0 / (epru * epru)) * atanepru - 3.0 / epru) / 2.0
    U = -GMdivElin * atanepru - wsq * (asq * q * (cossqdelta - 1 / 3.0) / q0) / 2.0

    # Centrifugal potential with taper (vectorized branch)
    Vc = np.where(xsq <= x0sq,
                  (wsq / 2.0) * xsq,
                  (wsq / 2.0) * (Hsq * np.tanh((xsq - x0sq) / Hsq) + x0sq))
    U = U - Vc

    return (U - U0) / g0 / 1000.0


def dilog_vec(x0: np.ndarray) -> np.ndarray:
    """Vectorized dilogarithm Li₂(x) for domain [0, 1).

    Args:
        x0: Input values in [0, 1), array.

    Returns:
        Dilogarithm values, same shape as x0.
    """
    pi2_6 = PI * PI / 6.0
    x0 = np.asarray(x0, dtype=float)
    mask_hi = x0 > 0.5

    # High branch (x > 0.5): transform xlo = 1 - x
    # Guard against log domain errors by clamping
    safe_x0 = np.where(mask_hi, x0, 1.0)            # for log(x0)
    safe_xlo = np.where(mask_hi, 1.0 - x0, 0.5)     # 1-x0 for hi; dummy for lo
    hi_lnx = np.log(safe_x0)
    hi_lnxlo = np.log(np.maximum(safe_xlo, 1e-300))
    hi_xx = safe_xlo * safe_xlo
    hi_x4 = 4.0 * safe_xlo
    hi_val = (pi2_6 - hi_lnx * hi_lnxlo -
              (4.0 * hi_xx * (23.0 / 16.0 + safe_xlo / 36.0
                              + hi_xx / 576.0 + hi_xx * safe_xlo / 3600.0)
               + hi_x4 + 3.0 * (1.0 - hi_xx) * hi_lnx) / (1.0 + hi_x4 + hi_xx))

    # Low branch (x <= 0.5): use x directly
    safe_x_lo = np.where(~mask_hi, x0, 0.0)          # for lo branch
    lo_1mx = np.where(~mask_hi, 1.0 - x0, 1.0)       # guard log(1-x)
    lo_xx = safe_x_lo * safe_x_lo
    lo_x4 = 4.0 * safe_x_lo
    lo_val = ((4.0 * lo_xx * (23.0 / 16.0 + safe_x_lo / 36.0
                               + lo_xx / 576.0 + lo_xx * safe_x_lo / 3600.0)
               + lo_x4 + 3.0 * (1.0 - lo_xx) * np.log(lo_1mx))
              / (1.0 + lo_x4 + lo_xx))

    return np.where(mask_hi, hi_val, lo_val)


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


def bspline_vec(x_arr: np.ndarray, nodes: np.ndarray, nd: int, kmax: int,
                eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Cox-de Boor B-spline evaluation.

    Computes the same per-point output as :func:`bspline` for each entry of
    ``x_arr``, in pure numpy with no Python loop over altitudes.

    Args:
        x_arr: 1D array of evaluation points, shape (N,).
        nodes: Spline node array (same as scalar bspline).
        nd: Number of spline nodes minus one.
        kmax: Maximum spline order (2..6).
        eta: Precomputed reciprocal node differences, shape (nd, 5).

    Returns:
        S_arr: shape (N, 6, 5). S_arr[n, offset+5, k-2] is the order-k B-spline
            weight at offset (-5..0) for point n.
        iz_arr: shape (N,), int. Knot interval index per point;
            -1 if x_arr[n] <= nodes[0]; nd if x_arr[n] >= nodes[nd].
    """
    x = np.asarray(x_arr, dtype=float).ravel()
    N = x.shape[0]
    S = np.zeros((N, 6, 5))

    # Knot interval via searchsorted (side='right' so x in [nodes[i], nodes[i+1])
    # → returns i+1, giving iz=i after the -1).
    iz_raw = np.searchsorted(nodes[:nd + 1], x, side='right') - 1

    # Match scalar's early-return behaviour: x <= nodes[0] → iz=-1, x >= nodes[nd] → iz=nd.
    high = x >= nodes[nd]
    low = x <= nodes[0]
    iz = np.where(high, nd, np.where(low, -1, iz_raw)).astype(np.int64)

    inside = ~(high | low)
    if not inside.any():
        return S, iz

    in_idx = np.flatnonzero(inside)
    iz_in = iz[in_idx]
    x_in = x[in_idx]
    Ni = iz_in.shape[0]

    # w[:, l+4] holds the weight at offset l (in {-4,...,0}), per-altitude.
    w = np.zeros((Ni, 5))
    Sl = np.zeros((Ni, 6, 5))

    def _safe_gather_node(idx):
        """Read nodes[idx] safely; entries with idx<0 are clamped to 0 and the
        caller is responsible for masking them with np.where."""
        return nodes[np.maximum(idx, 0)]

    def _safe_gather_eta(idx, col):
        return eta[np.maximum(idx, 0), col]

    # ----- k=2 (linear) -----
    w0 = (x_in - nodes[iz_in]) * eta[iz_in, 0]
    w[:, 4] = w0
    Sl[:, 5, 0] = np.where(iz_in < (nd - 1), w0, 0.0)          # S(0, 2)
    Sl[:, 4, 0] = np.where(iz_in > 0, 1.0 - w0, 0.0)           # S(-1, 2)

    # ----- k=3 (quadratic) -----
    w[:, 3] = np.where(iz_in > 0,
                       (x_in - _safe_gather_node(iz_in - 1)) * _safe_gather_eta(iz_in - 1, 1),
                       0.0)
    w[:, 4] = (x_in - nodes[iz_in]) * eta[iz_in, 1]
    Sl[:, 5, 1] = np.where(iz_in < (nd - 2),
                           w[:, 4] * Sl[:, 5, 0], 0.0)         # S(0, 3)
    valid = (iz_in > 0) & ((iz_in - 1) < (nd - 2))
    Sl[:, 4, 1] = np.where(valid,
                           w[:, 3] * Sl[:, 4, 0] + (1.0 - w[:, 4]) * Sl[:, 5, 0],
                           0.0)                                # S(-1, 3)
    Sl[:, 3, 1] = np.where(iz_in > 1,
                           (1.0 - w[:, 3]) * Sl[:, 4, 0], 0.0) # S(-2, 3)

    if kmax >= 4:
        # ----- k=4..kmax (generic) -----
        for k in range(4, kmax + 1):
            col_w = k - 2          # eta column index
            col_prev = k - 3       # previous order column in Sl
            col_new = k - 2        # current order column in Sl

            # w[:, l+4] for offsets l in 0, -1, ..., -(k-2)
            for l in range(0, -(k - 1), -1):
                l_idx = l + 4
                w[:, l_idx] = np.where(iz_in + l >= 0,
                                       (x_in - _safe_gather_node(iz_in + l)) *
                                       _safe_gather_eta(iz_in + l, col_w),
                                       0.0)

            # S(0, k)  = w[:, 4] * S(0, k-1);   valid if iz < nd-(k-1)
            S_prev_top = Sl[:, 5, col_prev]
            Sl[:, 5, col_new] = np.where(iz_in < (nd - (k - 1)),
                                          w[:, 4] * S_prev_top, 0.0)

            # Middle: S(m, k) = w[m+4]*S(m,k-1) + (1 - w[m+5])*S(m+1,k-1)
            #   for m in -1, ..., -(k-2)
            for m in range(-1, -(k - 1), -1):
                row = m + 5
                valid = (iz_in + m >= 0) & ((iz_in + m) < (nd - (k - 1)))
                Sl[:, row, col_new] = np.where(
                    valid,
                    w[:, m + 4] * Sl[:, row, col_prev]
                    + (1.0 - w[:, m + 5]) * Sl[:, row + 1, col_prev],
                    0.0,
                )

            # S(-(k-1), k) = (1 - w[6-k]) * S(-(k-2), k-1);   valid if iz-(k-1) >= 0
            m_last = -(k - 1)
            row_last = m_last + 5                # = 6-k
            row_prev_bot = m_last + 6            # = 7-k
            w_idx_for_bot = 6 - k                # = m_last + 5 (clearer this way)
            Sl[:, row_last, col_new] = np.where(
                iz_in + m_last >= 0,
                (1.0 - w[:, w_idx_for_bot]) * Sl[:, row_prev_bot, col_prev],
                0.0,
            )

    # Scatter back to full N-array
    S[in_idx] = Sl
    return S, iz


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
