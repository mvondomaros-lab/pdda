from typing import Callable, Tuple

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline
from scipy.linalg import null_space, solve


def propagate_smoluchowski(
        xmin: float,
        xmax: float,
        dx: float,
        t: float,
        dt: float,
        fD: Callable[[np.ndarray], np.ndarray],
        fW: Callable[[np.ndarray], np.ndarray],
        pcut: float = 1.0e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute propagator using an implicit scheme for the Smoluchowski equation.

    Parameters
    ----------
    xmin
        Minimum boundary of the domain.
    xmax
        Maximum boundary of the domain.
    dx
        Spatial step size.
    t
        Total propagation time.
    dt
        Time step size.
    fD
        Function returning diffusion coefficient D(x).
    fW
        Function returning potential W(x).
    pcut
        Threshold for pruning small probability values in output. Default is 1e-4.

    Returns
    -------
    x_masked
        Grid points where probability exceeds pcut.
    p_masked
        Probability density values at x_masked.
    """
    K = int(np.ceil((xmax - xmin) / dx))
    N = int(np.ceil(t / dt)) + 1

    xouter = np.linspace(xmin, xmax, K + 1)
    xinner = xouter[1:-1]
    xshifted = xmin + (np.arange(K) + 0.5) * dx

    p0 = np.zeros_like(xshifted)
    p0[K // 2] = 1.0 / dx

    D = fD(xinner)
    W = fW(xinner)
    F = -np.gradient(W, xinner)

    D *= dt / dx ** 2
    F *= 0.5 * dx
    d, dl, du = _fA(F, D)

    p = p0.copy()
    for n in range(N):
        scipy.linalg.lapack.dgtsv(dl=dl, d=d, du=du, b=p, overwrite_b=1)

    mask = p > pcut

    return xshifted[mask], p[mask]


def _fA(F, D):
    """
    Internal helper to construct tridiagonal matrix components for the Smoluchowski equation.

    Parameters
    ----------
    F
        Advection-like terms at grid points.
    D
        Diffusion coefficients scaled by dt/dx^2.

    Returns
    -------
    d
        Main diagonal of the matrix.
    dl
        Lower diagonal of the matrix.
    du
        Upper diagonal of the matrix.
    """
    K = F.size + 1
    d = np.zeros(K)
    dl = np.zeros(K - 1)
    du = np.zeros(K - 1)

    d[0] = 1.0 + D[0] + D[0] * F[0]
    for j in range(1, K - 1):
        d[j] = 1.0 + D[j] + D[j] * F[j] + D[j - 1] - D[j - 1] * F[j - 1]
    d[K - 1] = 1.0 + D[K - 2] - D[K - 2] * F[K - 2]

    for j in range(K - 1):
        dl[j] = -(1.0 + F[j]) * D[j]
        du[j] = -(1.0 - F[j]) * D[j]
    return d, dl, du


def smoothing_spline_zero_boundary(
        x: ArrayLike,
        y: ArrayLike,
        lam: float = 1e-2,
        k: int = 3,
        n_internal: int = 25,
        quad_n: int = 50,
) -> BSpline:
    """
    Fit a smoothing spline with zero first and second derivatives at boundaries.

    Parameters
    ----------
    x
        Independent variable data points.
    y
        Dependent variable data points.
    lam
        Smoothing parameter (penalty weight). Default is 1e-2.
    k
        Degree of the spline. Default is 3.
    n_internal
        Number of internal knots. Default is 25.
    quad_n
        Number of quadrature points for penalty matrix integration. Default is 50.

    Returns
    -------
    BSpline
        The fitted smoothing spline.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x0, x1 = x[0], x[-1]

    # 1) clamped knots, interior strictly inside
    t_internal = np.linspace(x0, x1, n_internal + 2)[1:-1]
    knots = np.concatenate((np.repeat(x0, k + 1), t_internal, np.repeat(x1, k + 1)))
    m = len(knots) - (k + 1)

    # 2) design matrix B
    B = np.vstack([BSpline(knots, np.eye(m)[j], k)(x) for j in range(m)]).T

    # 3) penalty matrix P = ∫ B_i'' B_j'' dt
    pts, w = np.polynomial.legendre.leggauss(quad_n)
    tq = 0.5 * (pts + 1) * (x1 - x0) + x0
    wq = w * 0.5 * (x1 - x0)

    # D2[j,ℓ] = B_j''(tq[ℓ])
    D2 = np.vstack(
        [BSpline(knots, np.eye(m)[j], k).derivative(2)(tq) for j in range(m)]
    )  # shape (m, quad_n)

    # weight each column ℓ by wq[ℓ]
    P = D2 @ (D2 * wq).T  # → (m,m)

    # 4) build constraint matrix A for first & second derivs at ends
    A = np.zeros((4, m))
    for i, (pt, d) in enumerate(((x0, 1), (x0, 2), (x1, 1), (x1, 2))):
        A[i] = [BSpline(knots, np.eye(m)[j], k).derivative(d)(pt) for j in range(m)]

    # 5) null-space basis N (size m×(m-4))
    N = null_space(A)

    # 6) solve reduced system
    BN = B @ N  # shape (N, m-4)
    PN = N.T @ P @ N  # shape (m-4, m-4)
    lhs = BN.T @ BN + lam * PN
    rhs = BN.T @ y
    u = solve(lhs, rhs)

    # 7) recover c & return spline
    c = N @ u
    return BSpline(knots, c, k)
