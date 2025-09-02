import numpy as np
from numpy.linalg import lstsq
from scipy.interpolate import interp1d


# --------------------
# Metalog basis + deriv
# --------------------
def _metalog_basis_cols(p, max_k=9):
    """Return list of basis columns evaluated at p (matching Keelin metalog ordering)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    L = np.log(p / (1 - p))
    v = p - 0.5
    cols = [
        np.ones_like(p),  # 1
        L,  # L
        v * L,  # vL
        v,  # v
        v**2,  # v^2
        (v**2) * L,  # v^2 L
        v**3,  # v^3
        (v**3) * L,  # v^3 L
        v**4,  # v^4
    ]
    return cols[:max_k]


def _metalog_basis_derivs(p, max_k=9):
    """Return derivatives wrt p of basis columns (same order as _metalog_basis_cols)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    L = np.log(p / (1 - p))
    v = p - 0.5
    dL = 1.0 / (p * (1 - p))
    dv = np.ones_like(p)
    cols_d = []
    cols_d.append(np.zeros_like(p))  # d(1)/dp = 0
    cols_d.append(dL)  # dL/dp
    cols_d.append(dv * L + v * dL)  # d(vL)/dp = L + v dL
    cols_d.append(dv)  # d(v)/dp = 1
    cols_d.append(2 * v * dv)  # d(v^2)/dp = 2 v
    cols_d.append(2 * v * dv * L + (v**2) * dL)  # d(v^2 L)/dp
    cols_d.append(3 * (v**2) * dv)  # d(v^3)/dp = 3 v^2
    cols_d.append(3 * (v**2) * dv * L + (v**3) * dL)  # d(v^3 L)/dp
    cols_d.append(4 * (v**3) * dv)  # d(v^4)/dp = 4 v^3
    return cols_d[:max_k]


# --------------------
# Main builder
# --------------------
def tempered_metalog_from_quantiles(
    p_grid, x_grid, T, k=7, ridge=1e-8, n_integration=10001, eps=1e-12
):
    """
    Fit a Metalog from (p_grid, x_grid) and return tempered inverse CDF q_T(u).

    Parameters
    ----------
    p_grid : array-like
        Strictly inside (0,1), e.g. [0.1,0.2,...,0.9]. Must be sorted ascending.
    x_grid : array-like
        Corresponding quantile values.
    T : float > 0
        Temperature. T==1 returns the fitted metalog inverse CDF (no tempering).
        T < 1 sharpens peaks; T > 1 flattens distribution.
    k : int
        Number of Metalog terms to use (2..9). Default 7.
    ridge : float
        L2 ridge regularization on least squares (helpful for stability).
    n_integration : int
        Number of grid points used for numeric integration/inversion on p in (eps, 1-eps).
        Must be odd >= 201 for decent resolution. Default 10001.
    eps : float
        Small clipping to avoid log(0). Default 1e-12.

    Returns
    -------
    q_T : callable
        Function mapping u in [0,1] (array or scalar) to tempered samples x.
    extras : dict
        Diagnostic objects: {'a': coefficients, 'M': M, 'dM_dp': dM_dp,
                             'p_grid_integration': p_grid_int, 'H_vals': H_vals, 'Z': Z}
    """
    # --- basic checks & arrays
    p_grid = np.asarray(p_grid, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if p_grid.ndim != 1 or x_grid.ndim != 1 or p_grid.shape[0] != x_grid.shape[0]:
        raise ValueError("p_grid and x_grid must be 1-D arrays of same length.")
    if np.any(p_grid <= 0) or np.any(p_grid >= 1):
        raise ValueError("p_grid values must lie strictly in (0,1).")
    if not np.all(np.diff(p_grid) > 0):
        raise ValueError("p_grid must be strictly increasing.")
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")

    # --- fit metalog coefficients a via least squares
    # build design matrix G (n x k)
    basis_cols = _metalog_basis_cols(p_grid, max_k=k)
    G = np.vstack(basis_cols).T  # shape (n, k)

    if ridge > 0:
        GTG = G.T @ G
        GTx = G.T @ x_grid
        a = np.linalg.solve(GTG + ridge * np.eye(k), GTx)
    else:
        a, *_ = lstsq(G, x_grid, rcond=None)

    # metalog quantile & derivative
    def M(p):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, eps, 1 - eps)
        cols = _metalog_basis_cols(p, max_k=k)
        return sum(ai * ci for ai, ci in zip(a, cols))

    def dM_dp(p):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, eps, 1 - eps)
        cols_d = _metalog_basis_derivs(p, max_k=k)
        return sum(ai * ci for ai, ci in zip(a, cols_d))

    # If T == 1, no tempering: return original quantile function (vectorized)
    if np.isclose(T, 1.0):

        def qT_identity(u):
            u = np.asarray(u, dtype=float)
            u = np.clip(u, 0.0, 1.0)
            return M(u)

        extras = {"a": a, "M": M, "dM_dp": dM_dp}
        return qT_identity, extras

    # Quick monotonicity check on coarse grid
    p_check = np.linspace(eps, 1 - eps, 201)
    qprime_check = dM_dp(p_check)
    if np.any(qprime_check <= 0):
        raise RuntimeError(
            "Fitted Metalog is not strictly increasing (q'(p)<=0). "
            "Reduce k or increase ridge to enforce monotonicity."
        )

    # --- build integration grid in p-space (avoid endpoints)
    if n_integration < 201:
        n_integration = 201
    # ensure odd so symmetrical but not necessary; create interior grid
    p_int = np.linspace(eps, 1.0 - eps, n_integration)

    # evaluate q'(p) on grid and compute w(p) = q'(p)^(1 - 1/T)
    qprime = dM_dp(p_int)
    if np.any(qprime <= 0):
        raise RuntimeError(
            "q'(p) <= 0 on integration grid; metalog not monotone enough."
        )

    exponent = 1.0 - 1.0 / T
    # protect from under/overflow by clipping qprime away from 0
    qprime_clipped = np.clip(qprime, 1e-300, 1e300)
    w = np.power(qprime_clipped, exponent)

    # cumulative integral H(p) via trapezoid rule
    dp = p_int[1] - p_int[0]
    # cumulative trapezoid:
    # H_vals[i] approximates integral from p_int[0] to p_int[i]
    H_vals = np.empty_like(p_int)
    H_vals[0] = 0.0
    H_vals[1:] = np.cumsum(0.5 * (w[:-1] + w[1:]) * dp)

    Z = H_vals[-1]
    if Z <= 0 or not np.isfinite(Z):
        raise RuntimeError("Normalization Z invalid; check q'(p) and T.")

    # normalized s(p) = H(p)/Z  (monotone from 0..1)
    s_vals = H_vals / Z
    # numerical monotone correction in case of tiny non-monotonicity
    s_vals = np.maximum.accumulate(s_vals)

    # build p(s) interpolator: map s in [0,1] -> p in [p_int[0], p_int[-1]]
    # ensure s strictly within [0,1]
    # To make interp1d happy, ensure s_vals is strictly increasing by small jitter if needed
    # but maximum_accumulate above helps; still safe-guard:
    s_eps = 1e-15
    if np.any(np.diff(s_vals) <= 0):
        # add tiny increasing jitter
        jitter = np.linspace(0, 1e-12, len(s_vals))
        s_vals = s_vals + jitter
        s_vals = np.maximum.accumulate(s_vals)
    p_of_s = interp1d(
        s_vals,
        p_int,
        bounds_error=False,
        fill_value=(p_int[0], p_int[-1]),
        assume_sorted=True,
    )

    # final tempered quantile: u in [0,1] -> p = p_of_s(u) -> x = M(p)
    def q_T(u):
        u = np.asarray(u, dtype=float)
        # flatten then restore shape
        orig_shape = u.shape
        u_flat = u.ravel()
        # clip u safely to [0,1]
        u_flat = np.clip(u_flat, 0.0, 1.0)
        p_mapped = p_of_s(u_flat)
        x_vals = M(p_mapped)
        return x_vals.reshape(orig_shape)

    extras = {
        "a": a,
        "M": M,
        "dM_dp": dM_dp,
        "p_integration_grid": p_int,
        "H_vals": H_vals,
        "Z": Z,
        "s_vals": s_vals,
    }
    return q_T, extras


def transform(x):
    x_red = (x - x.mean()) / x.std()
    c = 2
    return np.sinh(c * x_red), lambda y: np.arcsinh(y) * x.std() / c + x.mean()


import torch
import tqdm


def quantile_trajectory_sampling(X_full, C, P, prediction_fn):

    F = 0
    Ymt = np.concatenate(
        [
            np.repeat(X_full[:, : C + F, None], 9, axis=2),
            prediction_fn(
                torch.Tensor(X_full[:, : C + F]),
            ).swapaxes(
                1, 2
            )[:, : P - F],
        ],
        axis=1,
    )

    Y = np.repeat(X_full[:, None, :C], 100, axis=1)
    # Y[:,:,-1] += np.random.normal(0,0.01, Y.shape[:2])
    W = np.zeros(Y.shape[:2] + (P,), dtype=int)

    T = 1.0
    preds = []
    full_ymt = []
    for i in tqdm.tqdm(range(P)):
        # if i < F:
        #     Y = np.concatenate([
        #         Y,
        #         np.repeat(X_full[:, None, C+i:C+i+1], 100, axis=1)
        #     ], axis=-1)
        #     continue

        Y = Y.reshape(1, -1, Y.shape[-1])

        y_mt = prediction_fn(
            torch.Tensor(Y[0]),
        )[None, :, :]
        y = y_mt[0, :, :, 0]

        preds.append(y.reshape(-1, 100, 9))

        y_mt = np.concatenate(
            [
                np.repeat(Y[:, :, :, None], 9, axis=3),
                y_mt.swapaxes(2, 3),
            ],
            axis=2,
        )

        full_ymt.append(y_mt[0, :, C : C + P, :].reshape(-1, 100, P, 9))

        if i > 0:

            subset = np.random.choice(100, 20, replace=False)
            subset_q = np.random.choice(9 * 100, 9 * 20, replace=False)

            Qtrue, Qemp = (
                # Ymt[:, C + i, :],
                np.quantile(
                    np.array(full_ymt[0:-1])[:, :, :, i, :]
                    .swapaxes(0, 1)
                    .reshape(64, -1),
                    np.linspace(0.1, 0.9, 9),
                    axis=1,
                ).T,
                np.quantile(
                    full_ymt[-1][:, :, i, :].reshape(64, -1)[:, :],
                    np.linspace(0.1, 0.9, 9),
                    axis=1,
                ).T,
            )

            A = np.sum(
                (Qemp - np.mean(Qemp, axis=1, keepdims=True))
                * (Qtrue - np.mean(Qtrue, axis=1, keepdims=True)),
                axis=1,
            ) / np.sum((Qemp - np.mean(Qemp, axis=1, keepdims=True)) ** 2, axis=1)

            B = np.mean(Qtrue, axis=1) - A * np.mean(Qemp, axis=1)

            A, B = A[:, None, None], B[:, None, None]

            y = y.reshape(-1, 100, 9)
            # y = y + 0.01 ** ((i + 1) / P) * ((A - np.ones_like(A)) * y + B)
            # y += Score_t.T.mean(axis=1) # Score_t[:, None, :]
            # y[:, subset, :] = y[:, subset, :] + (
            #     (A - np.ones_like(A)) * y[:, subset, :] + B
            # )
            y = y.reshape(-1, 9)

        sampled = []
        for ys, ymt in zip(y, np.repeat(Ymt[:, C + i], 100, axis=0)):
            Qs, itransform = transform(ys)
            ridge_reg = 1e-7
            try:
                s = tempered_metalog_from_quantiles(
                    np.linspace(0.1, 0.9, 9),
                    Qs,
                    T,
                    n_integration=10001,
                    ridge=ridge_reg,
                )[0](np.random.uniform(0, 1, 1))[None, None, :]
            except:
                ridge_reg = 1e-1
                try:
                    s = tempered_metalog_from_quantiles(
                        np.linspace(0.1, 0.9, 9),
                        Qs,
                        T,
                        n_integration=10001,
                        ridge=ridge_reg,
                    )[0](np.random.uniform(0, 1, 1))[None, None, :]
                except:
                    ridge_reg = 1e0
                    try:
                        s = tempered_metalog_from_quantiles(
                            np.linspace(0.1, 0.9, 9),
                            Qs,
                            T,
                            n_integration=10001,
                            ridge=ridge_reg,
                        )[0](np.random.uniform(0, 1, 1))[None, None, :]
                    except:
                        ridge_reg = 1e6
                        s = tempered_metalog_from_quantiles(
                            np.linspace(0.1, 0.9, 9),
                            Qs,
                            T,
                            n_integration=10001,
                            ridge=ridge_reg,
                        )[0](np.random.uniform(0, 1, 1))[None, None, :]

            sampled.append(itransform(s))

        y = np.concatenate(sampled, axis=0)
        Y = np.concatenate([Y, y[None, :, 0, :]], axis=-1)

        Y = Y.reshape(-1, 100, Y.shape[-1])

        # for j in range(len(Y)):
        #     bins = np.concatenate([Ymt[j, C + i], np.array([np.inf])])
        #     for k in range(1, len(bins)):
        #         if bins[k - 1] > bins[k]:
        #             bins[k] = bins[k - 1]

        #     digitize = np.digitize(
        #         Y[j, :, C + i],
        #         bins=bins,
        #     )
        #     # bc = np.bincount(digitize) / 100
        #     # W[j, :, i] = np.log(1 / bc)[digitize]

        #     W[j, :, i] = digitize

        # Wbc = np.array(
        #     [
        #         [np.log(1 / np.bincount(wi).astype(float))[wi] for wi in w.T]
        #         for w in W[:, :, : i + 1]
        #     ]
        # )

        # Wbc =

        # Wi = torch.softmax(
        #     torch.Tensor(Wbc.mean(axis=1)) * 0.2,
        #     dim=-1,
        # ).numpy()

        # # Wi = torch.softmax(torch.Tensor(W.mean(axis=2)), dim=-1).numpy()

        # print(np.sort(Wi, axis=1)[:,::-1])

        # Wi = np.

        # S = np.array([np.random.multinomial(1, w, 100).argmax(axis=-1) for w in Wi])

        # Y = np.array([y[s] for y, s in zip(Y, S)])
        # W = np.array([w[s] for w, s in zip(W, S)])

    return Y, Ymt, full_ymt
