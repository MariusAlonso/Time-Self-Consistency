import numpy as np
import torch
import tqdm

from .metalog import tempered_metalog_from_quantiles, transform


def sample_tempered_metalog(Qs, ridge):
    return tempered_metalog_from_quantiles(
        np.linspace(0.1, 0.9, 9),
        Qs,
        1.0,
        ridge=ridge,
    )[0](np.random.uniform(0, 1, 1))[None, None, :]


def quantile_remap(Qemp, Qtrue)
    A = np.sum(
        (Qemp - np.mean(Qemp, axis=1, keepdims=True))
        * (Qtrue - np.mean(Qtrue, axis=1, keepdims=True)),
        axis=1,
    ) / np.sum((Qemp - np.mean(Qemp, axis=1, keepdims=True)) ** 2, axis=1)

    B = np.mean(Qtrue, axis=1) - A * np.mean(Qemp, axis=1)

    A, B = A[:, None, None], B[:, None, None]

    return A, B

def quantile_trajectory_sampling(
    X_full,
    context_length,
    prediction_length,
    prediction_fn,
    n_trajectories=100,
    frac_quantile_remap=0.0,
):
    """
    Samples time series trajectories based a 0.1 ... 0.9 quantile predictor

    Parameters
    ----------
    X_full : array-like (Nb series x Nb timesteps)
        Array containing univariate time series
    context_length : int
        Length of context for inference
    porediction_length : int
        Length of prediction
    prediction_fn : callable
        Prediction model that returns quanatiles
    n_trajectories : int
        Number of sampled trajectories
    frac_quantile_remap : float
        Fraction of trajectories that are corrected per step to match
        first multistep quantile prediction

    Returns
    -------
    Y : array-like (Nb series x Nb trajectories x Nb timesteps)
        Array containing all sampled trajectories
    Ymt : array-like (Nb series x Nb timesteps)
        Array containing original multistep forecasting
    full_ymt : array-like (Nb timesteps x Nb series x Nb trajectories x Nb timesteps)
        Array containing all multistep forecasted quantiles for every trajectory and every timestep
    """

    C = context_length
    P = prediction_length
    N = n_trajectories

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

    Y = np.repeat(X_full[:, None, :C], N, axis=1)

    full_ymt = []

    for i in tqdm.tqdm(range(P)):

        Y = Y.reshape(1, -1, Y.shape[-1])

        y_mt = prediction_fn(
            torch.Tensor(Y[0]),
        )[None, :, :]
        y = y_mt[0, :, :, 0]

        y_mt = np.concatenate(
            [
                np.repeat(Y[:, :, :, None], 9, axis=3),
                y_mt.swapaxes(2, 3),
            ],
            axis=2,
        )

        full_ymt.append(y_mt[0, :, C : C + P, :].reshape(-1, N, P, 9))

        if i > 0 and frac_quantile_remap:

            Qtrue, Qemp = (
                Ymt[:, C + i, :],
                np.quantile(
                    full_ymt[-1][:, :, i, :].reshape(64, -1)[:, :],
                    np.linspace(0.1, 0.9, 9),
                    axis=1,
                ).T,
            )

            A, B = quantile_remap(Qemp, Qtrue)
            subset = np.random.choice(
                N,
                int(frac_quantile_remap * N),
                replace=False
            )

            y = y.reshape(-1, N, 9)
            y[:, subset, :] = y[:, subset, :] + (
                (A - np.ones_like(A)) * y[:, subset, :] + B
            )
            y = y.reshape(-1, 9)

        sampled = []
        for ys, ymt in zip(y, np.repeat(Ymt[:, C + i], N, axis=0)):
            Qs, itransform = transform(ys)
            ridge_reg = 1e-6

            while True:
                try:
                    s = sample_tempered_metalog(Qs, ridge_reg)
                    break
                except:
                    ridge_reg = 10 * ridge_reg

            sampled.append(itransform(s))

        y = np.concatenate(sampled, axis=0)
        Y = np.concatenate([Y, y[None, :, 0, :]], axis=-1)

        Y = Y.reshape(-1, N, Y.shape[-1])

    return Y, Ymt, full_ymt
