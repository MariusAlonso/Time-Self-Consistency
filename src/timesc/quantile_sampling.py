import numpy as np
import torch
import tqdm

from .metalog import tempered_metalog_from_quantiles, transform


def quantile_trajectory_sampling(
    X_full, context_length, prediction_length, prediction_fn
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
