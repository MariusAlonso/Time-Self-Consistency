import numpy as np
import random
import math


# -------------------------
# Synthetic Dataset
# -------------------------
def generate_sinusoid(length, freq, phase, amp, noise_std):
    t = np.arange(length)
    return amp * np.sin(2 * np.pi * freq * t + phase) + np.random.normal(
        0, noise_std, size=length
    )


def generate_ar(length, coef, noise_std):
    x = np.zeros(length, dtype=np.float32)
    for i in range(1, length):
        x[i] = coef * x[i - 1] + np.random.normal(0, noise_std)
    return x


def make_synthetic_series(num_series: int, length: int) -> np.ndarray:
    """
    Return shape (num_series, length) of float values in [-1,1]
    Each series is random mixture of sinusoids. A random walk is added.
    """
    out = np.zeros((num_series, length), dtype=np.float32)
    for i in range(num_series):
        # sinusoidal dominant
        n_components = random.randint(1, 3)
        s = np.zeros(length, dtype=np.float32)
        for _ in range(n_components):
            freq = random.uniform(0.001, 0.05)
            phase = random.uniform(0, 2 * math.pi)
            amp = random.uniform(0.1, 1.0)
            noise = random.uniform(0.0, 0.1)
            s += generate_sinusoid(length, freq, phase, amp, noise)

        rw = np.cumsum(np.random.normal(0, random.uniform(0.001, 0.05), length))
        s = s + rw

        # normalize to [-1, 1] robustly: subtract mean and scale to max abs
        s = s - np.mean(s)
        maxabs = max(1e-6, np.max(np.abs(s)))
        s = s / maxabs

        # p = np.random.beta(2, 0.5, 1)
        # m = np.repeat(np.random.binomial(1, p, length), int(1 / freq), axis=0)[:length]
        # m_fill = np.random.uniform(-1, 1)
        # s = s * m + m_fill * (1 - m)

        s = np.clip(s, -1.0, 1.0)
        out[i] = s.astype(np.float32)
    return out
