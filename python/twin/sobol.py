"""Variance-based (Sobol) sensitivity analysis: Saltelli sampling, first-order and total-order indices, bootstrap
confidence intervals."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
import operator
import numbers


@dataclass(frozen=True)
class SobolResult:
    """Result of sobol_indices. d is the number of inputs and q the number of outputs.

    first_order, total_order         S_i and ST_i, shape (d,) for a model with one output, (d, q) otherwise
    first_order_ci, total_order_ci   bootstrap intervals, shape (2, d) or (2, d, q): [0] lower bound, [1] upper bound
    mean, variance                   mean and variance of the outputs, shape () or (q,)
    n_evaluations                    number of model evaluations, n (d + 2)
    """
    first_order: np.ndarray      # S_i, structure (d,) or (d, q)
    total_order: np.ndarray      # ST_i, same structure
    first_order_ci: np.ndarray   # (2, d) or (2, d, q): [0] lower bound, [1] upper one
    total_order_ci: np.ndarray
    mean: np.ndarray             # f0, struct () or (q,)
    variance: np.ndarray         # V, same struct
    n_evaluations: int           # n * (d + 2)



def sobol_indices(model, low, high, n, seed=0, n_bootstrap=200, confidence=0.95):
    """Sobol sensitivity indices of a model whose inputs are independent and uniform on a box.

    model        function of an (n, d) array of points, returning the outputs of all of them: shape (n,) or (n, q)
    low, high    (d,) bounds of the box: input i is uniform on [low[i], high[i]]
    n            base sample size, a power of 2 and at least 2; the model is evaluated n (d + 2) times
    seed         seed of the scrambled Sobol sequence and of the bootstrap
    n_bootstrap  number of bootstrap resamples used for the intervals
    confidence   level of the intervals, strictly between 0 and 1

    Two independent blocks A and B of n points come from a scrambled Sobol sequence, and AB_i is A with its column i
    taken from B (Saltelli's scheme). With V the variance of the outputs of A and B together, the first-order index is
        S_i = mean( f(B) (f(AB_i) - f(A)) ) / V          (Saltelli et al., 2010)
    and the total-order index
        ST_i = mean( (f(A) - f(AB_i))^2 ) / (2 V)        (Jansen, 1999).
    The intervals are percentile intervals of n_bootstrap resamples of the n rows.

    Returns a SobolResult. Raises TypeError for arguments of the wrong type and ValueError for invalid values, or when
    the output of the model has a wrong or changing shape, contains NaN or Inf, or is constant in some column.
    """
    if not callable(model):
        raise TypeError("model must be callable.")

    for name, val in (("n", n), ("seed", seed), ("n_bootstrap", n_bootstrap)):
        if isinstance(val, bool):
            raise TypeError(f"{name} cannot be boolean")
        try:
            operator.index(val)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer") from exc

    n = operator.index(n)
    seed = operator.index(seed)
    n_bootstrap = operator.index(n_bootstrap)

    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
        raise TypeError("confidence must be a real number")
    confidence = float(confidence)

    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    if low.ndim != 1 or high.ndim != 1:
        raise ValueError("low and high must be 1-D arrays")
    if low.size == 0 or low.shape != high.shape:
        raise ValueError("low and high must be non-empty and have the same length")
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("low and high must have finite values")
    if np.any(low >= high):
        raise ValueError("every element of low must be smaller than the matching element of high")

    # n must be >= 2 and a power of 2 (n & (n - 1) == 0)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2 and at least 2")

    if seed < 0:
        raise ValueError("seed must be >= 0")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")

    d = low.size
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    samples = sampler.random(n=n)

    A = qmc.scale(samples[:, :d], low, high)
    B = qmc.scale(samples[:, d:], low, high)

    shapes = []

    def evaluate(points):
        out = np.array(model(points.copy()), dtype=float, copy=True)

        if out.ndim not in (1, 2) or out.shape[0] != n or out.size == 0:
            raise ValueError("The output of the model must have 1 or 2 dimensions, first dimension n and size > 0.")

        if not np.isfinite(out).all():
            raise ValueError("The output of the model contains non-finite values (NaN or Inf).")

        if not shapes:
            shapes.append(out.shape)
        elif out.shape != shapes[0]:
            raise ValueError(f"Inconsistent output shape: expected {shapes}, received {out.shape}.")

        return out

    fA = evaluate(A)
    fB = evaluate(B)

    fAB = []
    for i in range(d):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        fAB.append(evaluate(AB_i))

    fAB = np.stack(fAB, axis=0)

    var_check = np.var(np.concatenate([fA, fB], axis=0), axis=0)
    if not np.all(var_check > 0.0):
        raise ValueError("The output of the model is constant in at least one column (zero variance).")

    def compute_indices(fa, fb, fab):
        combined = np.concatenate([fa, fb], axis=0)
        f0 = np.mean(combined, axis=0)
        V = np.var(combined, axis=0, ddof=0)

        a = fa - f0
        b = fb - f0
        c = fab - f0

        S = np.mean(b * (c - a), axis=1) / V
        ST = np.mean((a - c) ** 2, axis=1) / (2.0 * V)
        return f0, V, S, ST

    mean, variance, first_order, total_order = compute_indices(fA, fB, fAB)

    rng = np.random.default_rng(seed)
    boot_S = np.empty((n_bootstrap,) + first_order.shape, dtype=float)
    boot_ST = np.empty((n_bootstrap,) + total_order.shape, dtype=float)

    for b_idx in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        _, _, s_b, st_b = compute_indices(fA[idx], fB[idx], fAB[:, idx])
        boot_S[b_idx] = s_b
        boot_ST[b_idx] = st_b

    alpha_low = 100.0 * (1.0 - confidence) / 2.0
    alpha_high = 100.0 * (1.0 + confidence) / 2.0

    first_order_ci = np.percentile(boot_S, [alpha_low, alpha_high], axis=0)
    total_order_ci = np.percentile(boot_ST, [alpha_low, alpha_high], axis=0)

    return SobolResult(
        first_order=first_order,
        total_order=total_order,
        first_order_ci=first_order_ci,
        total_order_ci=total_order_ci,
        mean=mean,
        variance=variance,
        n_evaluations=n * (d + 2),)
