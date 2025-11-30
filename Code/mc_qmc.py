import numpy as np
from time import perf_counter
from scipy.stats import qmc # for Sobol

def mc_estimate(f, d, N, rng=None):
    """
    Plain Monte Carlo estimator for integral over [0,1]^d.
    Returns (estimate, elapsed_time).
    """
    if rng is None:
        rng = np.random.default_rng()
    t0 = perf_counter()

    # Uniform samples in [0,1]^d
    X = rng.random((N, d))
    vals = f(X)
    est = np.mean(vals)
    t1 = perf_counter()

    return est, (t1 - t0)

def sobol_qmc_estimate(f, d, N, scramble=False, seed=None):
    """
    Sobol quasi-Monte Carlo estimator using scipy.stats.qmc.Sobol.

    Assumes N is a power of 2 and uses random_base2(m) so that we get
    a properly balanced Sobol net of size N = 2^m.
    """
    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)

    m = int(np.log2(N))
    assert 2**m == N, "N must be a power of 2 for random_base2"

    t0 = perf_counter()
    X = sampler.random_base2(m)
    vals = f(X)
    est = np.mean(vals)
    t1 = perf_counter()

    return est, (t1 - t0)
