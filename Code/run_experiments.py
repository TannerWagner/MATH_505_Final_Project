import numpy as np
from integrands import f_A, f_B, f_C, f_D, I_A_exact, I_B_exact, I_C_exact, I_D_exact
from mc_qmc import mc_estimate, sobol_qmc_estimate

dims = [5, 10, 15, 20]
Ns = [2**k for k in range(7, 16)]
R = 30  # MC repetitions

integrands = {
    "A": (f_A, I_A_exact),
    "B": (f_B, I_B_exact),
    "C": (f_C, I_C_exact),
    "D": (f_D, I_D_exact),
}

rng = np.random.default_rng(12345)

results = []

for name, (f, I_exact_fun) in integrands.items():
    for d in dims:
        I_true = I_exact_fun(d)
        for N in Ns:
            # MC: R runs
            mc_errors = []
            mc_times = []
            for r in range(R):
                est, t = mc_estimate(f, d, N, rng=rng)
                mc_errors.append(abs(est - I_true))
                mc_times.append(t)
            mc_errors = np.array(mc_errors)
            mc_times = np.array(mc_times)

            mc_mean_err = mc_errors.mean()
            mc_std_err = mc_errors.std(ddof=1)
            mc_mean_time = mc_times.mean()

            # Sobol QMC: 1 run (can change)
            qmc_est, qmc_time = sobol_qmc_estimate(f, d, N, scramble=False)
            qmc_err = abs(qmc_est - I_true)

            results.append({
                "integrand": name,
                "d": d,
                "N": N,
                "mc_mean_err": mc_mean_err,
                "mc_std_err": mc_std_err,
                "mc_mean_time": mc_mean_time,
                "qmc_err": qmc_err,
                "qmc_time": qmc_time,
            })

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("results_mc_qmc.csv", index=False, float_format="%.6e")
