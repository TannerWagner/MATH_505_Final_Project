import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results_mc_qmc.csv")

integrands = ["A", "B", "C", "D"]
dims = [5, 10, 15, 20]

# log-log error vs N and error vs time
for name in integrands:
    for d in dims:
        sub = df[(df["integrand"] == name) & (df["d"] == d)].sort_values("N")

        N = sub["N"].to_numpy()
        mc_mean_err = sub["mc_mean_err"].to_numpy()
        mc_std_err  = sub["mc_std_err"].to_numpy()
        mc_mean_time = sub["mc_mean_time"].to_numpy()
        qmc_err = sub["qmc_err"].to_numpy()
        qmc_time = sub["qmc_time"].to_numpy()

        # 1) Error vs N (log-log)
        plt.figure()
        plt.loglog(N, mc_mean_err, marker="o", linestyle="-", label="MC mean error")

        plt.loglog(N, qmc_err, marker="s", linestyle="-", label="QMC error")

        plt.xlabel("N (number of samples)")
        plt.ylabel("absolute error")
        plt.title(f"Error vs N (integrand {name}, d = {d})")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(f"error_vs_N_integrand_{name}_d{d}.png")
        plt.close()

        # 2) Error vs time
        plt.figure()
        plt.loglog(mc_mean_time, mc_mean_err, marker="o", linestyle="-", label="MC mean error")
        plt.loglog(qmc_time, qmc_err, marker="s", linestyle="-", label="QMC error")

        plt.xlabel("wall-clock time (seconds)")
        plt.ylabel("absolute error")
        plt.title(f"Error vs time (integrand {name}, d = {d})")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(f"error_vs_time_integrand_{name}_d{d}.png")
        plt.close()
