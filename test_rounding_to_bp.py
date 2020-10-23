import os
import numpy as np
import pandas as pd
from collections import defaultdict
from test_helpers import *

tests = collection("qaplib")
solvers = ["sfw", "tos-bp-project"]
k_s = [1]
n = 10

columns = [
    "collection", "test_name", "solver", "k", "n", "cost_pre_rounded", "cost_best_known",
    "cost_relative_gap", "time_mean", "time_stdev", "iters_best_mean", "iters_min_of_k_mean",
    "iters_mean_of_k_mean", "max_iters_reached(1e5)", "birkhoff_infeasibility"
]

results = pd.DataFrame(columns=columns)
for test_idx in range(len(tests)):
    test = tests[test_idx]
    collection, test_name = test
    print(collection, test_name, f"({test_idx+1} / {len(tests)})")
    for solver in solvers:
        for k in k_s:
            result = summarize(min_of_k_n_times((collection, test_name), solver, k, n))
            results = results.append({
                "collection": collection,
                "test_name": test_name,
                "solver": solver,
                "k": k,
                "n": n,

                "cost_pre_rounded": result["cost_pre_rounded"],
                "cost_best_known": result["cost_best_known"],
                "cost_relative_gap":
                  (result["cost"]["mean"]-result["cost_best_known"])/
                  max(result["cost_best_known"],np.spacing(1)),

                "time_mean": result["time"]["mean"],
                "time_stdev": result["time"]["stdev"],

                "iters_best_mean": result["iters"]["best_mean"],
                "iters_min_of_k_mean": result["iters"]["min_of_k_mean"],
                "iters_mean_of_k_mean": result["iters"]["mean_of_k_mean"],
                "max_iters_reached(1e5)": result["iters"]["max_reached(1e5)"],

                "birkhoff_infeasibility": result["birkhoff_infeasibility"]
            }, ignore_index=True)

results.to_csv("results/qaplib_on_bp.csv", index=False)
