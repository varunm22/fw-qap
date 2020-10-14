import os
import numpy as np
import pandas as pd
from collections import defaultdict
from test_helpers import *

tests = collection("qaplib")
solvers = ["sfw", "tos"]
k_s = [1] # try more ks, good to see tradeoffs
n = 1

columns = [
    "collection", "test_name", "solver", "k", "n", "cost_mean", "cost_stdev",
    "cost_best_known", "cost_relative_gap", "cost_pre_rounded",
    "norm_diff_best_prerounded", "norm_diff_best_rounded", "norm_diff_prerounded_rounded",
    "time_mean", "time_stdev", "iters_mean", "iters_min", "iters_max",
    "max_iters_reached(1e5)"
]

# TODO: discuss why sfw is bad at tai15b
# TODO: what is the whole TOS rounding process we want to do?
results = pd.DataFrame(columns=columns)
for test in tests:
    collection, test_name = test
    for solver in solvers:
        for k in k_s:
            print(collection, test_name, solver)
            result = summarize(min_of_k_n_times((collection, test_name), solver, k, n))
            results = results.append({
                "collection": collection,
                "test_name": test_name,
                "solver": solver,
                "k": k,
                "n": n,

                "cost_mean": result["cost"]["mean"],
                "cost_stdev": result["cost"]["stdev"],
                "cost_best_known": result["cost_best_known"],
                "cost_relative_gap":
                  (result["cost"]["mean"]-result["cost_best_known"])/
                  max(result["cost_best_known"],np.spacing(1)),
                "cost_pre_rounded": result["cost_pre_rounded"],

                "norm_diff_best_prerounded": result["norm_diff"]["best_prerounded"],
                "norm_diff_best_rounded": result["norm_diff"]["best_rounded"],
                "norm_diff_prerounded_rounded": result["norm_diff"]["prerounded_rounded"],

                "time_mean": result["time"]["mean"],
                "time_stdev": result["time"]["stdev"],

                "iters_mean": result["iters"]["mean"],
                "iters_min": result["iters"]["min"],
                "iters_max": result["iters"]["max"],
                "max_iters_reached(1e5)": result["iters"]["max"] == 1e5
            }, ignore_index=True)

results.to_csv("results/test.csv", index=False)
