import os
import numpy as np
import pandas as pd
from collections import defaultdict
from test_helpers import *

tests = collection("qaplib")
solvers = ["sfw", "tos"] # include sfw too
k_s = [1] # try more ks, good to see tradeoffs
n = 1

# Columns of output table are:
# collection, test_name, solver, k, n, cost_mean, cost_stdev, time_mean, time_stdev

results = defaultdict(list)
for test in tests:
    collection, test_name = test
    for solver in solvers:
        for k in k_s:
            print(collection, test_name, solver)
            result = summarize(min_of_k_n_times((collection, test_name), solver, k, n))
            results["collection"].append(collection)
            results["test_name"].append(test_name)
            results["solver"].append(solver)
            results["k"].append(k)
            results["n"].append(n)
            results["cost_mean"].append(result["cost"]["mean"])
            results["cost_stdev"].append(result["cost"]["stdev"])
            results["time_mean"].append(result["time"]["mean"])
            results["time_stdev"].append(result["time"]["stdev"])

results_df = pd.DataFrame(results)
results_df.to_csv("results/qaplib_k_1.csv", index=False)
