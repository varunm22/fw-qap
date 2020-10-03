## Overview

This project contains two basic solvers for QAP, one using Frank-Wolfe (sfw) and the other using Triple Operator Splitting (TOS). The main solvers are in solvers.py, though some of the logic to choose an initial X0 and to project the final output onto the set of permutation matrices is done in test\_helpers.py:solve\_qap.

The remainder of the test\_helpers.py file contains various convenience functions to run multiple tests with various modifiers including the ability to run k trials and take the minimum cost found (and sum of elapsed time between all) or the ability to run n trials (either of single tests or of min\_of\_k trials) and output the mean and standard deviation of the achieved costs and elapsed times.

## Testing

The data subdirectory contains 2 collections of tests currently. The neos-guide collection contains 7 toy examples taken from the neos-guide website. The qaplib collection contains 136 tests from the public QAPLIB set of QAP examples. Individual tests can be run by calling any of the testing functions in test\_helpers.py with the collection and specific test name provided as a tuple to the test argument. 

For example, let's say you want to run the Triple Operator Splitting solver on the chr12a test in qaplib. In addition, let's say you want to take the min cost of every 3 runs to consolidate into 1 datapoint, and you want 20 datapoints. Let's also say instead of seeing all 20 datapoints, you want the summary of mean/stdev of cost/time for each. Then, you can call:
```
print(summarize(min_of_k_n_times(("qaplib", "chr12a"), "tos", 3, 20)))
```
