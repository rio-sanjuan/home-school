import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.stats import chi2_contingency, mannwhitneyu, norm, t, ttest_ind


def hypergeom(k, K, n, N):
    """Probability mass function of the hypergeometric distribution"""
    return binom(K, k) * binom(N - K, n - k) / binom(N, n)


def fisher_prob(m):
    """Probability of a given observed contingency table according to Fisher's exact test"""
    ((a, b), (c, d)) = m
    return hypergeom(k=a, K=a + b, n=a + c, N=a + b + c + d)


def fisher_probs_histogram(m):
    """Computes probability mass function histogram according to Fisher's exact test"""
    neg_val = -min(m[0, 0], m[1, 1])
    pos_val = min(m[1, 0], m[0, 1])
    probs = []
    for k in range(neg_val, pos_val + 1):
        m1 = m + np.array([[1, -1], [-1, 1]]) * k
        probs.append(fisher_prob(m1))
    return probs


def run_discrete():
    np.random.seed(42)
    x = np.random.binomial(n=1, p=0.6, size=15)
    y = np.random.binomial(n=1, p=0.4, size=19)

    _, (a, c) = np.unique(x, return_counts=True)
    _, (b, d) = np.unique(y, return_counts=True)

    df = pd.DataFrame(
        data=[[a, b], [c, d]], index=["click", "no click"], columns=["A", "B"]
    )

    m = df.values
    print("- Observations:")
    print(f"  - Version A: = {x}")
    print(f"  - Version B: = {y}")
    print("")
    print("- Contingency table:")
    print(df)


if __name__ == "__main__":
    run_discrete()
