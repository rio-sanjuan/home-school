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
