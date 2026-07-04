"""
em_algorithm.py

From-scratch implementation of the Expectation-Maximization (EM) algorithm
for a 2-component 1-D Gaussian Mixture Model (GMM).

No external ML libraries (no sklearn.mixture, etc.) are used for the actual
EM math -- only NumPy for array/vector operations, which is standard practice
for implementing algorithms "from scratch" (we do not call any pre-built
GMM/EM routine).

This module is intentionally kept modular / DRY: each mathematical step of
EM (initialization, E-step, M-step, log-likelihood, classification) is its
own small function that can be tested and reused independently.
"""

import numpy as np


def gaussian_pdf(x, mu, sigma_sq):
    """
    Probability density function of a univariate Normal distribution.

        N(x; mu, sigma^2) = 1/sqrt(2*pi*sigma^2) * exp(-(x-mu)^2 / (2*sigma^2))

    x can be a scalar or a NumPy array (vectorized).
    """
    sigma_sq = max(sigma_sq, 1e-12)  # numerical safety, avoid divide-by-zero
    coeff = 1.0 / np.sqrt(2.0 * np.pi * sigma_sq)
    exponent = -((x - mu) ** 2) / (2.0 * sigma_sq)
    return coeff * np.exp(exponent)


def initialize_parameters(data):
    """
    Initialize the 6 GMM parameters: mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2.

    Strategy: start from the global mean/std of the *whole* (unlabeled)
    dataset, and offset the two means by one standard deviation in each
    direction. Both components start with equal variance (the global
    variance) and equal mixing weight (0.5 / 0.5).

    This is a standard, unbiased way to initialize EM -- it does NOT use
    any label information, and is different from "hard splitting" the data
    at the mean (which is a one-shot deterministic assignment, not a
    probabilistic model).
    """
    global_mean = np.mean(data)
    global_var = np.var(data)
    global_std = np.sqrt(global_var)

    mu1 = global_mean - global_std / 2.0   # lower cluster starting guess
    mu2 = global_mean + global_std / 2.0   # upper cluster starting guess
    sigma1_sq = global_var
    sigma2_sq = global_var
    pi1 = 0.5
    pi2 = 0.5

    return mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2


def e_step(data, params):
    """
    Expectation step.

    For every data point, compute the "responsibility" that each of the two
    Gaussian components takes for having generated it:

        r_i1 = pi1 * N(x_i; mu1, sigma1_sq) / [pi1*N(x_i;mu1,sigma1_sq) + pi2*N(x_i;mu2,sigma2_sq)]
        r_i2 = 1 - r_i1

    These responsibilities are soft (probabilistic) cluster assignments,
    which is the key difference from a hard mean-split.
    """
    mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2 = params

    weighted_p1 = pi1 * gaussian_pdf(data, mu1, sigma1_sq)
    weighted_p2 = pi2 * gaussian_pdf(data, mu2, sigma2_sq)
    total = weighted_p1 + weighted_p2
    total = np.where(total == 0, 1e-12, total)  # numerical safety

    r1 = weighted_p1 / total
    r2 = weighted_p2 / total
    return r1, r2


def m_step(data, r1, r2):
    """
    Maximization step.

    Using the responsibilities from the E-step, recompute the parameters
    that maximize the expected log-likelihood:

        N1 = sum(r1), N2 = sum(r2)
        mu_k      = sum(r_k * x) / N_k
        sigma_k^2 = sum(r_k * (x - mu_k)^2) / N_k
        pi_k      = N_k / N
    """
    n = len(data)
    n1 = np.sum(r1)
    n2 = np.sum(r2)

    mu1 = np.sum(r1 * data) / n1
    mu2 = np.sum(r2 * data) / n2

    sigma1_sq = np.sum(r1 * (data - mu1) ** 2) / n1
    sigma2_sq = np.sum(r2 * (data - mu2) ** 2) / n2

    pi1 = n1 / n
    pi2 = n2 / n

    return mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2


def log_likelihood(data, params):
    """
    Total log-likelihood of the dataset under the current mixture model:

        LL = sum_i log( pi1*N(x_i;mu1,sigma1_sq) + pi2*N(x_i;mu2,sigma2_sq) )

    This is the quantity EM guarantees to never decrease. We use it both to
    populate the tracking table and as the convergence criterion.
    """
    mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2 = params
    mix = pi1 * gaussian_pdf(data, mu1, sigma1_sq) + pi2 * gaussian_pdf(data, mu2, sigma2_sq)
    mix = np.where(mix <= 0, 1e-12, mix)
    return float(np.sum(np.log(mix)))


def run_em(data, max_iter=50, tol=1e-6):
    """
    Run the full EM loop until convergence (or max_iter reached).

    Returns:
        history: list of dicts, one per iteration (including iteration 0,
                 the initialization), each containing all tracking-table
                 columns: iteration, mu1, mu2, sigma1_sq, sigma2_sq, pi1,
                 pi2, log_likelihood.
        final_params: tuple (mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2)
    """
    data = np.asarray(data, dtype=float)

    params = initialize_parameters(data)
    ll = log_likelihood(data, params)

    history = [_record(0, params, ll)]

    converged = False
    prev_ll = ll
    for iteration in range(1, max_iter + 1):
        r1, r2 = e_step(data, params)
        params = m_step(data, r1, r2)
        ll = log_likelihood(data, params)
        history.append(_record(iteration, params, ll))

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    return history, params, converged



def _record(iteration, params, ll):
    mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2 = params
    return {
        "iteration": iteration,
        "mu1": mu1,
        "mu2": mu2,
        "sigma1_sq": sigma1_sq,
        "sigma2_sq": sigma2_sq,
        "pi1": pi1,
        "pi2": pi2,
        "log_likelihood": ll,
    }


def posterior_probabilities(x, params):
    """
    Given a single new height value x and the final fitted parameters,
    compute the posterior probability that x belongs to component 1
    ("Children") vs. component 2 ("Parents / Pros"):

        P(k | x) = pi_k * N(x; mu_k, sigma_k^2) / sum_j pi_j * N(x; mu_j, sigma_j^2)
    """
    mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2 = params
    p1 = pi1 * gaussian_pdf(x, mu1, sigma1_sq)
    p2 = pi2 * gaussian_pdf(x, mu2, sigma2_sq)
    total = p1 + p2
    return p1 / total, p2 / total


def naive_mean_split(data):
    """
    The "naive" approach the assignment asks us to critique: draw a single
    line at the dataset's global mean, split the data into two piles based
    on that hard threshold, and compute the mean of each pile.

    Returns (global_mean, pile1_mean, pile2_mean, pile1_size, pile2_size).
    """
    data = np.asarray(data, dtype=float)
    global_mean = np.mean(data)
    pile1 = data[data <= global_mean]   # "lower" pile
    pile2 = data[data > global_mean]    # "upper" pile
    return {
        "global_mean": global_mean,
        "pile1_mean": np.mean(pile1) if len(pile1) else float("nan"),
        "pile2_mean": np.mean(pile2) if len(pile2) else float("nan"),
        "pile1_size": len(pile1),
        "pile2_size": len(pile2),
    }
