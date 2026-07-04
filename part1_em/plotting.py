"""
plotting.py

Presentation-ready plots for Part 1 (EM Algorithm):

1. Histogram of the pooled (unlabeled) height data with the two fitted
   Gaussian components overlaid, so the audience can visually see the
   "mixture" the model recovered.
2. Log-likelihood vs. iteration curve, showing convergence.

Uses only Matplotlib (visualization), not any ML library.
"""

import numpy as np
import matplotlib.pyplot as plt

from em_algorithm import gaussian_pdf


def plot_mixture_fit(data, params, save_path="mixture_fit.png"):
    mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2 = params

    x = np.linspace(data.min() - 2, data.max() + 2, 500)
    pdf1 = pi1 * gaussian_pdf(x, mu1, sigma1_sq)
    pdf2 = pi2 * gaussian_pdf(x, mu2, sigma2_sq)

    plt.figure(figsize=(9, 5))
    plt.hist(data, bins=40, density=True, alpha=0.4, color="gray", label="Pooled height data")
    plt.plot(x, pdf1, lw=2, color="tab:blue", label=f"Component 1 (Children): μ={mu1:.2f}, σ²={sigma1_sq:.2f}")
    plt.plot(x, pdf2, lw=2, color="tab:red", label=f"Component 2 (Parents): μ={mu2:.2f}, σ²={sigma2_sq:.2f}")
    plt.plot(x, pdf1 + pdf2, lw=2, ls="--", color="black", label="Mixture (sum)")
    plt.title("EM-Fitted Gaussian Mixture on Height Data")
    plt.xlabel("Height (inches)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_log_likelihood(history, save_path="log_likelihood.png"):
    iterations = [row["iteration"] for row in history]
    ll = [row["log_likelihood"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, ll, marker="o", color="tab:green")
    plt.title("EM Convergence: Log-Likelihood per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()
