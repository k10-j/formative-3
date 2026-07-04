"""
main.py

Entry point for Part 1: EM Algorithm.

Run this script to:
1. Load the pooled (unlabeled) Galton heights data (Fathers + Children by
   default -- change PARENT_COL to "mother" to compare mothers instead).
2. Run the naive "split at the global mean" approach and print its result
   (used to justify, in the presentation, why this is NOT a good idea).
3. Run our from-scratch EM algorithm and print the full tracking table
   (Iteration 0, 1, 2, ... until convergence).
4. Classify a coach-given test height live, printing exact posterior
   probabilities P(Child) vs P(Parent/Pro).
5. Save two plots (mixture fit + log-likelihood curve) for the slides.

Usage:
    python main.py                  -> runs full pipeline with a default demo height
    python main.py 70.5             -> runs full pipeline and classifies height 70.5
"""

import sys
import numpy as np

from data_loader import load_heights
from em_algorithm import run_em, naive_mean_split, posterior_probabilities
from plotting import plot_mixture_fit, plot_log_likelihood

CSV_PATH = "../GaltonFamilies.csv"
PARENT_COL = "father"   # change to "mother" to compare Mothers vs Children


def print_tracking_table(history, max_rows=None):
    rows = history if max_rows is None else history[:max_rows]
    header = f'{"Iter":>4} | {"mu1 (Children)":>15} | {"mu2 (Parents)":>15} | {"sigma1^2":>10} | {"sigma2^2":>10} | {"pi1":>7} | {"pi2":>7} | {"Log-Likelihood":>16}'
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f'{row["iteration"]:>4} | '
            f'{row["mu1"]:>15.4f} | '
            f'{row["mu2"]:>15.4f} | '
            f'{row["sigma1_sq"]:>10.4f} | '
            f'{row["sigma2_sq"]:>10.4f} | '
            f'{row["pi1"]:>7.4f} | '
            f'{row["pi2"]:>7.4f} | '
            f'{row["log_likelihood"]:>16.4f}'
        )


def main():
    test_height = float(sys.argv[1]) if len(sys.argv) > 1 else 66.0

    print("=" * 70)
    print(f"Loading data: children vs. {PARENT_COL}s (pooled, treated as unlabeled)")
    print("=" * 70)
    data, true_labels = load_heights(csv_path=CSV_PATH, parent_col=PARENT_COL)
    print(f"Total data points: {len(data)}  |  Global mean height: {np.mean(data):.4f}\n")

    # ---- Step 1: naive mean-split approach (for the "should we?" discussion) ----
    print("=" * 70)
    print("NAIVE APPROACH: Split at the global mean, average each pile")
    print("=" * 70)
    naive = naive_mean_split(data)
    print(f"Global mean               : {naive['global_mean']:.4f}")
    print(f"Pile 1 (<= mean) size/mean: {naive['pile1_size']} / {naive['pile1_mean']:.4f}")
    print(f"Pile 2 (>  mean) size/mean: {naive['pile2_size']} / {naive['pile2_mean']:.4f}")
    print(
        "\n>> Discussion point: this hard split ignores the overlap between the\n"
        ">> two Gaussian populations. Points near the boundary get force-assigned\n"
        ">> to one pile with 100% certainty even when they are almost equally\n"
        ">> likely to belong to either group. EM instead computes a *soft*,\n"
        ">> probabilistic responsibility for every point, which is statistically\n"
        ">> principled and far less sensitive to overlap / outliers.\n"
    )

    # ---- Step 2: run EM ----
    print("=" * 70)
    print("EM ALGORITHM: Tracking Table")
    print("=" * 70)
    history, final_params = run_em(data, max_iter=100, tol=1e-6)
    print(f"Converged after {history[-1]['iteration']} iterations "
          f"(showing first 3 rows for the presentation table):\n")
    print_tracking_table(history, max_rows=3)
    print("\nFull convergence history (all iterations):\n")
    print_tracking_table(history)

    mu1, mu2, sigma1_sq, sigma2_sq, pi1, pi2 = final_params
    print("\nFinal fitted parameters:")
    print(f"  mu1 (Children) = {mu1:.4f}   sigma1^2 = {sigma1_sq:.4f}   pi1 = {pi1:.4f}")
    print(f"  mu2 (Parents)  = {mu2:.4f}   sigma2^2 = {sigma2_sq:.4f}   pi2 = {pi2:.4f}")

    # ---- Step 3: live classification demo ----
    print("\n" + "=" * 70)
    print(f"LIVE CLASSIFICATION DEMO: test height = {test_height}")
    print("=" * 70)
    p_child, p_parent = posterior_probabilities(test_height, final_params)
    print(f"P(Child  | height={test_height}) = {p_child:.6f}")
    print(f"P(Parent/Pro | height={test_height}) = {p_parent:.6f}")
    verdict = "CHILD" if p_child > p_parent else "PARENT/PRO"
    print(f"--> Model classifies this height as: {verdict}")

    # ---- Step 4: plots for slides ----
    print("\n" + "=" * 70)
    print("Saving presentation plots...")
    print("=" * 70)
    plot_mixture_fit(data, final_params, save_path="mixture_fit.png")
    plot_log_likelihood(history, save_path="log_likelihood.png")


if __name__ == "__main__":
    main()
