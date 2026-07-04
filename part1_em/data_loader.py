"""
data_loader.py

Loads the Galton Families dataset (GaltonFamilies.csv) and builds the
"pooled, unlabeled" dataset required by the assignment: a single 1-D array
of heights that is secretly a mixture of two populations -- Fathers
("Parents/Pros") and Children -- but which we treat as having NO labels
when we hand it to the EM algorithm.

We also keep the true hidden labels around separately (NOT used by EM) so
that we can sanity-check our clustering results and build a nice
presentation plot.
"""

import csv
import numpy as np


def load_heights(csv_path="../GaltonFamilies.csv", parent_col="father"):
    """
    Read the Galton Families CSV and return:

        pooled_heights : 1-D np.ndarray of all heights (parents + children),
                          this is what gets fed into EM (no labels used).
        true_labels    : 1-D np.ndarray of same length, 0 = "child",
                          1 = "parent" -- kept ONLY for our own validation /
                          plotting, never used by the EM algorithm itself.

    parent_col: "father" (default) or "mother" -- which parent population to
    compare against the children, per the assignment's choice.

    Each family's parent height is only counted ONCE (not once per child),
    since the same father/mother height is repeated on every child row of
    that family in the raw CSV.
    """
    seen_families = set()
    parent_heights = []
    child_heights = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            family_id = row["family"]
            if family_id not in seen_families:
                seen_families.add(family_id)
                parent_heights.append(float(row[parent_col]))
            child_heights.append(float(row["childHeight"]))

    parent_heights = np.array(parent_heights, dtype=float)
    child_heights = np.array(child_heights, dtype=float)

    pooled_heights = np.concatenate([child_heights, parent_heights])
    true_labels = np.concatenate([
        np.zeros(len(child_heights), dtype=int),   # 0 = child
        np.ones(len(parent_heights), dtype=int),   # 1 = parent
    ])

    return pooled_heights, true_labels


if __name__ == "__main__":
    heights, labels = load_heights(csv_path="../GaltonFamilies.csv", parent_col="father")
    print(f"Total pooled data points: {len(heights)}")
    print(f"  Children : {np.sum(labels == 0)}")
    print(f"  Fathers  : {np.sum(labels == 1)}")
    print(f"Global mean height: {np.mean(heights):.3f}")
