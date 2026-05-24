"""
dataset_generator.py
====================
Reproducible synthetic dataset generator for ML Assignment 2.
Seed = 122 (last 3 digits of roll number 23122).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global seed
# ---------------------------------------------------------------------------
SEED = 122
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _add_noise_features(X_informative, n_noise, noise_scale=1.0):
    """Append n_noise random Gaussian features."""
    n_samples = X_informative.shape[0]
    noise = np.random.normal(0, noise_scale, size=(n_samples, n_noise))
    return np.hstack([X_informative, noise])


def _make_labels_binary(y_multi):
    """Convert multi-class labels to binary (0/1) if needed."""
    return (y_multi % 2).astype(int)


# ---------------------------------------------------------------------------
# Base / Mandatory Datasets
# ---------------------------------------------------------------------------

def get_low_noise_dataset():
    """
    Low-noise dataset.
    n=5000, d=20 (15 informative + 5 noise)
    4 well-separated Gaussian blobs for classification.
    """
    n = 5000
    centers = np.array([
        [0, 0, 0, 0, 0],
        [5, 5, 5, 5, 5],
        [10, 0, 10, 0, 10],
        [0, 10, 0, 10, 0]
    ])
    informative_per_cluster = 5
    X_list, y_list = [], []
    for i, c in enumerate(centers):
        size = n // len(centers)
        X_cluster = np.random.normal(loc=c[:informative_per_cluster], scale=0.8, size=(size, informative_per_cluster))
        X_list.append(X_cluster)
        y_list.append(np.full(size, i))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    # Add 10 more informative features (linear combos + noise) to reach 15 informative
    X_extra = np.random.normal(0, 1, size=(n, 10))
    X = np.hstack([X, X_extra])
    # Add 5 noise features
    X = _add_noise_features(X, 5, noise_scale=1.0)
    return X, y


def get_high_noise_dataset():
    """
    High-noise dataset.
    n=3000, d=25 (10 informative + 15 noise)
    Same structure as low-noise but with heavy noise.
    """
    n = 3000
    centers = np.array([
        [0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4],
    ])
    X_list, y_list = [], []
    for i, c in enumerate(centers):
        size = n // len(centers)
        X_cluster = np.random.normal(loc=c[:5], scale=1.2, size=(size, 5))
        X_list.append(X_cluster)
        y_list.append(np.full(size, i))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    # 5 more informative features
    X_extra = np.random.normal(0, 1, size=(n, 5))
    X = np.hstack([X, X_extra])
    # 15 noise features with HIGH variance
    X = _add_noise_features(X, 15, noise_scale=3.0)
    return X, y


def get_high_dimensional_dataset():
    """
    High-dimensional dataset.
    n=5000, d=60 (10 informative + 50 noise)
    Tests curse of dimensionality and overfitting.
    """
    n = 5000
    centers = np.array([
        [0]*5,
        [4]*5,
    ])
    X_list, y_list = [], []
    for i, c in enumerate(centers):
        size = n // len(centers)
        X_cluster = np.random.normal(loc=c, scale=1.0, size=(size, 5))
        X_list.append(X_cluster)
        y_list.append(np.full(size, i))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    # 5 more informative features
    X_extra = np.random.normal(0, 1, size=(n, 5))
    X = np.hstack([X, X_extra])
    # 50 noise features
    X = _add_noise_features(X, 50, noise_scale=1.0)
    return X, y


# ---------------------------------------------------------------------------
# Q1: k-Means Datasets
# ---------------------------------------------------------------------------

def get_kmeans_friendly_dataset():
    """
    (Q1-B-a) k-Means performs extremely well.
    4 spherical, equal-variance, balanced, well-separated clusters.
    d=15 (5 informative positions + 10 extra informative + 5 noise)
    """
    n = 2000
    d_info = 10
    centers = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [2, 8, 2, 8, 2, 8, 2, 8, 2, 8],
        [8, 2, 8, 2, 8, 2, 8, 2, 8, 2],
    ])
    X_list, y_list = [], []
    for i, c in enumerate(centers):
        size = n // len(centers)
        X_cluster = np.random.normal(loc=c, scale=0.5, size=(size, d_info))
        X_list.append(X_cluster)
        y_list.append(np.full(size, i))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    # Add 5 noise features to reach d=15
    X = _add_noise_features(X, 5, noise_scale=0.5)
    return X, y


def get_kmeans_adversarial_dataset():
    """
    (Q1-B-b) k-Means fails despite visually separable clusters.
    Non-spherical clusters: elongated (different covariance) + imbalanced sizes.
    d=15 (5 informative structure + 10 noise)
    """
    n = 2000
    # Core 2D structure: cross-shaped clusters
    c0_2d = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[4, 0], [0, 0.3]],
        size=int(n * 0.5)
    )
    c1_2d = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[0.3, 0], [0, 4]],
        size=int(n * 0.5)
    )
    # Add 3 more dimensions that preserve the same anisotropic structure
    c0_extra = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
        size=c0_2d.shape[0]
    )
    c1_extra = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[0.3, 0, 0], [0, 3, 0], [0, 0, 0.3]],
        size=c1_2d.shape[0]
    )
    c0 = np.hstack([c0_2d, c0_extra])
    c1 = np.hstack([c1_2d, c1_extra])
    X = np.vstack([c0, c1])
    y = np.hstack([np.zeros(c0.shape[0]), np.ones(c1.shape[0])])
    # Add 10 noise features to reach d=15
    X = _add_noise_features(X, 10, noise_scale=0.5)
    return X, y


# ---------------------------------------------------------------------------
# Q2: Gaussian Naive Bayes Datasets
# ---------------------------------------------------------------------------

def get_nb_correlated_dataset():
    """
    (Q2-B) Two features highly correlated, one redundant.
    Plus additional independent informative features.
    Binary classification.
    d=15 (5 structured + 10 noise)
    """
    n = 2000
    # Class-dependent means
    mean_0, mean_1 = -1.5, 1.5
    y = np.random.randint(0, 2, size=n)
    X1 = np.where(y == 0, np.random.normal(mean_0, 1, n), np.random.normal(mean_1, 1, n))
    X2 = X1 + np.random.normal(0, 0.1, n)   # highly correlated with X1
    X3 = X1 + np.random.normal(0, 0.1, n)   # redundant (correlated to X1)
    # 2 additional independent informative features
    X4 = np.where(y == 0, np.random.normal(mean_0, 1, n), np.random.normal(mean_1, 1, n))
    X5 = np.where(y == 0, np.random.normal(mean_0, 1, n), np.random.normal(mean_1, 1, n))
    X = np.column_stack([X1, X2, X3, X4, X5])
    # Add 10 noise features
    X = _add_noise_features(X, 10, noise_scale=1.0)
    return X, y


def get_nb_success_dataset():
    """
    (Q2-C-a) NB performs surprisingly well despite violated independence.
    Features correlated but class separation is along principal direction.
    d=15 (5 structured + 10 noise)
    """
    n = 2000
    y = np.random.randint(0, 2, size=n)
    mean = np.where(y == 0, -2.0, 2.0)
    X1 = np.random.normal(mean, 1.0, n)
    X2 = X1 + np.random.normal(0, 0.5, n)   # correlated
    X3 = X1 + np.random.normal(0, 0.5, n)   # correlated
    # 2 additional independent informative features
    X4 = np.random.normal(mean, 1.0, n)
    X5 = np.random.normal(mean, 1.0, n)
    X = np.column_stack([X1, X2, X3, X4, X5])
    # 10 noise features
    X = _add_noise_features(X, 10, noise_scale=1.0)
    return X, y


def get_nb_failure_dataset():
    """
    (Q2-C-b) NB completely fails.
    XOR-like structure: each feature individually uninformative,
    but together they separate classes perfectly.
    d=15 (5 structured XOR + 10 noise)
    """
    n = 2000
    # 5D parity structure: no single feature is informative
    # 4D parity gives perfect 50/50 balance; 5th feature is also uninformative
    X_core = np.random.uniform(-1, 1, size=(n, 5))
    y = (
        (X_core[:, 0] > 0) ^
        (X_core[:, 1] > 0) ^
        (X_core[:, 2] > 0) ^
        (X_core[:, 3] > 0)
    ).astype(int)
    # Add 10 noise features
    X = _add_noise_features(X_core, 10, noise_scale=1.0)
    return X, y


def get_overfitting_dataset():
    """
    Dataset specifically designed to show overfitting in decision trees.
    n=3000, d=25 (5 informative + 20 noise)
    Strong signal from 5 features, but many noise features tempt overfitting.
    """
    n = 3000
    # 5 informative features with clear but not perfect separation
    X_info = np.random.normal(0, 1, size=(n, 5))
    # Deterministic label based on thresholds (tree can learn this)
    y = (
        (X_info[:, 0] > 0.5).astype(int) |
        (X_info[:, 1] > 0.8).astype(int) |
        (X_info[:, 2] < -0.5).astype(int)
    ).astype(int)
    # 10% label noise to prevent perfect separation
    flip_idx = np.random.choice(n, size=int(0.10 * n), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    # 20 noise features to tempt overfitting at high depth
    X = _add_noise_features(X_info, 20, noise_scale=1.0)
    return X, y


# ---------------------------------------------------------------------------
# Q3: Decision Tree Datasets
# ---------------------------------------------------------------------------

def get_tree_friendly_dataset():
    """
    (Q3-B/C) Tree-friendly: axis-aligned decision boundaries.
    n=3000, d=20 (10 informative thresholds + 10 noise)
    Labels depend on simple threshold conditions that trees can capture greedily.
    """
    n = 3000
    d = 10
    X_info = np.random.normal(0, 1, size=(n, d))
    # Tree-friendly label: nested thresholds (perfectly learnable by a shallow tree)
    # If X0 > 0.5:
    #   class = 1 if X1 > 0.5 else 0
    # else:
    #   class = 1 if X2 > 0.0 else 0
    y = np.where(
        X_info[:, 0] > 0.5,
        np.where(X_info[:, 1] > 0.5, 1, 0),
        np.where(X_info[:, 2] > 0.0, 1, 0)
    )
    X = _add_noise_features(X_info, 10, noise_scale=1.0)
    return X, y


def get_greedy_counterexample_dataset():
    """
    (Q3-D) Greedy splitting counterexample.
    XOR-like structure in 5D expanded to d=15.
    The best first split is weak; optimal tree needs multiple levels.
    """
    n = 2000
    # 5D parity structure where no single feature gives high purity
    X_core = np.random.uniform(-1, 1, size=(n, 5))
    y = (
        (X_core[:, 0] > 0) ^
        (X_core[:, 1] > 0) ^
        (X_core[:, 2] > 0) ^
        (X_core[:, 3] > 0)
    ).astype(int)
    # Add 10 noise features
    X = _add_noise_features(X_core, 10, noise_scale=0.3)
    return X, y


def get_tree_noisy_dataset():
    """
    (Q3-E) Same as tree-friendly but with label noise and outliers.
    """
    n = 3000
    d = 10
    X_info = np.random.normal(0, 1, size=(n, d))
    # Same tree-friendly label structure
    y = np.where(
        X_info[:, 0] > 0.5,
        np.where(X_info[:, 1] > 0.5, 1, 0),
        np.where(X_info[:, 2] > 0.0, 1, 0)
    )
    # Label noise: flip 8% of labels
    flip_idx = np.random.choice(n, size=int(0.08 * n), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    # Outliers: inject extreme values
    outlier_idx = np.random.choice(n, size=int(0.02 * n), replace=False)
    X_info[outlier_idx] = np.random.normal(0, 10, size=(len(outlier_idx), d))
    X = _add_noise_features(X_info, 10, noise_scale=1.0)
    return X, y


# ---------------------------------------------------------------------------
# Master loader for convenience
# ---------------------------------------------------------------------------

DATASET_MAP = {
    "low_noise": get_low_noise_dataset,
    "high_noise": get_high_noise_dataset,
    "high_dimensional": get_high_dimensional_dataset,
    "kmeans_friendly": get_kmeans_friendly_dataset,
    "kmeans_adversarial": get_kmeans_adversarial_dataset,
    "nb_correlated": get_nb_correlated_dataset,
    "nb_success": get_nb_success_dataset,
    "nb_failure": get_nb_failure_dataset,
    "tree_friendly": get_tree_friendly_dataset,
    "greedy_counterexample": get_greedy_counterexample_dataset,
    "tree_noisy": get_tree_noisy_dataset,
    "overfitting": get_overfitting_dataset,
}


def load_dataset(name):
    """Load a dataset by name. Returns (X, y)."""
    if name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_MAP.keys())}")
    return DATASET_MAP[name]()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name in DATASET_MAP:
        X, y = load_dataset(name)
        print(f"{name:25s}: X.shape={X.shape}, y.shape={y.shape}, n_classes={len(np.unique(y))}")
