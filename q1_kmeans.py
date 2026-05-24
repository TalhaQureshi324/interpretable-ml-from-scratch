"""
q1_kmeans.py
============
Question 1: k-Means Clustering [30 Marks]
- Part A: Core vectorized implementation
- Part B: Adversarial dataset construction
- Part C: Initialization sensitivity (20 runs)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import get_kmeans_friendly_dataset, get_kmeans_adversarial_dataset
from utils import savefig, plot_scatter_2d
from sklearn.metrics import silhouette_score

SEED = 122
np.random.seed(SEED)

# =============================================================================
# Part A: Core Implementation
# =============================================================================

class KMeans:
    """Vectorized k-Means from scratch."""

    def __init__(self, k=3, max_iter=300, tol=1e-4, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.cost_history = []

    def _init_centroids(self, X):
        """Random centroid initialization: pick k unique data points."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, size=self.k, replace=False)
        return X[idx].copy()

    def _compute_distances(self, X, centroids):
        """
        Vectorized Euclidean distance.
        Returns (n_samples, k) matrix.
        """
        # Using broadcasting: (n, 1, d) - (1, k, d) -> (n, k, d) -> sum -> (n, k)
        return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

    def _assign_clusters(self, distances):
        """Assign each point to nearest centroid."""
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points."""
        new_centroids = np.zeros((self.k, X.shape[1]))
        for j in range(self.k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = points_in_cluster.mean(axis=0)
            else:
                # Re-initialize empty cluster randomly
                new_centroids[j] = X[np.random.choice(X.shape[0])]
        return new_centroids

    def _compute_cost(self, X, labels, centroids):
        """Sum of squared distances to assigned centroids (inertia)."""
        cost = 0.0
        for j in range(self.k):
            points = X[labels == j]
            if len(points) > 0:
                cost += np.sum((points - centroids[j]) ** 2)
        return cost

    def fit(self, X, track_history=True):
        """Fit k-Means."""
        self.centroids = self._init_centroids(X)
        self.cost_history = []

        for iteration in range(self.max_iter):
            distances = self._compute_distances(X, self.centroids)
            labels = self._assign_clusters(distances)
            new_centroids = self._update_centroids(X, labels)
            cost = self._compute_cost(X, labels, new_centroids)
            if track_history:
                self.cost_history.append(cost)

            # Check convergence
            centroid_shift = np.sqrt(((new_centroids - self.centroids) ** 2).sum(axis=1)).max()
            self.centroids = new_centroids
            self.labels = labels

            if centroid_shift < self.tol:
                break

        return self

    def predict(self, X):
        """Predict cluster labels for new data."""
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)


# =============================================================================
# Part B: Adversarial Dataset Construction
# =============================================================================

def experiment_part_b():
    """
    (a) Dataset where k-Means performs extremely well
    (b) Dataset where k-Means fails despite visually separable clusters
    """
    print("=" * 70)
    print("Q1 - Part B: Adversarial Dataset Construction")
    print("=" * 70)

    # -------- (a) Friendly Dataset --------
    X_friendly, y_true_friendly = get_kmeans_friendly_dataset()
    print(f"\n[Friendly Dataset] X.shape={X_friendly.shape}")

    # Use only first 2 dims for visualization, but fit on full data
    kmeans_good = KMeans(k=4, random_state=SEED)
    kmeans_good.fit(X_friendly)
    y_pred_good = kmeans_good.labels

    sil_good = silhouette_score(X_friendly, y_pred_good)
    print(f"Silhouette Score (friendly): {sil_good:.4f}")

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label in np.unique(y_true_friendly):
        plt.scatter(X_friendly[y_true_friendly == label, 0],
                    X_friendly[y_true_friendly == label, 1],
                    label=f"True {label}", alpha=0.5, edgecolors='k', s=15)
    plt.title("Q1-B(a): Friendly Dataset (True Labels)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for label in np.unique(y_pred_good):
        plt.scatter(X_friendly[y_pred_good == label, 0],
                    X_friendly[y_pred_good == label, 1],
                    label=f"Cluster {label}", alpha=0.5, edgecolors='k', s=15)
    plt.scatter(kmeans_good.centroids[:, 0], kmeans_good.centroids[:, 1],
                c='red', marker='X', s=200, edgecolors='black', label='Centroids')
    plt.title(f"Q1-B(a): k-Means Result (Silhouette={sil_good:.3f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    savefig("q1b_friendly.png", subdir="q1_plots")

    # -------- (b) Adversarial Dataset --------
    X_adv, y_true_adv = get_kmeans_adversarial_dataset()
    print(f"\n[Adversarial Dataset] X.shape={X_adv.shape}")

    kmeans_bad = KMeans(k=2, random_state=SEED)
    kmeans_bad.fit(X_adv)
    y_pred_bad = kmeans_bad.labels

    sil_bad = silhouette_score(X_adv, y_pred_bad)
    print(f"Silhouette Score (adversarial): {sil_bad:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label in np.unique(y_true_adv):
        plt.scatter(X_adv[y_true_adv == label, 0],
                    X_adv[y_true_adv == label, 1],
                    label=f"True {label}", alpha=0.5, edgecolors='k', s=15)
    plt.title("Q1-B(b): Adversarial Dataset (True Labels)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for label in np.unique(y_pred_bad):
        plt.scatter(X_adv[y_pred_bad == label, 0],
                    X_adv[y_pred_bad == label, 1],
                    label=f"Cluster {label}", alpha=0.5, edgecolors='k', s=15)
    plt.scatter(kmeans_bad.centroids[:, 0], kmeans_bad.centroids[:, 1],
                c='red', marker='X', s=200, edgecolors='black', label='Centroids')
    plt.title(f"Q1-B(b): k-Means Result (Silhouette={sil_bad:.3f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    savefig("q1b_adversarial.png", subdir="q1_plots")

    # Analysis printout
    print("\n--- Analysis ---")
    print("Friendly dataset: Spherical, equal variance, balanced clusters.")
    print("  -> k-Means assumptions SATISFIED. Performs well.")
    print("Adversarial dataset: Elongated, different covariance shapes.")
    print("  -> k-Means assumes spherical clusters with equal variance.")
    print("  -> Violation leads to poor boundaries despite visual separability.")
    print("  -> Feature scaling (StandardScaler) does NOT fix this because")
    print("     the issue is shape/anisotropy, not scale.")

    return {
        "friendly_silhouette": sil_good,
        "adversarial_silhouette": sil_bad,
    }


# =============================================================================
# Part C: Initialization Sensitivity (20 runs)
# =============================================================================

def experiment_part_c():
    """
    Run k-Means 20 times with different initializations on friendly dataset.
    Plot convergence behavior and final cost histogram.
    """
    print("\n" + "=" * 70)
    print("Q1 - Part C: Initialization Sensitivity (20 runs)")
    print("=" * 70)

    X, _ = get_kmeans_friendly_dataset()
    k = 4
    n_runs = 20

    final_costs = []
    all_histories = []
    selected_runs = [0, 1, 2, 3, 4]  # Plot convergence for first 5 runs

    for run in range(n_runs):
        km = KMeans(k=k, random_state=SEED + run)
        km.fit(X, track_history=True)
        final_costs.append(km.cost_history[-1])
        if run in selected_runs:
            all_histories.append(km.cost_history)
        print(f"Run {run+1:2d}: iterations={len(km.cost_history):3d}, final_cost={km.cost_history[-1]:.2f}")

    # Plot convergence curves
    plt.figure(figsize=(8, 5))
    for i, hist in enumerate(all_histories):
        plt.plot(range(1, len(hist)+1), hist, marker='o', label=f"Run {selected_runs[i]+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (Inertia)")
    plt.title("Q1-C: Convergence Behavior Across Different Initializations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("q1c_convergence.png", subdir="q1_plots")

    # Plot histogram of final costs
    plt.figure(figsize=(8, 5))
    plt.hist(final_costs, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel("Final Cost (Inertia)")
    plt.ylabel("Frequency")
    plt.title(f"Q1-C: Distribution of Final Costs ({n_runs} Runs)")
    plt.axvline(np.mean(final_costs), color='red', linestyle='--', label=f"Mean={np.mean(final_costs):.1f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("q1c_cost_histogram.png", subdir="q1_plots")

    print(f"\nMean final cost: {np.mean(final_costs):.2f}")
    print(f"Std  final cost: {np.std(final_costs):.2f}")
    print(f"Min  final cost: {np.min(final_costs):.2f}")
    print(f"Max  final cost: {np.max(final_costs):.2f}")

    print("\n--- Analysis ---")
    print("Different initializations lead to different local minima because")
    print("k-Means optimizes a non-convex objective (sum of squared distances).")
    print("The coordinate descent (Lloyd's algorithm) is guaranteed to converge")
    print("only to a local minimum, not the global optimum.")
    print("Poor initialization can trap centroids in suboptimal partitions,")
    print("especially when clusters are not well-separated.")

    return {
        "final_costs": final_costs,
        "mean_cost": float(np.mean(final_costs)),
        "std_cost": float(np.std(final_costs)),
    }


# =============================================================================
# Main runner
# =============================================================================

def run_q1():
    """Execute all Q1 experiments."""
    print("\n" + "=" * 70)
    print("QUESTION 1: k-MEANS CLUSTERING")
    print("=" * 70)
    results_b = experiment_part_b()
    results_c = experiment_part_c()
    return {"part_b": results_b, "part_c": results_c}


if __name__ == "__main__":
    run_q1()
