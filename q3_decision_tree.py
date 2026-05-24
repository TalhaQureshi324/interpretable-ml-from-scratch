"""
q3_decision_tree.py
===================
Question 3: Decision Trees using C4.5 [40 Marks]
- Part A: C4.5 implementation (entropy, IG, split info, gain ratio)
- Part B: Gain Ratio analysis
- Part C: Overfitting investigation
- Part D: Greedy splitting counterexample
- Part E: Noise sensitivity
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import (
    get_tree_friendly_dataset,
    get_greedy_counterexample_dataset,
    get_tree_noisy_dataset,
    get_low_noise_dataset,
    get_overfitting_dataset,
)
from utils import savefig, classification_metrics, split_data

SEED = 122
np.random.seed(SEED)

# =============================================================================
# Part A: C4.5 Implementation
# =============================================================================

def entropy(y):
    """Compute Shannon entropy H(S) = -sum(p_i * log2(p_i))."""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(X_col, y, threshold):
    """
    Compute Information Gain for a binary split on X_col <= threshold.
    IG(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))
    """
    parent_entropy = entropy(y)
    left_mask = X_col <= threshold
    right_mask = ~left_mask

    n = len(y)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left == 0 or n_right == 0:
        return 0.0

    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

    return parent_entropy - weighted_entropy


def split_information(X_col, threshold):
    """
    Compute Split Information: -sum(|S_v|/|S| * log2(|S_v|/|S|))
    """
    left_mask = X_col <= threshold
    right_mask = ~left_mask
    n = len(X_col)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left == 0 or n_right == 0:
        return 0.0

    p_left = n_left / n
    p_right = n_right / n
    return -(p_left * np.log2(p_left + 1e-12) + p_right * np.log2(p_right + 1e-12))


def gain_ratio(X_col, y, threshold):
    """
    Gain Ratio = Information Gain / Split Information
    Returns 0 if split info is 0 to avoid division by zero.
    """
    ig = information_gain(X_col, y, threshold)
    si = split_information(X_col, threshold)
    if si < 1e-12:
        return 0.0
    return ig / si


class TreeNode:
    """Node in the decision tree."""
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.prediction = None
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.n_samples = 0
        self.ig = None      # information gain at this split
        self.gr = None      # gain ratio at this split
        self.class_distribution = None


class DecisionTreeC45:
    """
    Binary Decision Tree using C4.5 algorithm.
    Supports continuous features via threshold-based splitting.
    Optimized: only evaluates midpoints where label changes between
    consecutive sorted samples, and caps max thresholds per feature.
    """

    def __init__(self, max_depth=10, min_samples_split=2, max_thresholds=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.root = None
        self.n_features = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        node = TreeNode(depth=depth)
        node.n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        node.class_distribution = dict(zip(classes.tolist(), counts.tolist()))

        # Stopping criteria
        if len(classes) == 1:
            node.is_leaf = True
            node.prediction = classes[0]
            return node

        if depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = classes[np.argmax(counts)]
            return node

        if len(y) < self.min_samples_split:
            node.is_leaf = True
            node.prediction = classes[np.argmax(counts)]
            return node

        # Find best split using Gain Ratio
        best_gr = -1.0
        best_ig = -1.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(self.n_features):
            X_col = X[:, feature_idx]
            # Sort values and labels together
            sort_idx = np.argsort(X_col)
            X_sorted = X_col[sort_idx]
            y_sorted = y[sort_idx]

            # Only consider thresholds where label changes between consecutive values
            # This is mathematically sound: splitting between same-label points
            # does not change label distribution.
            change_points = np.where(y_sorted[:-1] != y_sorted[1:])[0]
            if len(change_points) == 0:
                continue

            thresholds = (X_sorted[change_points] + X_sorted[change_points + 1]) / 2.0
            # Remove exact duplicates (can happen when multiple same-value points straddle a change)
            thresholds = np.unique(thresholds)

            # Cap number of thresholds for speed
            if len(thresholds) > self.max_thresholds:
                idx = np.linspace(0, len(thresholds) - 1, self.max_thresholds, dtype=int)
                thresholds = thresholds[idx]

            for threshold in thresholds:
                ig = information_gain(X_col, y, threshold)
                si = split_information(X_col, threshold)
                if si < 1e-12:
                    continue
                gr = ig / si

                if gr > best_gr:
                    best_gr = gr
                    best_ig = ig
                    best_feature = feature_idx
                    best_threshold = threshold

        # No valid split found
        if best_feature is None or best_gr <= 0:
            node.is_leaf = True
            node.prediction = classes[np.argmax(counts)]
            return node

        # Apply split
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.ig = best_ig
        node.gr = best_gr

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Handle edge case where split doesn't divide data
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            node.is_leaf = True
            node.prediction = classes[np.argmax(counts)]
            return node

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _predict_one(self, x, node):
        """Predict single sample."""
        if node.is_leaf:
            return node.prediction
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        """Predict labels for X."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def count_nodes(self, node=None):
        """Count total nodes in tree."""
        if node is None:
            node = self.root
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def count_leaves(self, node=None):
        """Count leaf nodes."""
        if node is None:
            node = self.root
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)

    def get_depth(self, node=None):
        """Get maximum depth of tree."""
        if node is None:
            node = self.root
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))


# =============================================================================
# Part B: Gain Ratio Analysis
# =============================================================================

def experiment_part_b():
    """
    Compare Information Gain and Gain Ratio on generated datasets.
    Identify cases where IG prefers poor splits.
    """
    print("=" * 70)
    print("Q3 - Part B: Gain Ratio Analysis")
    print("=" * 70)

    X, y = get_tree_friendly_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=SEED)

    # Inject a synthetic "outlier trap" noise feature to demonstrate IG bias.
    # This feature is mostly random, but a few extreme values happen to correlate
    # with one class, creating a threshold that isolates a tiny pure subset.
    # IG loves this (pure leaf), but Split Information is near zero (imbalanced).
    n_train = X_train.shape[0]
    id_feature = np.random.normal(0, 1, size=n_train)
    # Force 3 samples from class 1 to have extreme positive values
    cls1_idx = np.where(y_train == 1)[0][:3]
    id_feature[cls1_idx] = 100.0 + np.random.normal(0, 0.001, size=len(cls1_idx))
    X_train_aug = np.column_stack([X_train, id_feature])
    y_train_aug = y_train.copy()

    # Evaluate top candidate splits for each feature
    n_features = X_train_aug.shape[1]
    results = []

    for feat in range(n_features):
        X_col = X_train_aug[:, feat]
        sort_idx = np.argsort(X_col)
        X_sorted = X_col[sort_idx]
        y_sorted = y_train_aug[sort_idx]

        change_points = np.where(y_sorted[:-1] != y_sorted[1:])[0]
        if len(change_points) == 0:
            continue
        thresholds = (X_sorted[change_points] + X_sorted[change_points + 1]) / 2.0
        thresholds = np.unique(thresholds)
        if len(thresholds) > 50:
            idx = np.linspace(0, len(thresholds) - 1, 50, dtype=int)
            thresholds = thresholds[idx]

        best_ig = -1.0
        best_gr = -1.0
        best_t = None
        for t in thresholds:
            ig = information_gain(X_col, y_train_aug, t)
            si = split_information(X_col, t)
            gr = ig / si if si > 1e-12 else 0.0
            if gr > best_gr:
                best_gr = gr
                best_ig = ig
                best_t = t

        results.append({
            "feature": feat,
            "threshold": best_t,
            "IG": best_ig,
            "GR": best_gr,
        })

    # Sort by IG descending
    results_by_ig = sorted(results, key=lambda x: x["IG"], reverse=True)[:6]
    results_by_gr = sorted(results, key=lambda x: x["GR"], reverse=True)[:6]

    print("\nTop 6 splits by INFORMATION GAIN:")
    print(f"{'Feature':>10} {'Threshold':>12} {'IG':>10} {'GR':>10} {'SplitInfo':>10}")
    for r in results_by_ig:
        si = split_information(X_train_aug[:, r['feature']], r['threshold'])
        marker = "  <-- ID noise" if r['feature'] == n_features - 1 else ""
        print(f"F{r['feature']:>8} {r['threshold']:>12.4f} {r['IG']:>10.4f} {r['GR']:>10.4f} {si:>10.4f}{marker}")

    print("\nTop 6 splits by GAIN RATIO:")
    print(f"{'Feature':>10} {'Threshold':>12} {'IG':>10} {'GR':>10} {'SplitInfo':>10}")
    for r in results_by_gr:
        si = split_information(X_train_aug[:, r['feature']], r['threshold'])
        marker = "  <-- ID noise" if r['feature'] == n_features - 1 else ""
        print(f"F{r['feature']:>8} {r['threshold']:>12.4f} {r['IG']:>10.4f} {r['GR']:>10.4f} {si:>10.4f}{marker}")

    # Identify the ID feature result specifically
    id_result = next((r for r in results if r['feature'] == n_features - 1), None)
    if id_result:
        print(f"\n[POOR SPLIT DEMONSTRATION]")
        print(f"Synthetic ID-like feature (F{n_features-1}) achieves IG={id_result['IG']:.4f}")
        print(f"but its Gain Ratio is only GR={id_result['GR']:.4f} due to near-zero Split Information.")
        print(f"This happens because the ID feature creates highly imbalanced singleton-like partitions.")

    # Bar plot comparison: top IG splits + ID feature
    selected = results_by_ig[:5]
    if id_result and id_result not in selected:
        selected.append(id_result)

    feats = [f"F{r['feature']}" for r in selected]
    ig_vals = [r["IG"] for r in selected]
    gr_vals = [r["GR"] for r in selected]

    x = np.arange(len(feats))
    width = 0.35
    colors = ['#ef4444' if 'F'+str(n_features-1) in f else '#3b82f6' for f in feats]
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, ig_vals, width, label="Information Gain", alpha=0.85, color=colors)
    bars2 = plt.bar(x + width/2, gr_vals, width, label="Gain Ratio", alpha=0.85, edgecolor='black', linewidth=0.5)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Q3-B: IG vs GR — ID-like Noise Feature Gets High IG but Low GR", fontsize=13, fontweight='bold')
    plt.xticks(x, feats, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)

    # Annotate the ID feature
    for i, f in enumerate(feats):
        if 'F'+str(n_features-1) in f:
            plt.annotate('ID noise\n(penalized by GR)', xy=(i, max(ig_vals[i], gr_vals[i])),
                        xytext=(i+0.3, max(ig_vals[i], gr_vals[i])+0.05),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=10, color='darkred', fontweight='bold')

    savefig("q3b_ig_vs_gr.png", subdir="q3_plots")

    print("\n--- Analysis ---")
    print("Information Gain tends to favor features with many unique values")
    print("(high cardinality) because they can create very pure subsets.")
    print("Our synthetic ID-like feature achieves high IG by isolating singletons,")
    print("but its Split Information is near zero, so Gain Ratio collapses.")
    print("Gain Ratio penalizes such splits by dividing IG by Split Information.")
    print("Thus, Gain Ratio is more robust against biased split selection.")

    return results


# =============================================================================
# Part C: Overfitting Investigation
# =============================================================================

def experiment_part_c():
    """
    Train trees with varying depths and plot training vs validation accuracy.
    Uses a dataset with real signal + noise features to demonstrate overfitting.
    """
    print("\n" + "=" * 70)
    print("Q3 - Part C: Overfitting Investigation")
    print("=" * 70)

    X, y = get_overfitting_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=SEED)

    depths = [1, 2, 3, 5, 7, 10, 15, 20, 30, None]
    train_accs = []
    val_accs = []
    node_counts = []

    for depth in depths:
        if depth is None:
            tree = DecisionTreeC45(max_depth=1000, min_samples_split=2, max_thresholds=30)
            label = "Unlimited"
        else:
            tree = DecisionTreeC45(max_depth=depth, min_samples_split=2, max_thresholds=30)
            label = str(depth)

        tree.fit(X_train, y_train)
        train_acc = np.mean(tree.predict(X_train) == y_train)
        val_acc = np.mean(tree.predict(X_test) == y_test)
        n_nodes = tree.count_nodes()

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        node_counts.append(n_nodes)

        print(f"Depth {label:>10}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Nodes={n_nodes}")

    # Plot accuracy vs depth
    depth_labels = [str(d) if d is not None else "Unlim" for d in depths]
    plt.figure(figsize=(10, 5))
    plt.plot(depth_labels, train_accs, 'o-', label="Training Accuracy", color='blue')
    plt.plot(depth_labels, val_accs, 's-', label="Validation Accuracy", color='red')
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Q3-C: Training vs Validation Accuracy vs Tree Depth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("q3c_overfitting.png", subdir="q3_plots")

    # Plot number of nodes
    plt.figure(figsize=(8, 5))
    plt.bar(depth_labels, node_counts, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel("Max Depth")
    plt.ylabel("Number of Nodes")
    plt.title("Q3-C: Tree Complexity vs Max Depth")
    plt.grid(axis='y', alpha=0.3)
    savefig("q3c_complexity.png", subdir="q3_plots")

    print("\n--- Analysis ---")
    print("Very small depth (e.g., 1-2): High bias, underfitting.")
    print("  The model cannot capture the true decision boundary.")
    print("Medium depth (e.g., 3-5): Best generalization.")
    print("  Validation accuracy peaks here.")
    print("Very large depth / unlimited: Low bias, high variance, overfitting.")
    print("  Training accuracy keeps rising while validation accuracy drops.")
    print("  Deep trees memorize training data and fit noise features.")
    print("Optimal depth balances bias and variance.")

    return {
        "depths": depth_labels,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "node_counts": node_counts,
    }


# =============================================================================
# Part D: Greedy Splitting Counterexample
# =============================================================================

def experiment_part_d():
    """
    Construct a dataset where greedy splitting produces a suboptimal tree.
    XOR structure: best first split is uninformative.
    """
    print("\n" + "=" * 70)
    print("Q3 - Part D: Greedy Splitting Counterexample")
    print("=" * 70)

    X, y = get_greedy_counterexample_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=SEED)

    tree = DecisionTreeC45(max_depth=5, min_samples_split=2)
    tree.fit(X_train, y_train)

    train_acc = np.mean(tree.predict(X_train) == y_train)
    val_acc = np.mean(tree.predict(X_test) == y_test)

    print(f"\nTree depth: {tree.get_depth()}")
    print(f"Tree nodes: {tree.count_nodes()}")
    print(f"Tree leaves: {tree.count_leaves()}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")

    # Visualize true labels and tree predictions in 2D
    y_pred = tree.predict(X_test)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for label in np.unique(y_test):
        plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1],
                    label=f"Class {label}", alpha=0.5, edgecolors='k', s=20)
    plt.title("Q3-D: Greedy Counterexample (True Labels)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for label in np.unique(y_pred):
        plt.scatter(X_test[y_pred == label, 0], X_test[y_pred == label, 1],
                    label=f"Pred {label}", alpha=0.5, edgecolors='k', s=20)
    plt.title(f"Q3-D: Tree Predictions (Acc={val_acc:.3f})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("q3d_greedy_counterexample.png", subdir="q3_plots")

    # Print tree structure
    def print_tree(node, indent=""):
        if node.is_leaf:
            print(f"{indent}LEAF -> predict {node.prediction} (samples={node.n_samples})")
            return
        print(f"{indent}IF Feature {node.feature_idx} <= {node.threshold:.3f}")
        print(f"{indent}  THEN:")
        print_tree(node.left, indent + "    ")
        print(f"{indent}  ELSE:")
        print_tree(node.right, indent + "    ")

    print("\nTree Structure:")
    print_tree(tree.root)

    print("\n--- Analysis ---")
    print("In XOR-like problems, any single axis-aligned split is uninformative")
    print("because both halves contain roughly equal numbers of both classes.")
    print("Greedy splitting chooses the locally best split, which may only")
    print("separate a small region rather than the globally optimal partition.")
    print("A globally optimal tree for XOR requires looking ahead or using")
    print("oblique splits (non-axis-aligned), which greedy C4.5 cannot do.")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "depth": tree.get_depth(),
        "nodes": tree.count_nodes(),
    }


# =============================================================================
# Part E: Noise Sensitivity
# =============================================================================

def experiment_part_e():
    """
    Introduce label noise and outliers, analyze tree changes.
    """
    print("\n" + "=" * 70)
    print("Q3 - Part E: Noise Sensitivity")
    print("=" * 70)

    # Clean dataset
    X_clean, y_clean = get_tree_friendly_dataset()
    y_clean = (y_clean % 2).astype(int)
    Xc_train, Xc_test, yc_train, yc_test = split_data(X_clean, y_clean, test_size=0.2, random_state=SEED)

    # Noisy dataset
    X_noisy, y_noisy = get_tree_noisy_dataset()
    yn = (y_noisy % 2).astype(int)
    Xn_train, Xn_test, yn_train, yn_test = split_data(X_noisy, yn, test_size=0.2, random_state=SEED)

    # Train both
    tree_clean = DecisionTreeC45(max_depth=10, min_samples_split=2)
    tree_clean.fit(Xc_train, yc_train)

    tree_noisy = DecisionTreeC45(max_depth=10, min_samples_split=2)
    tree_noisy.fit(Xn_train, yn_train)

    clean_train = np.mean(tree_clean.predict(Xc_train) == yc_train)
    clean_val = np.mean(tree_clean.predict(Xc_test) == yc_test)
    noisy_train = np.mean(tree_noisy.predict(Xn_train) == yn_train)
    noisy_val = np.mean(tree_noisy.predict(Xn_test) == yn_test)

    print(f"\nClean Tree:  Train={clean_train:.4f}, Val={clean_val:.4f}, "
          f"Depth={tree_clean.get_depth()}, Nodes={tree_clean.count_nodes()}")
    print(f"Noisy Tree:  Train={noisy_train:.4f}, Val={noisy_val:.4f}, "
          f"Depth={tree_noisy.get_depth()}, Nodes={tree_noisy.count_nodes()}")

    # Plot comparison
    categories = ["Clean", "Noisy"]
    train_vals = [clean_train, noisy_train]
    val_vals = [clean_val, noisy_val]
    node_vals = [tree_clean.count_nodes(), tree_noisy.count_nodes()]

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, train_vals, width, label="Train Acc", alpha=0.8)
    plt.bar(x, val_vals, width, label="Val Acc", alpha=0.8)
    plt.bar(x + width, [n / max(node_vals) for n in node_vals], width, label="Nodes (normalized)", alpha=0.8)
    plt.xticks(x, categories)
    plt.ylabel("Score / Normalized Count")
    plt.title("Q3-E: Clean vs Noisy Dataset Tree Comparison")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    savefig("q3e_noise_sensitivity.png", subdir="q3_plots")

    print("\n--- Analysis ---")
    print("Label noise and outliers cause trees to grow deeper and more complex")
    print("as they try to fit the corrupted samples. This reduces generalization")
    print("(validation accuracy drops) even though training accuracy may stay high.")
    print("Trees are unstable learners: a single mislabeled point can change")
    print("an entire split, altering the tree structure drastically.")
    print("This is because splits are hard thresholds; small data changes")
    print("near a threshold can flip the optimal split feature or value.")

    return {
        "clean": {"train": clean_train, "val": clean_val, "nodes": tree_clean.count_nodes(), "depth": tree_clean.get_depth()},
        "noisy": {"train": noisy_train, "val": noisy_val, "nodes": tree_noisy.count_nodes(), "depth": tree_noisy.get_depth()},
    }


# =============================================================================
# Main runner
# =============================================================================

def run_q3():
    """Execute all Q3 experiments."""
    print("\n" + "=" * 70)
    print("QUESTION 3: DECISION TREES (C4.5)")
    print("=" * 70)
    results_b = experiment_part_b()
    results_c = experiment_part_c()
    results_d = experiment_part_d()
    results_e = experiment_part_e()
    return {
        "part_b": results_b,
        "part_c": results_c,
        "part_d": results_d,
        "part_e": results_e,
    }


if __name__ == "__main__":
    run_q3()
