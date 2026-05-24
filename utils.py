"""
utils.py
========
Shared utilities for plotting, metrics, and experiment bookkeeping.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"


def savefig(name, subdir="", dpi=300):
    """Save a figure to results/subdir/name."""
    path = os.path.join(RESULTS_DIR, subdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"[Saved] {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation wrappers
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, average='binary'):
    """Return a dict of standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def split_data(X, y, test_size=0.2, random_state=122):
    """Train-test split wrapper."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_scatter_2d(X, y, title, xlabel="X1", ylabel="X2", subdir="", fname="scatter.png"):
    """Scatter plot for 2D data (uses first 2 dims if higher)."""
    plt.figure(figsize=(6, 5))
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}", alpha=0.5, edgecolors='k', s=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(fname, subdir)


def plot_confusion_matrix(cm, title, subdir="", fname="confusion_matrix.png"):
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(cm))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Annotate cells
    thresh = np.array(cm).max() / 2.
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, format(cm[i][j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i][j] > thresh else "black")
    savefig(fname, subdir)


def plot_bar_comparison(categories, values_dict, title, ylabel, subdir="", fname="bar.png"):
    """Grouped bar chart."""
    x = np.arange(len(categories))
    width = 0.8 / len(values_dict)
    plt.figure(figsize=(8, 5))
    for i, (label, vals) in enumerate(values_dict.items()):
        plt.bar(x + i * width, vals, width, label=label)
    plt.xticks(x + width * (len(values_dict) - 1) / 2, categories)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    savefig(fname, subdir)


def plot_line(x, y_dict, title, xlabel, ylabel, subdir="", fname="line.png"):
    """Multi-line plot."""
    plt.figure(figsize=(7, 5))
    for label, y_vals in y_dict.items():
        plt.plot(x, y_vals, marker='o', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(fname, subdir)


# ---------------------------------------------------------------------------
# Reliability diagram (for calibration analysis)
# ---------------------------------------------------------------------------

def plot_reliability_diagram(y_true, y_prob, n_bins=10, subdir="", fname="reliability.png"):
    """
    Plot reliability diagram for binary classifier.
    y_prob: predicted probability for the positive class.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    accuracies = []
    confidences = []
    counts = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        count = np.sum(in_bin)
        counts.append(count)
        if count > 0:
            accuracies.append(np.mean(y_true[in_bin]))
            confidences.append(np.mean(y_prob[in_bin]))
        else:
            accuracies.append(0)
            confidences.append(0)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(confidences, accuracies, 'o-', label='Model')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(fname, subdir)
