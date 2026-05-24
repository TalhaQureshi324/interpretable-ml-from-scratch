"""
q2_naive_bayes.py
=================
Question 2: Gaussian Naive Bayes [30 Marks]
- Part A: Implementation from scratch
- Part B: Correlated features experiment
- Part C: Counterexample challenge
- Part D: Conceptual analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import (
    get_nb_correlated_dataset,
    get_nb_success_dataset,
    get_nb_failure_dataset,
)
from utils import savefig, classification_metrics, plot_reliability_diagram, split_data
from sklearn.metrics import confusion_matrix

SEED = 122
np.random.seed(SEED)

# =============================================================================
# Part A: Gaussian Naive Bayes Implementation
# =============================================================================

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes from scratch.
    Uses log-probabilities for numerical stability.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = None       # log priors
        self.means_ = None        # (n_classes, n_features)
        self.vars_ = None         # (n_classes, n_features)

    def fit(self, X, y):
        """Estimate priors, means, and variances per class."""
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = X_c.mean(axis=0)
            self.vars_[idx, :] = X_c.var(axis=0) + self.var_smoothing
            self.priors_[idx] = np.log(X_c.shape[0] / X.shape[0])

        return self

    def _log_likelihood(self, X, class_idx):
        """
        Compute log P(X|y=class_idx) under Gaussian assumption.
        log P(x|y) = -0.5*log(2*pi*var) - (x-mean)^2 / (2*var)
        """
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        log_prob = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
        return np.sum(log_prob, axis=1)

    def predict_log_proba(self, X):
        """Compute log posterior for each class."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            log_probs[:, idx] = self.priors_[idx] + self._log_likelihood(X, idx)

        return log_probs

    def predict_proba(self, X):
        """Compute posterior probabilities using log-sum-exp trick."""
        log_probs = self.predict_log_proba(X)
        # log-sum-exp trick for stability
        max_log = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_log)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """Predict class labels."""
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


# =============================================================================
# Part B: Correlated Features Experiment
# =============================================================================

def experiment_part_b():
    """
    Train NB on dataset with highly correlated features + redundant feature.
    Analyze confidence scores, prediction errors, miscalibration.
    """
    print("=" * 70)
    print("Q2 - Part B: Correlated Features Experiment")
    print("=" * 70)

    X, y = get_nb_correlated_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=SEED)

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred)
    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

    # Reliability diagram
    plot_reliability_diagram(y_test, y_prob, n_bins=10,
                             subdir="q2_plots", fname="q2b_reliability.png")

    # Confidence score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_test == 0], bins=20, alpha=0.6, label="Class 0", edgecolor='k')
    plt.hist(y_prob[y_test == 1], bins=20, alpha=0.6, label="Class 1", edgecolor='k')
    plt.xlabel("Predicted Probability (Positive Class)")
    plt.ylabel("Count")
    plt.title("Q2-B: Confidence Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("q2b_confidence_dist.png", subdir="q2_plots")

    # Correlation matrix visualization for first 5 features
    df_corr = np.corrcoef(X_train[:, :5].T)
    plt.figure(figsize=(5, 4))
    plt.imshow(df_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(5), [f"F{i}" for i in range(5)])
    plt.yticks(range(5), [f"F{i}" for i in range(5)])
    plt.title("Q2-B: Feature Correlation Matrix (First 5)")
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{df_corr[i,j]:.2f}", ha="center", va="center",
                     color="white" if abs(df_corr[i,j]) > 0.5 else "black")
    savefig("q2b_correlation_matrix.png", subdir="q2_plots")

    print("\n--- Analysis ---")
    print("Features X1 and X2 are highly correlated; X3 is redundant.")
    print("Naive Bayes treats them as independent, double-counting evidence.")
    print("This leads to overconfident predictions (extreme probabilities).")
    print("The reliability diagram shows miscalibration:")
    print("  model confidence >> actual accuracy in extreme bins.")

    return metrics


# =============================================================================
# Part C: Counterexample Challenge
# =============================================================================

def experiment_part_c():
    """
    (a) Dataset where NB performs surprisingly well despite violated assumptions
    (b) Dataset where NB completely fails
    """
    print("\n" + "=" * 70)
    print("Q2 - Part C: Counterexample Challenge")
    print("=" * 70)

    results = {}

    # -------- (a) Surprising Success --------
    print("\n--- (a) Surprising Success ---")
    X_s, y_s = get_nb_success_dataset()
    X_train, X_test, y_train, y_test = split_data(X_s, y_s, test_size=0.2, random_state=SEED)

    model_s = GaussianNaiveBayes()
    model_s.fit(X_train, y_train)
    y_pred_s = model_s.predict(X_test)
    metrics_s = classification_metrics(y_test, y_pred_s)
    print(f"Accuracy: {metrics_s['accuracy']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics_s['confusion_matrix'])}")

    # Visualize decision boundary in first 2 dims
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for label in np.unique(y_test):
        plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1],
                    label=f"True {label}", alpha=0.5, edgecolors='k', s=20)
    plt.title("Q2-C(a): Success Dataset (True Labels)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for label in np.unique(y_pred_s):
        plt.scatter(X_test[y_pred_s == label, 0], X_test[y_pred_s == label, 1],
                    label=f"Pred {label}", alpha=0.5, edgecolors='k', s=20)
    plt.title(f"Q2-C(a): NB Predictions (Acc={metrics_s['accuracy']:.3f})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("q2c_success.png", subdir="q2_plots")

    # -------- (b) Complete Failure --------
    print("\n--- (b) Complete Failure ---")
    X_f, y_f = get_nb_failure_dataset()
    X_train, X_test, y_train, y_test = split_data(X_f, y_f, test_size=0.2, random_state=SEED)

    model_f = GaussianNaiveBayes()
    model_f.fit(X_train, y_train)
    y_pred_f = model_f.predict(X_test)
    metrics_f = classification_metrics(y_test, y_pred_f)
    print(f"Accuracy: {metrics_f['accuracy']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics_f['confusion_matrix'])}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for label in np.unique(y_test):
        plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1],
                    label=f"True {label}", alpha=0.5, edgecolors='k', s=20)
    plt.title("Q2-C(b): Failure Dataset (True Labels) - XOR Structure")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for label in np.unique(y_pred_f):
        plt.scatter(X_test[y_pred_f == label, 0], X_test[y_pred_f == label, 1],
                    label=f"Pred {label}", alpha=0.5, edgecolors='k', s=20)
    plt.title(f"Q2-C(b): NB Predictions (Acc={metrics_f['accuracy']:.3f})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("q2c_failure.png", subdir="q2_plots")

    print("\n--- Analysis ---")
    print("Success case: Class separation is along the principal direction")
    print("  where correlated features jointly point. Even though independence")
    print("  is violated, the marginal distributions are still informative.")
    print("Failure case: XOR structure. Each feature individually has identical")
    print("  marginal distributions for both classes. NB cannot learn the")
    print("  interaction, so it performs no better than random guessing.")
    print("Decision boundaries become problematic because NB can only learn")
    print("  axis-aligned, univariate thresholds, not diagonal/XOR boundaries.")

    results["success"] = metrics_s
    results["failure"] = metrics_f
    return results


# =============================================================================
# Part D: Conceptual Analysis
# =============================================================================

def experiment_part_d():
    """
    Print conceptual analysis for the report.
    """
    print("\n" + "=" * 70)
    print("Q2 - Part D: Conceptual Analysis")
    print("=" * 70)

    analysis = """
(a) Why can Naive Bayes outperform more sophisticated models on small datasets?

    Naive Bayes makes a strong conditional independence assumption,
    which dramatically reduces the number of parameters to estimate.
    For d features and C classes:
      - NB parameters: O(C * d)  (means + variances + priors)
      - Full Gaussian classifier: O(C * d^2)  (covariance matrices)
      - Logistic Regression with interactions: O(d^2) or more

    With small n, high-variance models (complex models) overfit because
    they cannot reliably estimate O(d^2) parameters. NB's high bias
    becomes a blessing: it generalizes better despite wrong assumptions.
    This is the classic bias-variance tradeoff.

(b) Why does correlated evidence lead to overconfident predictions?

    Mathematically, NB approximates the joint likelihood:
      P(x1, x2 | y)  ~  P(x1|y) * P(x2|y)

    If x1 and x2 are positively correlated, the true joint likelihood
    is more constrained than the product of marginals. The product
    overestimates the evidence because it double-counts the same
    information.

    In log-space:
      log P(x1,x2|y)  vs  log P(x1|y) + log P(x2|y)

    The variance of the RHS (sum of correlated log-likelihoods) is
    larger than the variance of the true joint log-likelihood.
    Consequently, the posterior log-odds have inflated variance,
    pushing predictions toward 0 or 1, leading to overconfidence.

    If rho is the correlation between x1 and x2 (given y), the effective
    variance of the evidence sum is approximately:
      Var[log P(x1|y) + log P(x2|y)] ~= sigma1^2 + sigma2^2 + 2*rho*sigma1*sigma2
    which exceeds the true information content when rho > 0.
    """
    print(analysis)
    return analysis


# =============================================================================
# Main runner
# =============================================================================

def run_q2():
    """Execute all Q2 experiments."""
    print("\n" + "=" * 70)
    print("QUESTION 2: GAUSSIAN NAIVE BAYES")
    print("=" * 70)
    results_b = experiment_part_b()
    results_c = experiment_part_c()
    results_d = experiment_part_d()
    return {"part_b": results_b, "part_c": results_c, "part_d": results_d}


if __name__ == "__main__":
    run_q2()
