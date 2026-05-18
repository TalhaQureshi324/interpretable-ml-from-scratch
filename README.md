# Machine Learning Assignment 2 — Implementation Plan

> **Course:** Machine Learning  
> **Assignment:** A02 — Unsupervised & Probabilistic Learning + Decision Trees  

---

## 1. Assignment Overview

This assignment requires implementing three core ML algorithms **from scratch** using only `NumPy`, `Pandas`, and `Matplotlib`. `scikit-learn` is allowed **only** for train-test split and evaluation metrics. All datasets must be **programmatically generated** with reproducible seeds.

### Allowed Libraries
- ✅ NumPy, Pandas, Matplotlib
- ✅ scikit-learn: `train_test_split`, evaluation metrics (`accuracy_score`, `confusion_matrix`, `silhouette_score`, etc.)

### Strictly Prohibited
- ❌ scikit-learn classifiers, clustering APIs, decision tree APIs, Naive Bayes APIs
- ❌ Direct use of Kaggle/public datasets (unless explicitly stated)
- ❌ Hardcoded outputs

---

## 2. Dataset Construction Plan

### 2.1 Reproducibility Seed
```python
seed = <last_3_digits_of_roll_number>
np.random.seed(seed)
```

### 2.2 Global Constraints (ALL datasets must satisfy)
| Constraint | Value |
|------------|-------|
| Min samples (`n`) | ≥ 1000 |
| Min features (`d`) | ≥ 15 |
| Informative features | ≥ 5 |
| Noisy/irrelevant features | ≥ 5 |
| At least one dataset with `d` | ≥ 50 |
| At least one dataset with `n` | ≥ 5000 |

### 2.3 Required Dataset Categories (≥ 3)

| Category | Purpose | Planned Specs |
|----------|---------|---------------|
| **Low-noise dataset** | Ideal conditions for all algorithms | `n=5000`, `d=20` (15 informative + 5 noise) |
| **High-noise dataset** | Tests robustness | `n=3000`, `d=25` (10 informative + 15 noise), high variance noise |
| **High-dimensional dataset** | Tests curse of dimensionality & overfitting | `n=5000`, `d=60` (10 informative + 50 noise) |

### 2.4 Special-Purpose Datasets (per-question)

| Question | Dataset | Description |
|----------|---------|-------------|
| Q1-B(a) | k-Means Friendly | Well-separated, spherical, balanced Gaussian clusters |
| Q1-B(b) | k-Means Adversarial | Visually separable but non-spherical / unequal variance / imbalanced |
| Q2-B | Correlated Features | Highly correlated pair + redundant feature to violate Naive Bayes independence |
| Q2-C(a) | NB Surprise Success | Correlated but decision boundary still separable |
| Q2-C(b) | NB Failure | Extreme correlation creating overlapping class-conditional distributions |
| Q3-B/C | Tree-Friendly | Mix of categorical-like thresholds in continuous space |
| Q3-D | Greedy Counterexample | XOR-like or parity structure where greedy splits fail |
| Q3-E | Noisy Labels | Same as low-noise but with 5-10% label noise & outliers |

### 2.5 Mandatory Analysis Per Dataset
For every constructed dataset, the report must explain:
- Which algorithmic assumptions are **satisfied**
- Which assumptions are **violated**
- Why the algorithm **succeeds or fails**
- Expected **bias-variance** behavior
- Expected **overfitting/underfitting** behavior

---

## 3. Algorithm Implementation Plan

### 3.1 Question 1: k-Means Clustering [30 Marks]

**File:** `q1_kmeans.py`

#### Part A — Core Implementation [~10 marks]
| Component | Implementation Detail |
|-----------|----------------------|
| `random_centroid_init(X, k)` | Randomly select `k` unique points from `X` as initial centroids |
| `compute_distances(X, centroids)` | Vectorized Euclidean distance matrix: `(n, k)` |
| `assign_clusters(distances)` | Argmin across columns to assign each point to nearest centroid |
| `update_centroids(X, labels, k)` | Mean of points in each cluster; handle empty clusters by re-initialization |
| `fit(X, k, max_iter=300, tol=1e-4)` | Loop until convergence (centroid movement < `tol` or max iterations) |
| `compute_cost(X, labels, centroids)` | Sum of squared distances to assigned centroids (inertia) |

> **Constraint:** Use NumPy vectorization everywhere possible. Avoid Python loops for distance computation and assignment.

#### Part B — Adversarial Dataset Construction [~10 marks]
- **Dataset (a) — k-Means performs well:**
  - Generate 3-4 spherical Gaussian clusters with equal variance, balanced sizes, clear separation.
  - Verify with silhouette score and visualization.
  
- **Dataset (b) — k-Means fails:**
  - Generate non-spherical clusters (e.g., elongated/anisotropic Gaussians, concentric circles, or moons).
  - OR: clusters with vastly different densities/scales.
  - Show that despite visual separability, k-Means produces poor cluster boundaries.
  - **Analysis:** Explain violation of the spherical/equal-variance assumption. Test with and without feature scaling (StandardScaler) to see if outcome changes.

#### Part C — Initialization Sensitivity [~10 marks]
- Run k-Means **20 times** with different random initializations on the same dataset.
- Record:
  - Convergence cost at each iteration (for a few runs to plot curves)
  - Final clustering cost for all 20 runs
- **Outputs:**
  - Plot: Convergence curves (cost vs. iteration) for selected runs
  - Plot: Histogram of final costs across 20 runs
  - Analysis: Why different initializations lead to different local minima

---

### 3.2 Question 2: Gaussian Naive Bayes [30 Marks]

**File:** `q2_naive_bayes.py`

#### Part A — Core Implementation [~10 marks]
| Component | Implementation Detail |
|-----------|----------------------|
| `fit(X, y)` | Compute per-class priors `P(y)`, per-feature means `μ`, and variances `σ²` |
| `_compute_log_likelihood(x, mean, var)` | Log Gaussian PDF: `-0.5*log(2πσ²) - (x-μ)²/(2σ²)` |
| `predict(X)` | `log_prior + sum(log_likelihood)` per class; argmax for prediction |
| `predict_proba(X)` | Return normalized probabilities (with log-sum-exp trick for stability) |

> **Numerical Stability:** Use log-probabilities throughout. Apply `log-sum-exp` trick when converting back to probabilities to avoid underflow.

#### Part B — Correlated Features Experiment [~8 marks]
- Generate dataset:
  ```python
  X1 = np.random.normal(0, 1, n)
  X2 = X1 + np.random.normal(0, 0.1, n)   # highly correlated
  X3 = np.random.normal(0, 1, n)           # independent
  ```
- Add remaining features to meet `d ≥ 15` requirement.
- Train Gaussian NB, analyze:
  - Confidence scores (predicted probabilities)
  - Prediction errors (confusion matrix, error rate)
  - Miscalibration: compare predicted confidence vs. actual accuracy (reliability diagram)

#### Part C — Counterexample Challenge [~7 marks]
- **(a) NB performs surprisingly well:**
  - Create a dataset where features are correlated but classes are still linearly separable along the principal axis.
  - NB ignores covariance but still finds the right direction due to marginal distributions.
  
- **(b) NB completely fails:**
  - Create XOR-like or parity structure where each feature individually is uninformative but jointly they are perfectly predictive.
  - OR: strong anti-correlation that flips the decision boundary.
- **Analysis:** Justify success/failure based on independence assumption violation and decision boundary geometry.

#### Part D — Conceptual Analysis [~5 marks]
- **(a)** Why NB outperforms sophisticated models on small datasets:
  - Fewer parameters (linear in `d` vs. quadratic/exponential)
  - Less variance / overfitting despite high bias
  - "Bias-variance tradeoff" argument
- **(b)** Mathematically explain why correlated evidence leads to overconfident predictions:
  - Joint evidence `P(x₁, x₂|y)` treated as `P(x₁|y)·P(x₂|y)` → double-counting information
  - Variance of log-odds inflates; posterior collapses to 0 or 1
  - Show with covariance matrix intuition

---

### 3.3 Question 3: Decision Trees using C4.5 [40 Marks]

**File:** `q3_decision_tree.py`

#### Part A — C4.5 Implementation [~15 marks]

**Splitting Criteria (MUST use):**
| Metric | Formula |
|--------|---------|
| Entropy | `H(S) = -Σ pᵢ log₂(pᵢ)` |
| Information Gain | `IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) · H(Sᵥ)` |
| Split Information | `SI(S, A) = -Σ (|Sᵥ|/|S|) · log₂(|Sᵥ|/|S|)` |
| Gain Ratio | `GR(S, A) = IG(S, A) / SI(S, A)` |

> **Critical:** Use Gain Ratio for split selection. ID3-only (Information Gain only) will receive deductions.

**Implementation Structure:**
```
class DecisionTreeC45:
    - __init__(max_depth, min_samples_split)
    - _entropy(y)
    - _information_gain(X_col, y, threshold)
    - _split_information(X_col, threshold)
    - _gain_ratio(X_col, y, threshold)
    - _find_best_split(X, y)          # iterate features & thresholds, pick max GR
    - _build_tree(X, y, depth)        # recursive splitting
    - fit(X, y)
    - predict(X)
    - predict_proba(X)                # optional but useful
```

**Handling Continuous Features (C4.5 style):**
- For each feature, sort values and evaluate midpoints between consecutive distinct values as candidate thresholds.
- Split: `X_col ≤ threshold` vs `X_col > threshold` (binary tree).

**Stopping Criteria:**
1. All labels in node are the same (pure)
2. `max_depth` reached
3. `min_samples_split` not satisfied
4. No split yields positive Gain Ratio

#### Part B — Gain Ratio Analysis [~7 marks]
- On generated datasets, compare **Information Gain** vs **Gain Ratio** for candidate splits.
- Identify cases where IG prefers poor splits (e.g., features with many unique values / high cardinality).
- Explain why Gain Ratio penalizes such splits via `Split Information` in denominator.
- **Output:** Table or plot showing IG and GR values for top candidate splits.

#### Part C — Overfitting Investigation [~8 marks]
- Train trees with varying depths: `[2, 5, 10, 20, 50, None]`
- Record training accuracy and validation accuracy for each depth.
- **Outputs:**
  - Plot: Training vs. Validation accuracy vs. Tree Depth
  - Analysis:
    - Very small depth → high bias, underfitting
    - Very large depth → low bias, high variance, memorization
    - Optimal depth region

#### Part D — Greedy Splitting Counterexample [~5 marks]
- Construct a dataset where greedy splitting is suboptimal.
  - **Classic example:** XOR problem or parity problem.
  - **Alternative:** A dataset where the best single split is uninformative, but a combination of two splits is perfect.
- **Output:**
  - Draw/visualize the resulting greedy tree structure.
  - Explain why greedy choice fails (myopic: optimizes immediate purity, not global structure).
  - Describe what globally optimal splitting would require (e.g., oblique splits, or looking ahead).

#### Part E — Noise Sensitivity [~5 marks]
- Start with a clean, tree-friendly dataset.
- Introduce:
  - **Label noise:** Randomly flip 5-10% of labels.
  - **Outliers:** Inject extreme values in a subset of samples.
- Train trees on clean vs. noisy data.
- **Analysis:**
  - Compare tree structures (depth, number of nodes, splits).
  - Explain why trees are unstable learners (small data changes → completely different splits).
  - Connect to high variance and lack of smoothness in decision boundaries.

---

## 4. File Structure

```
A02_ML/
├── README.md                  # This file
├── dataset_generator.py       # All dataset generation functions (reproducible, seeded)
├── q1_kmeans.py               # k-Means implementation + experiments
├── q2_naive_bayes.py          # Gaussian NB implementation + experiments
├── q3_decision_tree.py        # C4.5 decision tree implementation + experiments
├── utils.py                   # Shared utilities (metrics, plotting helpers, etc.)
├── notebooks/
│   └── analysis.ipynb         # Optional: consolidated visualizations
├── results/
│   ├── q1_plots/              # Convergence curves, cluster visualizations
│   ├── q2_plots/              # Calibration plots, confusion matrices
│   └── q3_plots/              # Tree depth plots, tree structures
└── report/
    └── report.pdf             # Final PDF report
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Day 1)
- [ ] Set up project structure and `dataset_generator.py`
- [ ] Implement base datasets: low-noise, high-noise, high-dimensional
- [ ] Implement shared utilities in `utils.py`

### Phase 2: k-Means (Day 1-2)
- [ ] Implement vectorized k-Means from scratch
- [ ] Construct friendly & adversarial datasets
- [ ] Run initialization sensitivity experiment (20 runs)
- [ ] Generate all Q1 plots and analysis

### Phase 3: Gaussian Naive Bayes (Day 2-3)
- [ ] Implement Gaussian NB with log-probability stability
- [ ] Run correlated features experiment
- [ ] Construct counterexample datasets
- [ ] Write conceptual analysis (Part D)
- [ ] Generate all Q2 plots and analysis

### Phase 4: C4.5 Decision Tree (Day 3-5)
- [ ] Implement entropy, IG, split information, gain ratio
- [ ] Implement recursive binary tree with continuous splits
- [ ] Implement stopping criteria and tree prediction
- [ ] Run gain ratio analysis (IG vs GR comparison)
- [ ] Run overfitting investigation across depths
- [ ] Construct greedy counterexample dataset
- [ ] Run noise sensitivity experiment
- [ ] Generate all Q3 plots and analysis

### Phase 5: Report & Submission (Day 5-6)
- [ ] Compile all results, plots, and observations into report
- [ ] Ensure every dataset has mandatory analysis (assumptions, bias-variance, etc.)
- [ ] Verify `dataset_generator.py` reproducibility with seed
- [ ] Final review: no hardcoded outputs, no prohibited library usage

---

## 6. Key Technical Notes

### k-Means Vectorization
```python
# Distance matrix: (n_samples, n_clusters)
distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
labels = np.argmin(distances, axis=1)
```

### Naive Bayes Log-Sum-Exp
```python
# For numerical stability when converting log-probs back to probs
log_probs = log_prior + log_likelihood_sum  # shape: (n_samples, n_classes)
max_log = np.max(log_probs, axis=1, keepdims=True)
probs = np.exp(log_probs - max_log)
probs /= np.sum(probs, axis=1, keepdims=True)
```

### C4.5 Continuous Split Search
```python
# For each feature, sort and evaluate midpoints
sorted_vals = np.sort(np.unique(X_col))
thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
# Evaluate gain ratio for each threshold, pick best
```

---

## 7. Submission Checklist

- [ ] `report.pdf` with experimental setup, graphs, observations, failure analysis, implementation discussion
- [ ] Source code (`.py` or `.ipynb`)
- [ ] `dataset_generator.py`
- [ ] All algorithms implemented from scratch (no sklearn classifiers/clusterers/trees/NB)
- [ ] C4.5 uses **Gain Ratio**, not just ID3/Information Gain
- [ ] All datasets satisfy minimum size requirements (`n≥1000`, `d≥15`, etc.)
- [ ] At least one dataset has `d≥50`
- [ ] At least one dataset has `n≥5000`
- [ ] Reproducible with `seed = last_3_digits_of_roll_number`
- [ ] No hardcoded outputs
- [ ] All code is explainable (viva-ready)
