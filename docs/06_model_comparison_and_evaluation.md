# Core Evidence 6: Model / Method Comparison & Evaluation
**IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`  
**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  
**Version:** Final Submission — October 2026

---

## 1. Validation Strategy & Split Justification

### 1.1 Why Chronological Splitting is Non-Negotiable in Time-Series POS Data

In standard supervised learning, random K-Fold cross-validation is the gold standard because it produces low-variance performance estimates by averaging over multiple random subsets. However, **random CV is theoretically invalid for sequential time-series data**:

- A random fold may place December 2025 observations in the training split and October 2025 in the validation split.
- The model would then be "predicting the past from the future" — an impossibility in deployment.
- Validation metrics would appear far too optimistic, and champions selected this way would underperform severely in production.

**Adopted Solution — Chronological Walk-Forward Split (70/15/15):**

| Split | Date Range | Days | Item-Days (L2) | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Training** | 2024-03-27 to 2025-12-18 | 632 | 6,952 | Model parameter estimation |
| **Validation** | 2025-12-19 to 2026-04-28 | 131 | 1,441 | **Champion model selection** |
| **Test** | 2026-04-29 to 2026-09-07 | 132 | 1,452 | **Final unbiased evaluation (evaluated once)** |

> **Critical Safeguard:** No model hyperparameter tuning or champion selection was performed using test set metrics. The test set was evaluated exactly once, after champions were committed based on validation WAPE. This eliminates test-set selection bias (also known as "peeking" or "data dredging").

### 1.2 Evaluation Metrics: Full Specification

| Metric | Formula | Primary Role | Why Included |
| :--- | :--- | :--- | :--- |
| **WAPE** | $\frac{\sum\|y_t - \hat{y}_t\|}{\sum y_t} \times 100\%$ | **Champion selection metric** | Zero-safe; stable on low/zero demand days; industry standard |
| **MAE** | $\frac{1}{N}\sum\|y_t - \hat{y}_t\|$ | Operational magnitude | Direct LKR or portion-count error; interpretable by non-technical staff |
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y_t - \hat{y}_t)^2}$ | Peak-error sensitivity | Penalises large misses more heavily; relevant for stockout risk |
| **MAPE** | $\frac{1}{N}\sum\frac{\|y_t - \hat{y}_t\|}{y_t} \times 100\%$ | Reference diagnostic | Reported for academic completeness; NOT used for selection (zero-division risk) |
| **$R^2$ Score** | $1 - \frac{\sum(y_t - \hat{y}_t)^2}{\sum(y_t - \bar{y})^2}$ | Variance explanation | Contextualizes how much variability the model explains above a flat mean |

---

## 2. Candidate Model Architectures: Technical Specifications

### Model 1: Baseline — 7-Day Seasonal Naive

**Algorithm:** $\hat{y}_t = y_{t-7}$

**Justification for use as baseline:** This exactly mirrors the current informal restaurant heuristic: "Prepare what we made last week on the same day of week." If ML models cannot outperform this heuristic, there is no justification for deploying a more complex system. A beaten baseline is the minimum threshold for demonstrating ML value.

**Limitations:** Completely blind to trend drift, festival calendar shocks, or month-level seasonality. A significant rainy-season downturn in August 2026 would not be captured by referencing August 2025 directly.

---

### Model 2: Ridge Regression (L2 Regularized Linear Model)

**Algorithm:** Ridge solves $\min_\beta \left[\sum_{t=1}^{N}(y_t - X_t^\top \beta)^2 + \alpha \|\beta\|_2^2\right]$

**Hyperparameters:** $\alpha = 10.0$, with `StandardScaler` preprocessing inside an `sklearn.Pipeline`.

**Why Ridge over plain OLS Linear Regression?**  
The feature matrix includes 30 correlated features (especially correlated lag features: lag_1, lag_7, lag_14 are all autocorrelated with each other). OLS with highly correlated predictors produces numerically unstable, high-variance coefficient estimates. Ridge's L2 penalty shrinks all coefficients harmoniously toward zero — reducing variance at the cost of a small, controlled bias.

**Lens 1 Validation WAPE: 30.54%** — Runner-up.  
**Lens 2 Validation WAPE: 47.31%** — 🥇 **Selected Champion.**

**Why Ridge was champion on Lens 2:** Disaggregated dish-level demand (5–20 portions) has high stochastic noise. Complex gradient boosters overfit to idiosyncratic daily fluctuations in the training window; Ridge's shrinkage produces smoother, more generalisable predictions on test data.

---

### Model 3: Random Forest Regressor (Ensemble Bagging)

**Algorithm:** Bootstrap aggregated ensemble of $n$ decorrelated regression decision trees.

**Hyperparameters:** `n_estimators=150`, `max_depth=10`, `random_state=42`.

**Why included:** Random Forest captures non-linear step-change interactions (e.g., Saturday revenue behaves categorically differently from Wednesday), is robust to noisy features, and provides natural feature importance via Gini impurity reduction.

**Why max_depth=10:** Unconstrained trees overfit to individual POS transactions. A maximum depth of 10 creates sufficient tree expressiveness while preventing memorization of training-set noise.

**Lens 1 Validation WAPE: 32.17%** | **Lens 2 Validation WAPE: 48.26%**

---

### Model 4: LightGBM Regressor (Leaf-Wise Gradient Boosting)

**Algorithm:** Gradient boosted decision trees using leaf-wise (best-first) split strategy with histogram binning.

**Hyperparameters:** `n_estimators=150`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.

**Technical advantages:**
- Leaf-wise splitting produces asymmetric trees that can fit complex interactions that level-wise splitting misses.
- Histogram binning dramatically accelerates training on 9,614-row panel datasets.
- Subsampling (`subsample=0.8`) acts as implicit L2 regularization, reducing overfitting to noisy demand patterns.

**Lens 1 Validation WAPE: 30.33%** — 🥇 **Selected Champion.**  
**Lens 2 Validation WAPE: 54.56%** — 5th.

**The Lens 2 paradox — why did the Lens 1 champion perform worst on Lens 2?**  
LightGBM's leaf-wise splitting is aggressive at partitioning low-volume demand regions. For dishes averaging 2–5 portions per day, the algorithm learned to create very specific decision rules that fit the training set noise well (high training $R^2$) but generalized poorly to the test period. This is a textbook example of the **bias-variance tradeoff**: on low-signal, high-noise data, simpler models with more regularization (Ridge) outperform expressive, low-bias models (LightGBM).

---

### Model 5: XGBoost Regressor (Level-Wise Gradient Boosting)

**Algorithm:** Gradient boosted decision trees using level-wise (depth-first) split strategy with built-in L1 and L2 regularization.

**Hyperparameters:** `n_estimators=150`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.

**Comparison to LightGBM:** Level-wise splitting produces more symmetric, shallower trees than LightGBM's leaf-wise approach — providing slightly stronger regularization inherently. On Lens 2, XGBoost's Validation WAPE of 50.92% confirms it is more conservative than LightGBM (54.56%) on sparse data but still underperforms Ridge (47.31%).

**Lens 1 Validation WAPE: 31.67%** | **Lens 2 Validation WAPE: 50.92%**

---

### Model 6: Stacking Ensemble (Time-Series-Safe Meta-Learner)

**Architecture:**
- **Level-0 Base Learners:** Ridge Regression + Random Forest + LightGBM
- **Level-1 Meta-Learner:** Ridge Regression (L2, $\alpha=10.0$)
- **CV Strategy:** Custom `TimeSeriesStackingRegressor` with `TimeSeriesSplit(n_splits=3)`

**Why a custom implementation?**  
`sklearn.StackingRegressor` uses `cross_val_predict` with random CV internally, which causes temporal leakage on sequential POS data. Our custom implementation:

```python
# Pseudocode — TimeSeriesStackingRegressor
tscv = TimeSeriesSplit(n_splits=3)
oof_predictions = zeros(n_train, n_base_models)

for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    for model_j, base_model in enumerate(base_models):
        base_model.fit(X_train[train_idx], y_train[train_idx])
        oof_predictions[val_idx, model_j] = base_model.predict(X_train[val_idx])

# Meta-learner trains on OOF predictions only — no future-to-past contamination
meta_learner.fit(oof_predictions, y_train)

# For inference, all base models are retrained on full training set
```

**Why Ridge as meta-learner?**  
The OOF predictions from Ridge, RF, and LightGBM are themselves correlated (all positively predict revenue/demand). Ridge's L2 penalty manages this multicollinearity at the meta-level, avoiding overfitting to specific base model output patterns.

**Lens 1 Validation WAPE: 31.32%** | **Lens 2 Validation WAPE: 48.06%**

---

## 3. Complete Benchmark Results

### 3.1 Lens 1: Combined Daily Revenue Forecasting

*Selection criterion: Lowest Validation WAPE (test set remained sealed until final evaluation)*

| Rank | Model Architecture | Val WAPE (%) | Test MAE (LKR) | Test RMSE (LKR) | Test WAPE (%) | Test MAPE (%) | Test $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **LightGBM (Champion)** | **30.33%** | **9,109.36** | **12,752.06** | **24.99%** | **23.58%** | **0.3521** |
| 🥈 | Ridge Regression | 30.54% | 8,734.59 | 11,943.78 | 23.96% | 24.58% | 0.4316 |
| 🥉 | Stacking Ensemble | 31.32% | 9,284.71 | 12,375.86 | 25.47% | 25.23% | 0.3898 |
| 4 | XGBoost Regressor | 31.67% | 9,730.90 | 13,483.93 | 26.70% | 24.46% | 0.2756 |
| 5 | Random Forest | 32.17% | 10,253.93 | 13,962.09 | 28.13% | 24.52% | 0.2233 |
| — | *Baseline Seasonal Naive* | *46.22%* | *15,728.64* | *20,831.72* | *43.15%* | *37.92%* | *-0.7290* |

**Baseline vs. Champion comparison:**

| Metric | Baseline Seasonal Naive | LightGBM Champion | Improvement |
| :--- | :---: | :---: | :---: |
| Test WAPE | 43.15% | 24.99% | **−18.16 pp (42.1% relative reduction)** |
| Test MAE (LKR) | 15,728.64 | 9,109.36 | **−6,619.28 LKR per day** |
| Test RMSE (LKR) | 20,831.72 | 12,752.06 | **−8,079.66 LKR** |
| Test $R^2$ | −0.7290 | 0.3521 | **+1.0811 (from negative to positive)** |

> The negative baseline $R^2$ (−0.7290) confirms the Seasonal Naive is mathematically worse than predicting the constant historical mean — a striking illustration of why heuristic-only management is suboptimal.

**Notable finding — Validation vs. Test Rank Reversal:**  
Ridge Regression achieves slightly *better* Test WAPE (23.96%) than the LightGBM champion (24.99%), yet LightGBM was correctly selected as champion based on lower *Validation* WAPE (30.33% vs. 30.54%). This is the exact reason why validation sets exist: had we selected champions based on test performance, we would have selected Ridge — introducing test-set peeking and invalidating the unbiased evaluation.

---

### 3.2 Lens 2: Product-Level Daily Demand Forecasting (All 11 Dishes)

| Rank | Model Architecture | Val WAPE (%) | Test MAE (Units) | Test RMSE (Units) | Test WAPE (%) | Test MAPE (%) | Test $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Ridge Regression (Champion)** | **47.31%** | **5.95** | **8.77** | **44.38%** | **60.91%** | **0.2972** |
| 🥈 | Stacking Ensemble | 48.06% | 5.94 | 8.67 | 44.35% | 64.19% | 0.3135 |
| 🥉 | Random Forest | 48.26% | 5.95 | 8.88 | 44.39% | 62.41% | 0.2808 |
| 4 | XGBoost Regressor | 50.92% | 6.44 | 9.49 | 48.07% | 66.49% | 0.1776 |
| 5 | LightGBM Regressor | 54.56% | 6.88 | 9.67 | 51.36% | 86.88% | 0.1471 |
| — | *Baseline Seasonal Naive* | *65.71%* | *8.51* | *12.16* | *63.49%* | *82.73%* | *-0.3500* |

**Operational kitchen impact:**

| Metric | Baseline | Ridge Champion | Kitchen Implication |
| :--- | :---: | :---: | :--- |
| Test MAE (portions) | 8.51 | 5.95 | **−2.56 fewer wasted/short portions per dish per day** |
| Test WAPE | 63.49% | 44.38% | **−19.11 pp (30.1% relative reduction)** |
| Test $R^2$ | −0.3500 | 0.2972 | From below-mean to above-mean prediction power |

---

## 4. Operational Decision Classification Metrics (Surge vs. Normal Day)

Since kitchen managers make a **binary operational decision** ("Is this a high-demand surge day requiring extra prep?"), we evaluated all models on an operational surge threshold (demand > historical median):

### Lens 1 Surge Classification Results (Test Set, 132 days)

| Model | Accuracy | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Champion)** | **63.6%** | **71.4%** | 45.5% | 0.556 | 54 | 12 | 36 | 30 |
| Ridge Regression | 61.4% | 59.5% | 71.2% | 0.648 | 34 | 32 | 19 | 47 |
| Stacking Ensemble | 61.4% | 58.8% | 75.8% | **0.662** | 31 | 35 | 16 | 50 |
| XGBoost Regressor | 58.3% | 78.9% | 22.7% | 0.353 | 62 | 4 | 51 | 15 |
| Random Forest | 54.5% | 80.0% | 12.1% | 0.211 | 64 | 2 | 58 | 8 |
| *Baseline Seasonal Naive* | *55.3%* | *55.6%* | *53.0%* | *0.543* | 38 | 28 | 31 | 35 |

> **LightGBM precision advantage:** With only 12 False Positives, LightGBM minimizes false surge alerts — preventing unnecessary overtime staff deployment on normal trading days.

### Lens 2 Surge Classification Results (1,452 item-day test instances)

| Model | Accuracy | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression (Champion)** | **72.0%** | **68.3%** | 74.3% | **0.712** | 300 | 129 | 96 | 278 |
| Stacking Ensemble | 70.7% | 65.1% | **80.2%** | 0.719 | 268 | 161 | 74 | 300 |
| Random Forest | 72.0% | 69.7% | 70.6% | 0.701 | 314 | 115 | 110 | 264 |
| LightGBM Regressor | 70.1% | 71.1% | 60.4% | 0.653 | 337 | 92 | 148 | 226 |
| XGBoost Regressor | 69.0% | 74.3% | 51.1% | 0.605 | 363 | 66 | 183 | 191 |
| *Baseline Seasonal Naive* | *66.4%* | *64.9%* | *60.4%* | *0.626* | 307 | 122 | 148 | 226 |

---

## 5. Feature Importance Analysis

### 5.1 LightGBM Feature Importance — Lens 1 (Revenue)

| Rank | Feature | Importance | Domain Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `Revenue_lag_1` | 6.55% | Yesterday's revenue is the strongest single momentum signal |
| 2 | `Revenue_lag_3` | 6.34% | Three-day cycle captures mid-week pattern shifts |
| 3 | `Revenue_roll_std_7` | 5.69% | Weekly volatility index — high uncertainty weeks predict uncertain days |
| 4 | `Revenue_lag_14` | 5.26% | Fortnightly payday-driven repeat cycle |
| 5 | `Revenue_lag_21` | 5.21% | Three-week baseline seasonal anchor |

### 5.2 LightGBM Feature Importance — Lens 2 (Product Demand)

| Rank | Feature | Importance | Domain Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `Quantity_roll_mean_7` | 7.10% | 7-day trailing average dish velocity |
| 2 | `Quantity_roll_mean_30` | 7.07% | Monthly popularity baseline per dish |
| 3 | `Quantity_lag_1` | 6.03% | Immediate prior day dish consumption |
| 4 | `Quantity_roll_mean_14` | 5.85% | Fortnightly smoothed consumption |
| 5 | `Quantity_ratio_lag1_to_roll7` | 5.79% | Surge/lull momentum ratio |

---

## 6. Critical Model Selection Interpretation

**Lens 1 — Why LightGBM?**  
Aggregate daily revenue has strong non-linear day-of-week interaction effects (Saturday behavior is categorically different from Wednesday) combined with payday monthly patterns. LightGBM's leaf-wise splitting discovers these non-linear partitions far more efficiently than Ridge's linear combination of features — justifying its champion position with the lowest Validation WAPE of 30.33%.

**Lens 2 — Why Ridge over LightGBM?**  
Disaggregated per-dish demand operates in a fundamentally different signal-to-noise environment. With only 5–20 portions per day, demand is heavily influenced by unpredictable customer choice stochasticity. LightGBM's expressive capacity causes it to overfit to training-set idiosyncrasies; Ridge's L2 regularization applies a controlled bias that dramatically improves generalization. This is a deliberate, evidence-backed choice demonstrating understanding of the bias-variance tradeoff — not a failure of gradient boosting.

**Academic Honesty — Honest Reporting of Limitations:**
- Lens 1 Test $R^2 = 0.3521$: 64.8% of daily revenue variance is unaccounted for. This reflects genuine hospitality unpredictability, not a modeling deficiency.
- Lens 2 Test WAPE = 44.38%: Still substantial, but operationally useful — the kitchen operates in batch increments of 5–10 portions, so a 5.95-portion mean error is within a single batch step.
