# Core Evidence 2: ML Workflow Diagram
**IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`  
**Version:** Final Submission — October 2026

---

## 1. End-to-End Machine Learning Workflow: ChefCast AI

The following narrative describes the complete ML workflow implemented in this project, following the structured, evidence-based process required by the IT3091 assessment rubric.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: PROBLEM FRAMING                     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Business Pain Point Identification                          │   │
│  │  ┌──────────────────┐      ┌──────────────────────────────┐ │   │
│  │  │  Perishable Food  │      │  Revenue Uncertainty         │ │   │
│  │  │  Waste (Overprep) │      │  (Cash-Flow Unpredictability) │ │   │
│  │  └──────────┬───────┘      └───────────────┬──────────────┘ │   │
│  │             └───────────────────────────────┘               │   │
│  │                           ↓                                  │   │
│  │         Dual-Lens Architecture Decision (DEC-02)             │   │
│  │  ┌────────────────────┐      ┌──────────────────────────┐   │   │
│  │  │  Lens 1: Revenue   │      │  Lens 2: Product Demand  │   │   │
│  │  │  Planning          │      │  Planning                │   │   │
│  │  │  (Daily LKR Total) │      │  (Daily Portions/Dish)   │   │   │
│  │  └────────────────────┘      └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 2: DATA ACQUISITION & AUDIT                  │
│                                                                     │
│  Input: pos_raw_dataset.csv (127,093 line-item transactions)        │
│                                                                     │
│  Quality Audit (src/data_pipeline.py):                             │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ Filter 1: Pre-commissioning records before 2024-03-27    │      │
│  │           → Removed 147 rows (system testing records)   │      │
│  │ Filter 2: Negative/zero quantity rows (voided bills)     │      │
│  │           → Removed 931 rows                            │      │
│  │ Filter 3: Negative/zero price rows (promotional lines)   │      │
│  │           → Removed 72 rows                             │      │
│  │ Filter 4: Extreme price/quantity anomalies               │      │
│  │           → Removed 52 rows (UnitPrice > 20,000 LKR)    │      │
│  │                                                          │      │
│  │ Result: 125,912 valid transactions (99.1% retained)      │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  Product Selection (DEC-03):                                        │
│  274 catalog items → 11 core fresh handmade items (47.0% revenue)  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 3: EXPLORATORY DATA ANALYSIS                 │
│                                                                     │
│  Notebooks: NB01 (Revenue EDA) | NB05 (Demand EDA)                 │
│                                                                     │
│  Key EDA Findings:                                                  │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ • Revenue Pareto: 11 items → 34.32M LKR (47.0%)          │      │
│  │ • Weekly Cyclicality: Fri/Sat/Sun surge 25%–40%           │      │
│  │ • Seasonal Peaks: Dec, Apr, Aug (festive calendar)        │      │
│  │ • Zero-demand Days: Poya holidays / POS downtime          │      │
│  │ • Intermittent Demand: DATE CAKE / WATALAPPAN (0–15/day)  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  Decision: WAPE over MAPE (DEC-08) — zero-safe aggregate metric    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│             STAGE 4: PREPROCESSING & FEATURE ENGINEERING            │
│                                                                     │
│  Notebooks: NB02 (Revenue Features) | NB06 (Demand Features)       │
│  Production: src/feature_engineering.py                            │
│                                                                     │
│  Temporal Aggregation:                                             │
│  Raw Transactions → Daily Revenue Series (895 days)                │
│  Raw Transactions → Daily Product Demand Panel (895 × 11)          │
│                                                                     │
│  Calendar Reindexing (DEC-05):                                     │
│  → Zero-fill missing calendar dates; preserve lag continuity       │
│                                                                     │
│  Leakage-Free Feature Engineering (DEC-06):                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ All features strictly past-only (shift=1 on all series)  │      │
│  │                                                          │      │
│  │ Calendar Features:    day_of_week, is_weekend, month ... │      │
│  │ Cyclical Encodings:   sin/cos(dow), sin/cos(month)       │      │
│  │ Autoregressive Lags:  lag_1, lag_2, lag_3, lag_7,       │      │
│  │                       lag_14, lag_21                     │      │
│  │ Rolling Statistics:   roll_mean_7/14/30, roll_std_7     │      │
│  │ Momentum Ratios:      ratio_lag1_to_roll7               │      │
│  │ Entity OHE (L2):      item_LUNCH, item_TEA, ... (11)    │      │
│  │                                                          │      │
│  │ Result: Lens 1 → 874 days × 30 features                 │      │
│  │         Lens 2 → 9,614 item-days × 41 features          │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│           STAGE 5: DATA SPLITTING — STRICT CHRONOLOGICAL           │
│                                                                     │
│  Decision: Whole-Date Chronological 70/15/15 Split (DEC-07)       │
│                                                                     │
│  ┌──────────────────┬────────────────┬──────────────────────┐      │
│  │  TRAINING (70%)  │  VALIDATION    │  TEST (15%)          │      │
│  │                  │  (15%)         │  ← UNTOUCHED         │      │
│  │  2024-03-27 to   │  2025-12-19 to │  2026-04-29 to       │      │
│  │  2025-12-18      │  2026-04-28    │  2026-09-07          │      │
│  │  (632 days)      │  (131 days)    │  (132 days)          │      │
│  │  Model Learning  │  Champion      │  Final Unbiased      │      │
│  │                  │  Selection     │  Evaluation          │      │
│  └──────────────────┴────────────────┴──────────────────────┘      │
│                                                                     │
│  ⚠️ CRITICAL: Test set evaluated EXACTLY ONCE per lens             │
│     on the validation-selected champion to prevent selection bias  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 6: MODEL TRAINING & EXPERIMENTATION           │
│                                                                     │
│  Notebooks: NB03 (Revenue Models) | NB07 (Demand Models)           │
│  Production: src/models.py, src/evaluate.py                        │
│                                                                     │
│  5 Candidate Architectures (DEC-09):                               │
│  ┌────────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Sensible Baseline  │  │  Linear Models   │  │  Tree Models  │  │
│  │  ─────────────────  │  │  ───────────────  │  │  ───────────  │  │
│  │  7-Day Seasonal     │  │  Ridge           │  │  Random       │  │
│  │  Naive              │  │  Regression      │  │  Forest       │  │
│  │  (ŷ = y_{t-7})      │  │  (α=10.0)        │  │  (n=150)      │  │
│  └────────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                     │
│  ┌──────────────────────┐  ┌───────────────────────────────────┐  │
│  │  Gradient Boosters   │  │  Stacking Ensemble                 │  │
│  │  ──────────────────   │  │  ─────────────────────────────── │  │
│  │  LightGBM (n=150)    │  │  Base: Ridge + RF + LightGBM      │  │
│  │  XGBoost (n=150)     │  │  Meta: Ridge (L2)                 │  │
│  │                      │  │  CV: TimeSeriesSplit(n_splits=3)  │  │
│  └──────────────────────┘  └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│             STAGE 7: VALIDATION & CHAMPION MODEL SELECTION          │
│                                                                     │
│  Notebooks: NB04 (Revenue Selection) | NB08 (Demand Selection)     │
│                                                                     │
│  Champion Selection Criterion: Lowest Validation WAPE              │
│                                                                     │
│  ┌────────────────────────────────────────────────┐               │
│  │  LENS 1 VALIDATION RESULTS (Ordered by Val WAPE)│               │
│  │  🥇 LightGBM:         30.33% ← SELECTED         │               │
│  │  🥈 Ridge Regression:  30.54%                   │               │
│  │  🥉 Stacking Ensemble: 31.32%                   │               │
│  │     XGBoost:           31.67%                   │               │
│  │     Random Forest:     32.17%                   │               │
│  │     Seasonal Naive:    46.22%                   │               │
│  └────────────────────────────────────────────────┘               │
│                                                                     │
│  ┌────────────────────────────────────────────────┐               │
│  │  LENS 2 VALIDATION RESULTS (Ordered by Val WAPE)│               │
│  │  🥇 Ridge Regression:  47.31% ← SELECTED        │               │
│  │  🥈 Stacking Ensemble: 48.06%                   │               │
│  │  🥉 Random Forest:     48.26%                   │               │
│  │     XGBoost:           50.92%                   │               │
│  │     LightGBM:          54.56%                   │               │
│  │     Seasonal Naive:    65.71%                   │               │
│  └────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 8: FINAL UNBIASED TEST EVALUATION                │
│                                                                     │
│  Notebook: NB09 — Ensemble Architecture & Comparative Benchmark    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  LENS 1 CHAMPION (LightGBM) — FINAL TEST RESULTS            │  │
│  │  Test WAPE: 24.99% | Test MAE: 9,109.36 LKR                 │  │
│  │  Test R²: 0.3521   | vs. Baseline: 43.15%                   │  │
│  │  Relative Error Reduction: 42.1% ✅ Exceeds 20% threshold   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  LENS 2 CHAMPION (Ridge Regression) — FINAL TEST RESULTS     │  │
│  │  Test WAPE: 44.38% | Test MAE: 5.95 portions                 │  │
│  │  Test R²: 0.2972   | vs. Baseline: 63.49%                   │  │
│  │  Relative Error Reduction: 30.1% ✅ Exceeds 20% threshold   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│            STAGE 9: DEPLOYMENT RETRAINING & PRODUCTIONIZATION       │
│                                                                     │
│  Production Code: src/evaluate.py, src/forecasting.py, app.py      │
│                                                                     │
│  • Champions retrained on Train + Validation (extended window)      │
│  • Models serialized to models/*.joblib artifacts                   │
│  • Recursive forward forecasting engine (src/forecasting.py)        │
│    → Feeds predicted values back as lag_1, lag_2 inputs            │
│    → Generates 7-day forward projections from Sep 8, 2026          │
│  • Interactive Streamlit Decision Support Dashboard (app.py)        │
│    → 6 operational tabs with real-time interactive controls         │
│  • Safety guardrails: ceiling at 1.3× rolling 30-day max           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 10: MONITORING & RETRAINING PROTOCOL            │
│                                                                     │
│  Recommended Production Operation Cycle:                           │
│  • Daily: Auto-ingest new POS transactions into rolling window     │
│  • Weekly: Retrain models on expanding window (+ 7 new days)       │
│  • Alert Trigger: Rolling 7-day WAPE > 35% (L1) or > 55% (L2)    │
│  • Manual Review: Any predicted demand > 1.3× monthly max         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Notebook-to-Stage Alignment

| Stage | Notebook | Author | Key Output |
| :--- | :--- | :--- | :--- |
| Problem Framing | Planning & Canvas | All | Decision Log, Problem Canvas |
| EDA (Lens 1) | NB01 | IT24102572 | Revenue EDA, anomaly findings |
| Feature Eng. (Lens 1) | NB02 | IT24102664 | 30-feature Lens 1 matrix |
| Modeling (Lens 1) | NB03 | IT24102726 | 5 candidate model experiments |
| Validation (Lens 1) | NB04 | IT24102913 | Champion: LightGBM |
| EDA (Lens 2) | NB05 | IT24102726 | Demand EDA, product selection |
| Feature Eng. (Lens 2) | NB06 | IT24102913 | 41-feature Lens 2 panel matrix |
| Modeling (Lens 2) | NB07 | IT24102572 | 5 candidate model experiments |
| Validation (Lens 2) | NB08 | IT24102664 | Champion: Ridge Regression |
| Benchmark Synthesis | NB09 | Joint | Cross-lens comparison & Ensemble |

---

## 3. Data Flow Summary

```
pos_raw_dataset.csv (127,093 rows)
         ↓ [data_pipeline.py: clean, filter, aggregate]
cleaned_pos_transactions.csv (125,912 rows)
         ↓ [group by date / group by date+item]
daily_revenue_series.csv         daily_product_demand_long.csv
(895 days × revenue)             (9,845 item-days × 11 dishes)
         ↓ [feature_engineering.py]
Lens 1 Feature Matrix            Lens 2 Feature Matrix
(874 days × 30 features)         (9,614 item-days × 41 features)
         ↓ [evaluate.py: train/val/test split + 5 models]
models/metrics.json              models/lens*.joblib (×12 files)
         ↓ [forecasting.py: recursive forward engine]
7-day forward forecast arrays
         ↓ [app.py: Streamlit UI]
Interactive Decision Support Dashboard (http://localhost:8501)
```
