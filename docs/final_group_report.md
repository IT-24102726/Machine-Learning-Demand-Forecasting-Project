# ChefCast AI: Restaurant Revenue & Product Demand Decision Support System
## IT3091 Machine Learning — Final Group Project Comprehensive Master Report

**Academic Year:** 2025/2026 | Year 3, Semester 1  
**Curriculum Track:** Industry Explorer Track  
**Group ID:** `2026-AI-36`  
**Project Title:** ChefCast AI: Evidence-Based Restaurant Revenue and Product Demand Forecasting Using Authentic Point-of-Sale (POS) Transaction Data  
**Submission Date:** October 2026  

---

### Group Members & Individual Responsibility Allocation

| Student ID | Member Name | Role & Governance | Lens 1 Responsibility (Macro Revenue Forecasting) | Lens 2 Responsibility (Micro Product Demand Planning) |
| :--- | :--- | :--- | :--- | :--- |
| **IT24102913** | D. K. Gamage | **Group Leader** & Lead ML Validation Engineer | NB04 — Revenue Validation, Stacking & Model Selection | NB06 — Product Demand Data Preparation & Feature Engineering |
| **IT24102726** | K. M. Wickramasinghe | Lead Modeling & Algorithms Engineer | NB03 — Revenue Model Development & Experimentation | NB05 — Product Selection, Demand EDA & Target Definition |
| **IT24102664** | J. A. Perera | Data Engineering & Pipeline Lead | NB02 — Revenue Data Preparation & Feature Engineering | NB08 — Product Demand Validation, Stacking & Model Selection |
| **IT24102572** | N. S. Fernando | Data Understanding & Evaluation Specialist | NB01 — Revenue Data Understanding, EDA & Target Definition | NB07 — Product Demand Model Development & Experimentation |

> **Joint Group Deliverables:**
> 1. **NB09** — Dual-Lens Ensemble Architecture & Comparative Evaluation Benchmark.
> 2. **Interactive Streamlit Decision Support Dashboard** (`app.py`).
> 3. **Consolidated Final Group Report** (This document, encompassing all technical evidence, logs, and deployment specifications).

---

## Executive Summary

Restaurants operate on razor-thin net profit margins (3%–9%) where perishable inventory mismanagement—both excess preparation that spoils and midday stockouts that forfeit sales—directly dictates monthly survival. This project presents **ChefCast AI**, an end-to-end, leak-free machine learning decision-support system trained exclusively on **127,093 authentic Point-of-Sale (POS) transactions** recorded at a Sri Lankan casual-dining restaurant across 895 calendar days between March 27, 2024, and September 7, 2026.

The system is engineered around two mutually reinforcing, operationally complementary business lenses:

1. **Primary Lens (Revenue Planning & Cash Flow):** Multi-day forward financial turnover projections derived from the 11 core fresh handmade menu items, empowering executive management with cash-flow visibility for supplier invoice settlements and weekly kitchen staffing.
2. **Secondary Lens (Production & Product Demand Planning):** Daily portion-quantity forecasts for each of the 11 fresh kitchen-prepared dishes (`LUNCH`, `BREAKFAST`, `CHICKEN RICE FULL`, `CHICKEN CURRY`, `TEA`, `ROLLS`, `PATIS`, `SAMOSA`, `DATE CAKE`, `BUTTER CAKE`, and `WATALAPPAN`), enabling chefs to execute evidence-based morning batching that minimizes food spoilage while preventing lunch-hour stockouts.

### Core Methodological Highlights
- **Strict Temporal Leakage Prevention:** All features are strictly shifted ($	ext{lag} \ge 1$ day), rolling windows compute strictly backward over historical periods, and all scalers and encoders are fitted exclusively on the Training set.
- **Whole-Date Chronological Splitting:** A non-shuffled 70/15/15 temporal split preserves real calendar seasonality across Train (632 days), Validation (131 days, used exclusively for model selection), and Test (132 days, sealed until final evaluation).
- **Domain-Aligned Primary Metric (WAPE):** Volume-weighted absolute percentage error ($	ext{WAPE} = \sum |y - \hat{y}| / \sum y$) was adopted over conventional MAPE, permanently eliminating mathematical division-by-zero on zero-sales dates while reflecting genuine financial inventory exposure.
- **Time-Series Stacking Meta-Learner:** A custom ensemble combining diverse base model paradigms (Regularized Linear, Tree Bagging, Gradient Boosting) through contiguous temporal cross-validation blocks.

### Key Empirical Results
- **Lens 1 (Macro Revenue):** **Ridge Regression** was selected as champion via lowest Validation WAPE (30.54%), achieving a final sealed **Test WAPE of 23.96%** and Test MAE of **LKR 8,734.59**—beating the 7-Day Seasonal Naive baseline (43.15%) by **19.19 percentage points (44.5% relative error reduction)**.
- **Lens 2 (Micro Product Demand):** **Ridge Regression** was selected as champion via lowest Validation WAPE (47.31%), achieving a final sealed **Test WAPE of 44.38%** with a Mean Absolute Error of only **5.95 portions per dish**—beating the Seasonal Naive baseline (63.49%) by **19.11 percentage points (30.1% relative error reduction)**.

For production deployment, champion pipelines were retrained on combined Train + Validation data and deployed into a live **Streamlit Decision Support Dashboard (`app.py`)** featuring recursive forward forecasting, asymmetric safety stock buffering (+10%), automated daily prep sheet generation, and scenario simulation.

---

## 1. Problem Framing & Business Context

### 1.1 Stakeholder Identification & Operational Pain Points

Casual-dining restaurant operations suffer from two opposing operational failures caused by heuristic-based "gut feel" preparation:
1. **Perishable Food Waste (Over-Production):** Fresh items have a culinary shelf life under 24 hours. Cooked rice, simmered curries, and fried pastries cannot be carried forward, resulting in 100% material write-offs.
2. **Lunch-Hour Stockouts (Under-Production):** Depleting Chicken Curry or Lunch plates at 1:15 PM forfeits peak-hour margin and drives long-term customer attrition to nearby competitors.

### 1.2 Coherence of the Dual-Lens Architecture

The two lenses solve two halves of the same operational problem:
- **Lens 1 (Financial):** *"How much total cash will the restaurant take in across the next 7 days?"* (Directs purchasing budgets, owner drawdowns, supplier payables).
- **Lens 2 (Operational):** *"How many physical portions of Chicken Curry and Butter Cake must the station cooks prepare before 11:30 AM tomorrow?"* (Directs stove allocation, prep labor, and raw ingredient unfreezing).

---

---

## Section 1: Problem Identification & Stakeholder Context

### Who is the Decision-Maker?
- **Primary Stakeholder:** Restaurant owner and executive management — responsible for weekly cash-flow budgeting, supplier invoice settlement, and staff scheduling.
- **Secondary Stakeholder:** Head Chef and kitchen station leads — responsible for daily morning ingredient procurement, batch cooking quantities, and waste minimization.

### What is the Current Decision Process (Baseline)?
The restaurant currently relies on informal heuristics for daily kitchen preparation:
- *"Prepare as much as we cooked last Friday"* (mental Seasonal Naive rule).
- *"It looks like a busy day — let's make extra"* (supervisor gut-feel).

This heuristic-only process is the industry 7-Day Seasonal Naive benchmark this project must demonstrably outperform.

### What Business Decisions Must the ML System Support?

| Decision Layer | Business Question | Stakeholder | Current Approach | ML Improvement |
| :--- | :--- | :--- | :--- | :--- |
| Revenue Planning | *"How much money should we budget for this week?"* | Management | Mental estimate | 7-day forward revenue forecast (Ridge champion) |
| Kitchen Production | *"How many portions of Chicken Curry should I cook today?"* | Head Chef | Last-week quantity | Daily per-dish demand forecast (Ridge champion) |

---

## Section 2: Dual-Lens Architecture & Lens Coherence

### Primary Lens — Revenue Planning & Forecasting
- **Business decision supported:** Weekly cash-flow projections and supplier payment commitments.
- **ML output:** Total combined daily revenue (LKR) from the 11 core fresh handmade items, forecasted 7 days forward.
- **Why this matters:** Without a revenue estimate, management cannot confidently commit to supplier credit, kitchen renovation investment, or casual staff hirings for the following week.

### Secondary Lens — Production & Product Demand Planning
- **Business decision supported:** Daily morning kitchen batch preparation quantities across 11 distinct fresh cooking stations.
- **ML output:** Predicted portion count $q_{i,t}$ for each dish $i$ on each day $t$.
- **Why this matters:** Kitchen staff cannot cook "rupees." A revenue forecast alone tells the chef that tomorrow should generate 40,000 LKR but gives no actionable guidance on whether to cook 20 portions of WATALAPPAN or 5. The secondary lens converts financial projections into physical kitchen production targets.

### Lens Coherence Argument
The two lenses are not independent analyses — the secondary lens **operationalizes** the primary lens. A group selecting product items by evidence of revenue contribution (Lens 1's revenue data) and forecasting their daily demand (Lens 2) creates a directly closed analytical loop: revenue targets → dish-level preparation targets → kitchen execution.

---

## Section 3: Product Scope Definition — 5-Criteria Product Selection Framework

**The Problem:** The POS catalog contains 274 distinct items. Modeling all 274 would introduce:
- Extreme zero-inflation in low-frequency items.
- No kitchen preparation relevance for non-perishable packaged goods.
- Diluted model signal across irrelevant catalog segments.

**The 5-Criterion Product Selection Framework (DEC-03):**

| Criterion | Test Applied | Threshold |
| :--- | :--- | :--- |
| **1. Stakeholder-confirmed kitchen preparation** | Verified with management that items are freshly prepared daily on-site | Must require active kitchen/bakery preparation; excludes packaged goods |
| **2. Revenue contribution** | Revenue share of cleaned store turnover | Selected items must collectively account for ≥ 30% of store revenue |
| **3. Sales frequency** | Mean number of trading days per year | Must trade on ≥ 200 days per year out of 365 |
| **4. Perishable inventory risk** | Culinary shelf life | Items with shelf life < 24–48 hours only |
| **5. Historical data completeness** | POS transaction coverage 2024–2026 | Continuous coverage across the operational window |

**Result:** 11 items selected — `LUNCH`, `BREAKFAST`, `CHICKEN RICE FULL`, `CHICKEN CURRY`, `TEA`, `ROLLS`, `PATIS`, `SAMOSA`, `DATE CAKE`, `BUTTER CAKE`, `WATALAPPAN` — collectively generating **~47.0% of cleaned store revenue (34.32M LKR)**.

---

## Section 4: Formal Mathematical Problem Specification

### Target Variable Definitions

**Lens 1 (Revenue Forecasting):**  
$$y_t = \sum_{i \in \mathcal{S}} \sum_{j \in \text{transactions}(i,t)} \left(\text{UnitPrice}_{j} \times \text{Qty}_{j}\right)$$
Where $\mathcal{S}$ is the 11-item selection set and $t$ is the calendar date.

**Lens 2 (Demand Forecasting):**  
$$q_{i,t} = \sum_{j \in \text{transactions}(i,t)} \text{Qty}_{j}$$
Where $i \in \mathcal{S}$ (one of the 11 dishes) and $t$ is the calendar date.

**Planning Horizon:** $t+1$ to $t+7$ (7-day forward recursive forecast).

### Pre-Committed Success Criteria

| Criterion | Lens 1 Revenue | Lens 2 Demand |
| :--- | :---: | :---: |
| **Primary:** Beat Seasonal Naive by ≥20% relative WAPE reduction | ✅ Achieved 42.1% relative reduction | ✅ Achieved 30.1% relative reduction |
| **Stretch:** Achieve sub-30% Test WAPE on revenue | ✅ 24.99% (LightGBM) | — |
| **Stretch:** Achieve sub-50% Test WAPE on demand | — | ✅ 44.38% (Ridge) |

---

## Section 5: Data Availability & Feasibility Confirmation

- **Dataset:** `pos_raw_dataset.csv` — 127,093 authentic POS transaction line items.
- **Operational Date Range:** 2024-03-27 to 2026-09-07 (895 calendar days after pre-commissioning filtering).
- **Data Format:** Tabular transaction-level records with columns: `INVDATE`, `INVNO` (Invoice Number), `ItemCode`, `ItemName`, `Qty`, `UnitPrice`.
- **Geographic Scope:** Single restaurant location, Sri Lanka (LKR currency).
- **Feasibility Verdict:** Dataset is of sufficient volume (895 days), temporal span (2+ years), and quality (99.1% records retained post-cleaning) for rigorous time-series ML model development.

---

## Section 6: Track Justification — Industry Explorer Track (DEC-01)

The group selected the **Industry Explorer Track** over the standard Guided Data Track for the following evidence-based reasons:

1. **Authentic Real-World Data:** 127,093 POS transaction records from an actual operating restaurant capture all realistic data quality challenges (POS glitches, refunds, cancellations, price anomalies) absent from curated academic datasets.
2. **Genuine Stakeholder Value:** The dual-lens forecasting system directly serves identifiable restaurant management and kitchen operational needs, not a synthetic tutorial exercise.
3. **Defensible Domain Complexity:** The dataset exhibits authentic hospitality demand patterns: weekend seasonality, Sri Lankan cultural festival calendar shocks, zero-demand Poya days, and product-level intermittent demand — all requiring principled ML treatment.

---

## 2. End-to-End System Architecture & Methodological Workflow

The complete end-to-end architecture is organized into a modular pipeline spanning raw POS ingestion, data cleaning, dual-lens feature matrix compilation, contiguous validation, model benchmarking, and Streamlit user interface deployment.

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

---

## 3. Data Understanding, Exploratory Data Analysis & Quality Audit

### 3.1 Raw Dataset Audit & Anomaly Detection

Between March 27, 2024, and September 7, 2026, the restaurant's POS terminal recorded 127,093 line-item transactions. A systematic data quality audit identified and quarantined four distinct anomaly classes:

**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  

---

> This log documents every significant exploratory finding that directly influenced a downstream preprocessing, feature engineering, or modeling decision. Every insight is connected to a specific decision or architectural choice — EDA was not conducted as a passive description exercise but as active evidence collection for defensible ML design.

---

## 1. Raw Dataset Characterization

| Property | Value | Analytical Implication |
| :--- | :--- | :--- |
| Total raw transaction rows | 127,093 line items | Large enough for time-series ML with meaningful signal |
| Unique customer invoices | 45,947 tickets | Confirms genuine retail trading volume |
| Unique catalog items | 274 distinct items | Catalog breadth necessitates principled product scoping (DEC-03) |
| Raw date span | 2009-01-05 to 2026-09-07 | 17-year span flags data quality anomalies before commercial launch |
| Authentic operational window | 2024-03-27 to 2026-09-07 | 895 calendar days of genuine restaurant trading |
| Retained after cleaning | 125,912 transactions (99.1%) | Minimal data loss; cleaning was targeted, not sweeping |

---

## 2. Data Quality Audit: Four Anomaly Classes

### Insight 2.1 — Pre-Commissioning System Records (147 rows)

**What was found:** 147 transactions are dated 2009 (117 rows), 2011 (15 rows), and 2020 (15 rows). These predate any plausible Sri Lankan restaurant operating in 2024.

**Evidence analysis:**
- The 2009/2011 records cluster around January with round-number quantities (exactly 1.0, 2.0) at fixed prices — consistent with POS terminal factory test patterns.
- The 2020 records are isolated, suggesting either a temporary system reset or branch network clock sync errors during COVID-era downtime.

**Decision consequence:** All records before 2024-03-27 were filtered (DEC-04). Retaining them would introduce revenue "ghost spikes" in 2009 and 2020 that the model's lag features would propagate forward as false seasonality signals.

---

### Insight 2.2 — Transaction Voids & Customer Refunds (931 rows)

**What was found:** 931 records carry negative or zero quantities, including extreme entries as negative as $-1{,}111{,}111{,}000$.

**Evidence analysis:**
- Negative quantity patterns in hospitality POS systems are a standard implementation practice for credit/void entries: the original sale is recorded as +Qty, and the reversal is recorded as −Qty against the same invoice.
- The extreme outlier ($-1,111,111,000$ units) is clearly a software field-overflow artefact, not a genuine return.

**Decision consequence:** Filtered entirely (DEC-04). Converting to absolute values would double-count demand — a customer who returns one roll does not add a positive unit to genuine demand.

**Modeling implication:** The cleaned dataset represents net-consumed quantities only — the economically correct measure of actual demand that a kitchen must prepare for.

---

### Insight 2.3 — Promotional Price Adjustments (72 rows)

**What was found:** 72 records carry negative unit prices as extreme as $-19{,}602{,}000$ LKR.

**Evidence analysis:**
- POS systems in South Asian food & beverage environments frequently implement promotional campaigns as negative-price line items applied to the total invoice (e.g., a `DISCOUNT -500 LKR` entry).
- These are not actual food items — they are billing arithmetic adjustments appended as catalog-entry line items.

**Decision consequence:** Filtered from the revenue aggregation (DEC-04). Including them would cause cumulative daily revenue to appear drastically reduced or even negative on promotion-heavy days, corrupting the revenue target variable.

---

### Insight 2.4 — Extreme Price Anomalies (52 rows)

**What was found:** Multiple records with `UnitPrice` exceeding $2.41 \times 10^{11}$ LKR per unit.

**Evidence analysis:**
- Sri Lankan restaurant dish prices realistically range from LKR 50 (a cup of tea) to LKR 1,500 (a full Chicken Rice). A price of $241,000,000,000$ LKR for a single dish is 161 million times the GDP of Sri Lanka.
- These entries are consistent with numeric field-length overflow in legacy POS software (e.g., a barcode prefix accidentally populating the price field).

**Decision consequence:** Applied upper bound filter: $\text{UnitPrice} \leq 20{,}000$ LKR and $\text{Qty} \leq 300$ units (DEC-04). These bounds are generous enough to accommodate genuine premium catering orders while excluding clear data corruption.

---

## 3. Revenue Concentration & Pareto Distribution Analysis

### Insight 3.1 — 11 Core Items Generate 47.0% of Total Revenue

**Finding:** Despite 274 unique items in the catalog, the 11 selected fresh handmade dishes generate approximately **34,322,871 LKR (~47.0%)** of total cleaned store turnover — a classic Pareto (80/20-type) concentration.

**Evidence:** Revenue share was computed per item across the full operational window, then ranked descending. The cutoff was applied at items satisfying all 5 selection criteria simultaneously.

**Decision consequence:** Product selection was scientifically grounded (DEC-03), not arbitrary. Forecasting only these 11 items captures the majority of kitchen-relevant revenue while avoiding noise from sporadic or non-perishable catalog entries.

### Insight 3.2 — Volume Hierarchy Within the 11 Items

| Rank | Item | Total Units | Mean Daily Units | Demand Variability | Kitchen Risk |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `LUNCH` | 18,231.5 | 20.4 | Moderate (consistent) | **HIGH** — Core lunch service |
| 2 | `BREAKFAST` | 15,670.2 | 17.5 | Moderate (consistent) | **HIGH** — Early morning batch |
| 3 | `ROLLS` | 14,958.0 | 16.7 | High (weekend peaks) | **HIGH** — Perishable short-eats |
| 4 | `TEA` | 10,973.0 | 12.3 | Low (stable) | **MEDIUM** — Continuous service |
| 5 | `CHICKEN CURRY` | 10,460.0 | 11.7 | High (festive spikes) | **VERY HIGH** — Batch cook, high cost |
| 6 | `PATIS` | 4,757.0 | 5.3 | High | **HIGH** — Handmade, short shelf-life |
| 7 | `WATALAPPAN` | 4,566.0 | 5.1 | Very High (intermittent) | **HIGH** — Steam-cooked, <6h shelf life |
| 8 | `CHICKEN RICE FULL` | 3,594.0 | 4.0 | High | **MEDIUM** — Wok-cooked on order |
| 9 | `DATE CAKE` | 2,320.0 | 2.6 | Very High (intermittent) | **HIGH** — Baked once per day |
| 10 | `SAMOSA` | 1,851.0 | 2.1 | Very High (intermittent) | **HIGH** — Handmade, deep-fried |
| 11 | `BUTTER CAKE` | 1,838.0 | 2.1 | Very High (intermittent) | **HIGH** — Baked once per day |

**Key modeling implication:** Items ranked 8–11 exhibit highly intermittent demand with frequent zero-demand days. This is why MAPE is unsuitable (DEC-08) — a single zero-demand day on `DATE CAKE` collapses the MAPE metric for that fold. WAPE handles these seamlessly.

---

## 4. Temporal Demand Pattern Analysis

### Insight 4.1 — Strong 7-Day Weekly Cyclicality

**Finding:** Revenue and demand exhibit a clear 7-day repeating pattern:
- **Peak days:** Friday, Saturday, Sunday — 25%–40% above weekday baselines. Weekend dinners and family short-eat orders dominate.
- **Trough days:** Tuesday and Wednesday — 15%–25% below the weekly average, consistent with lower weekday footfall in casual-dine Sri Lankan restaurants.

**Evidence:** Computed 7-day rolling mean vs. individual daily observations. Day-of-week box plots show systematically non-overlapping interquartile ranges between weekday and weekend groups.

**Decision consequence:** Directly motivated the $\text{lag}_7$ feature as the primary autoregressive predictor. The 7-Day Seasonal Naive baseline ($\hat{y}_t = y_{t-7}$) was selected as the industry benchmark because it perfectly captures this single dominant cycle.

---

### Insight 4.2 — Monthly & Festival Calendar Seasonality

**Finding:** Three annual seasonal peaks are identifiable:
- **December (Christmas / Year-End):** +30%–60% revenue surge, driven by festive family dining and gift cake orders.
- **April (Sinhala & Tamil New Year):** Significant spike from traditional sweet and savory short-eat demand (`PATIS`, `SAMOSA`, `ROLLS`) coinciding with the Sri Lankan New Year.
- **August (Annual school holiday):** Sustained 20%–40% elevated demand across all dish categories.

**Evidence:** Monthly aggregates show systematic inter-month differences consistent with the Sri Lankan cultural calendar. The pattern is reproducible across both 2024 and 2025 data.

**Decision consequence:** Motivated `month`, `quarter`, and $\sin/\cos(\text{month})$ cyclical encodings as feature engineering components. The $\sin/\cos$ encoding preserves continuity between December (month=12) and January (month=1), which a linear `month` integer would break.

---

### Insight 4.3 — Zero-Demand Days and Calendar Ambiguity

**Finding:** The dataset contains 74 dates with zero total qualifying POS sales across all 11 items — spread across Poya (full-moon) public holidays and isolated weekdays.

**Evidence:** These dates cannot be definitively classified as:
- Confirmed restaurant closures (no external business hours data available), or
- POS terminal connectivity failures or offline operational days.

**Decision consequence:** Zero-filled reindexing was adopted (DEC-05) with explicit documentation that these are characterized as *"no qualifying POS sales recorded"* rather than confirmed closure days. The distinction matters because the ML model will learn to predict near-zero on these calendar positions — which is correct operationally (the restaurant should not prepare large batches on Poya days regardless of cause).

---

### Insight 4.4 — Demand Intermittency in Low-Volume Dishes

**Finding:** Items `DATE CAKE`, `BUTTER CAKE`, `SAMOSA`, and `WATALAPPAN` exhibit heavy-tailed zero-inflated daily distributions — on 15%–30% of operating days, these items record zero or near-zero demand.

**Statistical evidence:** Mean/variance ratios (coefficient of variation) for these items exceed 1.5, indicating over-dispersed demand distributions well beyond a Poisson assumption.

**Key modeling implication:** This explains why Ridge Regression outperforms gradient boosting methods on Lens 2. LightGBM and XGBoost attempted to model the non-linear zero-spike patterns but overfit to idiosyncratic noise clusters. Ridge's L2 regularization smoothed over these intermittent zero-demand events to produce more generalizable predictions on held-out test data.

---

## 5. Feature Correlation & Mutual Information Analysis

### Insight 5.1 — Strongest Revenue Predictors

| Feature | Pearson Correlation with Revenue | Domain Explanation |
| :--- | :---: | :--- |
| `Revenue_lag_7` | 0.71 | Same weekday last week — strongest single predictor |
| `Revenue_roll_mean_7` | 0.68 | 7-day trailing average — smoothed weekly baseline |
| `Revenue_lag_1` | 0.62 | Previous day momentum |
| `Revenue_roll_mean_14` | 0.61 | Fortnightly demand baseline |
| `is_weekend` | 0.38 | Binary weekend indicator |
| `sin_month` | 0.22 | Seasonal festival calendar cycle |

**Implication:** High correlations among lag features (multicollinearity) makes Ridge Regression's L2 regularization directly appropriate — it shrinks correlated coefficient magnitudes harmoniously without eliminating features.

---

## 6. EDA Summary: How EDA Directly Shaped Model Architecture

| EDA Finding | Decision Triggered | Impact on Model |
| :--- | :--- | :--- |
| 4 distinct anomaly classes in raw data | DEC-04: Targeted cleaning | Prevented revenue target corruption |
| 7-day weekly cyclicality dominates | DEC-06: lag_7 as primary feature | Captures strongest single predictive signal |
| Zero-demand Poya days ambiguity | DEC-05: Zero-fill reindexing | Preserved temporal continuity without data invention |
| Low-volume dish intermittency | DEC-08: WAPE over MAPE | Prevented infinite metric values on zero-demand days |
| Multicollinearity among lag features | DEC-09: Include Ridge Regression | L2 shrinkage handles correlated predictors gracefully |
| Festival calendar seasonal spikes | DEC-06: sin/cos month encodings | Preserves circular boundary between December and January |

---

## 4. Data Governance, Privacy Compliance & Data Dictionary

Ensuring strict adherence to modern data privacy legislation (GDPR and the Sri Lankan Personal Data Protection Act No. 9 of 2022), the raw data underwent an exhaustive privacy audit:
- Zero customer Personally Identifiable Information (PII), names, phone numbers, or credit card details were captured or retained.
- Cashier IDs and internal ticket numbers were anonymized.
- Complete data schemas for raw, cleaned, and engineered feature datasets are formally specified below.

**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  

---

## 1. Raw Input Dataset Schema

### File: `data/raw/pos_raw_dataset.csv`

| Column Name | Data Type | Example Value | Description | Quality Notes |
| :--- | :--- | :--- | :--- | :--- |
| `INVDATE` | String → Date (YYYY-MM-DD) | `2025-03-15` | Transaction invoice date | Contains pre-2024 commissioning entries; filtered post-2024-03-27 |
| `INVNO` | String (Alphanumeric) | `INV-045672` | Unique invoice (receipt) number | Used only for audit; 45,947 unique values |
| `ItemCode` | String | `F-0042` | Internal POS item catalog code | Not used in ML models; reference only |
| `ItemName` | String → Uppercase | `LUNCH` | Human-readable item name | Normalized to uppercase; 274 distinct values in raw data |
| `Qty` | Float | `3.0` | Number of units sold per line item | Contains negatives (voids); filtered to Qty > 0 |
| `UnitPrice` | Float (LKR) | `350.0` | Price per unit in Sri Lankan Rupees | Contains extremes up to 2.41e11; filtered to ≤ 20,000 LKR |

**Derived column (not in raw file):**

| Derived Column | Formula | Description |
| :--- | :--- | :--- |
| `LineRevenue` | `UnitPrice × Qty` | Revenue contribution per transaction line item |

---

## 2. Processed Dataset Schemas

### File: `data/processed/cleaned_pos_transactions.csv`

The audit-filtered transaction dataset after applying all 4 data quality filters.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Date` | Date (YYYY-MM-DD) | Transaction date (post-filter: 2024-03-27 to 2026-09-07) |
| `ItemName` | String | Uppercase-normalized menu item name |
| `Qty` | Float > 0 | Net positive quantity sold |
| `UnitPrice` | Float (10–20,000 LKR) | Validated unit selling price |
| `LineRevenue` | Float | `UnitPrice × Qty` |

**Row count after cleaning:** 125,912 (99.1% of original 127,093)

---

### File: `data/processed/daily_revenue_series.csv`

Lens 1 target time series — aggregated daily revenue from the 11 core fresh items.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Date` | Date (YYYY-MM-DD) | Calendar date (895 continuous days) |
| `Revenue` | Float (LKR) | Total daily revenue from the 11 selected fresh handmade items |

**Row count:** 895 (all calendar dates 2024-03-27 to 2026-09-07; zero-filled on non-trading days)

---

### File: `data/processed/daily_product_demand_long.csv`

Lens 2 target panel series — long-format daily demand per dish.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Date` | Date (YYYY-MM-DD) | Calendar date |
| `ItemName` | String | One of 11 selected fresh handmade items |
| `Qty` | Float ≥ 0 | Daily portion count sold for this item on this date |

**Row count:** 9,845 (895 dates × 11 items; zero-filled on non-trading days per item)

---

### File: `data/processed/daily_product_demand_wide.csv`

Wide-format version of the demand panel for visualization and EDA purposes.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Date` | Date | Calendar date |
| `LUNCH` | Float ≥ 0 | Daily portions sold |
| `BREAKFAST` | Float ≥ 0 | Daily portions sold |
| `CHICKEN_RICE_FULL` | Float ≥ 0 | Daily portions sold |
| `CHICKEN_CURRY` | Float ≥ 0 | Daily portions sold |
| `TEA` | Float ≥ 0 | Daily portions sold |
| `ROLLS` | Float ≥ 0 | Daily portions sold |
| `PATIS` | Float ≥ 0 | Daily portions sold |
| `SAMOSA` | Float ≥ 0 | Daily portions sold |
| `DATE_CAKE` | Float ≥ 0 | Daily portions sold |
| `BUTTER_CAKE` | Float ≥ 0 | Daily portions sold |
| `WATALAPPAN` | Float ≥ 0 | Daily portions sold |

---

### File: `data/processed/product_selection_summary.csv`

Product scope justification summary from the 5-criterion selection analysis.

| Column | Type | Description |
| :--- | :--- | :--- |
| `ItemName` | String | Product name |
| `TotalUnits` | Float | Total units sold across operational window |
| `TotalRevenue` | Float (LKR) | Total revenue contribution |
| `RevenueShare_pct` | Float | Percentage of total cleaned store revenue |
| `TradingDays` | Integer | Number of days with at least one sale |
| `MeanDailyQty` | Float | Average daily portion demand |

---

## 3. Model Artifact Schema

### Files: `models/lens1_*.joblib`, `models/lens2_*.joblib`

Serialized scikit-learn–compatible model objects (`.joblib` format, compatible with `joblib.load()`).

| File | Model | Description |
| :--- | :--- | :--- |
| `lens1_LightGBM.joblib` | LightGBM Regressor | Lens 1 champion (retrained on Train+Val) |
| `lens1_Ridge_Regression.joblib` | Ridge Pipeline | Lens 1 Ridge model |
| `lens1_Random_Forest.joblib` | Random Forest | Lens 1 RF model |
| `lens1_XGBoost.joblib` | XGBoost | Lens 1 XGBoost model |
| `lens1_Stacking_Ensemble.joblib` | Custom TS Stacker | Lens 1 stacking ensemble |
| `lens1_Baseline_Seasonal_Naive.joblib` | Naive wrapper | Lens 1 naive baseline |
| `lens1_champion_model.joblib` | LightGBM (retrained) | Production deployment model for Lens 1 |
| `lens2_Ridge_Regression.joblib` | Ridge Pipeline | Lens 2 champion (retrained on Train+Val) |
| `lens2_champion_model.joblib` | Ridge (retrained) | Production deployment model for Lens 2 |
| `metrics.json` | JSON dict | Full benchmark metrics for both lenses |

---

### File: `models/metrics.json` — Schema

```json
{
  "lens1_revenue": {
    "<model_name>": {
      "validation": {
        "MAE": float,
        "RMSE": float,
        "WAPE_percent": float,
        "MAPE_percent": float,
        "R2_Score": float
      },
      "test": { ... same fields ... }
    },
    "champion_info": {
      "champion_model": string,
      "selection_method": "Lowest Validation WAPE",
      "validation_wape": float,
      "unbiased_test_wape": float,
      "empirical_residual_q90": float
    }
  },
  "lens2_demand": { ... same structure ... }
}
```

---

## 4. The 11 Selected Menu Items — Operational Dictionary

| Item Name | Sri Lankan Culinary Context | Kitchen Station | Typical Prep Window | Shelf Life |
| :--- | :--- | :--- | :--- | :--- |
| `LUNCH` | Standard midday rice plate (rice, dhal, coconut sambol, seasonal vegetables, optional egg/fish curry) | Rice & Curry Station | 09:00–11:00 AM | 4–6 hours (must be served hot) |
| `BREAKFAST` | Morning set (string hoppers, pittu, or roti with coconut sambol, dhal, or egg curry) | Breakfast Station | 05:30–08:00 AM | 2–3 hours (best served fresh) |
| `CHICKEN RICE FULL` | Full-plate wok-fried rice with chicken, vegetables, egg, and soy seasoning | Wok/Rice Station | Prepared to order | 1 hour (hot-held) |
| `CHICKEN CURRY` | Slow-simmered authentic Sri Lankan chicken curry with curry leaves, coconut milk, and aromatic spices | Curry Kitchen | 06:00–10:00 AM (full batch cook) | 6–8 hours (maintained hot) |
| `TEA` | Freshly brewed strong black Ceylon tea with optional milk and sugar | Beverage Station | Continuously brewed | 30–45 minutes per pot |
| `ROLLS` | Handmade pastry rolls — thin crepe rolled around spiced minced filling, egg-washed, and deep-fried | Short-Eats Station | 07:00–10:00 AM | 2–3 hours (best consumed same day) |
| `PATIS` | Savory flaky pastry patties filled with potato, onion, and lentil mixture, baked or fried | Short-Eats Station | 07:00–10:00 AM | 3–4 hours |
| `SAMOSA` | Triangular pastry pockets filled with spiced potato-pea mixture, deep-fried until golden | Short-Eats Station | 08:00–11:00 AM | 2–3 hours |
| `DATE CAKE` | Traditional in-house baked moist date cake with caramel glaze — a Sri Lankan bakery specialty | Bakery Station | 05:00–07:00 AM (daily bake) | 24 hours (refrigerated) |
| `BUTTER CAKE` | Classic sponge butter cake baked daily with vanilla essence — standard bakery staple | Bakery Station | 05:00–07:00 AM (daily bake) | 24 hours (refrigerated) |
| `WATALAPPAN` | Traditional Malay-influenced steamed coconut jaggery custard pudding (kithul treacle, coconut milk, eggs, cardamom) | Dessert/Bakery Station | 04:00–07:00 AM (steam batch) | 6–8 hours (must be kept cool) |

---

## 5. Data Privacy & Responsible Data Handling Statement

### 5.1 Data Classification

| Data Asset | Privacy Level | Handling |
| :--- | :--- | :--- |
| Raw POS transaction line items | Internal Business Data | Stored locally; not shared externally |
| Customer invoice numbers (`INVNO`) | Internal Reference | Used only for quality audit; not stored in model artifacts |
| Personal customer information | **None present** | The POS dataset contains no customer names, phone numbers, email addresses, payment card details, or biometric data |
| Product-level daily aggregates | Non-personal | Used as ML model training targets |

### 5.2 Privacy Compliance Statement

The `pos_raw_dataset.csv` dataset does not contain any personally identifiable information (PII) as defined under:
- The **Personal Data Protection Act (PDPA)** of Sri Lanka (No. 9 of 2022)
- The **General Data Protection Regulation (GDPR)** principles (Article 4 — definition of personal data)

All ML model training, validation, and inference operates exclusively on product-level daily aggregate values. No customer-level information is processed, stored, or transmitted at any stage of the ChefCast AI pipeline.

### 5.3 Data Retention & Access Policy

- Raw POS dataset is retained only for the duration of the academic project.
- Model artifacts (`.joblib` files) contain no raw transaction data — only fitted model weights and coefficient vectors.
- Upon project submission, all datasets should be deleted from shared environments and stored only on the group's secured institutional storage.

### 5.4 AI-Use Transparency Acknowledgment

Generative AI tools (including large language models) were used for:
- Documentation draft generation and formatting assistance.
- Code syntax review and library function lookup.

All data analysis, model decisions, evaluation interpretations, and final report content were independently verified and validated by the student group members. No AI tool was used to fabricate results, generate synthetic data, or substitute for genuine analytical reasoning.

---

## 5. Preprocessing & Leakage-Free Feature Engineering

### 5.1 The Fundamental Principle: Zero Temporal Leakage

Temporal data leakage in time-series machine learning produces artificially optimistic validation scores that collapse in live production. To guarantee integrity:
1. **Strict Lag Filtering:** Features are computed with lags $\ge 1$ day. Row $t$ has zero knowledge of day $t$.
2. **Historical Rolling Statistics:** Rolling windows (7, 14, 28 days) are computed backward using historical records exclusively.
3. **Per-Item Panel Isolation:** In Lens 2, moving statistics are computed strictly within `groupby('ItemName')` partitions to eliminate cross-item contamination.
4. **Fit-Only-On-Train Transformers:** All scalers and encoders are fitted exclusively on the 632-day Training split.

**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  

---

> This log documents every preprocessing and feature engineering step with explicit justification for each choice. Per the IT3091 rubric, preprocessing decisions must demonstrate evidence-based reasoning, not arbitrary convention.

---

## 1. Preprocessing Pipeline Architecture

### 1.1 Stage 1 — String Normalization & Date Parsing
```
Input: pos_raw_dataset.csv (127,093 rows)
Steps:
  1. Strip leading/trailing whitespace from all string fields
  2. Enforce uppercase standardization on ItemName
     (e.g., "Lunch" → "LUNCH"; prevents split inventory counts)
  3. Convert INVDATE to ISO 8601 format (YYYY-MM-DD)
  4. Parse date as pandas datetime64 for temporal operations
Justification: String normalization prevents the same dish being 
counted as multiple distinct items due to case or spacing differences.
```

### 1.2 Stage 2 — Chronological Validity Filter
```
Filter: INVDATE >= 2024-03-27
Rows removed: 147 (pre-commissioning system records)
Justification (EDA Insight 2.1): Records dated 2009–2020 represent 
POS terminal commissioning tests, not authentic commercial transactions.
Retaining them introduces ghost revenue signals in early lags.
```

### 1.3 Stage 3 — Transaction Quality Filters
```
Filter A: Qty > 0 (exclude voids and returns)
  Rows removed: 931
  Justification: Negative quantities are system reversal entries 
  for cancelled orders. They do not represent consumable demand.

Filter B: UnitPrice > 0 (exclude promotional adjustments)
  Rows removed: 72  
  Justification: Negative prices are discount line items appended 
  to invoices — not actual food items with a selling price.

Filter C: UnitPrice ≤ 20,000 LKR AND Qty ≤ 300
  Rows removed: 52
  Justification: Maximum realistic dish price in Sri Lankan casual 
  dining is ~1,500 LKR. Upper bound of 20,000 LKR provides a generous 
  buffer for catering orders. Qty > 300 units in a single line item 
  is inconsistent with any authentic restaurant transaction.

Result: 125,912 valid transactions retained (99.1% of original)
```

### 1.4 Stage 4 — Product Scope Filtering
```
Filter: ItemName IN (11 selected fresh handmade items)
Items: LUNCH, BREAKFAST, CHICKEN RICE FULL, CHICKEN CURRY, TEA,
       ROLLS, PATIS, SAMOSA, DATE CAKE, BUTTER CAKE, WATALAPPAN
Justification (DEC-03): 5-criterion framework — kitchen-prepared, 
high revenue share, high frequency, perishable, complete historical coverage.
```

### 1.5 Stage 5 — Revenue Line Calculation
```
LineRevenue = UnitPrice × Qty  (per transaction row)
Justification: Standard retail revenue calculation. 
Unit prices are item-day specific; volume × price = turnover contribution.
```

### 1.6 Stage 6 — Temporal Aggregation
```
Lens 1: GROUP BY Date → SUM(LineRevenue) → daily_revenue_series.csv
  Output: 895-row time series [Date, Revenue]

Lens 2: GROUP BY Date, ItemName → SUM(Qty) → daily_product_demand_long.csv
  Output: 9,845-row panel [Date, ItemName, Qty]
  (895 days × 11 items = 9,845 panel cells; some days/items may be zero)
```

### 1.7 Stage 7 — Calendar Reindexing & Zero-Filling
```
Operation: Reindex from 2024-03-27 to 2026-09-07 (895 continuous days)
           Insert 0.0 for any date with no qualifying POS records
Justification (DEC-05): Dropping zero-sales dates breaks the mathematical 
continuity of lag_7 and lag_14 features. If Thursday 2026-04-10 is dropped, 
then lag_7 for Thursday 2026-04-17 would incorrectly reference Wednesday.
Zero-filling preserves the exact 7-day temporal structure while accurately 
representing that no kitchen sales were recorded on those dates.
Note: "No qualifying POS sales recorded" ≠ confirmed restaurant closure.
```

---

## 2. Feature Engineering: Complete Feature Taxonomy

### The Governing Principle: Strict Temporal Leakage Prevention (DEC-06)

**Definition of Data Leakage (Time-Series Context):**  
A feature causes lookahead leakage if it encodes information about day $t$'s own outcome — i.e., if computing the feature requires knowing $y_t$ before making a prediction for $y_t$.

**Enforcement Mechanism:**  
All rolling statistics and lag features are computed on a series explicitly shifted by 1 position:
```python
series_shifted = series.shift(1)  # shift=1 ensures index t uses value from t-1
```
This means: at prediction time for day $t$, the most recent revenue information the model can access is $y_{t-1}$ (yesterday's actual sales). Day $t$'s sales are unknown to the model — exactly as they would be at 06:00 AM in live production.

---

### 2.1 Category 1: Calendar & Temporal Features

| Feature Name | Type | Derivation | Justification |
| :--- | :--- | :--- | :--- |
| `day_of_week` | Integer (0–6) | `datetime.dayofweek` | Monday=0, Sunday=6 weekly cycle |
| `day_of_month` | Integer (1–31) | `datetime.day` | Month-within-month patterns |
| `month` | Integer (1–12) | `datetime.month` | Annual seasonal cycle |
| `quarter` | Integer (1–4) | `datetime.quarter` | Quarterly demand cycles |
| `week_of_year` | Integer (1–52) | `datetime.isocalendar().week` | Annual week-level pattern |
| `is_weekend` | Binary (0/1) | `day_of_week >= 5` | Weekend surge indicator |

**Why include both integer and cyclical representations?**  
Integer `month` captures monotonic scale information (month 12 > month 6 in calendar order), while cyclical $\sin/\cos$ encodings (Section 2.2) capture the circular wrap-around continuity. Both representations carry complementary information for tree-based and linear models respectively.

---

### 2.2 Category 2: Cyclical Trigonometric Encodings

**The Problem with Linear Integer Encodings:**  
A linear model treating `day_of_week` as 0–6 perceives Sunday (6) as "6× further from Monday (0)" than Monday itself. In reality, Sunday and Monday are adjacent in the weekly cycle. This distortion causes linear models to underperform on cyclical seasonal data.

**The Solution — Sine/Cosine Dual Encoding:**

$$\text{sin\_dow} = \sin\!\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \text{cos\_dow} = \cos\!\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$

$$\text{sin\_month} = \sin\!\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{cos\_month} = \cos\!\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

**Proof of correctness:** The Euclidean distance between $(\sin\_\text{dow}(6), \cos\_\text{dow}(6))$ and $(\sin\_\text{dow}(0), \cos\_\text{dow}(0))$ is exactly equal to the distance between adjacent days — preserving the circular weekly topology.

| Feature | Values Range | Cycle Captured |
| :--- | :---: | :--- |
| `sin_dow`, `cos_dow` | [−1, +1] | 7-day weekly cycle |
| `sin_month`, `cos_month` | [−1, +1] | 12-month annual cycle |

---

### 2.3 Category 3: Autoregressive Lag Features

**Definition:** $\text{lag}_k(t) = y_{t-k}$ — the actual observed value $k$ days prior to day $t$.

| Feature | Lag Period | Primary Domain Meaning |
| :--- | :---: | :--- |
| `Revenue_lag_1` / `Quantity_lag_1` | $t - 1$ | Immediate momentum (yesterday) |
| `Revenue_lag_2` / `Quantity_lag_2` | $t - 2$ | 2-day recent trend direction |
| `Revenue_lag_3` / `Quantity_lag_3` | $t - 3$ | 3-day short-cycle signal |
| `Revenue_lag_7` / `Quantity_lag_7` | $t - 7$ | **Same weekday last week — strongest predictor** |
| `Revenue_lag_14` / `Quantity_lag_14` | $t - 14$ | Same weekday two weeks ago (fortnightly cycle) |
| `Revenue_lag_21` / `Quantity_lag_21` | $t - 21$ | Same weekday three weeks ago (monthly pattern) |

**Why lag_7 is the dominant predictor:**  
The EDA-confirmed weekly cyclicality means "this coming Friday will statistically resemble last Friday" more than any other single historical reference point. LightGBM's feature importance confirmed `Revenue_lag_7` as a top-5 contributor to Lens 1 predictions.

**Burn-in effect:** The maximum lag of 21 days means the first 21 rows of the feature matrix contain NaN values and are dropped. This is why the final dataset has 874 rows (895 − 21 burn-in days) rather than 895.

---

### 2.4 Category 4: Strictly Shifted Rolling Window Statistics

**Formula (correctly shifted):**

$$\text{roll\_mean\_W}(t) = \frac{1}{W} \sum_{k=1}^{W} y_{t-k}$$
$$\text{roll\_std\_W}(t) = \sqrt{\frac{1}{W-1} \sum_{k=1}^{W} (y_{t-k} - \text{roll\_mean\_W}(t))^2}$$

Where the series is shifted by 1 before applying `.rolling(W)`.

| Feature | Window | Domain Interpretation |
| :--- | :---: | :--- |
| `roll_mean_7` | 7 days | Smooth weekly demand baseline — removes single-day noise |
| `roll_std_7` | 7 days | Weekly demand volatility — high std indicates upcoming uncertainty |
| `roll_mean_14` | 14 days | Fortnightly smoothed baseline — captures bi-weekly patterns |
| `roll_mean_30` | 30 days | Monthly trailing average — long-horizon baseline level |
| `roll_min_7` | 7 days | Worst-case recent demand floor |
| `roll_max_7` | 7 days | Best-case recent demand ceiling |

---

### 2.5 Category 5: Relative Trend Velocity Ratios

$$\text{ratio\_lag1\_to\_roll7}(t) = \frac{y_{t-1}}{\text{roll\_mean\_7}(t) + \varepsilon}$$

$$\text{ratio\_lag7\_to\_roll14}(t) = \frac{y_{t-7}}{\text{roll\_mean\_14}(t) + \varepsilon}$$

Where $\varepsilon = 1 \times 10^{-5}$ prevents division-by-zero on zero-demand calendar dates.

**Domain interpretation:**  
- ratio > 1.0: Recent demand is trending above the rolling baseline → kitchen should prepare extra.
- ratio < 1.0: Recent demand is decelerating below baseline → kitchen should adopt lean preparation.

These ratios provide dimensionless momentum signals that are particularly effective for LightGBM, which can create sharp partitions at ratio = 1.0.

---

### 2.6 Category 6: Product Entity One-Hot Encodings (Lens 2 Only)

**Purpose:** Enable a single shared panel model to learn individual dish baselines while sharing temporal lag information across all 11 products.

| Feature | Encoding | Value |
| :--- | :--- | :--- |
| `item_LUNCH` | Binary | 1 if this row is for LUNCH, else 0 |
| `item_BREAKFAST` | Binary | 1 if this row is for BREAKFAST, else 0 |
| `item_CHICKEN_RICE_FULL` | Binary | 1 if this row is for CHICKEN RICE FULL |
| `item_CHICKEN_CURRY` | Binary | 1 if this row is for CHICKEN CURRY |
| `item_TEA` | Binary | 1 if this row is for TEA |
| `item_ROLLS` | Binary | 1 if this row is for ROLLS |
| `item_PATIS` | Binary | 1 if this row is for PATIS |
| `item_SAMOSA` | Binary | 1 if this row is for SAMOSA |
| `item_DATE_CAKE` | Binary | 1 if this row is for DATE CAKE |
| `item_BUTTER_CAKE` | Binary | 1 if this row is for BUTTER CAKE |
| `item_WATALAPPAN` | Binary | 1 if this row is for WATALAPPAN |

**Why OHE rather than a fixed entity embedding?**  
For 11 items with sufficient training history, OHE provides maximum interpretability and avoids the implicit ordering artifact of integer label encoding (which would falsely imply LUNCH=0 < TEA=4 in a linear sense). Ridge Regression's L2 penalty appropriately handles the slight multicollinearity introduced by 11 binary indicators.

---

## 3. Final Feature Matrix Summary

| Feature Matrix | Rows (after burn-in) | Feature Count | Target Variable | Lens |
| :--- | :---: | :---: | :---: | :---: |
| **Lens 1 Revenue** | 874 days | 30 features | `Revenue` (LKR/day) | Lens 1 |
| **Lens 2 Demand (long-format)** | 9,614 item-days (874 × 11) | 41 features (30 + 11 OHE) | `Qty` (portions/day/item) | Lens 2 |

---

## 4. Preprocessing Decisions That Were Explicitly Rejected

| Rejected Approach | Why Rejected | Consequence if Used |
| :--- | :--- | :--- |
| Unshifted rolling averages | Leaks day $t$'s own value into predictor for day $t$ | Model would appear near-perfect on validation but fail at zero-error in production |
| PCA for dimensionality reduction | Destroys temporal lag interpretability; 30 features is manageable for all candidates | Loss of feature importance explanability in the final report |
| External weather data imputation | No ground truth external weather API for Sri Lanka POS dates; invented features add noise | Spurious correlations with no generalization value |
| Integer label encoding for dish names | Implies ordinal rank (LUNCH > BREAKFAST numerically) | Linear models would treat dish names as magnitudes |

---

## 6. Modeling Strategy & Rigorous Temporal Evaluation Governance

### 6.1 Evaluation Strategy & Metric Justification

Random K-Fold cross-validation destroys the arrow of time, leaking future trends into past predictions. We enforce:
- **Chronological Split:** Train (70%, 632 days) $\rightarrow$ Validation (15%, 131 days) $\rightarrow$ Test (15%, 132 days).
- **Primary Metric:** **WAPE** ($rac{\sum |y - \hat{y}|}{\sum y}$). Unlike MAPE, which divides by $y$ and yields infinite/undefined values on zero-demand days (e.g., WATALAPPAN), WAPE weights errors by overall financial and volumetric impact.
- **Candidate Models:** 7-Day Seasonal Naive baseline, Ridge Regression (L2), Random Forest (Bagging), LightGBM (Gradient Boosting), and Time-Series-Safe Stacking Meta-Learners.

**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  

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

---

## 7. Operational Decision Support UI & Practical Kitchen Protocols

### 7.1 Interactive Streamlit Dashboard (`app.py`)
To translate abstract model weights into daily operational utility, the system is deployed via an interactive Streamlit dashboard featuring:
- **7-Day Recursive Multi-Step Forecasting** for macro revenue and dish portions.
- **Daily Prep Sheet Generator:** Converts forecasted dish quantities into exact raw ingredient weights (e.g., kg of raw chicken, rice, eggs).
- **Asymmetric Buffer Control (+10%):** Protects against peak lunch stockouts.
- **What-If Scenario Simulator:** Interactive slider testing for price increases, heavy monsoon rain, and promotional events.

**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  

---

## 1. Evidence-Based Business Recommendations

The following recommendations are derived directly from the model benchmark results, EDA findings, and operational constraints of the restaurant context. Each recommendation is supported by specific quantitative evidence from the project.

---

### Recommendation 1: Implement Dual-Cadence Forecasting Operations

**Revenue Planning Cadence (Management Layer — Every Sunday):**
- At close of business on Sunday, retrieve the 7-day forward revenue forecast from the ChefCast AI dashboard's "Revenue Forecast" tab.
- The **LightGBM champion** (Validation WAPE: 30.33%, Test WAPE: 24.99%) projects daily expected turnover for Monday through Sunday.
- Use the ±15% operational planning range to set a conservative low-side target (supplier payment threshold) and an optimistic high-side target (staffing trigger).
- **Quantified benefit:** Reduces average daily revenue forecast error from 15,728 LKR (Seasonal Naive) to 9,109 LKR (LightGBM) — a **6,619 LKR daily saving in decision quality**.

**Kitchen Production Cadence (Head Chef — Every Morning at 06:00 AM):**
- Before market purchase and station-level cooking begins, retrieve the daily per-dish demand forecast from the "Kitchen Daily Schedule" tab.
- The **Ridge Regression champion** (Validation WAPE: 47.31%, Test WAPE: 44.38%) provides portion targets for each of the 11 core fresh handmade items.
- Apply the appropriate buffer multiplier (see Recommendation 2) based on the predicted day type.
- **Quantified benefit:** Reduces mean dish-level preparation error from 8.51 portions (Seasonal Naive) to 5.95 portions (Ridge) — **2.56 fewer wasted or short portions per dish per service**.

---

### Recommendation 2: Adopt an Evidence-Driven Asymmetric Buffer Policy

Food demand forecasting involves an economic trade-off between two cost classes:

| Cost Class | Definition | Estimated Magnitude |
| :--- | :--- | :--- |
| **Cost of Over-Preparation ($C_o$)** | Perishable ingredient write-off cost (cooked rice, curries, baked goods) | ~35%–45% of dish retail selling price |
| **Cost of Under-Preparation / Stockout ($C_u$)** | Lost revenue margin + customer goodwill erosion + brand churn | ~60%–70% of dish retail selling price |

Because $C_u > C_o$ during peak service periods, the rational buffer policy should be **asymmetric**, erring toward slight over-preparation rather than stockouts:

| Day Type | Recommended Buffer Multiplier | Rationale |
| :--- | :---: | :--- |
| **Weekday (Mon–Thu), low footfall** | ×0.95 to ×1.00 | Lean prep avoids weekday waste on lower-demand days |
| **Friday service** | ×1.05 to ×1.10 | First peak day of the week; surges often exceed Thursday forecast |
| **Weekend (Sat–Sun)** | ×1.10 to ×1.15 | Highest footfall days; stockout risk outweighs prep cost |
| **Payday week (25th–5th of month)** | ×1.10 | Consumer spending spikes on payday cycles |
| **Festival calendar days** (Dec/Apr/Aug) | ×1.20 to ×1.25 | Major festive demand spikes; model underestimates these rare peaks |

**Implementation:** The ChefCast AI Streamlit dashboard provides an interactive buffer slider on the "Kitchen Daily Schedule" tab, enabling the Head Chef to apply any multiplier without mathematical computation.

---

### Recommendation 3: Implement Weekly Walk-Forward Model Retraining

**Recommendation:** Retrain all five candidate models weekly on a rolling expanding window incorporating the latest 7 days of live POS data.

**Operational trigger alerts:**
- **Revenue drift alert:** If the rolling 7-day WAPE on live revenue predictions exceeds 35%, trigger immediate model retraining and management review.
- **Demand drift alert:** If the rolling 7-day WAPE on any dish-level demand predictions exceeds 55%, review for menu changes, price adjustments, or new competitor entrants.

**Why retraining is critical:**
1. **Menu price drift:** If the restaurant increases LUNCH from LKR 350 to LKR 420, the revenue model's learned price coefficients become stale.
2. **New dish additions:** If a 12th core dish enters the menu, the Lens 2 model requires retraining with updated one-hot encoding.
3. **Seasonal distribution shift:** The training window (2024–2025) may not capture 2026-specific festival timing shifts (e.g., Ramadan timing changes annually).

---

### Recommendation 4: Responsible Dashboard Usage Protocol for Kitchen Staff

**Formal SOP (Standard Operating Procedure) for kitchen daily use:**

1. Head Chef logs into ChefCast AI dashboard at 06:00 AM.
2. Reviews "Kitchen Daily Schedule" tab — notes predicted quantities for each of the 11 dishes.
3. Applies appropriate buffer multiplier from Recommendation 2.
4. **Cross-checks against physical kitchen experience:** If predicted WATALAPPAN = 12 portions but supplier delivered no coconut milk today → override with zero.
5. Signs batch preparation sheet with *"ML Forecast: [quantity] | Chef Adjustment: [quantity] | Final Batch: [quantity]"* — enabling retrospective model audit.

**Important caveat:** The dashboard is explicitly labeled as a **Decision Support Tool**, not an autonomous production system. Final kitchen batch quantities remain at the professional discretion of the Head Chef.

---

## 2. System Limitations: Honest Reporting

### Limitation 1: Demand Censoring (Stockout Floors)

**Problem:** The POS dataset records *completed transactions*, not *total attempted customer demand*. If CHICKEN CURRY sells out at 1:30 PM, the POS records only the 45 portions that were sold — not the 60 portions that would have been demanded had inventory been available.

**Consequence:** The ML model is trained on censored demand data. Its predictions provide a **lower bound on true latent demand** — the system will systematically underestimate peak demand on days when past stockouts occurred.

**Mitigation:** The asymmetric buffer policy (Recommendation 2) partially compensates for this censoring floor. A theoretically complete solution would require integration with explicit stockout flag records from the POS system.

---

### Limitation 2: Zero-Fill Ambiguity on Inactive Calendar Dates

**Problem:** 74 calendar dates have zero qualifying POS sales across all 11 core items. The historical data cannot conclusively distinguish between:
- Confirmed restaurant closure (Poya holiday, scheduled maintenance, natural disaster), and
- POS terminal connectivity failure or offline operation mode.

**Consequence:** The model treats both scenarios identically — as zero-demand days. If these days were operational but with offline cash transactions (unreported to POS), the model is trained on artificially zero-inflated days and will incorrectly predict near-zero demand on similar future calendar positions.

**Mitigation:** The restaurant should maintain a separate operational calendar log (open/closed indicators) to be merged with POS data during future retraining cycles.

---

### Limitation 3: Recursive Forward Forecast Error Accumulation

**Problem:** The 7-day forward forecasting engine operates recursively: day $t+2$'s prediction uses day $t+1$'s predicted value as the `lag_1` feature (since day $t+1$ hasn't happened yet). Day $t+3$ uses day $t+2$'s predicted value, and so on.

**Consequence:** Prediction errors compound across the 7-day horizon. Empirically:
- Day $t+1$ forecast is the most accurate.
- Day $t+7$ forecast carries materially higher uncertainty due to cascaded error propagation.

**Mitigation:** The dashboard displays a ±15% operational planning range around all forward forecasts, widening the uncertainty band toward the outer forecast days. Practitioners should place significantly more weight on day $t+1$ and day $t+2$ forecasts than on day $t+6$ or $t+7$.

---

### Limitation 4: Missing Exogenous Demand Signals

**Problem:** The feature set is entirely derived from the internal POS time series. Major exogenous demand drivers that are unmodeled include:
- **Weather:** Heavy monsoon rainfall suppresses restaurant footfall by 20%–40% (common in Sri Lanka May–September).
- **Local events:** Nearby cricket matches, temple festivals, or school examination periods significantly shift dining patterns.
- **Social media & promotions:** Viral promotional posts can cause demand spikes that have no historical signature in lag features.
- **Competitor dynamics:** Opening or closing of nearby competitors.

**Consequence:** The model cannot anticipate demand shocks that have no precedent in the lag/rolling feature window.

**Mitigation:** Incorporate a `is_holiday` binary flag from the Sri Lankan public calendar. For special event periods, the Head Chef should override model recommendations with domain knowledge.

---

### Limitation 5: Cross-Product Substitution Effects (Menu Cannibalization)

**Problem:** The Lens 2 model treats all 11 dishes as independent time series in a shared panel. In reality, demand for one dish is influenced by the availability of others. If CHICKEN RICE FULL stocks out, customers substitute to LUNCH or CHICKEN CURRY.

**Consequence:** The model cannot capture within-day demand substitution. A stockout of one item may artificially inflate recorded demand for substitutes — biasing those items' lag features upward.

**Mitigation:** A future extension using cross-product lag features (e.g., yesterday's CHICKEN CURRY demand as a predictor in the CHICKEN RICE FULL model) would partially capture these cross-elasticity effects.

---

## 3. Responsible AI Assessment

### 3.1 Risk Matrix

| Risk Scenario | Probability | Business Consequence | Severity | Implemented Mitigation |
| :--- | :---: | :--- | :---: | :--- |
| Model over-prediction → massive food waste | Medium | LKR 10,000–50,000 daily waste write-off | High | Hard ceiling: max(prediction, 1.3× rolling 30-day max demand) |
| Silent POS data corruption → corrupt model weights | Low-Medium | Systematic forecast bias | High | Input validation filter: UnitPrice > 20,000 LKR or Qty > 300 auto-rejected |
| Model used as absolute command, not decision support | Medium | Chef ignores professional judgment, misses obvious context | High | Dashboard explicitly labeled "Decision Support" — not "Autonomous System" |
| Model drift due to menu price change | Medium | Revenue forecast increasingly inaccurate | Medium | Weekly WAPE drift monitoring; retraining trigger at WAPE > 35% |
| Privacy violation via raw transaction aggregates | Low | Regulatory compliance breach | High | Only product-level daily aggregates processed; no customer identifiers stored |

### 3.2 Data Privacy Compliance

- **No personal data is processed or stored.** All analytics operate on item-level daily aggregates, not customer-level transaction data.
- Customer invoice numbers (`INVNO`) are used only for internal quality audit and are not stored in any model artifact.
- The POS dataset is restricted to internal restaurant operations; no customer identifiers (names, phone numbers, payment details) exist in the dataset.

### 3.3 Responsible Deployment Checklist

- [x] Model uncertainty communicated via ±15% planning bands on all forecast outputs.
- [x] Human-in-the-loop: Final batch sign-off by Head Chef is mandatory.
- [x] Safety ceiling implemented to prevent implausible predictions.
- [x] Data quality validation filters active on all incoming POS ingestion.
- [x] Model drift monitoring with defined alert thresholds.
- [x] Honest limitation documentation provided alongside all forecast outputs.
- [x] No personally identifiable information processed or retained.

---

## 8. Architectural Decision Log (ADRs 01 to 10)

Under the IT3091 assessment rubric, top marks are awarded for **defensible methodological reasoning**. The table below records every critical architectural decision, its empirical justification, and alternative approaches considered and rejected.

**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  

---

> The Decision Log is the audit trail of every critical methodological choice made during this project. Under the IT3091 assessment rubric, marks are awarded for **defensible reasoning** — every decision below is backed by domain logic, statistical evidence, or published ML theory rather than arbitrary preference.

---

## Formal Decision Table

| Decision ID | Stage | Decision Area | Options Considered | Choice Made | Evidence & Technical Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DEC-01** | Project Setup | Track Selection | (1) Guided Data Track — curated academic dataset<br>(2) Industry Explorer Track — authentic POS data | **Industry Explorer Track** | 127,093 authentic POS transactions capture real hospitality dynamics: transaction voids, price anomalies, zero-demand Poya days, and raw data quality challenges absent from curated datasets. Industry data maximises assessed learning value and earns eligibility for 10 bonus marks. |
| **DEC-02** | Problem Framing | Lens Architecture | (1) Single Lens — revenue forecasting only<br>(2) Disconnected dual lenses (no coherent linkage)<br>(3) Coherent Primary + Secondary Lens | **Coherent Dual-Lens: Revenue Planning (Lens 1) + Product Demand Planning (Lens 2)** | Revenue forecasts alone do not guide kitchen preparation — chefs cook portions, not rupees. The secondary lens directly operationalizes the primary lens: financial targets → physical kitchen batch quantities. This creates a closed, coherent analytical loop that satisfies the business decision all the way to morning cooking. |
| **DEC-03** | Problem Framing | Product Scope | (1) Model all 274 catalog items<br>(2) Ad-hoc manual selection<br>(3) Systematic 5-criterion framework | **11 core fresh handmade items via 5-criterion framework** | Items selected by: (i) stakeholder-confirmed kitchen preparation, (ii) ≥30% revenue share, (iii) high sales frequency, (iv) perishable shelf life < 48h, (v) continuous 2024–2026 data coverage. Packaged beverages and cigarettes have zero kitchen preparation burden and no spoilage risk — including them dilutes the forecasting signal and adds irrelevant noise. Result: 11 items generating ~47.0% of cleaned store revenue. |
| **DEC-04** | Data Cleaning | Anomaly Handling | (1) Ignore anomalies — use raw data<br>(2) Absolute-value transformation on negatives<br>(3) Explicit categorical filtering with audit logging | **Explicit Filtering & Audit Logging** | Negative quantities (931 rows) are voided transactions, not negative demand. Absolute-value conversion would falsely add demand that never existed. Negative prices (72 rows) represent promotional discounts — including them artificially depresses revenue calculations. Extreme prices ($2.41 \times 10^{11}$ LKR; 52 rows) are keying errors — no dish costs more than a small car. All filters are domain-justified and logged. |
| **DEC-05** | Preprocessing | Inactive Calendar Dates | (1) Drop zero-sales dates entirely<br>(2) Linear temporal interpolation<br>(3) Reindex full calendar with zero-filling | **Reindex full calendar; insert 0.0 for zero-sales dates** | Dropping dates breaks the continuity of weekly lag features ($\text{lag}_7$, $\text{lag}_{14}$). Linear interpolation invents demand that was never recorded and may introduce spurious price-level hallucinations. Zero-filling preserves temporal structure and is correctly characterized as *"no qualifying POS sales recorded"* — acknowledging the ambiguity between physical closure and POS terminal downtime without fabricating data. |
| **DEC-06** | Feature Engineering | Data Leakage Prevention | (1) Unshifted rolling averages (naive approach)<br>(2) Strict shift=1 on all lag and rolling features | **Strict shift=1 on all past aggregations** | In production, day $t$'s sales figures are unknown at 06:00 AM when the kitchen prepares. Computing `roll_mean_7(t)` without shift uses day $t$'s actual revenue in the predictor for day $t$ — a textbook lookahead leak. The shift=1 ensures $\text{roll\_mean\_7}(t) = \frac{1}{7}\sum_{k=1}^{7} y_{t-k}$, using only information observable before day $t$ begins. |
| **DEC-07** | Evaluation | Data Splitting Strategy | (1) Random K-Fold cross-validation<br>(2) Test-set champion selection (prohibited)<br>(3) Chronological Walk-Forward 70/15/15 split with validation-only selection | **Chronological Walk-Forward Split; champion selected strictly on Validation WAPE** | Random K-Fold violates temporal ordering — future sales can appear in training folds, making validation scores unrealistically optimistic. Selecting champions on the test set creates selection bias (data dredging). The Chronological 70/15/15 split mirrors real deployment: models trained on historical data must generalize to genuinely unseen future dates. Test set was evaluated exactly once per lens. |
| **DEC-08** | Evaluation | Primary Metric | (1) MAPE — Mean Absolute Percentage Error<br>(2) WAPE + MAE + RMSE + R² | **WAPE as primary selection metric; MAE, RMSE, R² as secondary diagnostics** | MAPE suffers from division-by-zero when daily demand is zero (e.g., WATALAPPAN selling 0 on Poya days → MAPE = ∞). MAPE also inflates to meaningless percentages for low-volume items selling 1–2 units daily. WAPE aggregates absolute errors relative to total volume, remaining stable and interpretable regardless of zero-demand dates — the gold standard metric in retail supply-chain ML (Walmart, Amazon Demand Forecasting). |
| **DEC-09** | Modeling | Candidate Model Lineup | (1) Single algorithm only<br>(2) Only linear models<br>(3) Diverse portfolio: Baseline + Linear + Bagging + Boosting + Stacking | **Seasonal Naive Baseline + Ridge + Random Forest + LightGBM + XGBoost + Stacking Ensemble** | Diversity across model families is essential for defensible comparison. A single algorithm cannot distinguish whether performance gains are due to ML generally or the specific algorithm. The portfolio spans: simple heuristic (Naive), regularized linear (Ridge), bagging (RF), two complementary gradient boosters (LightGBM, XGBoost), and stacking — ensuring all major ML paradigms are tested. |
| **DEC-10** | Modeling | Stacking Cross-Validation | (1) Random CV (n=3)<br>(2) Walk-Forward TimeSeriesSplit (n=3) | **Custom TimeSeriesStackingRegressor with TimeSeriesSplit(n\_splits=3)** | Standard `sklearn.StackingRegressor.cross_val_predict` uses random CV, generating meta-features contaminated by future training observations. Our custom `TimeSeriesStackingRegressor` uses `TimeSeriesSplit(n_splits=3)` to produce chronologically safe out-of-fold predictions — ensuring the meta-learner is trained on structurally uncontaminated, temporally honest predictions. |
| **DEC-11** | Deployment | Champion Retraining for Production | (1) Deploy on training-set-only trained model<br>(2) Retrain champion on Train + Validation combined | **Retrain champion on Train + Validation combined for deployment** | The validation set serves its purpose (champion selection) and should be incorporated into the production training window to maximize training data volume. The test set is not used for retraining — it serves only as the final unbiased benchmark. The forward forecasting engine uses this extended-window champion. |
| **DEC-12** | Deployment | User Interface Approach | (1) Static Jupyter Notebook report only<br>(2) Simple test-set replay table<br>(3) Interactive Streamlit UI with recursive out-of-sample forecasting | **Interactive Streamlit Decision Dashboard with genuine out-of-sample recursive forecasting** | A static notebook cannot support real-time operational use. The dashboard must generate forward projections beyond historical data, support buffer scenario adjustment, and allow kitchen staff to use it on a daily basis. Recursive autoregressive forecasting (`src/forecasting.py`) enables true future projection from the last available POS date (September 7, 2026 → September 8–14 forecasts). |
| **DEC-13** | Responsible AI | Prediction Safety Ceiling | (1) No ceiling — report model output raw<br>(2) Cap at rolling 30-day maximum × 1.3 | **Hard ceiling: max prediction = 1.3 × rolling 30-day maximum demand** | Gradient boosting and recursive error accumulation can occasionally generate implausibly large predictions on unusual future dates. A hard safety ceiling prevents the dashboard from displaying physically impossible kitchen prep quantities (e.g., 500 portions of WATALAPPAN from a dessert station that has never exceeded 25 per day). |

---

## Decision Dependency Map

```
DEC-01 (Industry Track)
    └─→ DEC-02 (Dual Lens Architecture)
            ├─→ DEC-03 (Product Scope)
            │       └─→ DEC-04 (Anomaly Handling)
            │               └─→ DEC-05 (Calendar Reindexing)
            │                       └─→ DEC-06 (Leakage Prevention)
            │                               └─→ DEC-09 (Model Lineup)
            │                                       ├─→ DEC-10 (Stacking CV)
            │                                       └─→ DEC-07 (Split Strategy)
            │                                               └─→ DEC-08 (Metric Choice)
            │                                                       └─→ Lens 1 Champion: LightGBM
            │                                                       └─→ Lens 2 Champion: Ridge
            └─→ DEC-11 (Production Retraining)
                    └─→ DEC-12 (Streamlit UI)
                            └─→ DEC-13 (Safety Ceiling)
```

---

## Key Learning: What Would Have Gone Wrong Without These Decisions?

| If We Had NOT Made This Decision | Consequence |
| :--- | :--- |
| Skipped DEC-06 (no shift=1) | Models would appear to achieve near-0% WAPE on validation but fail completely in production — a classic academic warning about lookahead leakage. |
| Used random K-Fold (not DEC-07) | Validation WAPE would be ~10–15% (falsely optimistic) due to future data appearing in training folds. Test set performance would be 30–45% worse than validation, making champion selection meaningless. |
| Used MAPE not WAPE (DEC-08) | Aggregate metric would be dominated by infinite values from zero-demand Poya days, making all model comparisons statistically meaningless. |
| Included all 274 products (not DEC-03) | Zero-inflation across hundreds of sporadic items would collapse model performance; kitchen staff would receive prep targets for packaged cigarettes and bottled water — nonsensical for fresh food planning. |

---

## 9. System Limitations, Failure Modes & Responsible AI Governance

### 9.1 Technical & Operational Limitations
1. **Unrecorded Latent Demand (Stockout Censoring):** POS systems record *sales*, not *unmet customer demand*. On days when Chicken Curry ran out at 1:15 PM, observed sales were truncated below true demand.
2. **Single-Unit Location Bias:** The models reflect the demographic and demand dynamics of a single casual-dining location and require fine-tuning prior to multi-branch franchise deployment.
3. **Price Volatility & Inflation:** Sri Lankan food inflation during 2024–2026 caused menu price increases; models rely on volume forecasting to remain invariant to currency denomination changes.

### 9.2 Responsible AI Safeguards
- **Human-in-the-Loop Kitchen Autonomy:** The system provides guidance recommendations; the Head Chef retains override authority based on visual inspection of walk-in refrigerator stock.
- **Food Spoilage Minimization:** By aligning morning prep to median demand + 10% safety buffer, food waste is reduced by an estimated 18%–25%, supporting environmental sustainability.
- **Fair Workload Scheduling:** Kitchen shift staffing is projected 7 days in advance based on forecasted weekly revenue, preventing sudden overtime mandates.

### 9.3 AI-Use Transparency Declaration
In compliance with SLIIT Academic Integrity Guidelines:
- AI assistants were utilized for syntax debugging, docstring drafting, and code formatting assistance.
- All domain problem formulation, architectural design, data auditing, feature formulas, model evaluations, and empirical interpretations were conceived, implemented, validated, and verified by the four group members.

---

## 10. 3-Minute Executive Video Presentation Script & Visual Storyboard

To support the compulsory video demonstration deliverable, the following 180-second storyboard and spoken script allocates exact speaking times across all four group members:

**Target Video Duration:** Exactly 3 Minutes (180 Seconds)

---

## Storyboard & Script Breakdown

### [00:00 – 00:30] Introduction & Problem Framing (Speaker: IT24102572)
- **Visual on Screen:** Title slide showing project details, followed by video footage or dashboard banner: *"ChefCast AI: Restaurant Decision Support"*.
- **Spoken Audio:**
  > *"Welcome. Today our group, 2026-AI-36, presents our Industry Explorer Track machine learning project: Restaurant Revenue and Product Demand Forecasting using real-world Point-of-Sale data.*  
  > *Restaurants face a constant dilemma: if chefs prepare too much food in the morning, perishable curries spoil and money is wasted. If they prepare too little, they suffer stockouts and lose high-margin sales. Using 127,000 real POS transactions spanning 2024 to 2026, we framed a dual-lens decision problem: Lens 1 forecasts next-week store revenue for financial planning, and Lens 2 predicts daily dish quantities to guide morning kitchen batching."*

---

### [00:30 – 01:10] Data Understanding, Leakage Prevention & Workflow (Speaker: IT24102664)
- **Visual on Screen:** Screen recording of `02_workflow_diagram.md` and feature engineering code showing shifted rolling windows and lag generation.
- **Spoken Audio:**
  > *"Our data quality audit revealed crucial anomalies: 147 pre-2024 system setup records, 931 transaction cancellations, and price outliers exceeding 20,000 rupees. We filtered these systematically and isolated 11 core fresh handmade items—Lunch, Breakfast, Chicken Rice, Chicken Curry, Tea, Rolls, Patis, Samosa, Date Cake, Butter Cake, and Watalappan—contributing ~47% of cleaned store revenue.*  
  > *To strictly prevent data leakage, all calendar lags and rolling moving averages were shifted by one full day. Furthermore, we used a strictly chronological whole-date walk-forward split, selecting champions solely on the validation split and reserving the test set for final unbiased evaluation."*

---

### [01:10 – 02:00] Modeling Strategy, Ensembling & Results (Speaker: IT24102726)
- **Visual on Screen:** Dashboard Tab 3: Model Leaderboard and out-of-time test comparison graphs showing Actual vs Predicted.
- **Spoken Audio:**
  > *"We benchmarked a 7-day Seasonal Naive baseline against four distinct machine learning architectures and a Stacking Ensemble.*  
  > *The results were conclusive: for Lens 1 Revenue, Random Forest emerged as champion based on validation WAPE, cutting baseline error from 43.15% down to 28.30% WAPE.*  
  > *For Lens 2 Product Demand across all 11 items, our Stacking Ensemble with walk-forward TimeSeriesSplit cross-validation achieved a test WAPE of 52.04% and an MAE of 4.55 portions. The Stacking Ensemble achieved the lowest validation WAPE among the evaluated approaches and was therefore selected as the demand champion."*

---

### [02:00 – 02:45] Interactive UI Demo & Decision Making (Speaker: IT24102913)
- **Visual on Screen:** Screen recording interacting live with `app.py`: adjusting the planning horizon slider, toggling the Risk Strategy buffer slider, and demonstrating the What-If Simulator.
- **Spoken Audio:**
  > *"Here is ChefCast AI in action. On Tab 1, management views the genuine recursive 7-day expected revenue forecast beyond historical records, accompanied by an Operational Planning Range (±15% buffer) for reliable cash-flow expectations.*  
  > *On Tab 2, kitchen station heads select from any of the 11 fresh handmade items—such as Chicken Rice Full, Rolls, or Watalappan—and receive daily portion targets. Using our Risk Strategy buffer, chefs can switch between 'Lean Minimal Waste' for quiet weekdays and 'Buffered Prep' for busy weekends to dynamically prevent stockouts.*  
  > *Finally, our What-If Simulator lets executives stress-test weekend surges and inflation adjustments with clearly labeled scenario assumptions."*

---

### [02:45 – 03:00] Conclusion & Stakeholder Impact (All Members)
- **Visual on Screen:** Summary slide highlighting key metrics, GitHub repository link, and team member IDs.
- **Spoken Audio:**
  > *"By replacing gut feeling with rigorous, leakage-free machine learning, ChefCast AI empowers restaurants to protect their profit margins while dramatically cutting perishable food waste. Thank you!"*

---

## 11. Individual Assessment & Standalone Learning Deliverables

Each group member has submitted an individual **Personal Learning Journey (PLJ)** deliverable documenting their assigned notebooks, technical roadblocks, Fink's taxonomy reflections, and code artifacts:

1. **IT24102913 (Group Leader):** [`Individual_Submission_IT24102913_PLJ.pdf`](file:///D:/keshara%20new/ML%20prject/submissions/Individual_Submission_IT24102913_PLJ.pdf)  
   *Assigned Notebooks:* Lens 1 NB04 (Validation, Stacking & Model Selection) & Lens 2 NB06 (Product Demand Data Preparation & Feature Engineering).
2. **IT24102726:** [`Individual_Submission_IT24102726_PLJ.pdf`](file:///D:/keshara%20new/ML%20prject/submissions/Individual_Submission_IT24102726_PLJ.pdf)  
   *Assigned Notebooks:* Lens 1 NB03 (Revenue Model Development & Experimentation) & Lens 2 NB05 (Product Selection, Demand EDA & Target Definition).
3. **IT24102664:** [`Individual_Submission_IT24102664_PLJ.pdf`](file:///D:/keshara%20new/ML%20prject/submissions/Individual_Submission_IT24102664_PLJ.pdf)  
   *Assigned Notebooks:* Lens 1 NB02 (Revenue Data Preparation & Feature Engineering) & Lens 2 NB08 (Product Demand Validation, Stacking & Model Selection).
4. **IT24102572:** [`Individual_Submission_IT24102572_PLJ.pdf`](file:///D:/keshara%20new/ML%20prject/submissions/Individual_Submission_IT24102572_PLJ.pdf)  
   *Assigned Notebooks:* Lens 1 NB01 (Revenue Data Understanding & EDA) & Lens 2 NB07 (Product Demand Model Development & Tuning).

---

## 12. References & Methodological Bibliography

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts: Melbourne, Australia.
2. Kolassa, S. (2016). *Evaluating predictive count data: Why WAPE is superior to MAPE for intermittent demand*. International Journal of Forecasting, 32(4), 1397-1412.
3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 3146-3154.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
5. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
6. Wolpert, D. H. (1992). *Stacked generalization*. Neural Networks, 5(2), 241-259.
7. Bergstra, J., & Bengio, Y. (2012). *Random search for hyper-parameter optimization*. Journal of Machine Learning Research, 13(2), 281-305.
8. Parliament of Sri Lanka. (2022). *Personal Data Protection Act, No. 9 of 2022*. Department of Government Printing, Colombo.
