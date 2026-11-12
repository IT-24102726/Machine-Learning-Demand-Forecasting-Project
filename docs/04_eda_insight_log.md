# Core Evidence 4: Exploratory Data Analysis (EDA) Insight Log
**IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`  
**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  
**Version:** Final Submission — October 2026

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
