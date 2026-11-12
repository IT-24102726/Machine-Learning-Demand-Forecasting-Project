# Personal Learning Journey Report
## IT3091 Machine Learning — Individual Assessment Deliverable

---

**Student ID:** IT24102572  
**Group ID:** 2026-AI-36  
**Project Title:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  
**Assigned Notebooks:** NB01 (Lens 1 — Revenue EDA & Target Definition) and NB07 (Lens 2 — Product Demand Model Development & Experimentation)  
**Submission Date:** October 2026

---

## Section 1: My Technical Role & Specific Contributions

### NB01 — Lens 1: Revenue Data Understanding, EDA & Target Definition

My responsibility in the first notebook was to serve as the data detective — interrogating the raw POS dataset from the ground up before any ML pipeline was designed. This was not a passive data description exercise; it was a structured investigation with the explicit goal of answering: *"What does this data actually represent, and what hidden problems does it contain that would break a naively implemented ML model?"*

**Specific work I performed:**

**1. Dataset Characterisation & Initial Audit:**  
I loaded and profiled the raw 127,093-row POS dataset, computing summary statistics across all six fields: `INVDATE`, `INVNO`, `ItemCode`, `ItemName`, `Qty`, and `UnitPrice`. My first diagnostic scan revealed an immediately alarming finding: dates spanning from 2009 to 2026 in a dataset representing a restaurant that only began operating in 2024. This was not immediately obvious — the 2009/2011 records did not carry any "test" label; they appeared as ordinary transactions with plausible-looking prices. I identified them as pre-commissioning system records through cross-referencing the date distribution against the stated operational opening date and noting the suspiciously round quantities and fixed prices in those early entries.

**2. Four-Class Anomaly Identification:**  
I systematically categorized all data quality anomalies into four distinct classes, each requiring different resolution strategies:
- Pre-operational records (147 rows): Filtered using date threshold.
- Transaction voids/refunds (931 rows): Identified by `Qty ≤ 0`, confirmed by POS accounting convention analysis.
- Promotional price adjustments (72 rows): Identified by `UnitPrice ≤ 0`, distinguished from zero-demand items.
- Extreme price outliers (52 rows): Identified via descriptive statistics showing values exceeding $2.41 \times 10^{11}$ LKR — I applied domain reasoning to establish a defensible upper bound of LKR 20,000 (generous enough for catering but clearly excluding data corruption artifacts).

**3. Revenue Target Variable Definition:**  
I formally defined the Lens 1 target variable: $y_t = \sum_{i \in \mathcal{S}} \text{LineRevenue}_{i,t}$ where $\text{LineRevenue} = \text{UnitPrice} \times \text{Qty}$. This sounds simple, but required careful decisions: Should I use total store revenue or only the 11-item revenue? Should zero-sales days be included? I resolved both questions with evidence — restricting to the 11 core items maximizes operational relevance and model signal; zero-filling preserves temporal lag continuity.

**4. EDA Visualisation & Pattern Discovery:**  
I produced time-series plots of aggregated daily revenue, day-of-week box plots, monthly seasonality heatmaps, and Pareto charts of revenue by item. The day-of-week box plot was particularly revealing: Friday and Saturday revenue interquartile ranges did not overlap with Tuesday and Wednesday ranges — confirming that weekday identity is a primary demand driver, directly motivating the `lag_7` and `sin/cos(dow)` features that became the top predictors in the final model.

---

### NB07 — Lens 2: Product Demand Model Development & Experimentation

My responsibility in NB07 was to design, train, and systematically experiment with five candidate machine learning architectures on the product-level daily demand forecasting problem.

**Specific work I performed:**

**1. Candidate Model Implementation:**  
I implemented all five candidate models on the Lens 2 feature matrix (9,614 item-days × 41 features):
- 7-Day Seasonal Naive baseline.
- Ridge Regression with StandardScaler pipeline.
- Random Forest Regressor with `n_estimators=150`, `max_depth=10`.
- LightGBM Regressor with `n_estimators=150`, `learning_rate=0.05`, `subsample=0.8`.
- XGBoost Regressor with `n_estimators=150`, `learning_rate=0.05`, `max_depth=6`.

**2. Hyperparameter Justification:**  
For gradient boosting models, I set `learning_rate=0.05` (smaller learning rates require more trees but generalise better — a well-established regularization tradeoff). The `subsample=0.8` on LightGBM applies stochastic gradient boosting, where only 80% of training data is used per tree — this implicit regularisation was critical for reducing overfitting on the noisy 5–20 portion daily demand counts.

**3. Systematic Validation Experiment:**  
I trained all five models on the Training set (2024-03-27 to 2025-12-18) and evaluated their performance on the Validation set (2025-12-19 to 2026-04-28) using WAPE, MAE, RMSE, and $R^2$. I did not touch the test set at this stage — disciplined adherence to the validation-only selection principle.

**4. Model Comparison & Preliminary Findings:**  
My experiments produced a counterintuitive finding: LightGBM, despite being the Lens 1 champion, ranked 5th on Lens 2 with a Validation WAPE of 54.56%. Ridge Regression ranked 1st with 47.31%. I documented this finding, providing the bias-variance interpretation: on high-noise, low-signal disaggregated dish data, L2 regularisation outperforms expressive gradient boosting because the stochastic noise in daily dish counts does not contain exploitable patterns beyond what Ridge can extract via smoothed lag features.

---

## Section 2: Critical Challenges I Overcame

### Challenge 1: Identifying Non-Obvious Data Quality Issues

The extreme price entries ($2.41 \times 10^{11}$ LKR) were immediately visible in a `.describe()` call. But the 2009-dated pre-commissioning records were subtle — they carried realistic-looking prices, standard quantities, and legitimate item names. Without domain reasoning (asking "when did this restaurant actually open?"), a naive analyst would include them and inadvertently introduce ghost revenue signals from 2009 into 2025 rolling window features. I overcame this by treating EDA as hypothesis testing rather than passive inspection.

### Challenge 2: The Zero-Demand Day Decision

When I first produced the daily revenue time series, I found 74 dates with zero sales. My initial instinct was to drop them (they contaminate the revenue distribution). But then I reasoned through the consequence: removing date 2025-08-14 means `lag_7` for 2025-08-21 would incorrectly reference 2025-08-13 (a Wednesday) instead of 2025-08-14 (a Thursday). The entire 7-day cycle alignment breaks. I tested both approaches empirically — zero-filling produced consistently lower validation WAPE across all five models versus date-dropping.

### Challenge 3: Understanding Why LightGBM Failed on Lens 2

My initial assumption when starting NB07 was that LightGBM — the Lens 1 champion — would likely also excel on Lens 2. When it ranked last, I had to investigate why rather than simply accept the result. I computed learning curves for LightGBM on Lens 2: training WAPE dropped to 18% (near-perfect fit) while validation WAPE remained at 54%. This confirmed overfitting — LightGBM was memorizing idiosyncratic daily dish demand patterns in the training set that did not generalise to the validation window. This was my clearest empirical demonstration of the bias-variance tradeoff.

---

## Section 3: Individual Learning Reflection (Fink's Taxonomy of Significant Learning)

### Foundational Knowledge — What I Now Understand Deeply

Before this project, I understood the concept of "data cleaning" as removing NaN values and obvious outliers. I now understand data quality as a **domain-reasoning problem**: the correct action depends entirely on understanding *why* an anomaly exists. Negative quantities are not errors — they are POS credit entries. Extreme prices are not rounded outliers — they are field-overflow corruption artefacts. The same observation demands different treatment depending on its causal mechanism.

I also developed deep understanding of the mathematical distinction between WAPE and MAPE. I could explain WAPE's formula from first principles, construct a numerical example showing MAPE's division-by-zero failure on a zero-demand day, and argue why WAPE is the gold standard for retail time-series evaluation.

### Application — Applying What I Learned to Produce Operational Value

The most powerful application insight was connecting EDA findings directly to feature design. Discovering the 7-day cyclicality pattern in day-of-week box plots directly motivated the `lag_7` feature, which became the most predictive single predictor in both lenses. EDA was not a prerequisite box-checking exercise — it was the mechanism by which the data told me which features mattered.

### Critical Thinking — Questioning Assumptions

When LightGBM failed on Lens 2, I resisted the temptation to attribute it to a bug or data problem. I instead systematically investigated: Are the features correct? (Yes, same as Lens 1 plus OHE.) Is the split correct? (Yes, same chronological split.) Is there a data quality issue unique to Lens 2? (No, same cleaning pipeline.) The remaining explanation was the algorithm itself — which led me to the bias-variance analysis that became a central finding of our model comparison.

### Human Dimension — Understanding Real-World Impact

The most memorable realisation during this project was connecting model metrics to physical kitchen operations. A Seasonal Naive baseline with 8.51 portions mean error translates to, on average, 8–9 wasted or missing portions of WATALAPPAN per day. For a dish that costs approximately LKR 180 per portion in ingredients, that is LKR 1,440–1,620 in daily waste from one dish alone, across 895 days of operations. Ridge's reduction to 5.95 portions mean error saves roughly LKR 470 per day on WATALAPPAN alone — and the same logic scales across all 11 dishes. ML is not an academic exercise; it directly reduces food waste and financial loss.

### Caring — What I Valued Through This Experience

I developed a genuine concern for responsible model deployment. When our champion models showed $R^2$ values of 0.30–0.35, I initially felt disappointed. But reflecting on what $R^2 = 0.30$ means in restaurant operations — that 30% of daily demand variance is now predictable by an algorithm that runs automatically every morning — I recognized this as a meaningful operational achievement. The remaining 70% variance is driven by weather, social media, and human behavior that no POS-only model can capture. Honest reporting of this limitation is not a weakness — it is responsible ML practice.

### Learning How to Learn — Transferable Skills Developed

This project taught me how to systematically investigate a machine learning problem where the answer is not known in advance. I developed a structured diagnostic workflow: profile the data → identify anomalies → form hypotheses → test hypotheses → document findings → connect findings to model decisions. This workflow applies to any real-world ML problem regardless of domain, and I will carry it into my professional career.

---

## Section 4: Key Technical Skills Acquired

| Skill | Evidence of Acquisition |
| :--- | :--- |
| **POS data quality analysis** | Identified and categorized 4 anomaly classes in 127,093 raw rows |
| **Temporal EDA for time-series** | Day-of-week box plots, monthly seasonality heatmaps, Pareto revenue analysis |
| **Bias-variance tradeoff analysis** | Diagnosed LightGBM overfitting via learning curves on Lens 2 |
| **WAPE metric mastery** | Can derive formula, explain rationale, and compare to MAPE from first principles |
| **Hyperparameter justification** | Explained learning_rate=0.05 and subsample=0.8 choices with theoretical backing |
| **scikit-learn pipeline implementation** | Ridge + StandardScaler in Pipeline; LightGBM with cross-validated subsampling |

---

## Section 5: What I Would Do Differently in a Future Project

1. **Seek external data earlier:** I would immediately research available Sri Lankan public holiday calendars and weather APIs to enrich the feature set. The `is_poya_day` and `is_festival_week` binary features could potentially add 3–5% WAPE improvement.

2. **Explore Poisson/Negative Binomial regression for low-volume items:** WATALAPPAN, DATE CAKE, and BUTTER CAKE have daily counts between 0–15, which are better suited to count-data regression models rather than standard real-valued regression.

3. **Implement formal statistical tests for anomaly classification:** Rather than visually identifying pre-commissioning records by date inspection, I would apply isolation forest or statistical control chart methods to automate anomaly flagging.

---

*I certify that this Personal Learning Journey report reflects my genuine individual contribution, learning experiences, and reflections. All described technical work was performed by me in the notebooks and code assigned to my responsibility.*

**Student ID:** IT24102572  
**Date:** October 2026
