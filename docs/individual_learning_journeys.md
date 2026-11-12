# Individual Personal Learning Journey Reports
**IT3091 Machine Learning — Final Individual Deliverable**  
**Group ID:** `2026-AI-36`

---

## Member 1: IT24102572
**Assigned Roles:**
- *Lens 1:* Revenue Data Understanding, EDA & Target Definition
- *Lens 2:* Product-Demand Model Development & Experimentation

### 1. Technical Contribution & Responsibility
I led the data understanding phase for Lens 1, dissecting the 127,093 raw POS transactions. I identified the pre-operational system setup and clock configuration anomalies in 2009, 2011, and 2020 and established the operational window from March 2024 to September 2026. For Lens 2, I developed and tuned the standalone machine learning regression models (LightGBM, XGBoost, and Random Forest), exploring hyperparameters like learning rates, tree depth, and minimum leaf samples to capture non-linear demand patterns across the 11 core fresh handmade items.

### 2. Critical Challenges Overcome
The raw data exhibited massive outlier spikes—for example, extreme price entries consistent with data entry or keying errors ($2.41 \times 10^{11}$ LKR). By analyzing descriptive statistics and box plots, I defined robust bounds ($UnitPrice \le 20,000$ LKR) that restored numerical validity. In model experimentation, standard gradient boosting overfitted high-frequency noise; I introduced subsampling (0.8) and regularization to stabilize generalization.

### 3. Key Learning & Fink's Taxonomy Reflection
- **Foundational Knowledge:** Learned the mathematical distinction between tree-based splitting on high-cardinality time series versus regularized linear models.
- **Application & Human Dimension:** Realized that an accurate model is useless if kitchen staff cannot interpret it; expressing forecasts in actionable batch portions is essential for real-world impact.

---

## Member 2: IT24102664
**Assigned Roles:**
- *Lens 1:* Revenue Data Preparation & Feature Engineering
- *Lens 2:* Product-Demand Validation, Stacking & Model Selection

### 1. Technical Contribution & Responsibility
I engineered the temporal feature pipeline for Lens 1, transforming raw transaction timestamps into 30 leak-free features, including cyclical trigonometric transforms ($\sin/\cos$ of day of week and month) and strictly shifted moving averages. For Lens 2, I orchestrated the model selection process, establishing the chronological walk-forward validation partition and implementing the Stacking Regressor meta-learner that unified Ridge, Random Forest, and LightGBM into a champion demand predictor. Crucially, I ensured champion model selection was executed strictly on the held-out validation set (reserving the test set solely for final unbiased evaluation) and configured stacking cross-validation using a custom `TimeSeriesStackingRegressor` with `TimeSeriesSplit(n_splits=3)` to generate chronological out-of-fold meta-features without future-to-past leakage.

### 2. Critical Challenges Overcome
A major risk in time-series engineering is lookahead bias (target leakage). I resolved this by enforcing an explicit $shift=1$ on all rolling windows so that day $t$'s sales were mathematically isolated from day $t$'s predictors. Furthermore, in stacking, standard random cross-validation was replaced with walk-forward `TimeSeriesSplit(n_splits=3)` cross-validation so that future folds never trained meta-weights evaluated on earlier folds.

### 3. Key Learning & Fink's Taxonomy Reflection
- **Integration:** Integrated feature engineering with ensemble theory, observing firsthand how combining structurally diverse models (linear + tree ensembles) cancelled out individual residual variances.
- **Learning How to Learn:** Mastered the practical implementation of `StackingRegressor` within scikit-learn pipelines with walk-forward `TimeSeriesSplit` cross-validation.

---

## Member 3: IT24102726
**Assigned Roles:**
- *Lens 1:* Revenue Model Development & Experimentation
- *Lens 2:* Product Selection, Demand EDA & Target Definition

### 1. Technical Contribution & Responsibility
For Lens 1, I formulated and benchmarked the candidate revenue models, developing the 7-day Seasonal Naive baseline, Ridge Regression, and tree-based regressors. For Lens 2, I spearheaded the product selection methodology, analyzing the 274 catalog items to filter out non-perishable packaged goods and pinpointing the 11 core fresh handmade items (`LUNCH`, `BREAKFAST`, `CHICKEN RICE FULL`, `CHICKEN CURRY`, `TEA`, `ROLLS`, `PATIS`, `SAMOSA`, `DATE CAKE`, `BUTTER CAKE`, and `WATALAPPAN`) that generate **~47.0%** of cleaned restaurant revenue and require active daily kitchen and bakery prep.

### 2. Critical Challenges Overcome
Initially, we considered modeling all 274 catalog items, but sporadic low-volume items (sold only once every 3 months) caused extreme zero-inflation and erratic model convergence. By formulating 5 defensible selection criteria (revenue share, daily frequency, prep freshness, and data completeness), I created a high-signal problem definition that directly solved the chef's daily cooking dilemma.

### 3. Key Learning & Fink's Taxonomy Reflection
- **Critical Thinking:** Understood that business problem framing is far more impactful than blindly throwing complex algorithms at noisy, unfiltered data.
- **Caring & Professional Practice:** Observed how thoughtful data curation directly prevents kilograms of edible cooked food from ending up in restaurant trash bins.

---

## Member 4: IT24102913
**Assigned Roles:**
- *Lens 1:* Revenue Validation, Stacking & Model Selection
- *Lens 2:* Product-Demand Data Preparation & Feature Engineering

### 1. Technical Contribution & Responsibility
For Lens 1, I designed the evaluation framework, implementing the out-of-time chronological split (70% train, 15% val, 15% test) and programming the metric evaluation suite (MAE, RMSE, WAPE, and $R^2$). For Lens 2, I constructed the long-format product demand dataset, generating product-specific autoregressive lags ($lag_1, lag_2, lag_7, lag_{14}, lag_{21}$) and one-hot entity embeddings across 4,370 item-day observations. I also co-developed the interactive Streamlit decision dashboard.

### 2. Critical Challenges Overcome
Standard MAPE generated infinite and undefined values when daily sales for specific items were zero on quiet days or holidays. I resolved this by championing WAPE (Weighted Absolute Percentage Error) as the primary assessment metric, which aggregates absolute residuals over total volume. In the dashboard, I built the interactive risk tolerance slider, allowing managers to dynamically apply asymmetric buffers against stockouts.

### 3. Key Learning & Fink's Taxonomy Reflection
- **Statistical Reasoning:** Gained deep insight into why standard regression metrics fail in real-world retail time-series and how metric selection directly governs business decisions.
- **Communication:** Bridged the gap between raw Python algorithms and end-user stakeholders through an interactive, executive-grade decision dashboard.
