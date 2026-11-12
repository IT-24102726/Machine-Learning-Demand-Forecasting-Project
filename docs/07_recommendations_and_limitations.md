# Core Evidence 7: Practical Recommendations & System Limitations
**IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`  
**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  
**Version:** Final Submission — October 2026

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
