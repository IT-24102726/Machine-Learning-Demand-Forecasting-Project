# Core Evidence 1: Problem Framing Canvas
**IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`  
**Project Title:** ChefCast AI: Restaurant Revenue & Product Demand Forecasting Using POS Data  
**Version:** Final Submission — October 2026

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
| Revenue Planning | *"How much money should we budget for this week?"* | Management | Mental estimate | 7-day forward revenue forecast (LightGBM champion) |
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
