# Data Dictionary & Privacy Statement
**IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`  
**Project:** ChefCast AI — Restaurant Revenue & Product Demand Forecasting  
**Version:** Final Submission — October 2026

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
