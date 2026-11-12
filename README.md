# ChefCast AI: Restaurant Revenue & Product Demand Decision Support System
**SLIIT IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`

---

## 📌 Project Overview
ChefCast AI is an evidence-based machine learning forecasting system that translates real-world restaurant Point-of-Sale (POS) transaction records into actionable weekly financial revenue expectations and kitchen portion preparation schedules.

- **Primary Lens (Revenue Planning & Forecasting):** Forecasts total expected revenue across the upcoming 7 days from 11 core fresh handmade kitchen dishes for executive cash-flow budgeting.
- **Secondary Lens (Production & Demand Planning):** Forecasts daily portion sales for all **11 core fresh handmade menu items** (`LUNCH`, `BREAKFAST`, `CHICKEN RICE FULL`, `CHICKEN CURRY`, `TEA`, `ROLLS`, `PATIS`, `SAMOSA`, `DATE CAKE`, `BUTTER CAKE`, and `WATALAPPAN`) to guide morning kitchen batching, reducing perishable food waste while mitigating stockouts. These fresh items generate **~47.0%** of cleaned restaurant turnover.

---

## 🏛️ Project Structure
```
./
├── app.py                      # Interactive Streamlit Decision Support Dashboard
├── requirements.txt            # Pinned Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
│
├── data/
│   ├── raw/                    # Raw POS transactions (pos_raw_dataset.csv)
│   └── processed/              # Cleaned temporal aggregations & summaries
│       ├── cleaned_pos_transactions.csv
│       ├── daily_revenue_series.csv
│       ├── daily_product_demand_long.csv
│       ├── daily_product_demand_wide.csv
│       └── product_selection_summary.csv
│
├── notebooks/                  # 9 Aligned Jupyter Notebooks (Group Responsibility Allocation)
│   ├── Lens1_NB01_Revenue_EDA_and_Target_Definition.ipynb                 # IT24102572
│   ├── Lens1_NB02_Revenue_Data_Preparation_and_Feature_Engineering.ipynb  # IT24102664
│   ├── Lens1_NB03_Revenue_Model_Development_and_Experimentation.ipynb     # IT24102726
│   ├── Lens1_NB04_Revenue_Validation_Stacking_and_Model_Selection.ipynb    # IT24102913
│   ├── Lens2_NB05_Product_Selection_Demand_EDA_and_Target_Definition.ipynb # IT24102726
│   ├── Lens2_NB06_Product_Demand_Data_Preparation_and_Feature_Engineering.ipynb # IT24102913
│   ├── Lens2_NB07_Product_Demand_Model_Development_and_Experimentation.ipynb    # IT24102572
│   ├── Lens2_NB08_Product_Demand_Validation_Stacking_and_Model_Selection.ipynb # IT24102664
│   └── NB09_Ensemble_Architecture_and_Comparative_Evaluation_Benchmark.ipynb    # Joint / Synthesized
│
├── src/                        # Modular production pipeline code
│   ├── __init__.py
│   ├── data_pipeline.py        # Ingestion, audit, cleaning & daily aggregation
│   ├── feature_engineering.py  # Leakage-free calendar, lag & rolling features
│   ├── models.py               # Ridge, RF, LightGBM, XGBoost & TimeSeries Stacking Ensemble
│   ├── evaluate.py             # Chronological validation, benchmark logging & model persistence
│   └── forecasting.py          # Recursive forward multi-step forecasting engine
│
├── models/                     # Serialized model artifacts (.joblib, metrics.json)
│   ├── lens1_*.joblib          # Trained Lens 1 models & champion
│   ├── lens2_*.joblib          # Trained Lens 2 models & champion
│   └── metrics.json            # Benchmarking and validation metrics
│
└── docs/                       # Assessment Evidence Deliverables (Untouched)
```

---

## 🚀 Quickstart Guide

### 1. Installation
Install pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
To reproduce all data preprocessing, leakage-safe feature engineering, and model training:
```bash
python -X utf8 -m src.data_pipeline
python -X utf8 -m src.evaluate
```

### 3. Launch Interactive Decision Support UI
To launch the executive decision dashboard:
```bash
python -X utf8 -m streamlit run app.py
```
Or open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏆 Model Benchmark & Evaluation Summary

Evaluated on an **out-of-time chronological test split** (132 continuous held-out days, April 29 to September 7, 2026), with champions selected strictly on the held-out validation set (131 days):

### Lens 1: Combined Daily Revenue Forecasting
| Model Architecture | Val WAPE (%) | Test MAE (LKR) | Test RMSE (LKR) | Test WAPE (%) | Test $R^2$ | Status & Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | **30.33%** | **9,109.36** | **12,752.06** | **24.99%** | **0.3521** | 🥇 **Champion (Selected via Val WAPE)** |
| Ridge Regression | 30.54% | 8,734.59 | 11,943.78 | 23.96% | 0.4316 | 🥈 Runner-up |
| Stacking Ensemble | 31.32% | 9,284.71 | 12,375.86 | 25.47% | 0.3898 | 🥉 3rd |
| XGBoost Regressor | 31.67% | 9,730.90 | 13,483.93 | 26.70% | 0.2756 | 4th |
| Random Forest | 32.17% | 10,253.93 | 13,962.09 | 28.13% | 0.2233 | 5th |
| *Baseline Seasonal Naive* | 46.22% | 15,728.64 | 20,831.72 | 43.15% | -0.7290 | *Baseline Benchmark (Beaten)* |

### Lens 2: Product-Level Daily Demand Forecasting (All 11 Dishes)
| Model Architecture | Val WAPE (%) | Test MAE (Units) | Test RMSE (Units) | Test WAPE (%) | Test $R^2$ | Status & Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ridge Regression** | **47.31%** | **5.95** | **8.77** | **44.38%** | **0.2972** | 🥇 **Champion (Selected via Val WAPE)** |
| Stacking Ensemble | 48.06% | 5.94 | 8.67 | 44.35% | 0.3135 | 🥈 Runner-up |
| Random Forest | 48.26% | 5.95 | 8.88 | 44.39% | 0.2808 | 🥉 3rd |
| XGBoost Regressor | 50.92% | 6.44 | 9.49 | 48.07% | 0.1776 | 4th |
| LightGBM Regressor | 54.56% | 6.88 | 9.67 | 51.36% | 0.1471 | 5th |
| *Baseline Seasonal Naive* | 65.71% | 8.51 | 12.16 | 63.49% | -0.3500 | *Baseline Benchmark (Beaten)* |

---

## 👥 Group Allocations (Group 2026-AI-36)
- **IT24102572:** Lens 1 NB01 (Revenue EDA & Target Definition) & Lens 2 NB07 (Product Demand Model Development)
- **IT24102664:** Lens 1 NB02 (Revenue Data Prep & Feature Engineering) & Lens 2 NB08 (Product Demand Validation & Model Selection)
- **IT24102726:** Lens 1 NB03 (Revenue Model Development) & Lens 2 NB05 (Product Selection, Demand EDA & Target Definition)
- **IT24102913:** Lens 1 NB04 (Revenue Validation & Model Selection) & Lens 2 NB06 (Product Demand Data Prep & Feature Engineering)
- **Joint / All:** NB09 (Ensemble Architecture & Comparative Evaluation Benchmark) & Decision Support Dashboard (`app.py`)

---

## 📂 Final Assessment Deliverables Directory (`submissions/` & `docs/`)

All deliverables have been compiled as both clean Markdown documents in `docs/` and publication-ready A4 PDF files in `submissions/` (and mirrored in `docs/`):

### 1. Final Group Submission (Submitted by Group Leader)
- **Primary Group Report:** [`Group_Final_Report_ChefCast_AI.pdf`](./submissions/Group_Final_Report_ChefCast_AI.pdf) | Source: [`final_group_report.md`](./docs/final_group_report.md)
- **Core Evidence 1 (Problem Framing):** [`CoreEvidence_01_Problem_Framing_Canvas.pdf`](./submissions/CoreEvidence_01_Problem_Framing_Canvas.pdf) | Source: [`01_problem_framing_canvas.md`](./docs/01_problem_framing_canvas.md)
- **Core Evidence 2 (ML Workflow Diagram):** [`CoreEvidence_02_Workflow_Diagram.pdf`](./submissions/CoreEvidence_02_Workflow_Diagram.pdf) | Source: [`02_workflow_diagram.md`](./docs/02_workflow_diagram.md)
- **Core Evidence 3 (Decision Log):** [`CoreEvidence_03_Decision_Log.pdf`](./submissions/CoreEvidence_03_Decision_Log.pdf) | Source: [`03_decision_log.md`](./docs/03_decision_log.md)
- **Core Evidence 4 (EDA Insight Log):** [`CoreEvidence_04_EDA_Insight_Log.pdf`](./submissions/CoreEvidence_04_EDA_Insight_Log.pdf) | Source: [`04_eda_insight_log.md`](./docs/04_eda_insight_log.md)
- **Core Evidence 5 (Feature Engineering Log):** [`CoreEvidence_05_Preprocessing_Feature_Log.pdf`](./submissions/CoreEvidence_05_Preprocessing_Feature_Log.pdf) | Source: [`05_preprocessing_feature_log.md`](./docs/05_preprocessing_feature_log.md)
- **Core Evidence 6 (Model Comparison):** [`CoreEvidence_06_Model_Comparison_and_Evaluation.pdf`](./submissions/CoreEvidence_06_Model_Comparison_and_Evaluation.pdf) | Source: [`06_model_comparison_and_evaluation.md`](./docs/06_model_comparison_and_evaluation.md)
- **Core Evidence 7 (Recommendations & Limitations):** [`CoreEvidence_07_Recommendations_and_Limitations.pdf`](./submissions/CoreEvidence_07_Recommendations_and_Limitations.pdf) | Source: [`07_recommendations_and_limitations.md`](./docs/07_recommendations_and_limitations.md)
- **Data Dictionary & Privacy Statement:** [`CoreEvidence_Data_Dictionary_and_Privacy.pdf`](./submissions/CoreEvidence_Data_Dictionary_and_Privacy.pdf) | Source: [`data_dictionary_and_privacy.md`](./docs/data_dictionary_and_privacy.md)
- **3-Minute YouTube Video Script:** [`CoreEvidence_Video_Presentation_Script.pdf`](./submissions/CoreEvidence_Video_Presentation_Script.pdf) | Source: [`video_presentation_script.md`](./docs/video_presentation_script.md)

### 2. Individual Submissions (Submitted by Each Group Member)
- **Student IT24102572:** [`Individual_Submission_IT24102572_PLJ.pdf`](./submissions/Individual_Submission_IT24102572_PLJ.pdf) | Source: [`individual_PLJ_IT24102572.md`](./docs/individual_PLJ_IT24102572.md)
- **Student IT24102664:** [`Individual_Submission_IT24102664_PLJ.pdf`](./submissions/Individual_Submission_IT24102664_PLJ.pdf) | Source: [`individual_PLJ_IT24102664.md`](./docs/individual_PLJ_IT24102664.md)
- **Student IT24102726:** [`Individual_Submission_IT24102726_PLJ.pdf`](./submissions/Individual_Submission_IT24102726_PLJ.pdf) | Source: [`individual_PLJ_IT24102726.md`](./docs/individual_PLJ_IT24102726.md)
- **Student IT24102913:** [`Individual_Submission_IT24102913_PLJ.pdf`](./submissions/Individual_Submission_IT24102913_PLJ.pdf) | Source: [`individual_PLJ_IT24102913.md`](./docs/individual_PLJ_IT24102913.md)
