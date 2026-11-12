# 3-Minute Demo Video Presentation Script & Visual Storyboard
**IT3091 Machine Learning — Final Group Submission Deliverable**  
**Group ID:** `2026-AI-36`  
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
