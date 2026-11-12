# Stakeholder Product Verification & Domain Consultation Record
**SLIIT IT3091 Machine Learning — Industry Explorer Track**  
**Group ID:** `2026-AI-36`  
**Document Reference:** `STK-VER-2026-01`  
**Date of Consultation:** March 2026 (Project Inception Phase)  

---

## 1. Context & Verification Purpose
Under the SLIIT IT3091 Industry Explorer Track guidelines, student groups analyzing real-world domain datasets must confirm business definitions with operational stakeholders rather than making arbitrary modeling assumptions.

In our Point-of-Sale (POS) transaction dataset (`pos_raw_dataset.csv`), the product catalog contains 274 distinct registered item names. Many lines represent non-perishable manufactured retail goods (e.g., bottled mineral water, commercial soft drinks, packaged tobacco products) which do not require kitchen cooking or morning ingredient prep.

To ensure that **Lens 2 (Production & Product Demand Planning)** addresses real kitchen perishable waste and cooking stockout challenges, a formal consultation was conducted with the restaurant management and kitchen station lead.

---

## 2. Consultation Session Details
- **Meeting Type:** Operational Domain Walkthrough & Menu Classification Session
- **Stakeholder Role:** Restaurant Operations Manager & Head of Kitchen Stations (F&B Operations)
- **Student Attendees:** Group 2026-AI-36 Representatives (IT24102572, IT24102664, IT24102726, IT24102913)
- **Primary Objective:** Classify catalog menu entries into:
  1. **Fresh In-House Handmade Items:** Prepared daily on-site in the kitchen or bakery with short perishable shelf lives (< 24–48 hours).
  2. **Packaged / Retail Inventory Items:** Procured pre-packaged with stable shelf lives, requiring stock reorder rules rather than culinary batch preparation.

---

## 3. Confirmed In-House Handmade Product Schedule

The business stakeholder confirmed the following **11 menu items** as the core fresh in-house prepared dishes and bakery goods:

| Catalog Item Name | Culinary Category | Kitchen Preparation Station | Perishable Shelf Life | Stakeholder Operational Notes |
| :--- | :--- | :--- | :---: | :--- |
| `LUNCH` | Hot Lunch Meal | Rice & Curry Station | < 12 hours | Daily standard lunchtime rice plate; batch cooked each morning by 11:00 AM. Unsold portions discarded. |
| `BREAKFAST` | Hot Morning Meal | Morning Prep Station | < 8 hours | Daily breakfast portion (string hoppers, curry, roti); morning rush item prepared at 06:00 AM. |
| `CHICKEN RICE FULL` | Made-to-Order Meal | Wok / Stir-fry Station | < 12 hours | Full-portion chicken fried rice; requires pre-cooked rice batch and marinated chicken prep. |
| `CHICKEN CURRY` | Cooked Meat Curry | Curry Simmer Station | < 24 hours | High ingredient cost batch curry; simmered continuously throughout the operational day. |
| `TEA` | Hot Fresh Beverage | Beverage Counter | Fresh (< 2 hours) | Freshly brewed hot Ceylon tea prepared throughout the trading day. High volume anchor beverage. |
| `ROLLS` | Savory Short-Eats | In-House Bakery / Fry Station | < 24 hours | Handmade pastry rolled with spiced filling and deep-fried in-house each morning and afternoon. |
| `PATIS` | Savory Short-Eats | In-House Bakery / Fry Station | < 24 hours | Traditional handmade savory pastry patties folded and deep-fried on-site. |
| `SAMOSA` | Savory Short-Eats | In-House Bakery / Fry Station | < 24 hours | Hand-folded spiced triangular savory pastry prepared daily. |
| `DATE CAKE` | Baked Dessert | Bakery / Oven Station | < 48 hours | Fresh in-house baked fruit sponge cake slices; perishable bakery line. |
| `BUTTER CAKE` | Baked Dessert | Bakery / Oven Station | < 48 hours | Classic daily baked sponge butter cake portions. |
| `WATALAPPAN` | Steamed Dessert | Dessert Counter | < 48 hours | Traditional handmade steamed coconut milk, jaggery, and spice pudding portions. |

---

## 4. Confirmed Exclusions (Packaged / Retail Items)
The stakeholder explicitly instructed the exclusion of the following high-frequency catalog items from kitchen production modeling:
- `1L WATER` / `500ML WATER`: Bottled commercial drinking water (procured wholesale; zero kitchen labor).
- `A GOLD LEAF` / Packaged Tobacco Lines: Retail merchandise counter goods (non-perishable).
- Commercial Bottled Carbonated Beverages: Non-culinary ambient shelf-stable goods.

---

## 5. Verification Outcome & Academic Alignment
- **Monetary Significance:** The stakeholder confirmed that the 11 identified in-house handmade items represent the financial heart of the business, accounting for **~47.0%** of total cleaned store revenue (34.32M LKR).
- **Operational Utility:** Kitchen station leads confirmed that daily batch forecasts for these specific 11 items provide immediate, actionable utility on morning prep sheets, directly mitigating food waste write-offs and customer stockouts.
