"""
Data Pipeline Module — ChefCast AI (NB01 / NB05 aligned)
Handles raw POS ingestion, validation, anomaly filtering,
handmade product selection, and daily time-series aggregation.

Authors: IT24102572 (NB01 Revenue EDA), IT24102726 (NB05 Demand EDA)
"""

import os
import json
import pandas as pd
import numpy as np

# ── Core item catalogue ────────────────────────────────────────────
CORE_ITEMS = [
    'LUNCH', 'BREAKFAST', 'CHICKEN RICE FULL', 'CHICKEN CURRY',
    'TEA', 'ROLLS', 'PATIS', 'SAMOSA',
    'DATE CAKE', 'BUTTER CAKE', 'WATALAPPAN'
]

# ── Validation thresholds (NB01 §4) ───────────────────────────────
MIN_YEAR       = 2024       # pre-2024 = terminal test / clock-reset records
MAX_UNIT_PRICE = 20_000     # LKR ceiling for any single menu item
MAX_QTY        = 300        # physiological ceiling for fresh-dish orders


def load_and_audit_raw_data(filepath='data/raw/pos_raw_dataset.csv'):
    """
    Loads raw POS transactions, standardises columns, parses dates,
    and returns the validated DataFrame together with a full audit log.

    Cleaning rules (NB01 §4–5):
        1. Restrict to year >= 2024   (pre-2024 = terminal reset rows)
        2. Remove Qty <= 0            (returns, voids, refund entries)
        3. Remove Qty > 300           (system glitch / extreme bulk)
        4. Remove UnitPrice <= 0      (cashier discount / giveaway)
        5. Remove UnitPrice > 20 000  (data-entry typos / bulk catering)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data not found at: {filepath}")

    df_raw = pd.read_csv(filepath)
    raw_count = len(df_raw)

    # ── Standardise ────────────────────────────────────────────────
    df_raw.columns  = [c.strip() for c in df_raw.columns]
    df_raw['ItemName'] = df_raw['ItemName'].astype(str).str.strip().str.upper()
    df_raw['INVDATE']  = pd.to_datetime(df_raw['INVDATE'], errors='coerce')

    # ── Audit log (before filtering) ───────────────────────────────
    audit = {
        'total_raw_rows'        : raw_count,
        'null_dates'            : int(df_raw['INVDATE'].isna().sum()),
        'pre_2024_records'      : int((df_raw['INVDATE'].dt.year < MIN_YEAR).sum()),
        'negative_zero_qty'     : int((df_raw['Qty'] <= 0).sum()),
        'extreme_qty_glitches'  : int((df_raw['Qty'] > MAX_QTY).sum()),
        'negative_zero_price'   : int((df_raw['UnitPrice'] <= 0).sum()),
        'extreme_price_anomalies': int((df_raw['UnitPrice'] > MAX_UNIT_PRICE).sum()),
    }

    # ── Apply filters ──────────────────────────────────────────────
    df_valid = df_raw[
        (df_raw['INVDATE'].dt.year >= MIN_YEAR) &
        (df_raw['Qty'] > 0)                     &
        (df_raw['Qty'] <= MAX_QTY)              &
        (df_raw['UnitPrice'] > 0)               &
        (df_raw['UnitPrice'] <= MAX_UNIT_PRICE)
    ].copy()

    df_valid['LineRevenue'] = df_valid['UnitPrice'] * df_valid['Qty']

    audit['clean_valid_rows']   = len(df_valid)
    audit['filtered_out_rows']  = raw_count - len(df_valid)
    audit['retention_pct']      = round(len(df_valid) / raw_count * 100, 2)
    audit['operational_start']  = str(df_valid['INVDATE'].min().date())
    audit['operational_end']    = str(df_valid['INVDATE'].max().date())
    audit['total_revenue_clean']= float(df_valid['LineRevenue'].sum())

    return df_valid, audit


def extract_handmade_datasets(df_valid, output_dir='data/processed'):
    """
    Produces the four processed artefacts needed by the 9 notebooks:
        1. data/processed/daily_revenue.csv            (Lens 1 target)
        2. data/processed/daily_product_demand.csv     (Lens 2 wide)
        3. data/processed/daily_product_demand_long.csv (Lens 2 long / panel)
        4. data/processed/data_summary.json            (metadata)
    """
    os.makedirs(output_dir, exist_ok=True)

    df_core = df_valid[df_valid['ItemName'].isin(CORE_ITEMS)].copy()
    df_core['Date'] = pd.to_datetime(df_core['INVDATE'].dt.date)

    min_date = df_core['Date'].min()
    max_date = df_core['Date'].max()
    full_calendar = pd.date_range(start=min_date, end=max_date, freq='D')

    # ── 1. Lens 1: daily revenue ───────────────────────────────────
    daily_rev = (
        df_core.groupby('Date')['LineRevenue']
        .sum()
        .reindex(full_calendar, fill_value=0.0)
        .rename_axis('Date')
        .rename('Revenue')
        .reset_index()
    )
    daily_rev_path = os.path.join(output_dir, 'daily_revenue.csv')
    daily_rev.to_csv(daily_rev_path, index=False)

    # ── 2. Lens 2: daily demand (wide format) ─────────────────────
    prod_daily = (
        df_core.groupby(['Date', 'ItemName'])['Qty']
        .sum()
        .reset_index()
    )
    pivot_demand = (
        prod_daily
        .pivot(index='Date', columns='ItemName', values='Qty')
        .reindex(full_calendar, fill_value=0.0)
        .rename_axis('Date')
        .reset_index()
    )
    # Ensure all 11 core items have a column (even if zero sales)
    for item in CORE_ITEMS:
        if item not in pivot_demand.columns:
            pivot_demand[item] = 0.0

    wide_path = os.path.join(output_dir, 'daily_product_demand.csv')
    pivot_demand.to_csv(wide_path, index=False)

    # ── 3. Lens 2: long (panel) format ────────────────────────────
    long_demand = pivot_demand.melt(
        id_vars=['Date'], var_name='ItemName', value_name='Qty'
    ).sort_values(['Date', 'ItemName']).reset_index(drop=True)
    long_path = os.path.join(output_dir, 'daily_product_demand_long.csv')
    long_demand.to_csv(long_path, index=False)

    # ── 4. Metadata / data_summary.json ───────────────────────────
    historical_avg_prices = {}
    for item in CORE_ITEMS:
        rows = df_core[df_core['ItemName'] == item]
        total_qty = rows['Qty'].sum()
        historical_avg_prices[item] = (
            round(float(rows['LineRevenue'].sum() / total_qty), 2)
            if total_qty > 0 else 200.0
        )

    summary = {
        'selected_items'          : CORE_ITEMS,
        'date_range'              : [str(min_date.date()), str(max_date.date())],
        'total_calendar_days'     : len(full_calendar),
        'total_revenue_handmade'  : float(daily_rev['Revenue'].sum()),
        'mean_daily_revenue'      : float(daily_rev['Revenue'].mean()),
        'total_valid_store_revenue': float(df_valid['LineRevenue'].sum()),
        'product_qty_totals'      : {
            item: float(pivot_demand[item].sum()) for item in CORE_ITEMS
        },
        'historical_avg_prices'   : historical_avg_prices,
    }
    summary_path = os.path.join(output_dir, 'data_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"[Pipeline] Saved → {daily_rev_path}  ({len(daily_rev)} days)")
    print(f"[Pipeline] Saved → {wide_path}  ({pivot_demand.shape})")
    print(f"[Pipeline] Saved → {long_path}  ({len(long_demand)} rows)")
    print(f"[Pipeline] Saved → {summary_path}")
    return daily_rev, pivot_demand, summary


if __name__ == '__main__':
    df_valid, audit = load_and_audit_raw_data()
    print("Audit:", json.dumps(audit, indent=2))
    daily_rev, pivot_demand, summary = extract_handmade_datasets(df_valid)
    print("Summary:", json.dumps(
        {k: v for k, v in summary.items() if k != 'historical_avg_prices'}, indent=2
    ))
