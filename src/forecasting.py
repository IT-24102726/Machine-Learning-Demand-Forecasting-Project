"""
Production Forecasting Engine — ChefCast AI
Generates genuine future out-of-sample predictions using recursive
multi-step autoregression.  Feature vector construction exactly
mirrors the NB02 / NB06 pipeline so model inputs are identical at
inference time to training time.

Authors: IT24102913 (NB04), IT24102664 (NB08)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def _load_meta(path: str) -> dict:
    with open(_resolve(path)) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────
# LENS 1 — Revenue Forecasting
# ─────────────────────────────────────────────────────────────────
def forecast_future_revenue(
        daily_rev_path: str = 'data/processed/daily_revenue.csv',
        meta_path: str      = 'data/processed/revenue_feature_meta.json',
        model_path: str     = 'models/lens1_champion_model.joblib',
        horizon: int        = 7,
        buffer_pct: float   = 0.15,
) -> pd.DataFrame:
    """
    Recursively produces `horizon` days of future daily revenue forecasts.
    Feature names are read from the saved meta JSON to guarantee exact
    column alignment with the trained model.
    """
    df = pd.read_csv(_resolve(daily_rev_path), parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    meta  = _load_meta(meta_path)
    model = joblib.load(_resolve(model_path))

    if hasattr(model, 'feature_names_in_'):
        FEATURE_COLS = list(model.feature_names_in_)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('scaler', None), 'feature_names_in_'):
        FEATURE_COLS = list(model.named_steps['scaler'].feature_names_in_)
    else:
        FEATURE_COLS = meta['feature_cols']
    LAG_DAYS     = meta.get('lag_days', [1, 2, 3, 7, 14, 21])
    ROLL_WINS    = meta.get('roll_wins', [7, 14, 21])
    MAX_LAG      = max(LAG_DAYS)
    MAX_ROLL     = max(ROLL_WINS)
    LOOKBACK     = max(MAX_LAG, MAX_ROLL)

    rev_hist = list(df['Revenue'].values)
    last_date = df['Date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D'
    )

    records = []
    for i, fdate in enumerate(future_dates):
        dow   = fdate.dayofweek
        month = fdate.month

        feat = {
            'dow_sin'        : np.sin(2 * np.pi * dow / 7),
            'dow_cos'        : np.cos(2 * np.pi * dow / 7),
            'month_sin'      : np.sin(2 * np.pi * month / 12),
            'month_cos'      : np.cos(2 * np.pi * month / 12),
            'dom_sin'        : np.sin(2 * np.pi * fdate.day / 31),
            'dom_cos'        : np.cos(2 * np.pi * fdate.day / 31),
            'is_weekend'     : int(dow >= 5),
            'is_friday'      : int(dow == 4),
            'is_month_start' : int(fdate.day == 1),
            'is_month_end'   : int(fdate.day == fdate.days_in_month),
            'quarter'        : fdate.quarter,
            't_index'        : len(df) + i,
            'is_closed'      : 0,
        }

        # Lag features
        for k in LAG_DAYS:
            feat[f'lag_{k}'] = rev_hist[-k] if len(rev_hist) >= k else 0.0

        # Rolling features (shifted — uses past values only)
        for w in ROLL_WINS:
            window = rev_hist[-w:]
            feat[f'roll{w}_mean'] = float(np.mean(window))
            feat[f'roll{w}_std']  = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
            feat[f'roll{w}_max']  = float(np.max(window))
            feat[f'roll{w}_min']  = float(np.min(window))

        feat['roll7_sum'] = float(np.sum(rev_hist[-7:]))

        row = pd.DataFrame([{c: feat.get(c, 0.0) for c in FEATURE_COLS}], columns=FEATURE_COLS)
        row = row.fillna(0.0)  # Ridge/any model: replace any residual NaN with 0
        pred = float(np.maximum(model.predict(row)[0], 0.0))

        records.append({
            'Date'                      : fdate,
            'DayOfWeek'                 : fdate.day_name(),
            'Forecasted Revenue (LKR)'  : round(pred, 2),
            'Planning Lower Bound'      : round(max(pred * (1 - buffer_pct), 0), 2),
            'Planning Upper Bound'      : round(pred * (1 + buffer_pct), 2),
        })

        rev_hist.append(pred)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# LENS 2 — Product Demand Forecasting
# ─────────────────────────────────────────────────────────────────
def forecast_future_demand(
        daily_demand_path: str = 'data/processed/daily_product_demand.csv',
        meta_path: str         = 'data/processed/demand_feature_meta.json',
        model_path: str        = 'models/lens2_champion_model.joblib',
        horizon: int           = 7,
        buffer_multiplier: float = 1.15,
) -> pd.DataFrame:
    """
    Recursively produces `horizon` days of future per-dish demand forecasts.
    """
    pivot_df = pd.read_csv(_resolve(daily_demand_path), parse_dates=['Date'])
    pivot_df = pivot_df.sort_values('Date').reset_index(drop=True)

    meta  = _load_meta(meta_path)
    model = joblib.load(_resolve(model_path))

    if hasattr(model, 'feature_names_in_'):
        FEATURE_COLS = list(model.feature_names_in_)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('scaler', None), 'feature_names_in_'):
        FEATURE_COLS = list(model.named_steps['scaler'].feature_names_in_)
    else:
        FEATURE_COLS = meta['feature_cols']
    LAG_DAYS      = meta.get('lag_days',     [1, 2, 3, 7, 14, 21])
    ROLL_WINS     = meta.get('roll_wins',    [7, 14])
    ITEM_FEATURES = meta.get('item_features', [])
    ALL_ITEMS     = meta.get('items', [c for c in pivot_df.columns if c != 'Date'])

    # Build per-item history dicts
    histories = {item: list(pivot_df[item].values) for item in ALL_ITEMS if item in pivot_df.columns}

    last_date    = pivot_df['Date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D'
    )

    # Previous-day total (cross-item feature seed)
    prev_day_total = float(pivot_df.iloc[-1][[c for c in pivot_df.columns if c != 'Date']].sum())

    records = []
    for i, fdate in enumerate(future_dates):
        dow   = fdate.dayofweek
        month = fdate.month

        # Current-day total will be accumulated for the next lag
        current_day_total = 0.0

        for item in ALL_ITEMS:
            series = histories.get(item, [0.0] * 30)

            feat = {
                'dow_sin'        : np.sin(2 * np.pi * dow / 7),
                'dow_cos'        : np.cos(2 * np.pi * dow / 7),
                'month_sin'      : np.sin(2 * np.pi * month / 12),
                'month_cos'      : np.cos(2 * np.pi * month / 12),
                'dom_sin'        : np.sin(2 * np.pi * fdate.day / 31),
                'dom_cos'        : np.cos(2 * np.pi * fdate.day / 31),
                'is_weekend'     : int(dow >= 5),
                'is_friday'      : int(dow == 4),
                'is_month_start' : int(fdate.day == 1),
                'is_month_end'   : int(fdate.day == fdate.days_in_month),
                'quarter'        : fdate.quarter,
                't_index'        : len(pivot_df) + i,
                'total_qty_lag1' : prev_day_total,
            }

            for k in LAG_DAYS:
                feat[f'lag_{k}'] = series[-k] if len(series) >= k else 0.0

            for w in ROLL_WINS:
                window = series[-w:]
                feat[f'roll{w}_mean'] = float(np.mean(window)) if window else 0.0
                feat[f'roll{w}_std']  = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
                feat[f'roll{w}_max']  = float(np.max(window)) if window else 0.0

            # Item OHE
            for c in FEATURE_COLS:
                if c.startswith('item_'):
                    tgt = c[len('item_'):]
                    feat[c] = 1 if (item == tgt or item.replace(' ', '_') == tgt) else 0

            row  = pd.DataFrame([{c: feat.get(c, 0.0) for c in FEATURE_COLS}], columns=FEATURE_COLS)
            row  = row.fillna(0.0)  # Ridge/any model: replace residual NaN with 0
            pred = float(np.maximum(model.predict(row)[0], 0.0))
            prep = float(np.ceil(pred * buffer_multiplier))

            current_day_total += pred

            records.append({
                'Date'                    : fdate,
                'DayOfWeek'               : fdate.day_name(),
                'ItemName'                : item,
                'Forecast_Demand'         : round(pred, 1),
                'Recommended_Kitchen_Prep': int(prep),
            })

            histories[item].append(pred)

        prev_day_total = current_day_total

    return pd.DataFrame(records)
