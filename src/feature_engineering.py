"""
Feature Engineering Module — ChefCast AI (NB02 / NB06 aligned)
Builds leakage-free feature matrices for Lens 1 (revenue) and Lens 2 (demand).

Authors: IT24102664 (NB02 Revenue FE), IT24102913 (NB06 Demand FE)

Feature naming conventions match notebooks exactly so that the forecasting
engine can reconstruct feature vectors identically at inference time.
"""

import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _resolve(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    full = os.path.join(PROJECT_ROOT, path)
    return full if os.path.exists(full) else path


# ── Calendar / cyclical feature names (shared by both lenses) ──────
CALENDAR_FEATURES = [
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'dom_sin', 'dom_cos', 'is_weekend', 'is_friday',
    'is_month_start', 'is_month_end', 'quarter', 't_index',
]
LAG_DAYS    = [1, 2, 3, 7, 14, 21]
ROLL_WINS   = [7, 14, 21]


def _add_calendar_features(df: pd.DataFrame, date_col: str = 'Date',
                            t_index_vals=None) -> pd.DataFrame:
    """Adds cyclical calendar and binary indicator features (NB02 §4, NB06 §4)."""
    df = df.copy()
    d = pd.to_datetime(df[date_col])

    df['dow_sin']       = np.sin(2 * np.pi * d.dt.dayofweek / 7)
    df['dow_cos']       = np.cos(2 * np.pi * d.dt.dayofweek / 7)
    df['month_sin']     = np.sin(2 * np.pi * d.dt.month / 12)
    df['month_cos']     = np.cos(2 * np.pi * d.dt.month / 12)
    df['dom_sin']       = np.sin(2 * np.pi * d.dt.day / 31)
    df['dom_cos']       = np.cos(2 * np.pi * d.dt.day / 31)
    df['is_weekend']    = (d.dt.dayofweek >= 5).astype(int)
    df['is_friday']     = (d.dt.dayofweek == 4).astype(int)
    df['is_month_start']= d.dt.is_month_start.astype(int)
    df['is_month_end']  = d.dt.is_month_end.astype(int)
    df['quarter']       = d.dt.quarter
    df['t_index']       = (
        t_index_vals if t_index_vals is not None
        else np.arange(len(df))
    )
    return df


def build_revenue_feature_matrix(
        filepath: str = 'data/processed/daily_revenue.csv'):
    """
    Builds the Lens 1 feature matrix (NB02 §3–9).
    Feature names exactly match NB02 decision log.

    Returns
    -------
    df_model : pd.DataFrame  — shape (N, 2 + n_features)
    feature_cols : list[str]
    """
    filepath = _resolve(filepath)
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df = _add_calendar_features(df, 'Date')
    df['is_closed'] = (df['Revenue'] == 0).astype(int)

    # Per-lag features (NB02 §5)
    for k in LAG_DAYS:
        df[f'lag_{k}'] = df['Revenue'].shift(k)

    # Rolling window features — shifted by 1 to prevent leakage (NB02 §6)
    rev_s1 = df['Revenue'].shift(1)
    for w in ROLL_WINS:
        df[f'roll{w}_mean'] = rev_s1.rolling(w, min_periods=1).mean()
        df[f'roll{w}_std']  = rev_s1.rolling(w, min_periods=2).std().fillna(0)
        df[f'roll{w}_max']  = rev_s1.rolling(w, min_periods=1).max()
        df[f'roll{w}_min']  = rev_s1.rolling(w, min_periods=1).min()
    df['roll7_sum'] = rev_s1.rolling(7, min_periods=1).sum()

    lag_features     = [f'lag_{k}' for k in LAG_DAYS]
    rolling_features = (
        [f'roll{w}_mean' for w in ROLL_WINS] +
        [f'roll{w}_std'  for w in ROLL_WINS] +
        [f'roll{w}_max'  for w in ROLL_WINS] +
        [f'roll{w}_min'  for w in ROLL_WINS] +
        ['roll7_sum']
    )
    all_features = CALENDAR_FEATURES + lag_features + rolling_features + ['is_closed']

    df_model = df[['Date', 'Revenue'] + all_features].dropna().reset_index(drop=True)

    # Persist metadata for forecasting engine
    meta = {
        'feature_cols': all_features,
        'target_col'  : 'Revenue',
        'date_col'    : 'Date',
        'lag_days'    : LAG_DAYS,
        'roll_wins'   : ROLL_WINS,
        'n_rows'      : len(df_model),
        'date_min'    : str(df_model['Date'].min().date()),
        'date_max'    : str(df_model['Date'].max().date()),
    }
    meta_path = _resolve('data/processed/revenue_feature_meta.json')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    feat_path = _resolve('data/processed/revenue_feature_matrix.csv')
    df_model.to_csv(feat_path, index=False)

    return df_model, all_features


def build_product_demand_feature_matrix(
        filepath: str = 'data/processed/daily_product_demand_long.csv'):
    """
    Builds the Lens 2 panel feature matrix (NB06 §3–11).
    Per-item lags, per-item rolling windows, cross-item feature,
    and item one-hot encoding.

    Returns
    -------
    df_model : pd.DataFrame
    feature_cols : list[str]
    """
    filepath = _resolve(filepath)
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values(['ItemName', 'Date']).reset_index(drop=True)

    all_items = sorted(df['ItemName'].unique().tolist())

    # Calendar features (NB06 §4)
    df = _add_calendar_features(df, 'Date',
                                  t_index_vals=df.groupby('ItemName').cumcount().values)

    # Per-item lag features (NB06 §5)
    for k in LAG_DAYS:
        df[f'lag_{k}'] = df.groupby('ItemName')['Qty'].shift(k)

    # Per-item rolling features — shifted to prevent leakage (NB06 §6)
    for w in [7, 14]:
        df[f'roll{w}_mean'] = (
            df.groupby('ItemName')['Qty']
            .shift(1)
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f'roll{w}_std'] = (
            df.groupby('ItemName')['Qty']
            .shift(1)
            .transform(lambda x: x.rolling(w, min_periods=2).std().fillna(0))
        )
        df[f'roll{w}_max'] = (
            df.groupby('ItemName')['Qty']
            .shift(1)
            .transform(lambda x: x.rolling(w, min_periods=1).max())
        )

    # Cross-item feature: previous day's total demand (NB06 §7)
    date_totals = df.groupby('Date')['Qty'].sum().rename('_date_total')
    df = df.merge(date_totals, on='Date', how='left')
    lag1_map = (
        date_totals
        .reset_index()
        .assign(Date=lambda d: d['Date'] + pd.Timedelta(days=1))
        .set_index('Date')['_date_total']
    )
    df['total_qty_lag1'] = df['Date'].map(lag1_map).fillna(0)
    df.drop(columns=['_date_total'], inplace=True)

    # Item one-hot encoding (NB06 §8)
    ohe = pd.get_dummies(df['ItemName'], prefix='item', drop_first=False).astype(int)
    df = pd.concat([df, ohe], axis=1)

    lag_features     = [f'lag_{k}' for k in LAG_DAYS]
    rolling_features = (
        ['roll7_mean', 'roll7_std', 'roll7_max',
         'roll14_mean', 'roll14_std', 'roll14_max']
    )
    item_features    = [c for c in ohe.columns]
    cross_features   = ['total_qty_lag1']

    all_features = (
        CALENDAR_FEATURES +
        lag_features +
        rolling_features +
        cross_features +
        item_features
    )

    df_model = (
        df[['Date', 'ItemName', 'Qty'] + all_features]
        .dropna()
        .reset_index(drop=True)
    )

    # Persist metadata
    meta = {
        'feature_cols': all_features,
        'target_col'  : 'Qty',
        'date_col'    : 'Date',
        'item_col'    : 'ItemName',
        'items'       : all_items,
        'lag_days'    : LAG_DAYS,
        'roll_wins'   : [7, 14],
        'item_features': item_features,
        'n_rows'      : len(df_model),
        'date_min'    : str(df_model['Date'].min().date()),
        'date_max'    : str(df_model['Date'].max().date()),
    }
    meta_path = _resolve('data/processed/demand_feature_meta.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    feat_path = _resolve('data/processed/demand_feature_matrix.csv')
    df_model.to_csv(feat_path, index=False)

    return df_model, all_features


if __name__ == '__main__':
    rev_df, rev_feats = build_revenue_feature_matrix()
    print(f"Revenue FM: {rev_df.shape} | {len(rev_feats)} features")
    prod_df, prod_feats = build_product_demand_feature_matrix()
    print(f"Demand FM : {prod_df.shape} | {len(prod_feats)} features")
