"""
Evaluation & Training Pipeline — ChefCast AI (NB03/NB04/NB07/NB08 aligned)
Trains all models, computes metrics, selects champions on validation WAPE,
saves artefacts, and dumps unified metrics.json.

Authors: IT24102726 (NB03), IT24102913 (NB04/NB08), IT24102572 (NB07), IT24102664 (NB08)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


from src.feature_engineering import (
    build_revenue_feature_matrix,
    build_product_demand_feature_matrix,
)
from src.models import get_model_registry, save_model


# -----------------------------------------------------------------
# Metrics (NB03 §3 compute_metrics)
# -----------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(
        np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-9) * 100
    )
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Safe MAPE on non-zero points only
    nz = y_true > 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else 0.0

    return {
        'MAE'         : round(mae,  2),
        'RMSE'        : round(rmse, 2),
        'WAPE_percent': round(wape, 2),
        'MAPE_percent': round(mape, 2),
        'R2_Score'    : round(r2,   4),
    }


# -----------------------------------------------------------------
# Temporal split helper
# -----------------------------------------------------------------
def temporal_split(df, date_col='Date', train_frac=0.70, val_frac=0.15):
    """
    Strict chronological 70/15/15 split by unique calendar dates.
    All items for a date stay in the same partition (critical for Lens 2).
    """
    dates = np.sort(df[date_col].unique())
    n = len(dates)
    n_tr = int(n * train_frac)
    n_va = int(n * (train_frac + val_frac))

    tr_d = set(dates[:n_tr])
    va_d = set(dates[n_tr:n_va])
    te_d = set(dates[n_va:])

    sort_cols = [date_col, 'ItemName'] if 'ItemName' in df.columns else [date_col]
    tr = df[df[date_col].isin(tr_d)].sort_values(sort_cols).reset_index(drop=True)
    va = df[df[date_col].isin(va_d)].sort_values(sort_cols).reset_index(drop=True)
    te = df[df[date_col].isin(te_d)].sort_values(sort_cols).reset_index(drop=True)
    return tr, va, te


# -----------------------------------------------------------------
# LENS 1
# -----------------------------------------------------------------
def train_and_evaluate_lens1(output_dir: str = 'models') -> tuple:
    print("\n" + "="*60)
    print("  LENS 1 — DAILY REVENUE FORECASTING")
    print("="*60)

    output_dir = _resolve(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rev_df, feature_cols = build_revenue_feature_matrix()
    train_df, val_df, test_df = temporal_split(rev_df, 'Date')

    X_tr, y_tr = train_df[feature_cols], train_df['Revenue']
    X_va, y_va = val_df[feature_cols],   val_df['Revenue']
    X_te, y_te = test_df[feature_cols],  test_df['Revenue']

    print(f"  Train: {len(train_df)}d | Val: {len(val_df)}d | Test: {len(test_df)}d")

    models  = get_model_registry()
    results = {}
    preds_df = pd.DataFrame({'Date': test_df['Date'].values, 'Actual': y_te.values})

    best_name = None
    best_val_wape = float('inf')
    best_obj  = None

    for name, model in models.items():
        print(f"  Training: {name} ...", end='', flush=True)

        # LightGBM / XGBoost need eval_set for early stopping
        if name == 'LightGBM':
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[
                          __import__('lightgbm').early_stopping(50, verbose=False),
                          __import__('lightgbm').log_evaluation(-1),
                      ])
        elif name == 'XGBoost':
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      verbose=False)
        else:
            model.fit(X_tr, y_tr)

        va_pred = np.maximum(model.predict(X_va), 0)
        te_pred = np.maximum(model.predict(X_te), 0)

        vm = compute_metrics(y_va, va_pred)
        tm = compute_metrics(y_te, te_pred)

        results[name] = {'validation': vm, 'test': tm}
        preds_df[name] = te_pred

        print(f"  Val WAPE={vm['WAPE_percent']:.1f}%  Test WAPE={tm['WAPE_percent']:.1f}%")

        if vm['WAPE_percent'] < best_val_wape:
            best_val_wape = vm['WAPE_percent']
            best_name = name
            best_obj  = model

        save_model(model, os.path.join(output_dir, f'lens1_{name}.joblib'))

    # Feature importance from LightGBM (if training succeeded)
    lgb_model = models['LightGBM']
    if hasattr(lgb_model, 'feature_importances_'):
        pd.DataFrame({
            'feature'   : feature_cols,
            'importance': lgb_model.feature_importances_,
        }).sort_values('importance', ascending=False).to_csv(
            os.path.join(output_dir, 'lens1_feature_importance.csv'), index=False
        )
    elif hasattr(models['Random_Forest'], 'feature_importances_'):
        pd.DataFrame({
            'feature'   : feature_cols,
            'importance': models['Random_Forest'].feature_importances_,
        }).sort_values('importance', ascending=False).to_csv(
            os.path.join(output_dir, 'lens1_feature_importance.csv'), index=False
        )

    preds_df.to_csv(os.path.join(output_dir, 'lens1_test_predictions.csv'), index=False)

    # Retrain champion on Train+Val for deployment (no eval_set = no early stopping)
    tv_df = pd.concat([train_df, val_df], ignore_index=True)
    deploy_models = get_model_registry()
    deploy = deploy_models[best_name]
    if best_name == 'LightGBM':
        # Remove early_stopping for final fit — no separate val set
        deploy.set_params(n_estimators=500)
        deploy.fit(tv_df[feature_cols], tv_df['Revenue'])
    elif best_name == 'XGBoost':
        deploy.set_params(early_stopping_rounds=None)
        deploy.fit(tv_df[feature_cols], tv_df['Revenue'])
    else:
        deploy.fit(tv_df[feature_cols], tv_df['Revenue'])
    save_model(deploy, os.path.join(output_dir, 'lens1_champion_model.joblib'))

    # 90th-percentile residual buffer (empirical confidence band)
    val_res = np.abs(y_va.values - np.maximum(best_obj.predict(X_va), 0))
    q90 = float(np.percentile(val_res, 90))

    results['champion_info'] = {
        'champion_model'       : best_name,
        'selection_method'     : 'Lowest Validation WAPE',
        'validation_wape'      : round(best_val_wape, 2),
        'unbiased_test_wape'   : results[best_name]['test']['WAPE_percent'],
        'empirical_residual_q90': round(q90, 2),
    }

    print(f"\n  [CHAMPION] {best_name} | Val WAPE={best_val_wape:.2f}% | Test WAPE={results[best_name]['test']['WAPE_percent']:.2f}%")
    return results, preds_df


# -----------------------------------------------------------------
# LENS 2
# -----------------------------------------------------------------
def train_and_evaluate_lens2(output_dir: str = 'models') -> tuple:
    print("\n" + "="*60)
    print("  LENS 2 — PRODUCT DEMAND FORECASTING")
    print("="*60)

    output_dir = _resolve(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prod_df, feature_cols = build_product_demand_feature_matrix()
    train_df, val_df, test_df = temporal_split(prod_df, 'Date')

    X_tr, y_tr = train_df[feature_cols], train_df['Qty']
    X_va, y_va = val_df[feature_cols],   val_df['Qty']
    X_te, y_te = test_df[feature_cols],  test_df['Qty']

    print(f"  Train: {len(train_df)} rows | Val: {len(val_df)} | Test: {len(test_df)}")

    models  = get_model_registry()
    results = {}
    preds_df = pd.DataFrame({
        'Date'    : test_df['Date'].values,
        'ItemName': test_df['ItemName'].values,
        'Actual'  : y_te.values,
    })

    best_name = None
    best_val_wape = float('inf')
    best_obj  = None

    for name, model in models.items():
        print(f"  Training: {name} ...", end='', flush=True)

        if name == 'LightGBM':
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[
                          __import__('lightgbm').early_stopping(50, verbose=False),
                          __import__('lightgbm').log_evaluation(-1),
                      ])
        elif name == 'XGBoost':
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr)

        va_pred = np.maximum(model.predict(X_va), 0)
        te_pred = np.maximum(model.predict(X_te), 0)

        vm = compute_metrics(y_va, va_pred)
        tm = compute_metrics(y_te, te_pred)

        results[name] = {'validation': vm, 'test': tm}
        preds_df[name] = te_pred

        print(f"  Val WAPE={vm['WAPE_percent']:.1f}%  Test WAPE={tm['WAPE_percent']:.1f}%")

        if vm['WAPE_percent'] < best_val_wape:
            best_val_wape = vm['WAPE_percent']
            best_name = name
            best_obj  = model

        save_model(model, os.path.join(output_dir, f'lens2_{name}.joblib'))

    # Feature importance from LightGBM (or RF fallback)
    lgb_model = models['LightGBM']
    if hasattr(lgb_model, 'feature_importances_'):
        pd.DataFrame({
            'feature'   : feature_cols,
            'importance': lgb_model.feature_importances_,
        }).sort_values('importance', ascending=False).to_csv(
            os.path.join(output_dir, 'lens2_feature_importance.csv'), index=False
        )
    elif hasattr(models['Random_Forest'], 'feature_importances_'):
        pd.DataFrame({
            'feature'   : feature_cols,
            'importance': models['Random_Forest'].feature_importances_,
        }).sort_values('importance', ascending=False).to_csv(
            os.path.join(output_dir, 'lens2_feature_importance.csv'), index=False
        )

    preds_df.to_csv(os.path.join(output_dir, 'lens2_test_predictions.csv'), index=False)

    # Retrain champion on Train+Val for deployment
    tv_df = pd.concat([train_df, val_df], ignore_index=True)
    deploy_models2 = get_model_registry()
    deploy = deploy_models2[best_name]
    if best_name == 'LightGBM':
        deploy.set_params(n_estimators=300)
        deploy.fit(tv_df[feature_cols], tv_df['Qty'])
    elif best_name == 'XGBoost':
        deploy.set_params(early_stopping_rounds=None)
        deploy.fit(tv_df[feature_cols], tv_df['Qty'])
    else:
        deploy.fit(tv_df[feature_cols], tv_df['Qty'])
    save_model(deploy, os.path.join(output_dir, 'lens2_champion_model.joblib'))

    val_res = np.abs(y_va.values - np.maximum(best_obj.predict(X_va), 0))
    q90 = float(np.percentile(val_res, 90))

    results['champion_info'] = {
        'champion_model'       : best_name,
        'selection_method'     : 'Lowest Validation WAPE',
        'validation_wape'      : round(best_val_wape, 2),
        'unbiased_test_wape'   : results[best_name]['test']['WAPE_percent'],
        'empirical_residual_q90': round(q90, 2),
    }

    print(f"\n  [CHAMPION] {best_name} | Val WAPE={best_val_wape:.2f}% | Test WAPE={results[best_name]['test']['WAPE_percent']:.2f}%")
    return results, preds_df


# -----------------------------------------------------------------
# Full pipeline entry point
# -----------------------------------------------------------------
def run_full_evaluation():
    res_l1, _ = train_and_evaluate_lens1()
    res_l2, _ = train_and_evaluate_lens2()

    all_metrics = {
        'lens1_revenue': res_l1,
        'lens2_demand' : res_l2,
    }
    out = _resolve('models/metrics.json')
    with open(out, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[Evaluate] All artefacts saved. Metrics → {out}")


if __name__ == '__main__':
    run_full_evaluation()
