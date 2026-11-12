"""
Model Registry - ChefCast AI (NB03 / NB07 aligned)
All 5 candidate architectures + Stacking Ensemble for both lenses.

Authors: IT24102726 (NB03 Revenue), IT24102572 (NB07 Demand),
         IT24102913 (NB04/NB08 Stacking)

NOTE: sklearn StackingRegressor raises ValueError when passed TimeSeriesSplit
because cross_val_predict requires disjoint folds. We use a hand-rolled
TimeSeriesStackingRegressor that generates OOF meta-features with walk-forward
windows, strictly respecting temporal ordering.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model    import Ridge
from sklearn.ensemble         import RandomForestRegressor
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.base             import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection  import TimeSeriesSplit
import lightgbm as lgb
import xgboost  as xgb


# ─────────────────────────────────────────────────────────────────
# Custom Time-Series Stacking (avoids sklearn's partition error)
# ─────────────────────────────────────────────────────────────────
class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Walk-forward stacking that strictly respects chronological order.

    Phase 1: Walk-forward OOF predictions on training set  -> meta-features
    Phase 2: Fit meta-learner on OOF meta-features
    Phase 3: Re-fit all base learners on full training data
    Inference: Stack base predictions -> meta-learner prediction
    """

    def __init__(self, estimators, final_estimator=None,
                 cv=None, random_state=42):
        self.estimators      = estimators
        self.final_estimator = (final_estimator if final_estimator is not None
                                else Ridge(alpha=1.0))
        self.cv              = cv if cv is not None else TimeSeriesSplit(n_splits=5)
        self.random_state    = random_state

    def fit(self, X, y):
        is_df = isinstance(X, pd.DataFrame)
        self.feature_names_in_ = np.asarray(X.columns) if is_df else None

        Xa = X.values if is_df else np.asarray(X)
        ya = y.values if isinstance(y, pd.Series) else np.asarray(y)

        # ── Phase 1: OOF meta-features ────────────────────────────
        meta_X, meta_y = [], []
        for tr_idx, va_idx in self.cv.split(Xa):
            Xtr = X.iloc[tr_idx] if is_df else Xa[tr_idx]
            ytr = ya[tr_idx]
            Xva = X.iloc[va_idx] if is_df else Xa[va_idx]
            yva = ya[va_idx]
            fold_preds = []
            for _, est in self.estimators:
                c = clone(est)
                c.fit(Xtr, ytr)
                fold_preds.append(c.predict(Xva).reshape(-1, 1))
            meta_X.append(np.hstack(fold_preds))
            meta_y.append(yva)

        # ── Phase 2: Meta-learner ──────────────────────────────────
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(np.vstack(meta_X), np.concatenate(meta_y))

        # ── Phase 3: Re-fit base learners on full data ─────────────
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            c = clone(est)
            c.fit(X, y)
            self.fitted_estimators_.append((name, c))

        return self

    def predict(self, X):
        if self.feature_names_in_ is not None and isinstance(X, pd.DataFrame):
            X = X[self.feature_names_in_]
        base_preds = [
            est.predict(X).reshape(-1, 1)
            for _, est in self.fitted_estimators_
        ]
        return self.final_estimator_.predict(np.hstack(base_preds))


# ─────────────────────────────────────────────────────────────────
# Seasonal Naive Baseline
# ─────────────────────────────────────────────────────────────────
class SeasonalNaiveRegressor(BaseEstimator, RegressorMixin):
    """Predict today = same weekday last week (lag-7). Irreducible baseline."""

    def __init__(self, seasonal_lag=7):
        self.seasonal_lag = seasonal_lag

    def fit(self, X, y):
        col = f'lag_{self.seasonal_lag}'
        self.lag_col_ = col if col in X.columns else None
        return self

    def predict(self, X):
        if self.lag_col_ and self.lag_col_ in X.columns:
            return np.maximum(X[self.lag_col_].values.astype(float), 0.0)
        return np.zeros(len(X))


# ─────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────
def get_model_registry(random_state=42):
    """All five candidate models + stacking ensemble. Hyperparameters
    match NB03 / NB07 decision logs."""

    models = {
        'Baseline_Seasonal_Naive': SeasonalNaiveRegressor(seasonal_lag=7),

        'Ridge_Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=100, random_state=random_state)),
        ]),

        'Random_Forest': RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3,
            max_features='sqrt', n_jobs=-1, random_state=random_state,
        ),

        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_samples=10,
            random_state=random_state, verbosity=-1,
        ),

        'XGBoost': xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_state, verbosity=0,
            early_stopping_rounds=50,
        ),
    }

    # Stacking Ensemble (NB04 §3 / NB08 §3)
    base_estimators = [
        ('ridge', Pipeline([
            ('sc', StandardScaler()),
            ('m',  Ridge(alpha=100, random_state=random_state)),
        ])),
        ('rf', RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3,
            max_features='sqrt', n_jobs=-1, random_state=random_state,
        )),
        ('lgbm', lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=random_state, verbosity=-1,
        )),
    ]
    models['Stacking_Ensemble'] = TimeSeriesStackingRegressor(
        estimators      = base_estimators,
        final_estimator = Pipeline([
            ('sc', StandardScaler()),
            ('m',  Ridge(alpha=10, random_state=random_state)),
        ]),
        cv           = TimeSeriesSplit(n_splits=5),
        random_state = random_state,
    )

    return models


def save_model(model, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Models] Saved -> {path}")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
