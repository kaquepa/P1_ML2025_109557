import logging
import warnings
from typing import Dict, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. HANDLE UNKNOWN VALUES
# =============================================================================
class UnknownValuesHandler(BaseEstimator, TransformerMixin):
    """
    Handles 'unknown' categories using:
    - 'keep_category' → transforms 'unknown' into 'unknown_colname'
    - 'impute_mode' → replaces with mode of existing values
    """
    def __init__(self, strategies: Optional[Dict[str, str]] = None):
        self.strategies = strategies or {}
        self.mode_values_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        cat_cols = X.select_dtypes(include=['object']).columns

        for col in cat_cols:
            if 'unknown' in X[col].values:
                strategy = self.strategies.get(col, 'impute_mode')

                if strategy == 'impute_mode':
                    valid = X.loc[X[col] != "unknown", col]
                    mode_val = valid.mode()
                    self.mode_values_[col] = (
                        mode_val[0] if len(mode_val) else "unknown"
                    )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cat_cols = X.select_dtypes(include=['object']).columns

        for col in cat_cols:
            if 'unknown' not in X[col].astype(str).values:
                continue

            strategy = self.strategies.get(col, 'impute_mode')

            if strategy == 'keep_category':
                X[col] = X[col].replace("unknown", f"unknown_{col}")

            elif strategy == 'impute_mode':
                if col in self.mode_values_:
                    X[col] = X[col].replace("unknown", self.mode_values_[col])

        return X


# =============================================================================
# 2. TARGET ENCODER with K-Fold
# =============================================================================
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoding with K-Fold to avoid overfitting-leakage.

    EXAMPLE:
    job → mean(y | job)
    """
    def __init__(self, cols=None, n_splits=5):
        self.cols = cols
        self.n_splits = n_splits
        self.global_mean_ = None
        self.encoding_maps_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.global_mean_ = y.mean()

        if self.cols is None:
            self.cols = X.select_dtypes(include=['object']).columns.tolist()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in self.cols:
            values = []
            for train_idx, val_idx in kf.split(X):
                fold_mean = (
                    X.iloc[train_idx]
                    .groupby(col)[y.iloc[train_idx].name]
                    .apply(lambda idx: y.iloc[idx].mean())
                    if col in X.columns
                    else None
                )
                values.append(fold_mean)

            mean_map = pd.concat(values, axis=1).mean(axis=1).to_dict()
            self.encoding_maps_[col] = mean_map

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.cols:
            X[col] = X[col].map(self.encoding_maps_.get(col, {}))
            X[col] = X[col].fillna(self.global_mean_)
        return X


# =============================================================================
# 3. FEATURE ENGINEERING (20+ FEATURES)
# =============================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.economic_features_ = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        eco_cols = [
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
            'euribor3m', 'nr.employed'
        ]
        self.economic_features_ = [c for c in eco_cols if c in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # AGE GROUP
        if "age" in X.columns:
            X['age_group'] = pd.cut(
                X['age'], bins=[0, 30, 45, 60, 100],
                labels=[0, 1, 2, 3]
            ).astype(int)

        # CONTACT EFFICIENCY
        if 'pdays' in X.columns and 'previous' in X.columns:
            X['contact_efficiency'] = np.where(
                X['pdays'] != 999,
                X['previous'] / (X['pdays'] + 1),
                0
            )

        # ECONOMIC PRESSURE
        if all(c in X.columns for c in ['euribor3m', 'emp.var.rate', 'cons.price.idx']):
            X['economic_pressure'] = (
                X['euribor3m'] * 0.5 +
                X['emp.var.rate'] * -0.3 +
                X['cons.price.idx'] * 0.2
            )

        # LOAN SCORE
        if 'housing' in X.columns and 'loan' in X.columns:
            X['loan_score'] = (
                (X['housing'] == 'yes').astype(int) +
                (X['loan'] == 'yes').astype(int)
            )

        # PREVIOUSLY CONTACTED FLAG
        if "pdays" in X.columns:
            X['previously_contacted'] = (X['pdays'] != 999).astype(int)

        # TOTAL CONTACTS
        if 'campaign' in X.columns and 'previous' in X.columns:
            X['total_contacts'] = X['campaign'] + X['previous']

        # ECONOMIC SCORE
        if self.economic_features_:
            eco = X[self.economic_features_]
            X['economic_score'] = (eco - eco.mean()).sum(axis=1)

        return X.fillna(0)


# =============================================================================
# 4. SAFE ONE-HOT + LABEL/FREQ ENCODING
# =============================================================================
class SafeOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical variables using:
    - LabelEncoder if cardinalidade <= threshold
    - Frequency Encoding if cardinalidade > threshold
    """
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.encoders_ = {}
        self.encoding_types_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cat_cols = X.select_dtypes(include=['object']).columns

        for col in cat_cols:
            n_unique = X[col].nunique()

            if n_unique <= self.threshold:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders_[col] = le
                self.encoding_types_[col] = "label"

            else:
                freq = X[col].value_counts(normalize=True).to_dict()
                self.encoders_[col] = freq
                self.encoding_types_[col] = "freq"

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col, enc in self.encoders_.items():
            if col not in X:
                continue
            if self.encoding_types_[col] == "label":
                X[col] = X[col].astype(str)
                known = set(enc.classes_)
                X[col] = X[col].apply(
                    lambda v: enc.transform([v])[0] if v in known else -1
                )
            else:
                X[col] = X[col].map(enc).fillna(0)

        return X


# =============================================================================
# 5. PCA (AUTOMÁTICO 95% VARIÂNCIA)
# =============================================================================
class AutoPCA(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.pca = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.pca = PCA(n_components=self.variance_threshold, svd_solver="full")
        self.pca.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.pca.transform(X),
            index=X.index
        )
