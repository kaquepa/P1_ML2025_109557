import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import json 

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. HANDLE UNKNOWN VALUES
# =============================================================================
class UnknownValuesHandler(BaseEstimator, TransformerMixin):
    """
    Handles 'unknown' categories using custom strategies:
    - keep_category → transforms 'unknown' → 'unknown_colname'
    - impute_mode → replaces with mode
    """
    def __init__(self, strategies: Optional[Dict[str, str]] = None):
        self.strategies = strategies or {}
        self.mode_values_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cat_cols = X.select_dtypes(include='object').columns

        for col in cat_cols:
            if "unknown" in X[col].values:
                strategy = self.strategies.get(col, "impute_mode")

                if strategy == "impute_mode":
                    mode = X.loc[X[col] != "unknown", col].mode()
                    self.mode_values_[col] = mode[0] if len(mode) else "unknown"

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cat_cols = X.select_dtypes(include='object').columns

        for col in cat_cols:

            if col not in self.mode_values_ and "unknown" not in X[col].values:
                continue

            strategy = self.strategies.get(col, "impute_mode")

            if strategy == "keep_category":
                X[col] = X[col].replace("unknown", f"unknown_{col}")

            elif strategy == "impute_mode":
                if col in self.mode_values_:
                    X[col] = X[col].replace("unknown", self.mode_values_[col])

        return X


# =============================================================================
# 2. FEATURE ENGINEERING (~20 FEATURES)
# =============================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.economic_cols = [
            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # 1. Age group
        if 'age' in X:
            X['age_group'] = pd.cut(
                X['age'],
                bins=[0, 30, 45, 60, 120],
                labels=[0, 1, 2, 3]
            ).astype(int)

        # 2. Euribor effect
        if 'euribor3m' in X:
            X['euribor_effect'] = (
                (X['euribor3m'] < 1.0) * 2 +
                ((X['euribor3m'] >= 1.0) & (X['euribor3m'] < 2)) * 1
            )

        # 3. Employment impact
        if 'emp.var.rate' in X:
            X['employment_impact'] = (
                (X['emp.var.rate'] < -1.5) * 2 +
                ((X['emp.var.rate'] >= -1.5) & (X['emp.var.rate'] < 0)) * 1
            )

        # 4. Economic pressure
        if all(col in X for col in self.economic_cols[:3]):
            X['economic_pressure'] = (
                X['euribor3m'] * 0.5 +
                X['emp.var.rate'] * -0.3 +
                X['cons.price.idx'] * 0.2
            )

        # 5. Contact efficiency
        if 'previous' in X and 'pdays' in X:
            X['contact_efficiency'] = 0.0
            valid = X['pdays'] != 999
            X.loc[valid, 'contact_efficiency'] = \
                X.loc[valid, 'previous'] / (X.loc[valid, 'pdays'] + 1)

        # 6. Client value
        if all(col in X for col in ['housing', 'loan', 'default']):
            X['client_value'] = (
                (X['housing'] == "yes").astype(int)*0.4 +
                (X['loan'] == "yes").astype(int)*0.3 +
                (X['default'] == "no").astype(int)*0.3
            )

        # 7. Previously contacted
        if 'pdays' in X:
            X['previously_contacted'] = (X['pdays'] != 999).astype(int)

        # 8. Total contacts
        if 'campaign' in X and 'previous' in X:
            X['total_contacts'] = X['campaign'] + X['previous']

        # 9. Any loan
        if 'housing' in X and 'loan' in X:
            X['has_any_loan'] = ((X['housing'] == 'yes') | (X['loan'] == 'yes')).astype(int)

        # 10+. Economic score
        eco_cols = [col for col in self.economic_cols if col in X]
        if eco_cols:
            eco = X[eco_cols]
            X['economic_score'] = (eco - eco.mean()).sum(axis=1) / len(eco_cols)

        return X.fillna(0)


# =============================================================================
# 3. TARGET ENCODER (K-FOLD)
# =============================================================================
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """K-Fold Target Encoding WITHOUT leakage."""
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.target_means_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)

        cat_cols = X.select_dtypes(include='object').columns

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in cat_cols:
            means = []
            for train_idx, val_idx in kf.split(X):
                means.append(
                    pd.DataFrame({
                        col: X.iloc[val_idx][col],
                        'target': y.iloc[train_idx].mean()
                    })
                )
            df = pd.concat(means)
            self.target_means_[col] = df.groupby(col)['target'].mean().to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col, mapping in self.target_means_.items():
            X[col] = X[col].map(mapping).fillna(np.mean(list(mapping.values())))

        return X


# =============================================================================
# 4. PREPROCESSOR PIPELINE (UNIQUE)
# =============================================================================
class BankMarketingProcessor:

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.dataset = df
        self.pipeline = None
        self.feature_names_ = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if self.dataset is None:
            raise ValueError("Dataset not loaded.")

        df = self.dataset.copy()

        # Separate X, y
        X = df.drop("y", axis=1)
        y = df["y"].replace({"yes": 1, "no": 0}).astype(int)

        # SALVAR A ORDEM ORIGINAL DAS FEATURES
        self.original_feature_order_ = X.columns.tolist()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Pipeline antes de PCA
        preprocessing_no_pca = Pipeline([
            ("unknown_handler", UnknownValuesHandler()),
            ("feature_engineer", FeatureEngineer()),
            ("target_encoder", KFoldTargetEncoder(n_splits=5)),
            ("scaler", StandardScaler())
        ])

        # Transformação sem PCA
        X_train_mid = preprocessing_no_pca.fit_transform(X_train, y_train)
        X_test_mid = preprocessing_no_pca.transform(X_test)

        # Nome das features antes da PCA
        self.feature_names_ = [f"f{i}" for i in range(X_train_mid.shape[1])]

        # PCA
        pca = PCA(n_components=0.95, svd_solver="full")
        X_train_pca = pca.fit_transform(X_train_mid)
        X_test_pca = pca.transform(X_test_mid)

        # Nome das features após PCA
        self.final_feature_names_ = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]

        # SMOTE
        sm = SMOTE(random_state=42)
        X_train_sm, y_train_sm = sm.fit_resample(X_train_pca, y_train)

        # Guardar resultados
        self.X_train = X_train_sm
        self.y_train = y_train_sm
        self.X_test = X_test_pca
        self.y_test = y_test

        # -------------------------------
        # FINAL PIPELINE PARA PREDIÇÃO
        # -------------------------------
        self.pipeline = Pipeline([
            ("unknown_handler", preprocessing_no_pca.named_steps["unknown_handler"]),
            ("feature_engineer", preprocessing_no_pca.named_steps["feature_engineer"]),
            ("target_encoder", preprocessing_no_pca.named_steps["target_encoder"]),
            ("scaler", preprocessing_no_pca.named_steps["scaler"]),
            ("pca", pca)
        ])

        # SALVAR A ORDEM NO PIPELINE PARA SER USADA NO PREDICT
        # Save to disk
        import cloudpickle
        from config import Config
        
        self.pipeline._original_feature_order = self.original_feature_order_
        
        self.original_feature_order_ = X.columns.tolist()
        order_path = Path(Config.MODELS_DIR) / "original_feature_order.json"
        with open(order_path, "w") as f:
            json.dump(self.original_feature_order_, f, indent=4)


        

        pipeline_path = Path(Config.MODELS_DIR) / "preprocessing_pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            cloudpickle.dump(self.pipeline, f)

        return self.X_train, self.X_test, self.y_train, self.y_test
