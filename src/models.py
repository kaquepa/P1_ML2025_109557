import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import cloudpickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ML
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import Config


class BankMarketingModeler:

    # =========================================================================
    # INIT
    # =========================================================================
    def __init__(self, processor, preprocessed=True):

        Config.create_directories()

        self.processor = processor

        if preprocessed:
            # Pré-processamento já foi realizado no Streamlit
            self.X_train = processor.X_train
            self.X_test = processor.X_test
            self.y_train = processor.y_train
            self.y_test = processor.y_test
            self.feature_names_ = processor.feature_names_

        else:
            # Caso contrário, forçar pré-processamento
            (
                self.X_train, self.X_test,
                self.y_train, self.y_test
            ) = processor.preprocess_data()
            self.feature_names_ = processor.feature_names_

        # Model store
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.cv_results = {}

        # Directories
        self.output_dir = Path(Config.OUTPUTS_DIR)
        self.fig_dir = Path(Config.FIGURES_DIR)
        self.model_dir = Path(Config.MODELS_DIR)

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.fig_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)



    # =========================================================================
    # HELPER
    # =========================================================================
    def calculate_scale_pos_weight(self):
        """Needed for XGBoost imbalance correction."""
        neg = np.sum(self.y_train == 0)
        pos = np.sum(self.y_train == 1)
        return neg / pos if pos > 0 else 1



    # =========================================================================
    # TRAIN MODELS
    # =========================================================================
    def train_models(self, optimize_hyperparams=True, cv_folds=5):

        model_configs = {
            "LogisticRegression": {
                "model": LogisticRegression(
                    random_state=42,
                    max_iter=1500,
                    class_weight="balanced",
                    solver="saga"
                ),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "l1_ratio": [0.1, 0.5, 0.9]
                }
            },

            "RandomForest": {
                "model": RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
                "params": {
                    "n_estimators": [200, 400],
                    "max_depth": [10, 20, None],
                    "max_features": ["sqrt", "log2"]
                }
            },

            "XGBoost": {
                "model": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1,
                    scale_pos_weight=self.calculate_scale_pos_weight()
                ),
                "params": {
                    "n_estimators": [300, 500],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.7, 1.0]
                }
            }
        }

        # Loop models
        for name, cfg in model_configs.items():

            if optimize_hyperparams:
                model = self._optimize_model(name, cfg["model"], cfg["params"], cv_folds)
            else:
                model = cfg["model"]
                model.fit(self.X_train, self.y_train)
                self.cv_results[name] = self._cross_validate(model, cv_folds)

            self.models[name] = model

            # Evaluate
            self._evaluate_model(name, model)

        return True



    # =========================================================================
    # CROSS VALIDATION
    # =========================================================================
    def _cross_validate(self, model, cv_folds):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="roc_auc")

        return {
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std())
        }


    def _optimize_model(self, name, model, param_grid, cv_folds):

        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=12,
            scoring="roc_auc",
            cv=cv_folds,
            n_jobs=-1,
            random_state=42
        )

        search.fit(self.X_train, self.y_train)

        self.cv_results[name] = {
            "mean_score": float(search.best_score_),
            "best_params": search.best_params_
        }

        return search.best_estimator_



    # =========================================================================
    # EVALUATION
    # =========================================================================
    def _evaluate_model(self, name, model):

        proba = model.predict_proba(self.X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        self.predictions[name] = {
            "pred": pred,
            "proba": proba
        }

        # Metrics
        auc = roc_auc_score(self.y_test, proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, pred, average="binary"
        )
        acc = accuracy_score(self.y_test, pred)

        self.metrics[name] = {
            "auc_roc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(acc),
        }



    # =========================================================================
    # GET BEST MODEL
    # =========================================================================
    def get_best_model(self):
        """Select model with best AUC."""
        best = max(self.metrics.items(), key=lambda x: x[1]["auc_roc"])[0]
        return best, self.models[best], self.metrics[best]



    # =========================================================================
    # SAVE BEST MODEL
    # =========================================================================
    def save_best_model(self):

        best_name, best_model, metrics = self.get_best_model()

        model_path = self.model_dir / f"best_model_{best_name}.pkl"
        metadata_path = self.model_dir / "best_model_metadata.json"

        with open(model_path, "wb") as f:
            cloudpickle.dump(best_model, f)

        # Save metadata
        meta = {
            "model_name": best_name,
            "metrics": metrics
        }

        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=4)

        return model_path



    # =========================================================================
    # MODEL COMPARISON TABLE
    # =========================================================================
    def compare_models(self):

        rows = []
        for name, m in self.metrics.items():

            cv = self.cv_results.get(name, {})
            cv_score = cv.get("mean_score", None)

            rows.append({
                "Model": name,
                "AUC ROC": m["auc_roc"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
                "CV Score": cv_score
            })

        df = pd.DataFrame(rows).sort_values("AUC ROC", ascending=False)

        df.to_csv(self.output_dir / "model_comparison.csv", index=False)

        return df



    # =========================================================================
    # PLOTS
    # =========================================================================
    def plot_roc_curves(self):

        plt.figure(figsize=(10, 7))

        for name, preds in self.predictions.items():
            fpr, tpr, _ = roc_curve(self.y_test, preds["proba"])
            auc = self.metrics[name]["auc_roc"]
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curves")
        plt.grid(True)
        plt.legend()

        path = self.output_dir / "roc_curves.png"
        plt.savefig(path)
        plt.close()

        return True, path


    def plot_confusion_matrices(self):

        best_name, _, _ = self.get_best_model()
        preds = self.predictions[best_name]["pred"]

        cm = confusion_matrix(self.y_test, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"Confusion Matrix — {best_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        path = self.output_dir / "confusion_matrix_best.png"
        plt.savefig(path)
        plt.close()

        return True, path



    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance after PCA.
        PCA removes original feature names → we use final_feature_names_.
        """
        best_name, model, _ = self.get_best_model()

        # --- Get importance according to model type ---
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            return None, "Model does not support feature importance."

        # --- Use PCA feature names ---
        feature_names = getattr(self.processor, "final_feature_names_", None)

        if feature_names is None:
            # Fallback if something failed in processor
            feature_names = [f"PC{i+1}" for i in range(len(importance))]

        # --- Ensure same length to avoid ValueError ---
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

        # --- Create DataFrame ---
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False).head(top_n)

        # --- Plot ---
        plt.figure(figsize=(8, 10))
        sns.barplot(data=df, x="importance", y="feature", palette="viridis")
        plt.title(f"Top {top_n} PCA Components — {best_name}")
        plt.tight_layout()

        path = self.output_dir / "feature_importance_best.png"
        plt.savefig(path)
        plt.close()

        return True, path



    # =========================================================================
    # REPORT
    # =========================================================================
    def generate_report(self):

        best_name, best_model, metrics = self.get_best_model()

        preds = self.predictions[best_name]["pred"]
        report = classification_report(self.y_test, preds)
        cm = confusion_matrix(self.y_test, preds)

        path = self.output_dir / "model_report.txt"

        with open(path, "w") as f:
            f.write(f"BEST MODEL: {best_name}\n\n")
            f.write(json.dumps(metrics, indent=4))
            f.write("\n\nCLASSIFICATION REPORT:\n")
            f.write(report)
            f.write("\n\nCONFUSION MATRIX:\n")
            f.write(np.array2string(cm))

        return True
