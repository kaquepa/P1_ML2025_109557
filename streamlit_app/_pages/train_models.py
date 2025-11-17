import streamlit as st
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure SRC is in import path
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]     # bank_marketing/
SRC = ROOT / "src"

# Clear previous imports (fixes hot-reload issues)
sys.modules.pop("models", None)

sys.path.insert(0, str(SRC))

from models import BankMarketingModeler


def show_train_model_page():
    st.title("Train Models")
    st.markdown(
        "Train and evaluate ML models using the fully processed dataset "
        "(Feature Engineering → Target Encoding → Scaling → PCA → SMOTE)."
    )

    # ------------------------------------------------------------------
    # CHECK PREPROCESSING
    # ------------------------------------------------------------------
    if (
        "processor" not in st.session_state
        or "X_train" not in st.session_state
        or "y_train" not in st.session_state
    ):
        st.warning("⚠ Please run **Pre-processing** first.")
        st.session_state["page"] = "preprocessing"
        st.rerun()

    processor = st.session_state["processor"]

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    # ------------------------------------------------------------------
    # TRAINING OPTIONS
    # ------------------------------------------------------------------
    optimize = st.checkbox("Optimize hyperparameters (slower)", value=False)
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5)

    # ------------------------------------------------------------------
    # TRAIN BUTTON
    # ------------------------------------------------------------------
    if st.button(" Train Models", use_container_width=True):

        with st.spinner("Training ML models… Please wait ⏳"):

            try:
                # Attach preprocessed data to processor
                processor.X_train = X_train
                processor.X_test = X_test
                processor.y_train = y_train
                processor.y_test = y_test
                processor.dataset = None     # prevent reprocessing

                # Create modeler
                modeler = BankMarketingModeler(
                    processor,
                    preprocessed=True
                )

                # Train models
                modeler.train_models(
                    optimize_hyperparams=optimize,
                    cv_folds=cv_folds
                )

                # Compare model performance
                results_df = modeler.compare_models()

                # Save best model to disk
                modeler.save_best_model()

                # Create plots
                _, roc_path = modeler.plot_roc_curves()
                _, cm_path = modeler.plot_confusion_matrices()
                _, fi_path = modeler.plot_feature_importance()

                # Report
                modeler.generate_report()

                # save everything
                st.session_state["modeler"] = modeler
                st.session_state["results_df"] = results_df
                st.session_state["roc_path"] = roc_path
                st.session_state["conf_matrix_path"] = cm_path
                st.session_state["feat_import_path"] = fi_path

            except Exception as e:
                st.error(" **Training error occurred**")
                import traceback
                st.code(traceback.format_exc())
                return

    # ------------------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------------------
    if "modeler" in st.session_state:
        st.markdown("## ✔ Model Evaluation")

        roc_path = st.session_state["roc_path"]
        cm_path = st.session_state["conf_matrix_path"]
        fi_path = st.session_state["feat_import_path"]

        col1, col2, col3 = st.columns(3)

        # ROC Curves
        with col1:
            st.subheader("ROC Curves")
            if roc_path.exists():
                st.image(str(roc_path), use_container_width=True)
            else:
                st.error("ROC curve image not found.")

        # Confusion Matrix
        with col2:
            st.subheader("Confusion Matrix")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.error("Confusion matrix not found.")

        # Feature Importance
        with col3:
            st.subheader("Feature Importance")
            if fi_path.exists():
                st.image(str(fi_path), use_container_width=True)
            else:
                st.error("Feature importance image not found.")

        st.markdown("##  Model Comparison")
        st.dataframe(st.session_state["results_df"], use_container_width=True)

        if st.button("Predict with trained model", use_container_width=True):
            st.session_state["page"] = "predictions"
            st.rerun()
