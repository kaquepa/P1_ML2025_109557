import streamlit as st
import sys
from pathlib import Path

# Add src/ to import path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))

from pre_processing import BankMarketingProcessor


def show_preprocessing_page():
    st.title("Data Preprocessing")
    st.write(
        "Run the full preprocessing pipeline: cleaning, handling unknowns, "
        "feature engineering, Target Encoding (K-Fold), scaling, PCA (95%), "
        "SMOTE in training, and Train/Test split."
    )

    # ---------------------------------------------------------
    # VALIDATE DATASET
    # ---------------------------------------------------------
    if "df" not in st.session_state:
        st.warning("Please load a dataset first on the **Load Dataset** page.")
        st.stop()

    df = st.session_state["df"]

    # ---------------------------------------------------------
    # RUN PIPELINE
    # ---------------------------------------------------------
    if st.button("Run Preprocessing", use_container_width=True):

        with st.spinner("Processing dataset... Please wait ⏳"):

            processor = BankMarketingProcessor()

            # Attach dataset to processor
            processor.dataset = df.copy()

            # Run full preprocessing
            X_train, X_test, y_train, y_test = processor.preprocess_data()

            # Save all artifacts in Streamlit state
            st.session_state["processor"] = processor
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["feature_names"] = processor.feature_names_
            st.session_state["final_feature_names"] = processor.final_feature_names_

            st.success("Preprocessing completed successfully!")

    # ---------------------------------------------------------
    # SHOW SUMMARY
    # ---------------------------------------------------------
    if "X_train" in st.session_state:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        processor = st.session_state["processor"]

        # CSS summary cards
        st.markdown("""
            <style>
            .sum-box {
                background: #fff;
                padding: 1.2rem;
                border-radius: 12px;
                border-left: 5px solid #3b82f6;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }
            .sum-title { font-size: .9rem; color: #475569; }
            .sum-value { font-size: 1.4rem; font-weight: bold; color: #1e293b; }
            </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("<div class='sum-box'><div class='sum-title'>Train Samples</div>"
                        f"<div class='sum-value'>{X_train.shape[0]:,}</div></div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='sum-box'><div class='sum-title'>Test Samples</div>"
                        f"<div class='sum-value'>{X_test.shape[0]:,}</div></div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='sum-box'><div class='sum-title'>Features Before PCA</div>"
                        f"<div class='sum-value'>{len(processor.feature_names_)}</div></div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='sum-box'><div class='sum-title'>Features After PCA</div>"
                        f"<div class='sum-value'>{len(processor.final_feature_names_)}</div></div>", unsafe_allow_html=True)

        st.divider()

        if st.button("train_model", use_container_width=True):
            st.session_state["page"] = "train_model"
            st.rerun()
