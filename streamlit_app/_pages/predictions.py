from config import Config
import streamlit as st
from pathlib import Path
import joblib
import cloudpickle
import json
import textwrap
import pandas as pd


# ======================================================================
# LOAD MODEL + PIPELINE
# ======================================================================
@st.cache_resource
def load_artifacts():
    """Load preprocessing pipeline + trained model + metadata"""

    models_dir = Path(Config.MODELS_DIR)

    # -------------------------
    # 1) Load preprocessing pipeline
    # -------------------------
    pipe_path = models_dir / "preprocessing_pipeline.pkl"
    if not pipe_path.exists():
        st.error(" Preprocessing pipeline not found. Train a model first.")
        return None, None, None

    try:
        with open(pipe_path, "rb") as f:
            pipeline = cloudpickle.load(f)
    except:
        pipeline = joblib.load(pipe_path)

    # -------------------------
    # 2) Load model
    # -------------------------
    model_files = list(models_dir.glob("best_model_*.pkl"))
    if not model_files:
        st.error(" No trained model found. Run training first.")
        return None, None, None

    model_path = model_files[0]

    try:
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
    except:
        model = joblib.load(model_path)

    # -------------------------
    # 3) Load metadata
    # -------------------------
    meta_path = models_dir / "best_model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_name": "unknown",
            "auc": 0,
            "metrics": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "auc_roc": 0
            }
        }

    return pipeline, model, metadata



# ======================================================================
# PREDICTION PAGE
# ======================================================================
def show_predictions_page():

    st.title("Client Subscription Prediction")
    st.markdown("Fill the form below to predict the probability of a client subscribing.")

    pipeline, model, metadata = load_artifacts()

    if not pipeline or not model:
        st.stop()

    metrics = metadata.get("metrics", {})

    # -----------------------------
    # METRIC CARDS
    # -----------------------------
    st.markdown("""
        <style>
        .cards-container {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: nowrap;
            margin-top: 1.5rem;
        }
        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: 12px;
            width: 19%;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
            border-top: 5px solid #3b82f6;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #1e293b;
        }
        .metric-label {
            font-size: .9rem;
            color: #64748b;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance Overview")

    st.markdown(f"""
    <div class="cards-container">
        <div class="metric-card">
            <div class="metric-value">{metrics.get("accuracy",0):.3f}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get("precision",0):.3f}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get("f1_score",0):.3f}</div>
            <div class="metric-label">F1-score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get("auc_roc",0):.3f}</div>
            <div class="metric-label">AUC-ROC</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get("recall",0):.3f}</div>
            <div class="metric-label">Recall</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ----------------------------------------------------------------------
    # FORM
    # ----------------------------------------------------------------------
    with st.form("prediction_form"):

        # ---------------------------
        # DEMOGRAPHICS
        # ---------------------------
        st.markdown("### 👤 Client Demographics")

        col = st.columns(7)
        age = col[0].number_input("Age", 18, 100, 35)
        job = col[1].selectbox("Occupation", [
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician',
            'unemployed', 'unknown'
        ])
        marital = col[2].selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
        education = col[3].selectbox("Education", [
            'university.degree', 'high.school', 'basic.9y', 'professional.course',
            'basic.6y', 'basic.4y', 'illiterate', 'unknown'
        ])
        default = col[4].selectbox("Credit Default", ["no", "yes", "unknown"])
        housing = col[5].selectbox("Housing Loan", ["no", "yes", "unknown"])
        loan = col[6].selectbox("Personal Loan", ["no", "yes", "unknown"])

        # ---------------------------
        # CAMPAIGN
        # ---------------------------
        st.markdown("### ☎️ Campaign Details")
        col = st.columns(8)
        contact = col[0].selectbox("Contact Type", ["cellular", "telephone"])
        month = col[1].selectbox("Month", [
            "may","june","july","august","november","april",
            "march","october","september","january","february","december"
        ])
        day_of_week = col[2].selectbox("Day", ["mon","tue","wed","thu","fri"])
        campaign = col[3].number_input("Campaign Contacts", 1, 50, 2)
        pdays = col[4].number_input("Days Since Last Contact", 0, 999, 999)
        previous = col[5].number_input("Previous Contacts", 0, 7, 0)
        poutcome = col[6].selectbox("Previous Outcome", ['nonexistent','failure','success'])
        duration = col[7].number_input("Call Duration (seconds)", 0, 5000, 180)

        # ---------------------------
        # ECONOMIC INDICATORS
        # ---------------------------
        st.markdown("### 📈 Economic Indicators")
        col = st.columns(5)
        emp_var_rate = col[0].number_input("Employment Var Rate", -5.0, 5.0, 1.1)
        cons_price = col[1].number_input("Price Index", 92.0, 95.0, 93.994)
        cons_conf = col[2].number_input("Confidence Index", -50.0, -26.0, -36.4)
        euribor3m = col[3].number_input("Euribor 3M", -1.0, 6.0, 4.857)
        nr_employed = col[4].number_input("Employees", 4900.0, 5300.0, 5191.0)

        submitted = st.form_submit_button(
            "Predict Subscription",
            type="primary",
            use_container_width=True
        )

    # ----------------------------------------------------------------------
    # PROCESS PREDICTION
    # ----------------------------------------------------------------------
    if submitted:

        # 1) Construir DataFrame EXACTAMENTE com as mesmas colunas do treino
        df = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,                
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price,
            "cons.conf.idx": cons_conf,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed
        }])


        if hasattr(pipeline, "feature_names_in_"):
            expected = list(pipeline.feature_names_in_)
            missing = set(expected) - set(df.columns)
            extra   = set(df.columns) - set(expected)

            if missing:
                st.error(f"Internal error: missing features in form: {missing}")
                st.stop()

       

        with st.spinner("Predicting…"):

            # Load feature order
            order_path = Path(Config.MODELS_DIR) / "original_feature_order.json"
            with open(order_path, "r") as f:
                original_order = json.load(f)

            # Reindex (reorder + add missing + ignore extra)
            df = df.reindex(columns=original_order, fill_value=0)

            #df = df.reindex(columns=pipeline._original_feature_order, fill_value=0)
            X_new = pipeline.transform(df)
            pred = model.predict(X_new)[0]
            prob = model.predict_proba(X_new)[0][1]

        interpretation = (
            "Client shows a high probability of subscribing. "
            if pred == 1 else
            "Client shows a low probability of subscribing. "
        )

        st.markdown(f"""
        <div style="
            background: #f8fafc;
            padding: 22px;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            text-align: center;
        ">
            <h3 style="color:#1e3a8a; font-size:1.4rem;">Prediction Score</h3>
            <div style="font-size:3rem; font-weight:700; color:#1e3a8a;">
                {prob:.1%}
            </div>
            <p style="color:#475569; font-size:1rem;">
                {interpretation}
            </p>
        </div>
        """, unsafe_allow_html=True)
