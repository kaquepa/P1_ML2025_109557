import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# -------------------------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------------------------
def show_data_explorer_page():
    st.title("📊 Data Explorer")
    st.markdown("Interactive exploratory analysis of the Bank Marketing Dataset")

    # ---------------------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def load_raw(df):
        return df.copy()

    if "df" not in st.session_state:
        st.warning("⚠ No dataset loaded. Please go to **Load Dataset** first.")
        st.stop()

    data = load_raw(st.session_state["df"])

    # ---------------------------------------------------------------------
    # SIDEBAR FILTERS
    # ---------------------------------------------------------------------
    st.sidebar.title("🔎 Filters")

    # Age
    if "age" in data.columns:
        age_min = int(data["age"].min())
        age_max = int(data["age"].max())
        age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
    else:
        age_range = None

    # Job
    job_col = "job"
    job_options = ["All"]
    if job_col in data.columns:
        job_options += sorted(data[job_col].dropna().unique())
    selected_job = st.sidebar.selectbox("Occupation", job_options)

    # Education
    edu_col = "education"
    edu_options = ["All"]
    if edu_col in data.columns:
        edu_options += sorted(data[edu_col].dropna().unique())
    selected_education = st.sidebar.selectbox("Education", edu_options)

    # ---------------------------------------------------------------------
    # APPLY FILTERS
    # ---------------------------------------------------------------------
    filtered = data.copy()

    if age_range and "age" in data.columns:
        filtered = filtered.query("age >= @age_range[0] and age <= @age_range[1]")

    if selected_job != "All" and job_col in filtered.columns:
        filtered = filtered[filtered[job_col] == selected_job]

    if selected_education != "All" and edu_col in filtered.columns:
        filtered = filtered[filtered[edu_col] == selected_education]

    st.sidebar.markdown(f"### Records: **{len(filtered):,} / {len(data):,}**")

    # ---------------------------------------------------------------------
    # TABS
    # ---------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview",
        "📈 Distributions",
        "🔗 Correlations",
        "🎯 Target Analysis",
        "📐 Advanced"
    ])

    # ---------------------------------------------------------------------
    # TAB 1 — Overview
    # ---------------------------------------------------------------------
    with tab1:
        st.subheader("Dataset Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(data):,}")
        with col2:
            st.metric("Columns", f"{data.shape[1]}")
        with col3:
            st.metric("Memory (MB)", f"{data.memory_usage(deep=True).sum()/1024**2:.2f}")
        with col4:
            st.metric("Duplicates", f"{data.duplicated().sum():,}")
        st.dataframe(filtered.head(50), use_container_width=True, height=350)

    # ---------------------------------------------------------------------
    # TAB 2 — Distributions
    # ---------------------------------------------------------------------
    with tab2:
        st.subheader("Feature Distributions")

        numeric_cols = filtered.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = filtered.select_dtypes(include=["object"]).columns.tolist()

        colA, colB = st.columns([1, 3])

        with colA:
            mode = st.radio("Feature Type", ["Numerical", "Categorical"])

        if mode == "Numerical" and numeric_cols:
            feature = st.selectbox("Select feature", numeric_cols)
            fig = px.histogram(
                filtered,
                x=feature,
                nbins=40,
                marginal="box",
                title=f"Distribution of {feature}",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif mode == "Categorical" and cat_cols:
            feature = st.selectbox("Select feature", cat_cols)
            value_counts = filtered[feature].value_counts().head(25)

            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation="h",
                labels={"x": "Count", "y": feature},
                title=f"Distribution of {feature}",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No valid features available.")

    # ---------------------------------------------------------------------
    # TAB 3 — Correlations
    # ---------------------------------------------------------------------
    with tab3:
        st.subheader("Correlation Heatmap")

        numeric_cols = filtered.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Not enough numerical features.")
        else:
            corr = filtered[numeric_cols].corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale="RdBu_r",
                    zmid=0,
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    showscale=True
                )
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 4 — Target Analysis
    # ---------------------------------------------------------------------
    with tab4:
        st.subheader("Target Variable")

        if "y" not in filtered.columns:
            st.error("Target column 'y' not found in dataset.")
        else:
            counts = filtered["y"].value_counts()

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.45,
                title="Target Distribution",
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 5 — Advanced Scatter
    # ---------------------------------------------------------------------
    with tab5:
        st.subheader("Scatter Analysis")

        if len(numeric_cols) < 2:
            st.info("Scatter requires at least two numerical features.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                x_var = st.selectbox("X-axis", numeric_cols)
            with col2:
                y_var = st.selectbox("Y-axis", numeric_cols, index=1)

            color_options = ["None"] + (["y"] if "y" in filtered.columns else []) + cat_cols
            color_by = st.selectbox("Color by", color_options)

            fig = px.scatter(
                filtered,
                x=x_var,
                y=y_var,
                color=None if color_by == "None" else color_by,
                opacity=0.65,
                title=f"{x_var} vs {y_var}",
                marginal_x="histogram",
                marginal_y="histogram"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation
    if st.button(" preprocessing", use_container_width=True):
        st.session_state["page"] = "preprocessing"
        st.rerun()
