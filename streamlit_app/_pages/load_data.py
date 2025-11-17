import streamlit as st
import pandas as pd
def show_load_data_page():
    """Dataset loading page"""
    st.markdown("## Load Dataset")
    st.write("Upload the dataset that will be used for analysis and model training.")
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    # 1) A new file was uploaded 
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
            # Save dataset in session state
            st.session_state["df"] = df  
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
    # 2) Dataset already exists in session_state
    elif "df" in st.session_state:
        df = st.session_state["df"]
        st.dataframe(df.head())
    # 3) No dataset loaded
    else:
        st.warning("Please upload a CSV file to begin.")
        return  # Prevent showing the Next button before loading data
    # If execution reaches here -> dataset is loaded
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("data_explorer", use_container_width=True):
        st.session_state["page"] = "data_explorer"
        st.rerun()
