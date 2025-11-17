"""
Bank Marketing ML Dashboard - Main Application
Streamlit app with JWT authentication
"""

import sys, os
from pathlib import Path
import streamlit as st
from typing import Dict, Tuple, Optional, Any
import jwt
import time
from datetime import datetime, timezone
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR  # streamlit_app/
PAGES_DIR = CURRENT_DIR / "_pages"
SRC_DIR = ROOT_DIR.parent / "src"

# Add required paths to Python
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PAGES_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR.parent))  # bank_marketing/

# ✅ FIX — IMPORT CONFIG
from config import Config

# Components and pages
from _pages.load_data import show_load_data_page
from _pages.data_explorer import show_data_explorer_page
from _pages.preprocessing import show_preprocessing_page
from _pages.train_models import show_train_model_page
from _pages.predictions import show_predictions_page


# Constants
SECRET_KEY = Config.SECRET_KEY
SESSION_TIMEOUT = 1500  # 25 minutes of inactivity
LOGIN_URL = os.getenv("LOGIN_URL", "http://localhost:8000")


# 
# Page Configuration
# 
st.set_page_config(
    layout="wide",
    page_title="Bank ML Dashboard",
    page_icon="streamlit_app/image.png",
    initial_sidebar_state="expanded",
)

 
# Page Configuration
 
st.set_page_config(
    layout="wide",
    page_title="Bank ML Dashboard",
    page_icon="streamlit_app/image.png",
    initial_sidebar_state="expanded",
)


 
# Authentication & Session Management
 
def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Validate and decode JWT token."""
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        exp_timestamp = payload.get("exp")
        if exp_timestamp and datetime.now(timezone.utc).timestamp() > exp_timestamp:
            st.error("Session expired. Please log in again.")
            return None
        return payload

    except jwt.ExpiredSignatureError:
        st.error("Session expired. Please log in again.")
        return None
    except jwt.InvalidTokenError as e:
        st.error(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error in token validate: {str(e)}")
        return None


def check_session_timeout() -> bool:
    """Check if session has timed out due to inactivity."""
    if "last_activity" not in st.session_state:
        st.session_state["last_activity"] = time.time()
        return True

    time_elapsed = time.time() - st.session_state["last_activity"]

    if time_elapsed > SESSION_TIMEOUT:
        st.warning(f"Session expired after {SESSION_TIMEOUT // 60} minutes of  inactivity.")
        return False

    st.session_state["last_activity"] = time.time()
    return True


def logout():
    """Clear session and redirect to login."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.info("Redirecting to login...")

    st.markdown(
        f'<meta http-equiv="refresh" content="2;url={LOGIN_URL}" />',
        unsafe_allow_html=True,
    )
    time.sleep(2)
    st.stop()


def initialize_session():
    """Initialize or validate user session."""
    token = st.session_state.get("token")

    if not token:
        query_params = st.query_params
        token = query_params.get("token")

        # # query_params may return a list → normalize
        if isinstance(token, list):
            token = token[0] if token else None

        if token:
            st.session_state["token"] = token
            st.query_params.clear()  # clean token  URL
            st.rerun()

    user = verify_token(token) if token else None

    if not user:
        st.warning("Invalid or expired session..")
        st.markdown(f"[Go to Login]({LOGIN_URL})")
        st.stop()

    if not check_session_timeout():
        logout()

    st.session_state["user"] = user
    return user

# UI Components
def render_header(user: Dict[str, Any]):
    """Render custom header with user info and logout button."""
    st.markdown(
        """
        <style>
        /* Adjust main content to avoid overlap with header */
        .main .block-container {
            padding-top: 80px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }
        section[data-testid="stSidebar"] > div:nth-child(1),
        [data-testid="stSidebar"] [data-testid="stSidebarContent"],
        aside[aria-label="sidebar"] [data-testid="stSidebarContent"] {
            padding-top: 80px !important;
        }
        header[data-testid="stHeader"] { display: none !important; }
        .custom-header {
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 70px;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 999;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-left { display: flex; align-items: center; gap: 0.75rem; }
        .header-title { color: white; font-size: 1.4rem; font-weight: 700; margin: 0; letter-spacing: -0.3px; }
        .header-right { display: flex; align-items: center; gap: 1rem; }
        .user-badge {
            display: flex; align-items: center; gap: 0.6rem;
            background: rgba(255,255,255,0.15);
            padding: 0.5rem 1rem; border-radius: 25px; backdrop-filter: blur(10px);
        }
        .user-name { color: white; font-size: 0.9rem; font-weight: 600; margin: 0; }
        .user-role { color: rgba(255,255,255,0.85); font-size: 0.75rem; margin: 0; }
        .logout-wrapper {
            background: rgba(255,255,255,0.15); padding: 0.5rem 1rem; border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.2); cursor: pointer; transition: all 0.2s;
            color: white; font-weight: 500; font-size: 0.85rem;
        }
        .logout-wrapper:hover { background: rgba(255,255,255,0.25); border-color: rgba(255,255,255,0.4); }
        .content-spacer { height: 70px; }
        button[data-testid="baseButton-secondary"][aria-label="Logout Hidden"] { 
            display: none !important; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="custom-header">
            <div class="header-left">
                <span class="header-icon">
                    <img src="http://localhost:8000/images/bank_logo.png" alt="Logo" style="height:32px;width:32px;">
                </span>
                <span class="header-title">Bank Marketing ML Dashboard</span>
            </div>
            <div class="header-right">
                <div class="user-badge">
                    <span class="user-icon">
                        <img src="http://localhost:8000/images/user_icon.png"
                             style="height:30px;width:30px;border-radius:50%;object-fit:cover;">
                    </span>
                    <div class="user-info">
                        <span class="user-name">{user.get('name', 'Utilizador')}</span>
                        <span class="user-role">{user.get('role', 'Visitante').title()}</span>
                    </div>
                </div>
                <div class="logout-wrapper" onclick="document.querySelector('[data-testid=\\'baseButton-secondary\\']').click()">
                   <img src="http://localhost:8000/images/logout_icon.png" alt="Logout"
                        style="height:28px;width:28px;border-radius:50%;object-fit:cover;margin:auto;">
                </div>
            </div>
        </div>
        <div class="content-spacer"></div>
        """,
        unsafe_allow_html=True,
    )

    # logout button  
    if st.button("Logout", key="logout", type="secondary"):
        logout()

def render_home_page(user: Dict[str, Any]):
    """Render home page with navigation cards."""
    st.markdown("###  Welcome to Machine Learning Dashboard !")
    st.write(f"HI, **{user.get('name')}**!")
    st.markdown(
        """
        <div style="font-size: 1rem; color: #555; margin: 1.5rem 0 2.5rem 0; 
                    padding: 1rem; background: #f8f9fa; border-left: 4px solid #3b82f6; border-radius: 4px;">
            This dashboard enables the analysis of bank marketing campaigns using Machine Learning..<br>
            Begin by clicking “Load Dataset”. From there, you can explore the data, train new models, or immediately generate predictions with the saved model.
        </div>
        """,
        unsafe_allow_html=True,
    )
    # CSS dos cards
    st.markdown(
        """
        <style>
        .cards-container {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            height: 200px;
            width: 18%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border-top: 4px solid transparent;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .feature-card img {
            height: 60px;
            width: 60px;
            object-fit: contain;
            margin-bottom: 0.5rem;
        }
        .feature-card h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e293b;
            margin: 0.5rem 0 0.25rem 0;
        }
        .feature-card p {
            font-size: 0.85rem;
            color: #64748b;
            line-height: 1.5;
            margin: 0;
        }
        .card-1 { border-top-color: #667eea; }
        .card-2 { border-top-color: #43cea2; }
        .card-3 { border-top-color: #f093fb; }
        .card-4 { border-top-color: #4facfe; }
        .card-5 { border-top-color: #fa709a; }

        @media (max-width: 1200px) { .feature-card { width: 30%; } }
        @media (max-width: 800px)  { .feature-card { width: 45%; } }
        @media (max-width: 600px)  { .feature-card { width: 100%; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Cards with images
    st.markdown(
        """
        <div class="cards-container">
            <div class="feature-card card-1">
                <img src="http://localhost:8000/images/loading_dataset.png">
                <h3>Load Dataset</h3>
                <p>Upload and validate the dataset</p>
            </div>
            <div class="feature-card card-2">
                <img src="http://localhost:8000/images/data_explorer.png">
                <h3>Data Explorer</h3>
                <p>Explore data with interactive charts</p>
            </div>
            <div class="feature-card card-3">
                <img src="http://localhost:8000/images/pre_processing.png">
                <h3>Pre-Processing</h3>
                <p>Prepare and transform the dataset</p>
            </div>
            <div class="feature-card card-4">
                <img src="http://localhost:8000/images/train_model.png">
                <h3>Train Model</h3>
                <p>Train and evaluate ML models</p>
            </div>
            <div class="feature-card card-5">
                <img src="http://localhost:8000/images/predict.png">
                <h3>Perform Predictions</h3>
                <p>Predict new client behavior</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Buttons under cards
    st.markdown("#### Quick Access")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("Load Dataset", key="card_load", use_container_width=True):
            st.session_state["page"] = "load_dataset"
            st.rerun()
    
    with col2:
        if st.button("Data Explorer", key="card_explorer", use_container_width=True):
            st.session_state["page"] = "data_explorer"
            st.rerun()
    
    with col3:
        if st.button("Pre-Processing", key="card_preproc", use_container_width=True):
            st.session_state["page"] = "preprocessing"
            st.rerun()
    
    with col4:
        if st.button("Train Model", key="card_train", use_container_width=True):
            st.session_state["page"] = "train_model"
            st.rerun()
    
    with col5:
        if st.button("Perform Predictions", key="card_predict", use_container_width=True):
            st.session_state["page"] = "predictions"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Tip: Use the cards to navigate through the modules.")


def main():
    """Main application entry point."""
    user = initialize_session()
    render_header(user)
    page = st.session_state.get("page", "home")
    try:
        if page == "home":
            render_home_page(user)
        elif page == "load_dataset":
            show_load_data_page()
        elif page == "data_explorer":
            show_data_explorer_page()
        elif page == "preprocessing":
            show_preprocessing_page()
        elif page == "train_model":
            show_train_model_page()
        elif page == "predictions":
            show_predictions_page()
        else:
            st.error(f"Page not found: {page}")
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
    
# Entry Point 
if __name__ == "__main__":
    main()