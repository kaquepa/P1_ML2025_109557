import streamlit as st
import time 
def show_header():
    """Renderiza o friso superior com nome do utilizador e botão Logout"""
    user = st.session_state.get("user")
    if not user or not isinstance(user, dict):
        return

    name = user.get("name","Admin")
    role = user.get("role", "Visitor")
    st.markdown(
        """
        <style>
        header {visibility: hidden;}
        .block-container {padding-top: 5rem !important;}
        .custom-header {
            position: fixed;
            top: 0; left: 0; right: 0;
            z-index: 9999;
            background-color: #111827;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 2rem;
            font-family: 'Segoe UI', Roboto, sans-serif;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }
        .app-title {
            font-size: 1.1rem;
            font-weight: 600;
        }
        .user-section {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .user-info {
            font-size: 0.9rem;
            color: #d1d5db;
        }
        .logout-btn {
            background-color: #ef4444;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .logout-btn:hover {
            background-color: #dc2626;
        }
        </style>

        <div class="custom-header">
            <div class="app-title">🏦 Bank Marketing ML Dashboard</div>
            <div class="user-section">
                <div class="user-info">👤 {name} ({role})</div>
                <form action="" method="post">
                    <button class="logout-btn" onclick="window.location.href='/?logout=true'">🚪 Logout</button>
                </form>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    query_params = st.query_params
    if query_params.get("logout") == "true":
        logout()
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.toast("Logout...")
    time.sleep(0.5)
    st.markdown(
        """
        <meta http-equiv="refresh" content="0;url=http://localhost:8000" />
        """,
        unsafe_allow_html=True
    )
    st.stop()
