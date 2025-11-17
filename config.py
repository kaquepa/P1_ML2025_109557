import os
from pathlib import Path


class Config:

    # ──────────────────────────────────────────────
    # BASE PATHS
    # ──────────────────────────────────────────────
    BASE_DIR = Path(__file__).resolve().parent

    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUTS_DIR = DATA_DIR / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    FIGURES_DIR = OUTPUTS_DIR / "figures"

    SRC_DIR = BASE_DIR / "src"
    STREAMLIT_APP_DIR = BASE_DIR / "streamlit_app"
    FRONTEND_DIR = BASE_DIR / "frontend"

    # ──────────────────────────────────────────────
    # STREAMLIT SETTINGS
    # ──────────────────────────────────────────────
    APP_TITLE = "Bank Marketing ML Dashboard"
    APP_ICON = "🏦"
    LAYOUT = "wide"

    # ──────────────────────────────────────────────
    # AUTH
    # ──────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "123456789qwertyuiop")

    USERS_DB = {
        "admin@bankml.com": {
            "password": "admin123",
            "role": "admin",
            "name": "Graciano"
        },
        "visit@bankml.com": {
            "password": "user123",
            "role": "Visitor",
            "name": "Visitor"
        }
    }

    # ──────────────────────────────────────────────
    # CREATE REQUIRED DIRECTORIES
    # ──────────────────────────────────────────────
    @classmethod
    def create_directories(cls):
        dirs = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUTS_DIR,
            cls.MODELS_DIR,
            cls.FIGURES_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
