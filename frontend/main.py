import subprocess
import time
import psutil
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
try:
    from config import Config
    FRONTEND_DIR = Config.FRONTEND_DIR
except ImportError:
    # Fallback
    from config import Config

from token_utils import generate_token

def is_streamlit_running(port=8501):
    """verify if streamlit is running"""
    for proc in psutil.process_iter(attrs=["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if not cmdline or not isinstance(cmdline, list):
                continue

            cmd_str = " ".join(cmdline)
            if "streamlit" in cmd_str and f"--server.port={port}" in cmd_str:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_streamlit():
    """Start Streamlit in background """
    if not is_streamlit_running():
        subprocess.Popen(
            [
                "streamlit",
                "run",
                "../streamlit_app/app.py",
                "--server.port=8501",
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
                "--browser.serverAddress=localhost",
                "--browser.serverPort=8501"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        time.sleep(3)  
    else:
        print("Streamlit is already running.")

app = FastAPI(title="Login - Bank ML")


start_streamlit()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory=FRONTEND_DIR / "images"), name="images")
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/", response_class=HTMLResponse)
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    user_data = Config.USERS_DB.get(email)
    if user_data and user_data["password"] == password:
        token = generate_token(email)
        streamlit_url = f"http://localhost:8501?token={token}"
        return RedirectResponse(url=streamlit_url, status_code=303)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid Credential"})
@app.get("/health")
async def health_check():
    return {"status": "ok"}
