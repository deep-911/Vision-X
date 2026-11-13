@echo off
TITLE VisionX - Tele-Ophthalmology Screening Platform
echo =========================================================================
echo   VisionX: Rural Tele-Ophthalmology Screening Kiosk
echo =========================================================================
echo.

:: Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating local virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

:: Verify dependencies without reaching out to network if already installed
echo [*] Checking Python dependencies...
python -c "import fastapi, torch, torchvision, cv2, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [*] Missing dependencies detected. Installing from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [*] Python dependencies verified.
)

echo.
echo [*] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "VisionX API Backend" cmd /k "if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat && python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000"

:: Wait for API startup
timeout /t 3 /nobreak >nul

echo [*] Starting Frontend Clinical Cockpit...
if exist "frontend\node_modules" (
    start "VisionX Frontend Cockpit" cmd /k "cd frontend && npm run dev"
    timeout /t 2 /nobreak >nul
    start http://localhost:5173
) else (
    echo [!] Node modules not found in frontend. Installing packages...
    start "VisionX Frontend Cockpit" cmd /k "cd frontend && npm install && npm run dev"
    timeout /t 5 /nobreak >nul
    start http://localhost:5173
)

echo.
echo =========================================================================
echo   VisionX is running!
echo   - Web Cockpit:  http://localhost:5173
echo   - Backend API:  http://127.0.0.1:8000
echo   - API Docs:     http://127.0.0.1:8000/api/v1/docs
echo =========================================================================
echo.
echo Press any key to close this launcher (servers will keep running in their windows)...
pause >nul
