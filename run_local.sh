#!/usr/bin/env bash
set -e

echo "========================================================================="
echo "  VisionX: Rural Tele-Ophthalmology Screening Kiosk"
echo "========================================================================="
echo ""

if [ -f ".venv/bin/activate" ]; then
    echo "[*] Activating local virtual environment (.venv)..."
    source .venv/bin/activate
fi

command -v python3 >/dev/null 2>&1 || { echo >&2 "[ERROR] Python3 is required but not installed."; exit 1; }

echo "[*] Checking Python dependencies..."
if python3 -c "import fastapi, torch, torchvision, cv2, uvicorn" >/dev/null 2>&1; then
    echo "[*] Python dependencies verified."
else
    echo "[*] Missing dependencies detected. Installing from requirements.txt..."
    pip install -r requirements.txt
fi

echo ""
echo "[*] Starting FastAPI Backend on http://127.0.0.1:8000 ..."
python3 -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

trap "kill $API_PID 2>/dev/null || true" EXIT

sleep 3

FRONTEND_DIR="frontend"
if [ -d "$FRONTEND_DIR" ]; then
    echo "[*] Launching Frontend Clinical Cockpit..."
    cd "$FRONTEND_DIR"
    if [ ! -d "node_modules" ]; then
        echo "[*] Installing frontend packages..."
        npm install
    fi
    npm run dev &
    cd ../..
fi

echo ""
echo "========================================================================="
echo "  VisionX is running!"
echo "  - Web Cockpit:  http://localhost:5173"
echo "  - Backend API:  http://127.0.0.1:8000"
echo "  - API Docs:     http://127.0.0.1:8000/api/v1/docs"
echo "========================================================================="
wait
