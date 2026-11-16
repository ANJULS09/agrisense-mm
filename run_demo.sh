#!/usr/bin/env bash
# ==============================================================================
# AgriSense-MM: One-Click Demo Launcher
# Starts FastAPI backend, launches the Streamlit dashboard, and opens the browser.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "  AGRISENSE-MM: MULTIMODAL CROP STRESS & SMART TRANSPORTATION"
echo "  Authors: Sohum Chandra Srivastava (562), Anjul Shukla (572)"
echo "======================================================================"

# 1. Virtual Environment Check
if [ -d ".venv" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"
    STREAMLIT_EXEC="$SCRIPT_DIR/.venv/bin/streamlit"
else
    PYTHON_EXEC="python3"
    STREAMLIT_EXEC="streamlit"
fi

# 2. Check if Database exists, initialize if not
if [ ! -f "agrisense_mm.db" ]; then
    echo "[*] Initializing database and generating baseline field data..."
    "$PYTHON_EXEC" run_all.py
fi

# 3. Check if Backend is running, start if needed
if ! lsof -i :8000 > /dev/null 2>&1; then
    echo "[*] Starting FastAPI Ingestion Backend on port 8000..."
    "$PYTHON_EXEC" serve.py > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "[+] Backend started (PID: $BACKEND_PID). Logs: backend.log"
    sleep 2
else
    echo "[✓] Backend already active on http://127.0.0.1:8000"
fi

# 4. Run a background vehicle simulation batch
echo "[*] Streaming live vehicle telemetry steps into system..."
"$PYTHON_EXEC" simulate.py 20 > simulator.log 2>&1 &

# 5. Open browser and start Streamlit Dashboard
echo "[*] Opening Dashboard at http://localhost:8501..."
if which open > /dev/null 2>&1; then
    open "http://localhost:8501" || true
fi

echo "[*] Starting Streamlit Dashboard..."
"$STREAMLIT_EXEC" run dashboard/app.py --server.port 8501

