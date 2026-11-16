"""
AgriSense-MM: Backend API Launcher.
Runs FastAPI with Uvicorn on port 8000.
"""

import uvicorn

if __name__ == "__main__":
    print("Starting AgriSense-MM Ingestion & Routing Backend on http://127.0.0.1:8000...")
    uvicorn.run("agrisense_mm.services.ingest_api:app", host="0.0.0.0", port=8000, reload=False)

