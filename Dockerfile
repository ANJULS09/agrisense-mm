# AgriSense-MM: Containerized Deployment
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .
RUN pip install -e .

EXPOSE 8000 8501 1883

CMD ["uvicorn", "agrisense_mm.services.ingest_api:app", "--host", "0.0.0.0", "--port", "8000"]

