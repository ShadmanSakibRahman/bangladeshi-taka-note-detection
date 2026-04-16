# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Install PyTorch CPU-only version first (much smaller than GPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights and source code
COPY best.pt .
COPY app.py .
COPY inference.py .

# Expose port 8000 for the API
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
