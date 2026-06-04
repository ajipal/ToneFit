FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install CLIP from GitHub
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Copy backend source
COPY backend/ ./backend/

# Models directory (populated at runtime via download_models.py)
RUN mkdir -p models

EXPOSE 8000

# Download models then start server
# PORT is injected by Render; falls back to 8000 for local use
CMD ["sh", "-c", "python backend/download_models.py && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
