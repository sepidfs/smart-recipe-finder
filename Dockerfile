# Dockerfile — Smart Recipe App (Streamlit, Cloud Run–ready)

# 1) Base image
FROM python:3.11-slim

# 2) Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 3) Working directory
WORKDIR /app

# 4) OS packages
#    - ffmpeg: for audio/TTS (pydub, gTTS)
#    - libgomp1: needed by xgboost/lightgbm if your pipeline uses them (harmless otherwise)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 5) Python dependencies (install first for better layer caching)
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# 6) Application sources and artifacts (model, labels, assets, app code)
COPY . /app

# 7) Use /bin/sh so ${PORT} expands at runtime (Cloud Run / Docker)
SHELL ["/bin/sh", "-c"]

# 8) Entrypoint — bind to 0.0.0.0 and use PORT if provided (default 8080)
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
