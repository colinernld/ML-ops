# Streamlit web app: a thin client of the API (no PyTorch, no model), so the image stays small.
#   docker build -f docker/webapp/Dockerfile -t ride-webapp .   (build context = repository root)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN pip install "streamlit>=1.40,<2" "requests>=2.32" "pandas>=2.1"

# app.py looks for the sample rides in ../examples (EXAMPLES_DIR overrides it).
COPY webapp/ webapp/
COPY examples/ examples/

RUN useradd --create-home --uid 10001 app
USER 10001

ENV API_URL=http://api:8000
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
