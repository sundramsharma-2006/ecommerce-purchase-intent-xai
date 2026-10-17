# One slim image for both services; the command decides which one runs (see docker-compose.yml).
# NOTE: written for Docker but NOT built in the original sandbox (no Docker there) - run `docker compose build` yourself.
FROM python:3.11-slim

# LightGBM needs libgomp1 (only relevant when LightGBM is installed via requirements.txt)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
COPY dashboard ./dashboard
COPY dataset ./dataset
COPY models ./models
COPY reports ./reports

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
