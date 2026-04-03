# Optional — Phase 10. Build from repo root:
#   docker build -f docker/Dockerfile -t churn-serve .
# Run with project root mounted so configs/, models/, and data/ resolve:
#   docker run --rm -p 8000:8000 -v "%cd%":/app -w /app churn-serve
# (PowerShell: use ${PWD} or full path instead of %cd%.)

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[portfolio]"

ENV CHURN_PROJECT_ROOT=/app

EXPOSE 8000

CMD ["churn-serve", "--host", "0.0.0.0", "--port", "8000"]
