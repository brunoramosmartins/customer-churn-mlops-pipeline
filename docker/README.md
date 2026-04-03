# Docker (optional — Phase 10)

The API is intended to run **locally** with `churn-serve` after `pip install -e ".[portfolio]"`. This image is optional for demos.

## Build

From the **repository root**:

```bash
docker build -f docker/Dockerfile -t churn-serve .
```

## Run

Mount the full project so `configs/champion.yaml`, `models/*.joblib`, and optional `data/processed/` resolve under `CHURN_PROJECT_ROOT=/app`:

**Linux / macOS**

```bash
docker run --rm -p 8000:8000 -v "$(pwd)":/app -w /app churn-serve
```

**Windows PowerShell**

```powershell
docker run --rm -p 8000:8000 -v "${PWD}:/app" -w /app churn-serve
```

Then: `curl http://127.0.0.1:8000/health`

Without a mounted `models/` tree containing the champion joblib, `/health` may still respond but `/predict` will fail at startup if the model file is missing.
