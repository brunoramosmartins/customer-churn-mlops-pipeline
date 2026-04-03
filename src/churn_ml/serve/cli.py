"""Run uvicorn for local serving: churn-serve."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastAPI churn inference (Phase 10).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload (not for production).")
    args = parser.parse_args()

    import uvicorn

    # Ensure env can point project root when cwd differs
    if os.environ.get("CHURN_PROJECT_ROOT") is None:
        from pathlib import Path

        from churn_ml.serve.state import serve_project_root

        os.environ.setdefault("CHURN_PROJECT_ROOT", str(serve_project_root()))

    uvicorn.run(
        "churn_ml.serve.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
