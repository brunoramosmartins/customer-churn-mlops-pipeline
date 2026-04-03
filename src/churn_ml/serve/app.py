"""FastAPI application — load champion at startup."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from churn_ml.serve.router import router
from churn_ml.serve.state import ChampionState, load_champion_state


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.churn = load_champion_state()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Telco churn inference",
        description="Phase 10 portfolio API — POST /predict with one Telco row JSON.",
        version="0.10.0",
        lifespan=_lifespan,
    )
    app.include_router(router)
    return app


def create_test_app(state: ChampionState) -> FastAPI:
    """FastAPI app with a pre-built champion state (no lifespan / disk load)."""
    app = FastAPI(title="Telco churn inference (test)", version="0.10.0")
    app.state.churn = state
    app.include_router(router)
    return app


app = create_app()
