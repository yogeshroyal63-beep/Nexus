"""Nexus — Autonomous Developer Agent. FastAPI entrypoint."""
from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nexus",
    description="Autonomous Developer Agent — detects, diagnoses, and autonomously remediates issues.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate limiter -----------------------------------------------------------
_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMITS = {
    "/api/run": (3, 60),
    "/api/incidents": (20, 60),
    "/api/investigate": (5, 60),
}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    for pattern, (limit, window) in RATE_LIMITS.items():
        if path.startswith(pattern):
            client_ip = request.client.host if request.client else "unknown"
            key = f"{client_ip}:{pattern}"
            now = time.time()
            _rate_store[key] = [t for t in _rate_store[key] if now - t < window]
            if len(_rate_store[key]) >= limit:
                return Response(
                    content='{"detail":"Rate limit exceeded. Please wait before retrying."}',
                    status_code=429,
                    media_type="application/json",
                )
            _rate_store[key].append(now)
            break
    return await call_next(request)


app.include_router(router, prefix="/api")
