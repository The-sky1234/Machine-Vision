"""Local inspection console. One process serves the page and the API."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fabqc.labels import SCENARIO_OPTIONS
from fabqc.simulator.line import SCENARIOS
from fabqc.station import decide_action, ensure_model, get_run, inspect_lot, serialize_run

WEB = Path(__file__).resolve().parent / "web"
_run_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_model()
    yield


app = FastAPI(title="产线质检台", version="0.2.0", lifespan=lifespan)


class RunRequest(BaseModel):
    scenario: str = Field(min_length=1, max_length=64)
    seed: int | None = Field(default=None, ge=0, le=1_000_000)


class DecisionRequest(BaseModel):
    approved: bool


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {
        "disclaimer": "这是模拟产线的样本数据，不是真实工厂。",
        "scenarios": [{"id": k, "label": v} for k, v in SCENARIO_OPTIONS],
    }


@app.post("/api/run")
def run_lot(body: RunRequest) -> dict:
    if body.scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail="unknown scenario")
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="busy")
    try:
        run = inspect_lot(body.scenario, seed=body.seed)
        return serialize_run(run)
    finally:
        _run_lock.release()


@app.get("/api/runs/{run_id}")
def read_run(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return serialize_run(run)


@app.post("/api/runs/{run_id}/actions/{action_id}")
def decide(run_id: str, action_id: str, body: DecisionRequest) -> dict:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        action = decide_action(run_id, action_id, approved=body.approved)
    except KeyError:
        raise HTTPException(status_code=404, detail="action not found") from None
    return {
        "action_id": action.action_id,
        "status": action.status,
        "title": action.title,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run("fabqc.webapp:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
