from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from codereviewenv import CodeReviewAction, CodeReviewEnv, CodeReviewObservation, TASKS

app = FastAPI(
    title="CodeReviewEnv",
    version="0.1.0",
    description="HTTP wrapper for the CodeReviewEnv OpenEnv environment.",
)

_SESSIONS: dict[str, CodeReviewEnv] = {}


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_list_utils")


class ResetResponse(BaseModel):
    session_id: str
    observation: dict[str, Any]
    available_tasks: dict[str, str]


class StepRequest(BaseModel):
    session_id: str
    action: CodeReviewAction


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: dict[str, Any]
    done: bool
    info: dict[str, Any]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "CodeReviewEnv",
        "version": "0.1.0",
        "status": "ok",
        "available_tasks": CodeReviewEnv.available_tasks(),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "name": "CodeReviewEnv",
        "description": "Deterministic pull-request code review environment with seeded bug checklists.",
        "version": "0.1.0",
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": CodeReviewAction.model_json_schema(),
        "observation": CodeReviewObservation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "difficulty": {"type": "string"},
                "step": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "done": {"type": "boolean"},
                "last_score": {"type": ["number", "null"]},
                "last_action_error": {"type": ["string", "null"]},
                "observation": {"type": ["object", "null"]},
            },
        },
    }


@app.get("/tasks")
def tasks() -> dict[str, str]:
    return CodeReviewEnv.available_tasks()


@app.post("/reset", response_model=ResetResponse)
def reset_env(payload: ResetRequest) -> ResetResponse:
    if payload.task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{payload.task_id}'.",
        )

    env = CodeReviewEnv(task_id=payload.task_id)
    observation = env.reset()
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = env
    return ResetResponse(
        session_id=session_id,
        observation=observation.model_dump(),
        available_tasks=CodeReviewEnv.available_tasks(),
    )


@app.post("/step", response_model=StepResponse)
def step_env(payload: StepRequest) -> StepResponse:
    env = _SESSIONS.get(payload.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    observation, reward, done, info = env.step(payload.action)
    return StepResponse(
        observation=observation.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def state(session_id: str = Query(...)) -> dict[str, Any]:
    env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    return env.state()


@app.get("/state/{session_id}")
def state_by_path(session_id: str) -> dict[str, Any]:
    env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    return env.state()


@app.post("/mcp")
def mcp_probe() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "server": "CodeReviewEnv",
            "status": "ok",
        },
    }


@app.delete("/session/{session_id}")
def close_session(session_id: str) -> dict[str, Any]:
    env = _SESSIONS.pop(session_id, None)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    env.close()
    return {"closed": True, "session_id": session_id}
