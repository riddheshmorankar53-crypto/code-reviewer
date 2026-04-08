import pytest
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "CodeReviewEnv"
    assert "description" in payload


def test_schema_endpoint_has_required_sections():
    response = client.get("/schema")
    assert response.status_code == 200
    payload = response.json()
    assert "action" in payload
    assert "observation" in payload
    assert "state" in payload


def test_reset_step_state_flow():
    reset_response = client.post("/reset", json={"task_id": "easy_list_utils"})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()

    session_id = reset_payload["session_id"]
    assert reset_payload["observation"]["task_id"] == "easy_list_utils"

    step_response = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": {
                "verdict": "request_changes",
                "summary": "Found an off-by-one in get_last_n and a missing zero-division guard.",
                "identified_bugs": [
                    "off_by_one_slice",
                    "missing_zero_division_guard",
                ],
                "inline_comments": [],
            },
        },
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["done"] is True
    assert step_payload["reward"]["score"] == pytest.approx(0.95)

    state_response = client.get("/state", params={"session_id": session_id})
    assert state_response.status_code == 200
    assert state_response.json()["done"] is True


def test_mcp_probe():
    response = client.post("/mcp", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
