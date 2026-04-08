"""
Three real-world PR code-review tasks (easy / medium / hard).

Each task is a dataclass that carries:
  - The PR metadata + file diffs presented to the agent
  - A seeded bug checklist used by the deterministic grader
  - The expected verdict
  - A seeded RNG for reproducible scoring

Bug checklist format:
  {bug_id: weight}   where weights sum to 1.0
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from ..models import (
    CodeReviewObservation,
    DifficultyLevel,
    FileDiff,
    PullRequestMeta,
    ReviewVerdict,
)

# ---------------------------------------------------------------------------
# Base task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: DifficultyLevel
    pr_meta: PullRequestMeta
    files: List[FileDiff]
    bug_checklist: Dict[str, float]          # bug_id -> weight (sum == 1.0)
    expected_verdict: ReviewVerdict
    seed: int = 42
    max_steps: int = 3

    def __post_init__(self) -> None:
        total = sum(self.bug_checklist.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Bug checklist weights must sum to 1.0, got {total:.4f}"
        )
        self._rng = random.Random(self.seed)

    def initial_observation(self) -> CodeReviewObservation:
        return CodeReviewObservation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            pr_meta=self.pr_meta,
            files=self.files,
            review_history=[],
            step_number=0,
            max_steps=self.max_steps,
        )


# ---------------------------------------------------------------------------
# TASK 1 — EASY
# A small Python utility with an obvious off-by-one error and a missing
# None-check. Perfect for a one-turn review.
# ---------------------------------------------------------------------------

EASY_TASK = Task(
    task_id="easy_list_utils",
    difficulty=DifficultyLevel.EASY,
    seed=1001,
    max_steps=1,
    pr_meta=PullRequestMeta(
        pr_id="PR-101",
        title="Add list utility helpers",
        description=(
            "Adds `get_last_n` and `safe_divide` helpers to list_utils.py. "
            "Fixes issue #88."
        ),
        author="alice",
        base_branch="main",
        head_branch="feature/list-utils",
        labels=["enhancement"],
    ),
    files=[
        FileDiff(
            filename="src/list_utils.py",
            language="python",
            additions=28,
            deletions=0,
            patch="""\
+def get_last_n(lst: list, n: int) -> list:
+    \"\"\"Return the last n elements of lst.\"\"\"
+    if n <= 0:
+        return []
+    # BUG: should be lst[-n:] not lst[-n-1:-1]
+    return lst[-n-1:-1]
+
+
+def safe_divide(a: float, b: float) -> float:
+    \"\"\"Return a / b, or 0.0 if b is zero.\"\"\"
+    # BUG: does not handle b == 0; will raise ZeroDivisionError
+    return a / b
+
+
+def flatten(nested: list) -> list:
+    \"\"\"Recursively flatten a nested list.\"\"\"
+    result = []
+    for item in nested:
+        if isinstance(item, list):
+            result.extend(flatten(item))
+        else:
+            result.append(item)
+    return result
+
+
+def chunk(lst: list, size: int) -> list:
+    \"\"\"Split lst into chunks of `size`.\"\"\"
+    if size <= 0:
+        raise ValueError("size must be positive")
+    return [lst[i:i + size] for i in range(0, len(lst), size)]
""",
        ),
        FileDiff(
            filename="tests/test_list_utils.py",
            language="python",
            additions=14,
            deletions=0,
            patch="""\
+import pytest
+from src.list_utils import get_last_n, safe_divide, flatten, chunk
+
+def test_get_last_n_basic():
+    # NOTE: This test is wrong — it passes because of the bug above
+    assert get_last_n([1, 2, 3, 4, 5], 2) == [4, 5]  # actually returns [4,5] by accident only for this input
+
+def test_flatten():
+    assert flatten([1, [2, [3]]]) == [1, 2, 3]
+
+def test_chunk():
+    assert chunk([1,2,3,4], 2) == [[1,2],[3,4]]
""",
        ),
    ],
    bug_checklist={
        "off_by_one_slice": 0.45,       # lst[-n-1:-1] instead of lst[-n:]
        "missing_zero_division_guard": 0.40,  # safe_divide no guard
        "misleading_test": 0.15,        # test accidentally passes due to bug
    },
    expected_verdict=ReviewVerdict.REQUEST_CHANGES,
)


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM
# A Flask REST endpoint with a SQL injection vulnerability, missing auth,
# and an unhandled exception path. Requires reading across two files.
# ---------------------------------------------------------------------------

MEDIUM_TASK = Task(
    task_id="medium_flask_api",
    difficulty=DifficultyLevel.MEDIUM,
    seed=2002,
    max_steps=2,
    pr_meta=PullRequestMeta(
        pr_id="PR-247",
        title="Add /users/search endpoint",
        description=(
            "Implements a new search endpoint that queries users by name. "
            "No authentication required for public profiles."
        ),
        author="bob",
        base_branch="main",
        head_branch="feature/user-search",
        labels=["feature", "api"],
    ),
    files=[
        FileDiff(
            filename="app/routes/users.py",
            language="python",
            additions=42,
            deletions=3,
            patch="""\
-from flask import Blueprint, jsonify
+from flask import Blueprint, jsonify, request
+from app.db import get_db
+
 users_bp = Blueprint("users", __name__, url_prefix="/users")
+
+
+@users_bp.route("/search", methods=["GET"])
+def search_users():
+    name = request.args.get("name", "")
+    db = get_db()
+    # BUG: SQL injection — user input concatenated directly into query
+    query = f"SELECT id, username, email FROM users WHERE username LIKE '%{name}%'"
+    cursor = db.execute(query)
+    rows = cursor.fetchall()
+    # BUG: exposes email addresses — should be filtered for public endpoint
+    results = [{"id": r[0], "username": r[1], "email": r[2]} for r in rows]
+    return jsonify(results)
+
+
+@users_bp.route("/<int:user_id>", methods=["DELETE"])
+def delete_user(user_id: int):
+    # BUG: no authentication / authorisation check before deleting
+    db = get_db()
+    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
+    db.commit()
+    return jsonify({"deleted": user_id})
""",
        ),
        FileDiff(
            filename="app/db.py",
            language="python",
            additions=18,
            deletions=0,
            patch="""\
+import sqlite3
+from flask import g, current_app
+
+
+def get_db():
+    if "db" not in g:
+        g.db = sqlite3.connect(
+            current_app.config["DATABASE"],
+            detect_types=sqlite3.PARSE_DECLTYPES,
+        )
+        g.db.row_factory = sqlite3.Row
+    return g.db
+
+
+def close_db(e=None):
+    db = g.pop("db", None)
+    if db is not None:
+        db.close()
""",
        ),
        FileDiff(
            filename="tests/test_users.py",
            language="python",
            additions=12,
            deletions=0,
            patch="""\
+import pytest
+from app import create_app
+
+@pytest.fixture
+def client():
+    app = create_app({"TESTING": True, "DATABASE": ":memory:"})
+    with app.test_client() as c:
+        yield c
+
+def test_search_returns_list(client):
+    # BUG: No test for SQL injection or auth on delete
+    resp = client.get("/users/search?name=alice")
+    assert resp.status_code == 200
""",
        ),
    ],
    bug_checklist={
        "sql_injection": 0.35,
        "sensitive_data_exposure": 0.20,
        "missing_authentication": 0.25,
        "missing_security_tests": 0.20,
    },
    expected_verdict=ReviewVerdict.REQUEST_CHANGES,
)


# ---------------------------------------------------------------------------
# TASK 3 — HARD
# An async data-pipeline module with a race condition, improper exception
# handling swallowing errors, a memory leak in a long-running worker, and a
# subtle type coercion bug. Spread across four files.
# ---------------------------------------------------------------------------

HARD_TASK = Task(
    task_id="hard_async_pipeline",
    difficulty=DifficultyLevel.HARD,
    seed=3003,
    max_steps=3,
    pr_meta=PullRequestMeta(
        pr_id="PR-512",
        title="Async event-pipeline v2 with worker pool",
        description=(
            "Rewrites the synchronous event pipeline as an async worker pool. "
            "Uses asyncio.Queue for back-pressure. Includes retry logic and "
            "metrics emission."
        ),
        author="carol",
        base_branch="main",
        head_branch="refactor/async-pipeline",
        labels=["refactor", "performance", "async"],
    ),
    files=[
        FileDiff(
            filename="pipeline/worker.py",
            language="python",
            additions=68,
            deletions=12,
            patch="""\
+import asyncio
+import logging
+from typing import Any, Callable, Coroutine
+
+logger = logging.getLogger(__name__)
+
+
+class Worker:
+    def __init__(self, queue: asyncio.Queue, handler: Callable, worker_id: int):
+        self.queue = queue
+        self.handler = handler
+        self.worker_id = worker_id
+        # BUG: results list grows unbounded — memory leak in long-running workers
+        self.results: list = []
+
+    async def run(self) -> None:
+        while True:
+            item = await self.queue.get()
+            try:
+                result = await self.handler(item)
+                self.results.append(result)
+            except Exception:
+                # BUG: bare except swallows ALL exceptions including
+                # KeyboardInterrupt / CancelledError; should re-raise
+                # CancelledError and at minimum log the error
+                pass
+            finally:
+                self.queue.task_done()
+
+    async def stop(self) -> None:
+        # BUG: does not cancel the running coroutine; just returns
+        return
""",
        ),
        FileDiff(
            filename="pipeline/coordinator.py",
            language="python",
            additions=55,
            deletions=8,
            patch="""\
+import asyncio
+from .worker import Worker
+
+
+class Coordinator:
+    def __init__(self, num_workers: int = 4):
+        self.num_workers = num_workers
+        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
+        self._workers: list[Worker] = []
+        self._counter: int = 0
+
+    async def start(self, handler) -> None:
+        for i in range(self.num_workers):
+            w = Worker(self._queue, handler, i)
+            self._workers.append(w)
+            asyncio.create_task(w.run())
+
+    async def submit(self, item: dict) -> None:
+        # BUG: race condition — _counter is read and written without a lock;
+        # two coroutines can submit simultaneously and produce duplicate IDs
+        self._counter += 1
+        item["seq"] = self._counter
+        await self._queue.put(item)
+
+    async def shutdown(self) -> None:
+        await self._queue.join()
+        for w in self._workers:
+            await w.stop()
""",
        ),
        FileDiff(
            filename="pipeline/metrics.py",
            language="python",
            additions=30,
            deletions=0,
            patch="""\
+import time
+from dataclasses import dataclass, field
+from typing import Dict
+
+
+@dataclass
+class MetricsCollector:
+    counters: Dict[str, int] = field(default_factory=dict)
+    _start: float = field(default_factory=time.monotonic, init=False)
+
+    def increment(self, key: str, value: int = 1) -> None:
+        # BUG: value is not validated — negative values silently corrupt counters
+        self.counters[key] = self.counters.get(key, 0) + value
+
+    def elapsed(self) -> float:
+        return time.monotonic() - self._start
+
+    def report(self) -> dict:
+        # BUG: implicit int->str coercion; keys that overlap with Python
+        # built-ins (e.g. "type") will silently shadow them in downstream JSON
+        return {k: str(v) for k, v in self.counters.items()}
""",
        ),
        FileDiff(
            filename="tests/test_pipeline.py",
            language="python",
            additions=22,
            deletions=0,
            patch="""\
+import asyncio
+import pytest
+from pipeline.coordinator import Coordinator
+
+
+@pytest.mark.asyncio
+async def test_basic_submit():
+    processed = []
+
+    async def handler(item):
+        processed.append(item)
+
+    coord = Coordinator(num_workers=2)
+    await coord.start(handler)
+    await coord.submit({"data": "hello"})
+    await asyncio.sleep(0.1)
+    assert len(processed) == 1
+    # BUG: test does not verify seq IDs are unique (race condition not caught)
+    # BUG: no test for error handling, memory bounds, or graceful shutdown
""",
        ),
    ],
    bug_checklist={
        "unbounded_memory_leak": 0.20,
        "swallowed_exception_cancellederror": 0.20,
        "race_condition_counter": 0.25,
        "broken_stop_method": 0.10,
        "negative_counter_corruption": 0.10,
        "type_coercion_report": 0.05,
        "insufficient_async_tests": 0.10,
    },
    expected_verdict=ReviewVerdict.REQUEST_CHANGES,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {
    EASY_TASK.task_id: EASY_TASK,
    MEDIUM_TASK.task_id: MEDIUM_TASK,
    HARD_TASK.task_id: HARD_TASK,
}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}"
        )
    return TASKS[task_id]
