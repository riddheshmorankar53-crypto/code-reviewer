# CodeReviewEnv

CodeReviewEnv is an OpenEnv-style reinforcement learning environment for pull-request code review. Agents receive realistic PR metadata and diffs, then submit review actions containing a verdict, summary, inline comments, and identified bug IDs. Rewards are deterministic and come from a seeded per-task checklist, so evaluation is reproducible without an LLM judge.

## Why this environment

Code review is a real-world software engineering task with clear value for training and evaluation. This environment models practical review behavior:

- read PR metadata and diffs
- identify correctness and security issues
- provide structured review feedback
- choose an overall review verdict

## Project structure

```text
CodeReviewEnv/
├── codereviewenv/
│   ├── __init__.py
│   ├── env.py
│   ├── grader.py
│   ├── models.py
│   └── tasks/
│       └── __init__.py
├── inference.py
├── scripts/
│   └── baseline_inference.py
├── tests/
│   └── test_env.py
├── Dockerfile
├── openenv.yaml
├── requirements.txt
└── setup.py
```

## Tasks

There are 3 deterministic tasks with increasing difficulty.

| Task ID | Difficulty | Max Steps | Seeded Bugs | Expected Verdict |
| --- | --- | ---: | ---: | --- |
| `easy_list_utils` | easy | 1 | 3 | `request_changes` |
| `medium_flask_api` | medium | 2 | 4 | `request_changes` |
| `hard_async_pipeline` | hard | 3 | 7 | `request_changes` |

### `easy_list_utils`

Small Python utility review with:

- `off_by_one_slice`
- `missing_zero_division_guard`
- `misleading_test`

### `medium_flask_api`

Flask API review with:

- `sql_injection`
- `sensitive_data_exposure`
- `missing_authentication`
- `missing_security_tests`

### `hard_async_pipeline`

Async worker-pool review with:

- `unbounded_memory_leak`
- `swallowed_exception_cancellederror`
- `race_condition_counter`
- `broken_stop_method`
- `negative_counter_corruption`
- `type_coercion_report`
- `insufficient_async_tests`

## Observation space

The observation is the typed Pydantic model `CodeReviewObservation` and contains:

- `task_id`
- `difficulty`
- `pr_meta`
- `files`
- `review_history`
- `step_number`
- `max_steps`

Each changed file is a `FileDiff` with:

- `filename`
- `language`
- `patch`
- `additions`
- `deletions`

## Action space

The action is the typed Pydantic model `CodeReviewAction` and contains:

- `verdict`
- `summary`
- `inline_comments`
- `identified_bugs`
- `reasoning` (optional, ignored by scoring)

Each inline comment is an `InlineComment` with:

- `filename`
- `line_number`
- `body`
- `severity`

## Reward model

The reward is the typed Pydantic model `CodeReviewReward` and includes:

- `score`
- `bug_detection_score`
- `verdict_correct`
- `coverage_ratio`
- `checklist_results`
- `feedback`
- `done`
- `info`

Scoring:

```text
bug_detection_score = sum(weight_i for each correctly identified bug_i)
verdict_bonus       = 0.10 if verdict matches expected_verdict else 0.0
final_score         = min(1.0, bug_detection_score + verdict_bonus)
```

This provides partial progress signal and deterministic reproducibility.

## Environment API

```python
from codereviewenv import CodeReviewEnv, CodeReviewAction, ReviewVerdict

env = CodeReviewEnv(task_id="easy_list_utils")
obs = env.reset()

action = CodeReviewAction(
    verdict=ReviewVerdict.REQUEST_CHANGES,
    summary="Found an off-by-one in get_last_n and a missing zero-division guard.",
    identified_bugs=["off_by_one_slice", "missing_zero_division_guard"],
)

next_obs, reward, done, info = env.step(action)
state = env.state()
env.close()
```

`step(action)` returns `(observation, reward, done, info)`.

## Setup

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Local usage

```bash
python inference.py --help
```

Quick smoke example:

```bash
python - <<'PY'
from codereviewenv import CodeReviewEnv, CodeReviewAction, ReviewVerdict

env = CodeReviewEnv(task_id="easy_list_utils")
obs = env.reset()
action = CodeReviewAction(
    verdict=ReviewVerdict.REQUEST_CHANGES,
    summary="Found an off-by-one in get_last_n and a missing zero-division guard.",
    identified_bugs=["off_by_one_slice", "missing_zero_division_guard"],
)
next_obs, reward, done, info = env.step(action)
print(obs.pr_meta.title)
print(reward.score)
print(done)
PY
```

## Submission inference script

The required submission entrypoint is the root-level `inference.py`.

Environment variables:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`

Defaults:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`

Example:

```bash
export HF_TOKEN="hf-..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
python inference.py --task hard_async_pipeline
```

The script emits strict structured logs in the required format:

```text
[START] task=<task_name> env=codereviewenv model=<model_name>
[STEP] step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

## Human-friendly baseline script

For local debugging, a more verbose runner is also available:

```bash
python scripts/baseline_inference.py --model "$MODEL_NAME" --verbose
```

## HTTP API

The repository also includes a lightweight FastAPI wrapper for containerized deployment and Hugging Face Spaces.

Endpoints:

- `GET /`
- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state/{session_id}`
- `DELETE /session/{session_id}`

Example:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy_list_utils"}'
```

## Docker

Build:

```bash
docker build -t codereviewenv .
```

Run the API server:

```bash
docker run --rm -p 7860:7860 codereviewenv
```

Run submission inference manually inside the image:

```bash
docker run --rm --entrypoint python \
  -e HF_TOKEN="$HF_TOKEN" \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  codereviewenv inference.py --task hard_async_pipeline
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
