#!/usr/bin/env python3
"""
Submission inference script for CodeReviewEnv.

Emits strict structured logs:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from typing import Optional

from openai import OpenAI

from codereviewenv import (
    CodeReviewAction,
    CodeReviewEnv,
    CodeReviewObservation,
    ReviewVerdict,
    TASKS,
)
from codereviewenv.models import InlineComment, Severity

BENCHMARK = "codereviewenv"
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert software engineer performing a pull-request code review.
    You will be given a PR description and one or more file diffs.

    Your job:
    1. Identify bugs, security issues, and correctness problems.
    2. Add canonical snake_case bug IDs to "identified_bugs".
    3. Write clear inline comments for each issue.
    4. Decide on a verdict: "approve", "request_changes", or "comment".
    5. Write a concise overall summary.

    Respond only with a valid JSON object:
    {
      "verdict": "<approve|request_changes|comment>",
      "summary": "<string, min 10 chars>",
      "identified_bugs": ["<bug_id>", ...],
      "inline_comments": [
        {
          "filename": "<string>",
          "line_number": <int>,
          "body": "<string>",
          "severity": "<info|warning|error|critical>"
        }
      ]
    }
    """
)


def build_user_message(obs: CodeReviewObservation) -> str:
    lines: list[str] = []
    pr = obs.pr_meta
    lines.extend(
        [
            f"Pull Request: {pr.title}",
            f"Author: {pr.author}",
            f"Branch: {pr.head_branch} -> {pr.base_branch}",
            f"Labels: {', '.join(pr.labels) or 'none'}",
            "",
            "Description:",
            pr.description,
            "",
            "Changed Files:",
        ]
    )

    for file_diff in obs.files:
        lines.extend(
            [
                f"File: {file_diff.filename} ({file_diff.language}) +{file_diff.additions} -{file_diff.deletions}",
                file_diff.patch,
                "",
            ]
        )

    if obs.review_history:
        lines.append("Review History:")
        for entry in obs.review_history:
            lines.append(f"Step {entry['step']}: score={entry['score']:.3f}")

    lines.append("Return a JSON object only.")
    return "\n".join(lines)


def parse_action(raw: str) -> CodeReviewAction:
    content = raw.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content)

    try:
        verdict = ReviewVerdict(data.get("verdict", "comment"))
    except ValueError:
        verdict = ReviewVerdict.COMMENT

    inline_comments = []
    for ic in data.get("inline_comments", []):
        try:
            severity = Severity(ic.get("severity", "warning"))
        except ValueError:
            severity = Severity.WARNING
        inline_comments.append(
            InlineComment(
                filename=ic["filename"],
                line_number=max(1, int(ic.get("line_number", 1))),
                body=ic["body"],
                severity=severity,
            )
        )

    return CodeReviewAction(
        verdict=verdict,
        summary=data.get("summary", "No summary provided."),
        identified_bugs=data.get("identified_bugs", []),
        inline_comments=inline_comments,
        reasoning=data.get("reasoning"),
    )


def call_model(
    client: OpenAI,
    obs: CodeReviewObservation,
    model_name: str,
    temperature: float,
) -> CodeReviewAction:
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(obs)},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    return parse_action(raw)


def compact_action(action: CodeReviewAction) -> str:
    payload = {
        "verdict": action.verdict.value,
        "identified_bugs": action.identified_bugs,
        "summary": action.summary,
    }
    action_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return re.sub(r"\s+", " ", action_str)


def emit_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}")


def emit_step(step: int, action: CodeReviewAction, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = "null" if not error else re.sub(r"\s+", " ", error)
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={compact_action(action)} "
        f"reward={reward:.2f} done={done_value} error={error_value}"
    )


def emit_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}"
    )


def run_task(task_name: str, client: OpenAI, model_name: str, temperature: float) -> float:
    env = CodeReviewEnv(task_id=task_name)
    rewards: list[float] = []
    success = False
    final_score = 0.0
    steps = 0
    emit_start(task_name, model_name)

    try:
        obs = env.reset()
        for step_number in range(1, env.task.max_steps + 1):
            steps = step_number
            action = call_model(client, obs, model_name, temperature)
            obs, reward, done, info = env.step(action)
            rewards.append(reward.score)
            final_score = reward.score
            emit_step(
                step=step_number,
                action=action,
                reward=reward.score,
                done=done,
                error=info.get("last_action_error"),
            )
            if done:
                success = True
                break
    except Exception as exc:
        fallback_action = CodeReviewAction(
            verdict=ReviewVerdict.COMMENT,
            summary="Inference failed before a valid review could be completed.",
        )
        emit_step(
            step=max(steps, 1),
            action=fallback_action,
            reward=0.0,
            done=True,
            error=str(exc),
        )
    finally:
        env.close()
        emit_end(success=success, steps=steps, score=final_score, rewards=rewards)

    return final_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CodeReviewEnv inference.")
    parser.add_argument("--task", choices=list(TASKS.keys()), default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME))
    args = parser.parse_args()

    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

    if not api_key:
        print("HF_TOKEN or OPENAI_API_KEY must be set.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    task_names = [args.task] if args.task else list(TASKS.keys())

    for task_name in task_names:
        run_task(task_name, client, args.model, args.temperature)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
