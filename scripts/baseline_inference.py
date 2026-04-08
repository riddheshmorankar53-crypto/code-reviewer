#!/usr/bin/env python3
"""
Baseline inference script for CodeReviewEnv.
--------------------------------------------
Uses the OpenAI Chat Completions API (gpt-4o by default) to run each
registered task and report scores.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python scripts/baseline_inference.py                      # all tasks
    python scripts/baseline_inference.py --task easy_list_utils
    python scripts/baseline_inference.py --model gpt-4-turbo --verbose

The script:
  1. Builds a structured prompt from the PR observation.
  2. Asks the model to respond ONLY with a JSON object matching
     CodeReviewAction's schema.
  3. Parses the response and feeds it to CodeReviewEnv.step().
  4. Prints per-task and aggregate scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the package is importable when run from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from codereviewenv import (
    CodeReviewAction,
    CodeReviewEnv,
    CodeReviewObservation,
    ReviewVerdict,
    TASKS,
)
from codereviewenv.models import InlineComment, Severity

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert software engineer performing a pull-request code review.
    You will be given a PR description and one or more file diffs.

    Your job:
    1. Identify ALL bugs, security issues, and correctness problems.
    2. For each bug found, add its canonical snake_case identifier to
       "identified_bugs" (e.g. "off_by_one_slice", "sql_injection").
    3. Write clear inline comments for each issue.
    4. Decide on a verdict: "approve", "request_changes", or "comment".
    5. Write a concise overall summary.

    Respond ONLY with a valid JSON object — no markdown fences, no preamble.
    The JSON must conform to this schema:
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
      ],
      "reasoning": "<optional chain-of-thought>"
    }
""")


def build_user_message(obs: CodeReviewObservation) -> str:
    lines: list[str] = []

    pr = obs.pr_meta
    lines += [
        f"## Pull Request: {pr.title}",
        f"**Author:** {pr.author}  |  **Branch:** `{pr.head_branch}` → `{pr.base_branch}`",
        f"**Labels:** {', '.join(pr.labels) or 'none'}",
        "",
        "### Description",
        pr.description,
        "",
        "### Changed Files",
    ]

    for f in obs.files:
        lines += [
            f"#### `{f.filename}` ({f.language})  +{f.additions} −{f.deletions}",
            "```diff",
            f.patch,
            "```",
            "",
        ]

    if obs.review_history:
        lines += ["### Review History (previous steps)"]
        for entry in obs.review_history:
            lines.append(f"Step {entry['step']}: score={entry['score']:.3f}")
        lines.append("")

    lines.append("Now provide your code review as a JSON object only.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

def call_openai(
    client: OpenAI,
    obs: CodeReviewObservation,
    model: str,
    temperature: float = 0.2,
) -> CodeReviewAction:
    """
    Call the OpenAI Chat Completions API and parse the response into
    a CodeReviewAction.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(obs)},
        ],
    )

    raw = response.choices[0].message.content or ""
    # Strip accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)

    # Coerce verdict to enum safely
    verdict_str = data.get("verdict", "comment")
    try:
        verdict = ReviewVerdict(verdict_str)
    except ValueError:
        verdict = ReviewVerdict.COMMENT

    inline_comments = []
    for ic in data.get("inline_comments", []):
        try:
            sev = Severity(ic.get("severity", "warning"))
        except ValueError:
            sev = Severity.WARNING
        inline_comments.append(
            InlineComment(
                filename=ic["filename"],
                line_number=max(1, int(ic.get("line_number", 1))),
                body=ic["body"],
                severity=sev,
            )
        )

    return CodeReviewAction(
        verdict=verdict,
        summary=data.get("summary", "No summary provided."),
        identified_bugs=data.get("identified_bugs", []),
        inline_comments=inline_comments,
        reasoning=data.get("reasoning"),
    )


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    client: OpenAI,
    model: str,
    verbose: bool = False,
) -> dict:
    env = CodeReviewEnv(task_id=task_id)
    obs = env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}  ({env.task.difficulty.value})")
        print(f"Max steps: {env.task.max_steps}")

    total_score = 0.0
    last_reward = None

    for step_idx in range(env.task.max_steps):
        if verbose:
            print(f"\n--- Step {step_idx + 1} / {env.task.max_steps} ---")

        try:
            action = call_openai(client, obs, model)
        except Exception as exc:
            print(f"  [ERROR] OpenAI call failed: {exc}")
            # Return a no-op action so the episode can continue
            action = CodeReviewAction(
                verdict=ReviewVerdict.COMMENT,
                summary="Error generating review.",
            )

        if verbose:
            print(f"  Verdict        : {action.verdict.value}")
            print(f"  Identified bugs: {action.identified_bugs}")
            if action.inline_comments:
                for ic in action.inline_comments:
                    print(f"  InlineComment  : {ic.filename}:{ic.line_number} [{ic.severity.value}]")

        obs, reward, done, _ = env.step(action)
        total_score += reward.score
        last_reward = reward

        if verbose:
            print(f"\n  {reward.feedback}")

        if done:
            break

    result = {
        "task_id": task_id,
        "difficulty": env.task.difficulty.value,
        "final_score": last_reward.score if last_reward else 0.0,
        "coverage_ratio": last_reward.coverage_ratio if last_reward else 0.0,
        "verdict_correct": last_reward.verdict_correct if last_reward else False,
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CodeReviewEnv baseline inference via OpenAI API."
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=list(TASKS.keys()) + [None],
        help="Run a single task (default: all tasks)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed step-level output",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    task_ids = [args.task] if args.task else list(TASKS.keys())

    print(f"\nCodeReviewEnv Baseline  |  model={args.model}  |  tasks={task_ids}")
    print("-" * 60)

    results = []
    for tid in task_ids:
        result = run_task(tid, client, args.model, verbose=args.verbose)
        results.append(result)
        score_bar = "█" * int(result["final_score"] * 20)
        print(
            f"  {tid:<30} "
            f"score={result['final_score']:.3f}  "
            f"coverage={result['coverage_ratio']:.2f}  "
            f"verdict_ok={result['verdict_correct']}  "
            f"|{score_bar}"
        )

    if len(results) > 1:
        avg_score = sum(r["final_score"] for r in results) / len(results)
        avg_coverage = sum(r["coverage_ratio"] for r in results) / len(results)
        verdict_acc = sum(1 for r in results if r["verdict_correct"]) / len(results)
        print("-" * 60)
        print(
            f"  {'AGGREGATE':<30} "
            f"score={avg_score:.3f}  "
            f"coverage={avg_coverage:.2f}  "
            f"verdict_acc={verdict_acc:.2f}"
        )

    print()


if __name__ == "__main__":
    main()
