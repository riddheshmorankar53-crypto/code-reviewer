"""
Deterministic grader for CodeReviewEnv.

Scores an agent's CodeReviewAction against a task's seeded bug checklist.
The grader is fully deterministic given the same task seed and action.

Scoring formula
---------------
  bug_detection_score  = sum(weight_i  for each found bug_i)
  verdict_bonus        = 0.10  if verdict matches expected else 0.0
  coverage_ratio       = bugs_found / total_bugs
  final_score          = min(1.0, bug_detection_score + verdict_bonus)
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from .models import (
    BugChecklistResult,
    CodeReviewAction,
    CodeReviewReward,
)

if TYPE_CHECKING:
    from .tasks import Task


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade(task: "Task", action: CodeReviewAction, done: bool = False) -> CodeReviewReward:
    """
    Score *action* against *task*'s seeded bug checklist.

    Parameters
    ----------
    task:
        The Task whose bug_checklist and expected_verdict are used.
    action:
        The agent's CodeReviewAction.
    done:
        Whether this is the terminal step of the episode.

    Returns
    -------
    CodeReviewReward with a scalar `score` in [0.0, 1.0].
    """
    checklist_results: list[BugChecklistResult] = []
    agent_bugs = set(action.identified_bugs)          # already lowercased by validator

    total_weighted = 0.0
    earned_weighted = 0.0

    for bug_id, weight in task.bug_checklist.items():
        found = bug_id in agent_bugs
        points = weight if found else 0.0
        total_weighted += weight
        earned_weighted += points
        checklist_results.append(
            BugChecklistResult(
                bug_id=bug_id,
                found=found,
                weight=weight,
                points_earned=points,
            )
        )

    # Normalise (weights already sum to 1.0 by construction, but be safe)
    bug_detection_score = earned_weighted / total_weighted if total_weighted else 0.0

    # Verdict bonus
    verdict_correct = action.verdict == task.expected_verdict
    verdict_bonus = 0.10 if verdict_correct else 0.0

    # Coverage ratio
    bugs_found = sum(1 for r in checklist_results if r.found)
    coverage_ratio = bugs_found / len(checklist_results) if checklist_results else 0.0

    final_score = min(1.0, bug_detection_score + verdict_bonus)

    # Build human-readable feedback
    found_ids = [r.bug_id for r in checklist_results if r.found]
    missed_ids = [r.bug_id for r in checklist_results if not r.found]

    feedback_lines = [
        f"Task: {task.task_id}  |  Difficulty: {task.difficulty.value}",
        f"Score: {final_score:.3f}  (bug_detection={bug_detection_score:.3f}, "
        f"verdict_bonus={verdict_bonus:.2f})",
        f"Coverage: {bugs_found}/{len(checklist_results)} bugs found "
        f"({coverage_ratio*100:.0f}%)",
    ]
    if found_ids:
        feedback_lines.append("✓ Found:  " + ", ".join(found_ids))
    if missed_ids:
        feedback_lines.append("✗ Missed: " + ", ".join(missed_ids))
    if not verdict_correct:
        feedback_lines.append(
            f"✗ Verdict '{action.verdict.value}' does not match "
            f"expected '{task.expected_verdict.value}'"
        )

    return CodeReviewReward(
        score=final_score,
        bug_detection_score=bug_detection_score,
        verdict_correct=verdict_correct,
        coverage_ratio=coverage_ratio,
        checklist_results=checklist_results,
        feedback="\n".join(feedback_lines),
        done=done,
        info={
            "task_id": task.task_id,
            "difficulty": task.difficulty.value,
            "bugs_found": bugs_found,
            "bugs_total": len(task.bug_checklist),
            "agent_identified_bugs": list(agent_bugs),
        },
    )
