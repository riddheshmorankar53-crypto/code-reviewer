"""
CodeReviewEnv — OpenEnv environment for pull-request code review.

Quick start
-----------
    from codereviewenv import CodeReviewEnv, CodeReviewAction, ReviewVerdict

    env = CodeReviewEnv(task_id="easy_list_utils")
    obs = env.reset()

    action = CodeReviewAction(
        verdict=ReviewVerdict.REQUEST_CHANGES,
        summary="Found an off-by-one error in get_last_n.",
        identified_bugs=["off_by_one_slice", "missing_zero_division_guard"],
    )
    reward = env.step(action)
    print(reward.score)   # 0.85 + 0.1 verdict bonus
"""

from .env import CodeReviewEnv
from .models import (
    BugChecklistResult,
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewReward,
    DifficultyLevel,
    FileDiff,
    InlineComment,
    PullRequestMeta,
    ReviewVerdict,
    Severity,
)
from .tasks import TASKS, Task, get_task

__all__ = [
    # Environment
    "CodeReviewEnv",
    # Models
    "CodeReviewObservation",
    "CodeReviewAction",
    "CodeReviewReward",
    "BugChecklistResult",
    "FileDiff",
    "InlineComment",
    "PullRequestMeta",
    "ReviewVerdict",
    "Severity",
    "DifficultyLevel",
    # Tasks
    "Task",
    "TASKS",
    "get_task",
]

__version__ = "0.1.0"
