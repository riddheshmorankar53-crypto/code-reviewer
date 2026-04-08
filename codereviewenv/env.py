"""
CodeReviewEnv — OpenEnv-compatible environment for pull-request code review.

Public interface
----------------
    env = CodeReviewEnv(task_id="easy_list_utils")
    obs = env.reset()
    next_obs, reward, done, info = env.step(action)
    state = env.state()
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from .grader import grade
from .models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewReward,
)
from .tasks import Task, get_task, TASKS


class CodeReviewEnv:
    """
    OpenEnv environment for pull-request code review.

    Parameters
    ----------
    task_id:
        One of the registered task IDs (see `codereviewenv.tasks.TASKS`).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, task_id: str = "easy_list_utils") -> None:
        self._task: Task = get_task(task_id)
        self._obs: Optional[CodeReviewObservation] = None
        self._last_reward: Optional[CodeReviewReward] = None
        self._done: bool = False
        self._step_count: int = 0
        self.last_action_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Core OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> CodeReviewObservation:
        """
        Reset the environment to its initial state and return the first
        observation.

        Returns
        -------
        CodeReviewObservation
            The PR context the agent should review.
        """
        self._obs = self._task.initial_observation()
        self._last_reward = None
        self._done = False
        self._step_count = 0
        self.last_action_error = None
        return copy.deepcopy(self._obs)

    def step(
        self, action: CodeReviewAction
    ) -> tuple[CodeReviewObservation, CodeReviewReward, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        action:
            The agent's CodeReviewAction for this step.

        Returns
        -------
        tuple[CodeReviewObservation, CodeReviewReward, bool, Dict[str, Any]]
            The next observation, scalar reward plus diagnostics, done flag,
            and info dict.

        Raises
        ------
        RuntimeError
            If `reset()` has not been called first or the episode is over.
        """
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError(
                "Episode is finished. Call reset() to start a new episode."
            )

        self.last_action_error = None
        self._step_count += 1
        done = self._step_count >= self._task.max_steps

        reward = grade(task=self._task, action=action, done=done)

        # Append this turn to review history so multi-turn tasks can refer back
        self._obs.review_history.append(
            {
                "step": self._step_count,
                "action": action.model_dump(exclude={"reasoning"}),
                "score": reward.score,
            }
        )
        self._obs.step_number = self._step_count

        self._last_reward = reward
        self._done = done
        next_obs = copy.deepcopy(self._obs)
        info = {
            **reward.info,
            "last_action_error": self.last_action_error,
        }
        return next_obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """
        Return a snapshot of the current environment state.

        Useful for serialisation, logging, and debugging.
        """
        return {
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty.value,
            "step": self._step_count,
            "max_steps": self._task.max_steps,
            "done": self._done,
            "last_score": self._last_reward.score if self._last_reward else None,
            "last_action_error": self.last_action_error,
            "observation": self._obs.model_dump() if self._obs else None,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def current_observation(self) -> Optional[CodeReviewObservation]:
        return copy.deepcopy(self._obs) if self._obs else None

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def task(self) -> Task:
        return self._task

    def close(self) -> None:
        """Release episode state."""
        self._obs = None
        self._last_reward = None
        self._done = False
        self._step_count = 0
        self.last_action_error = None

    # ------------------------------------------------------------------
    # Class-level utilities
    # ------------------------------------------------------------------

    @classmethod
    def available_tasks(cls) -> Dict[str, str]:
        """Return a mapping of task_id -> difficulty for all registered tasks."""
        return {tid: t.difficulty.value for tid, t in TASKS.items()}

    def __repr__(self) -> str:
        return (
            f"CodeReviewEnv(task={self._task.task_id!r}, "
            f"step={self._step_count}/{self._task.max_steps}, "
            f"done={self._done})"
        )
