"""
Tests for CodeReviewEnv.

Run with:  pytest tests/ -v
"""

import pytest

from codereviewenv import (
    CodeReviewAction,
    CodeReviewEnv,
    CodeReviewObservation,
    CodeReviewReward,
    ReviewVerdict,
    TASKS,
)
from codereviewenv.grader import grade
from codereviewenv.models import InlineComment, Severity
from codereviewenv.tasks import get_task


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_action_lowercases_bug_ids(self):
        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Some issue found.",
            identified_bugs=["  Off_By_One  ", "SQL_INJECTION"],
        )
        assert action.identified_bugs == ["off_by_one", "sql_injection"]

    def test_action_requires_non_empty_summary(self):
        with pytest.raises(Exception):
            CodeReviewAction(
                verdict=ReviewVerdict.APPROVE,
                summary="ok",  # too short (< 10 chars)
            )

    def test_inline_comment_line_ge_1(self):
        with pytest.raises(Exception):
            InlineComment(filename="a.py", line_number=0, body="bad")

    def test_reward_score_range(self):
        with pytest.raises(Exception):
            CodeReviewReward(
                score=1.5,  # > 1.0 — invalid
                bug_detection_score=1.0,
                verdict_correct=True,
                coverage_ratio=1.0,
            )


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGrader:
    def setup_method(self):
        self.task = get_task("easy_list_utils")

    def test_perfect_score(self):
        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Found off-by-one and missing zero-division guard.",
            identified_bugs=list(self.task.bug_checklist.keys()),
        )
        reward = grade(self.task, action, done=True)
        # All bugs found + correct verdict = 1.0 + 0.10 capped at 1.0
        assert reward.score == pytest.approx(1.0, abs=1e-4)
        assert reward.coverage_ratio == pytest.approx(1.0)
        assert reward.verdict_correct is True

    def test_zero_score_no_bugs(self):
        action = CodeReviewAction(
            verdict=ReviewVerdict.APPROVE,
            summary="Looks good to me, no issues.",
            identified_bugs=[],
        )
        reward = grade(self.task, action, done=True)
        assert reward.score == pytest.approx(0.0)
        assert reward.coverage_ratio == pytest.approx(0.0)
        assert reward.verdict_correct is False

    def test_partial_score(self):
        # Only find the heaviest bug (off_by_one_slice, weight=0.45)
        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Found an off-by-one error in the slice.",
            identified_bugs=["off_by_one_slice"],
        )
        reward = grade(self.task, action)
        # 0.45 bug + 0.10 verdict bonus = 0.55
        assert reward.score == pytest.approx(0.55, abs=1e-4)
        assert reward.bug_detection_score == pytest.approx(0.45, abs=1e-4)

    def test_unknown_bugs_ignored(self):
        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Found issues in the code. Needs fixes.",
            identified_bugs=["nonexistent_bug_xyz"],
        )
        reward = grade(self.task, action)
        assert reward.bug_detection_score == pytest.approx(0.0)

    def test_checklist_results_populated(self):
        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Found off-by-one slice error.",
            identified_bugs=["off_by_one_slice"],
        )
        reward = grade(self.task, action)
        assert len(reward.checklist_results) == len(self.task.bug_checklist)
        found = [r for r in reward.checklist_results if r.found]
        assert len(found) == 1
        assert found[0].bug_id == "off_by_one_slice"


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_available_tasks(self):
        tasks = CodeReviewEnv.available_tasks()
        assert "easy_list_utils" in tasks
        assert "medium_flask_api" in tasks
        assert "hard_async_pipeline" in tasks

    def test_reset_returns_observation(self):
        env = CodeReviewEnv(task_id="easy_list_utils")
        obs = env.reset()
        assert isinstance(obs, CodeReviewObservation)
        assert obs.step_number == 0
        assert len(obs.files) >= 1

    def test_step_without_reset_raises(self):
        env = CodeReviewEnv(task_id="easy_list_utils")
        action = CodeReviewAction(
            verdict=ReviewVerdict.COMMENT,
            summary="No reset called before step.",
        )
        with pytest.raises(RuntimeError, match="reset"):
            env.step(action)

    def test_step_after_done_raises(self):
        env = CodeReviewEnv(task_id="easy_list_utils")
        env.reset()
        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Terminal step reached; episode done.",
        )
        env.step(action)          # easy task has max_steps=1 → done
        assert env.is_done
        with pytest.raises(RuntimeError, match="finished"):
            env.step(action)

    def test_full_easy_episode(self):
        env = CodeReviewEnv(task_id="easy_list_utils")
        obs = env.reset()
        assert obs.task_id == "easy_list_utils"

        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Found off-by-one and zero-division issues.",
            identified_bugs=["off_by_one_slice", "missing_zero_division_guard"],
        )
        _, reward, done, info = env.step(action)
        assert isinstance(reward, CodeReviewReward)
        assert 0.0 <= reward.score <= 1.0
        assert done is True
        assert reward.done is True
        assert info["task_id"] == "easy_list_utils"
        assert env.is_done

    def test_multi_step_medium_episode(self):
        env = CodeReviewEnv(task_id="medium_flask_api")
        obs = env.reset()
        assert env.task.max_steps == 2

        for step in range(2):
            action = CodeReviewAction(
                verdict=ReviewVerdict.REQUEST_CHANGES,
                summary="SQL injection and missing auth found in this PR.",
                identified_bugs=["sql_injection", "missing_authentication"],
            )
            _, reward, done, _ = env.step(action)
            if step < 1:
                assert not done
                assert not reward.done
            else:
                assert done
                assert reward.done

    def test_state_snapshot(self):
        env = CodeReviewEnv(task_id="easy_list_utils")
        env.reset()
        state = env.state()
        assert state["task_id"] == "easy_list_utils"
        assert state["step"] == 0
        assert state["done"] is False
        assert state["last_score"] is None

    def test_reset_clears_history(self):
        env = CodeReviewEnv(task_id="easy_list_utils")
        env.reset()
        action = CodeReviewAction(
            verdict=ReviewVerdict.REQUEST_CHANGES,
            summary="Some review comment here.",
        )
        env.step(action)
        obs2 = env.reset()
        assert obs2.step_number == 0
        assert obs2.review_history == []

    def test_hard_task_all_bugs(self):
        env = CodeReviewEnv(task_id="hard_async_pipeline")
        env.reset()
        all_bugs = list(env.task.bug_checklist.keys())

        for _ in range(env.task.max_steps):
            action = CodeReviewAction(
                verdict=ReviewVerdict.REQUEST_CHANGES,
                summary="Multiple concurrency and memory issues found.",
                identified_bugs=all_bugs,
            )
            _, reward, _, _ = env.step(action)
            if env.is_done:
                break

        assert reward.score == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("task_id", list(TASKS.keys()))
    def test_each_task_runs_to_completion(self, task_id):
        env = CodeReviewEnv(task_id=task_id)
        obs = env.reset()
        for _ in range(env.task.max_steps):
            action = CodeReviewAction(
                verdict=ReviewVerdict.REQUEST_CHANGES,
                summary="Reviewing this PR carefully for issues.",
            )
            _, reward, _, _ = env.step(action)
            if env.is_done:
                break
        assert env.is_done
        assert 0.0 <= reward.score <= 1.0

    def test_invalid_task_id(self):
        with pytest.raises(KeyError):
            CodeReviewEnv(task_id="nonexistent_task")
