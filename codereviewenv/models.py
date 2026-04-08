"""
Pydantic models for CodeReviewEnv.
Defines the observation, action, and reward data structures used
throughout the environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    COMMENT = "comment"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class FileDiff(BaseModel):
    """A single file changed in the pull request."""
    filename: str = Field(..., description="Relative path to the file")
    language: str = Field(..., description="Programming language (e.g. 'python')")
    patch: str = Field(..., description="Unified diff patch text")
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)


class PullRequestMeta(BaseModel):
    """Metadata about the pull request under review."""
    pr_id: str
    title: str
    description: str
    author: str
    base_branch: str = "main"
    head_branch: str = "feature"
    labels: List[str] = Field(default_factory=list)


class CodeReviewObservation(BaseModel):
    """
    The full observation handed to the agent at each step.

    Contains the PR metadata, all changed files, and the conversation
    history of review comments so far (for multi-turn tasks).
    """
    task_id: str
    difficulty: DifficultyLevel
    pr_meta: PullRequestMeta
    files: List[FileDiff] = Field(..., min_length=1)
    review_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Previous (role, content) exchanges in this episode",
    )
    step_number: int = Field(default=0, ge=0)
    max_steps: int = Field(default=3, ge=1)

    @property
    def is_done(self) -> bool:
        return self.step_number >= self.max_steps


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class InlineComment(BaseModel):
    """A comment attached to a specific line in a file."""
    filename: str
    line_number: int = Field(ge=1)
    body: str = Field(..., min_length=1)
    severity: Severity = Severity.WARNING


class CodeReviewAction(BaseModel):
    """
    The action produced by the agent (or baseline) at each step.

    A complete review action consists of:
    - A high-level verdict
    - A summary comment
    - Zero or more inline comments on specific lines
    - An optional list of identified bug IDs (used by the grader)
    """
    verdict: ReviewVerdict
    summary: str = Field(
        ...,
        min_length=10,
        description="Overall review summary written in natural language",
    )
    inline_comments: List[InlineComment] = Field(default_factory=list)
    identified_bugs: List[str] = Field(
        default_factory=list,
        description=(
            "Canonical bug IDs the agent claims to have found, e.g. "
            "['off_by_one', 'sql_injection']. Used for deterministic scoring."
        ),
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought reasoning (not scored)",
    )

    @field_validator("identified_bugs", mode="before")
    @classmethod
    def lowercase_bug_ids(cls, v: List[str]) -> List[str]:
        return [b.strip().lower() for b in v]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class BugChecklistResult(BaseModel):
    """Per-bug scoring detail."""
    bug_id: str
    found: bool
    weight: float = Field(ge=0.0, le=1.0)
    points_earned: float = Field(ge=0.0, le=1.0)


class CodeReviewReward(BaseModel):
    """
    Structured reward returned by the environment after each step.

    The scalar `score` is the primary RL signal (0.0 – 1.0).
    Additional sub-scores are provided for interpretability.
    """
    score: float = Field(ge=0.0, le=1.0, description="Primary reward signal")
    bug_detection_score: float = Field(ge=0.0, le=1.0)
    verdict_correct: bool
    coverage_ratio: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of seeded bugs found by the agent",
    )
    checklist_results: List[BugChecklistResult] = Field(default_factory=list)
    feedback: str = Field(default="", description="Human-readable grader feedback")
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
