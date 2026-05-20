"""Pydantic models and structured output schemas for PaperBot."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


# ── LLM Structured Outputs ────────────────────────────────────────────────────

class IntentClassification(BaseModel):
    """
    Structured output for classify_intent() LLM call.

    .with_structured_output(IntentClassification) converts this
    Pydantic class to JSON Schema, sends it as a tool definition
    in the API payload, forces LLM to return structured args,
    Pydantic validates, returns a typed Python object.

    This is NOT prompt engineering.
    This is a transport-level contract between LLM and code.
    """
    intent: str = Field(
        description=(
            "The intent of the user question. "
            "Must be exactly one of: image, table, text, greeting, error. "
            "image = user wants to see a figure/diagram/formula/chart. "
            "table = user wants numerical data/metrics/benchmarks. "
            "text = user wants explanation/definition/summary. "
            "greeting = hi/bye/thanks/who are you. "
            "error = unrelated or incomprehensible."
        )
    )
    needs_table_augment: bool = False
    greeting_subtype: str = "greeting"

    @field_validator("intent", mode="before")
    @classmethod
    def validate_intent(cls, v):
        """
        Runs BEFORE Pydantic type check.
        Cleans LLM output. Prevents crashes on unexpected values.
        """
        if isinstance(v, str):
            v = v.lower().strip()
            if v in {"image", "table", "text", "greeting", "error"}:
                return v
        return "error"

    @field_validator("needs_table_augment", mode="before")
    @classmethod
    def clean_needs_table_augment(cls, v):
        if isinstance(v, bool):
            return v
        return False

    @field_validator("greeting_subtype", mode="before")
    @classmethod
    def clean_greeting_subtype(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            if v == "farewell":
                return "farewell"
        return "greeting"


class ContentSuggestions(BaseModel):
    """
    Structured output for _check_content_suggestions() LLM call.

    Decides if a relevant image or table exists for the question.
    Strict rules — does not say yes for everything.
    """
    has_image: bool = Field(
        description=(
            "True ONLY if question is about: architecture diagrams, "
            "system diagrams, CNN/neural network structure, electrode "
            "placement, confusion matrix, formulas, charts, plots, "
            "visual concepts that benefit from a diagram. "
            "False for pure conceptual explanations."
        )
    )
    has_table: bool = Field(
        description=(
            "True ONLY if question is about: accuracy scores, "
            "benchmark results, F1, precision, recall, latency, "
            "performance metrics, algorithm comparison, numerical results. "
            "False for pure conceptual explanations."
        )
    )

    @field_validator("has_image", "has_table", mode="before")
    @classmethod
    def validate_bool(cls, v):
        """
        Handles cases where LLM returns string 'true'/'false'
        instead of actual boolean. Prevents type errors.
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower().strip() == "true"
        return False


# ── API Request Models ────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """What the frontend sends to /ask."""
    question: str
    session_id: str


class ResumeRequest(BaseModel):
    """What the frontend sends to /resume."""
    session_id: str
    approved: bool


# ── API Sub-models ────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """A single retrieved source chunk shown as citation."""
    source: str
    page: int
    text_preview: str


# ── API Response Models ───────────────────────────────────────────────────────

class AskResponse(BaseModel):
    """What the backend always returns from /ask."""
    # Text answer
    answer: str = ""

    # Image fields
    needs_image: bool = False
    has_relevant_image: bool = False
    image_base64: str | None = None
    image_source: str = ""
    image_page: int = 0

    # Table fields
    has_relevant_table: bool = False

    # Source citations
    sources: List[SourceChunk] = []

    # Session
    interrupt_pending: bool = False
    session_id: str = ""


class ImageResponse(BaseModel):
    """What the backend returns from /resume."""
    image_path: str | None = None
    image_base64: str | None = None
    image_source: str = ""
    image_page: int = 0


class HealthResponse(BaseModel):
    """What the backend returns from /health."""
    status: str
