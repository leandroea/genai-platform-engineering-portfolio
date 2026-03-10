"""
Data models for Prompt Engineering Playground.
"""
from pydantic import BaseModel
from typing import Optional


class LLMParameters(BaseModel):
    """LLM parameters for API requests."""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class Experiment(BaseModel):
    """Experiment model for database storage."""
    id: int
    created_at: str
    prompt: str
    parameters: LLMParameters
    response: Optional[str] = None
    rating: Optional[int] = None
    feedback: Optional[str] = None
    experiment_type: str
    name: Optional[str] = None


class ComparisonResult(BaseModel):
    """Comparison result model for storing multiple prompt responses."""
    id: int
    experiment_id: int
    prompt_index: int
    response: str
    rating: Optional[int] = None
