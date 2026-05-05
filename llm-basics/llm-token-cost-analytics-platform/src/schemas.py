"""
Pydantic schemas for API request/response validation
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class LLMRequestCreate(BaseModel):
    """Schema for creating a new LLM request tracking entry"""
    model: str = Field(..., description="Model identifier (e.g., gpt-4, gpt-3.5-turbo)")
    input_tokens: int = Field(..., ge=0, description="Number of input tokens")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens")
    latency_ms: float = Field(..., ge=0, description="Request latency in milliseconds")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class LLMRequestResponse(BaseModel):
    """Schema for LLM request response"""
    id: int
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost: float
    timestamp: datetime
    user_id: Optional[str] = None
    
    class Config:
        from_attributes = True


class CostByModel(BaseModel):
    """Schema for cost breakdown by model"""
    model: str
    total_cost: float
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    avg_cost_per_request: float


class CostByDay(BaseModel):
    """Schema for cost breakdown by day"""
    date: str
    total_cost: float
    request_count: int
    total_tokens: int


class CostByUser(BaseModel):
    """Schema for cost breakdown by user"""
    user_id: str
    total_cost: float
    request_count: int
    total_tokens: int


class TokenStats(BaseModel):
    """Schema for token statistics"""
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    request_count: int


class LatencyStats(BaseModel):
    """Schema for latency statistics"""
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class ModelPerformance(BaseModel):
    """Schema for model performance metrics"""
    model: str
    request_count: int
    avg_latency_ms: float
    avg_cost: float
    avg_tokens: float
    total_cost: float


class DashboardSummary(BaseModel):
    """Schema for dashboard summary"""
    total_requests: int
    total_cost: float
    total_tokens: int
    avg_latency_ms: float
    models_used: List[str]
