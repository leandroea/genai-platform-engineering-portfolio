"""
LLM Token & Cost Analytics Platform - Main Application
FastAPI server for tracking and analyzing LLM usage
"""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.database import get_db, init_db
from src.models import LLMRequest
from src.schemas import (
    LLMRequestCreate, LLMRequestResponse, CostByModel, CostByDay,
    CostByUser, TokenStats, LatencyStats, DashboardSummary, ModelPerformance
)
from src.cost_calculator import calculate_cost
from src.token_analyzer import TokenAnalyzer, LatencyAnalyzer, ModelPerformanceAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="LLM Token & Cost Analytics Platform",
    description="Real-time analytics for tracking token usage, latency, and cost across multiple LLM providers",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    """Initialize database on startup"""
    init_db()


# ============== Tracking Endpoints ==============

@app.post("/track", response_model=LLMRequestResponse, status_code=201)
def track_request(request: LLMRequestCreate, db: Session = Depends(get_db)):
    """
    Track a new LLM request
    
    Example request:
    ```json
    {
        "model": "gpt-4",
        "input_tokens": 1200,
        "output_tokens": 400,
        "latency_ms": 2100,
        "user_id": "user123"
    }
    ```
    """
    # Calculate cost
    cost = calculate_cost(
        request.model,
        request.input_tokens,
        request.output_tokens
    )
    
    # Calculate total tokens
    total_tokens = request.input_tokens + request.output_tokens
    
    # Create database record
    db_request = LLMRequest(
        model=request.model,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        total_tokens=total_tokens,
        latency_ms=request.latency_ms,
        cost=cost,
        user_id=request.user_id,
        timestamp=datetime.utcnow()
    )
    
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    
    return db_request


@app.get("/requests", response_model=List[LLMRequestResponse])
def list_requests(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model: Optional[str] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List tracked LLM requests with optional filtering"""
    query = db.query(LLMRequest)
    
    if model:
        query = query.filter(LLMRequest.model == model)
    if user_id:
        query = query.filter(LLMRequest.user_id == user_id)
    
    return query.order_by(desc(LLMRequest.timestamp)).offset(skip).limit(limit).all()


# ============== Analytics Endpoints ==============

@app.get("/analytics/summary", response_model=DashboardSummary)
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get summary statistics for dashboard"""
    result = db.query(
        func.count(LLMRequest.id).label("total_requests"),
        func.sum(LLMRequest.cost).label("total_cost"),
        func.sum(LLMRequest.total_tokens).label("total_tokens"),
        func.avg(LLMRequest.latency_ms).label("avg_latency"),
        func.group_concat(LLMRequest.model).label("models")
    ).first()
    
    # Get unique models
    models = db.query(LLMRequest.model).distinct().all()
    models_used = [m[0] for m in models]
    
    return DashboardSummary(
        total_requests=int(result.total_requests or 0),
        total_cost=float(result.total_cost or 0),
        total_tokens=int(result.total_tokens or 0),
        avg_latency_ms=float(result.avg_latency or 0),
        models_used=models_used
    )


@app.get("/analytics/cost/by-model", response_model=List[CostByModel])
def get_cost_by_model(db: Session = Depends(get_db)):
    """Get cost breakdown by model"""
    results = db.query(
        LLMRequest.model,
        func.sum(LLMRequest.cost).label("total_cost"),
        func.count(LLMRequest.id).label("request_count"),
        func.sum(LLMRequest.input_tokens).label("total_input_tokens"),
        func.sum(LLMRequest.output_tokens).label("total_output_tokens"),
        func.avg(LLMRequest.cost).label("avg_cost_per_request")
    ).group_by(LLMRequest.model).all()
    
    return [
        CostByModel(
            model=r.model,
            total_cost=float(r.total_cost or 0),
            request_count=r.request_count,
            total_input_tokens=int(r.total_input_tokens or 0),
            total_output_tokens=int(r.total_output_tokens or 0),
            avg_cost_per_request=float(r.avg_cost_per_request or 0)
        )
        for r in results
    ]


@app.get("/analytics/cost/by-day", response_model=List[CostByDay])
def get_cost_by_day(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get cost breakdown by day"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    results = db.query(
        func.date(LLMRequest.timestamp).label("date"),
        func.sum(LLMRequest.cost).label("total_cost"),
        func.count(LLMRequest.id).label("request_count"),
        func.sum(LLMRequest.total_tokens).label("total_tokens")
    ).filter(
        LLMRequest.timestamp >= start_date
    ).group_by("date").order_by("date").all()
    
    return [
        CostByDay(
            date=str(r.date),
            total_cost=float(r.total_cost or 0),
            request_count=r.request_count,
            total_tokens=int(r.total_tokens or 0)
        )
        for r in results
    ]


@app.get("/analytics/cost/by-user", response_model=List[CostByUser])
def get_cost_by_user(db: Session = Depends(get_db)):
    """Get cost breakdown by user"""
    results = db.query(
        LLMRequest.user_id,
        func.sum(LLMRequest.cost).label("total_cost"),
        func.count(LLMRequest.id).label("request_count"),
        func.sum(LLMRequest.total_tokens).label("total_tokens")
    ).filter(
        LLMRequest.user_id.isnot(None)
    ).group_by(LLMRequest.user_id).all()
    
    return [
        CostByUser(
            user_id=r.user_id or "anonymous",
            total_cost=float(r.total_cost or 0),
            request_count=r.request_count,
            total_tokens=int(r.total_tokens or 0)
        )
        for r in results
    ]


@app.get("/analytics/tokens", response_model=TokenStats)
def get_token_stats(db: Session = Depends(get_db)):
    """Get overall token statistics"""
    analyzer = TokenAnalyzer(db)
    return analyzer.get_overall_stats()


@app.get("/analytics/tokens/{model}", response_model=TokenStats)
def get_token_stats_by_model(model: str, db: Session = Depends(get_db)):
    """Get token statistics for a specific model"""
    analyzer = TokenAnalyzer(db)
    return analyzer.get_stats_by_model(model)


@app.get("/analytics/latency", response_model=LatencyStats)
def get_latency_stats(db: Session = Depends(get_db)):
    """Get overall latency statistics"""
    analyzer = LatencyAnalyzer(db)
    return analyzer.get_overall_latency_stats()


@app.get("/analytics/latency/{model}", response_model=LatencyStats)
def get_latency_stats_by_model(model: str, db: Session = Depends(get_db)):
    """Get latency statistics for a specific model"""
    analyzer = LatencyAnalyzer(db)
    return analyzer.get_latency_by_model(model)


@app.get("/analytics/models/performance", response_model=List[ModelPerformance])
def get_models_performance(db: Session = Depends(get_db)):
    """Get performance metrics for all models"""
    analyzer = ModelPerformanceAnalyzer(db)
    return analyzer.get_all_models_performance()


# ============== Utility Endpoints ==============

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "name": "LLM Token & Cost Analytics Platform",
        "version": "1.0.0",
        "endpoints": {
            "track": "POST /track - Track a new LLM request",
            "requests": "GET /requests - List tracked requests",
            "analytics_summary": "GET /analytics/summary - Dashboard summary",
            "cost_by_model": "GET /analytics/cost/by-model - Cost by model",
            "cost_by_day": "GET /analytics/cost/by-day - Cost by day",
            "cost_by_user": "GET /analytics/cost/by-user - Cost by user",
            "token_stats": "GET /analytics/tokens - Token statistics",
            "latency_stats": "GET /analytics/latency - Latency statistics",
            "model_performance": "GET /analytics/models/performance - Model performance"
        }
    }
