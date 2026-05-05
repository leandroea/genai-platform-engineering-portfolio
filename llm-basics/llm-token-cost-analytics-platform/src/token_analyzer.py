"""
Token analysis module for LLM requests
Provides statistical analysis of token usage patterns
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import LLMRequest
from src.schemas import TokenStats, LatencyStats, ModelPerformance


class TokenAnalyzer:
    """Analyzer for token usage patterns"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_overall_stats(self) -> TokenStats:
        """Get overall token statistics"""
        result = self.db.query(
            func.sum(LLMRequest.input_tokens).label("total_input"),
            func.sum(LLMRequest.output_tokens).label("total_output"),
            func.sum(LLMRequest.total_tokens).label("total_tokens"),
            func.avg(LLMRequest.input_tokens).label("avg_input"),
            func.avg(LLMRequest.output_tokens).label("avg_output"),
            func.avg(LLMRequest.total_tokens).label("avg_total"),
            func.count(LLMRequest.id).label("count")
        ).first()
        
        return TokenStats(
            total_input_tokens=int(result.total_input or 0),
            total_output_tokens=int(result.total_output or 0),
            total_tokens=int(result.total_tokens or 0),
            avg_input_tokens=float(result.avg_input or 0),
            avg_output_tokens=float(result.avg_output or 0),
            avg_total_tokens=float(result.avg_total or 0),
            request_count=int(result.count or 0)
        )
    
    def get_stats_by_model(self, model: str) -> TokenStats:
        """Get token statistics for a specific model"""
        result = self.db.query(
            func.sum(LLMRequest.input_tokens).label("total_input"),
            func.sum(LLMRequest.output_tokens).label("total_output"),
            func.sum(LLMRequest.total_tokens).label("total_tokens"),
            func.avg(LLMRequest.input_tokens).label("avg_input"),
            func.avg(LLMRequest.output_tokens).label("avg_output"),
            func.avg(LLMRequest.total_tokens).label("avg_total"),
            func.count(LLMRequest.id).label("count")
        ).filter(LLMRequest.model == model).first()
        
        return TokenStats(
            total_input_tokens=int(result.total_input or 0),
            total_output_tokens=int(result.total_output or 0),
            total_tokens=int(result.total_tokens or 0),
            avg_input_tokens=float(result.avg_input or 0),
            avg_output_tokens=float(result.avg_output or 0),
            avg_total_tokens=float(result.avg_total or 0),
            request_count=int(result.count or 0)
        )
    
    def get_stats_by_user(self, user_id: str) -> TokenStats:
        """Get token statistics for a specific user"""
        result = self.db.query(
            func.sum(LLMRequest.input_tokens).label("total_input"),
            func.sum(LLMRequest.output_tokens).label("total_output"),
            func.sum(LLMRequest.total_tokens).label("total_tokens"),
            func.avg(LLMRequest.input_tokens).label("avg_input"),
            func.avg(LLMRequest.output_tokens).label("avg_output"),
            func.avg(LLMRequest.total_tokens).label("avg_total"),
            func.count(LLMRequest.id).label("count")
        ).filter(LLMRequest.user_id == user_id).first()
        
        return TokenStats(
            total_input_tokens=int(result.total_input or 0),
            total_output_tokens=int(result.total_output or 0),
            total_tokens=int(result.total_tokens or 0),
            avg_input_tokens=float(result.avg_input or 0),
            avg_output_tokens=float(result.avg_output or 0),
            avg_total_tokens=float(result.avg_total or 0),
            request_count=int(result.count or 0)
        )
    
    def get_stats_by_timerange(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> TokenStats:
        """Get token statistics for a time range"""
        result = self.db.query(
            func.sum(LLMRequest.input_tokens).label("total_input"),
            func.sum(LLMRequest.output_tokens).label("total_output"),
            func.sum(LLMRequest.total_tokens).label("total_tokens"),
            func.avg(LLMRequest.input_tokens).label("avg_input"),
            func.avg(LLMRequest.output_tokens).label("avg_output"),
            func.avg(LLMRequest.total_tokens).label("avg_total"),
            func.count(LLMRequest.id).label("count")
        ).filter(
            LLMRequest.timestamp >= start_time,
            LLMRequest.timestamp <= end_time
        ).first()
        
        return TokenStats(
            total_input_tokens=int(result.total_input or 0),
            total_output_tokens=int(result.total_output or 0),
            total_tokens=int(result.total_tokens or 0),
            avg_input_tokens=float(result.avg_input or 0),
            avg_output_tokens=float(result.avg_output or 0),
            avg_total_tokens=float(result.avg_total or 0),
            request_count=int(result.count or 0)
        )
    
    def get_token_distribution(self, model: Optional[str] = None) -> Dict:
        """Get token distribution (percentiles)"""
        query = self.db.query(LLMRequest.total_tokens)
        
        if model:
            query = query.filter(LLMRequest.model == model)
        
        tokens = [r[0] for r in query.all()]
        
        if not tokens:
            return {"p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}
        
        tokens.sort()
        n = len(tokens)
        
        def percentile(p):
            idx = int(n * p)
            if idx >= n:
                idx = n - 1
            return tokens[idx]
        
        return {
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "min": tokens[0],
            "max": tokens[-1],
            "mean": sum(tokens) / n
        }
    
    def get_requests_per_minute(self, minutes: int = 60) -> List[Dict]:
        """Get request count per minute for the last N minutes"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=minutes)
        
        results = self.db.query(
            func.date_trunc('minute', LLMRequest.timestamp).label('minute'),
            func.count(LLMRequest.id).label('count')
        ).filter(
            LLMRequest.timestamp >= start_time,
            LLMRequest.timestamp <= end_time
        ).group_by('minute').order_by('minute').all()
        
        return [
            {"timestamp": r.minute.isoformat(), "count": r.count}
            for r in results
        ]


class LatencyAnalyzer:
    """Analyzer for latency patterns"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_overall_latency_stats(self) -> LatencyStats:
        """Get overall latency statistics"""
        result = self.db.query(
            func.avg(LLMRequest.latency_ms).label("avg"),
            func.min(LLMRequest.latency_ms).label("min"),
            func.max(LLMRequest.latency_ms).label("max")
        ).first()
        
        latencies = [r[0] for r in self.db.query(LLMRequest.latency_ms).all()]
        
        if not latencies:
            return LatencyStats(
                avg_latency_ms=0, min_latency_ms=0, max_latency_ms=0,
                p50_latency_ms=0, p95_latency_ms=0, p99_latency_ms=0
            )
        
        latencies.sort()
        n = len(latencies)
        
        def percentile(p):
            idx = int(n * p)
            if idx >= n:
                idx = n - 1
            return latencies[idx]
        
        return LatencyStats(
            avg_latency_ms=float(result.avg or 0),
            min_latency_ms=float(result.min or 0),
            max_latency_ms=float(result.max or 0),
            p50_latency_ms=percentile(0.50),
            p95_latency_ms=percentile(0.95),
            p99_latency_ms=percentile(0.99)
        )
    
    def get_latency_by_model(self, model: str) -> LatencyStats:
        """Get latency statistics for a specific model"""
        result = self.db.query(
            func.avg(LLMRequest.latency_ms).label("avg"),
            func.min(LLMRequest.latency_ms).label("min"),
            func.max(LLMRequest.latency_ms).label("max")
        ).filter(LLMRequest.model == model).first()
        
        latencies = [r[0] for r in 
                     self.db.query(LLMRequest.latency_ms)
                     .filter(LLMRequest.model == model).all()]
        
        if not latencies:
            return LatencyStats(
                avg_latency_ms=0, min_latency_ms=0, max_latency_ms=0,
                p50_latency_ms=0, p95_latency_ms=0, p99_latency_ms=0
            )
        
        latencies.sort()
        n = len(latencies)
        
        def percentile(p):
            idx = int(n * p)
            if idx >= n:
                idx = n - 1
            return latencies[idx]
        
        return LatencyStats(
            avg_latency_ms=float(result.avg or 0),
            min_latency_ms=float(result.min or 0),
            max_latency_ms=float(result.max or 0),
            p50_latency_ms=percentile(0.50),
            p95_latency_ms=percentile(0.95),
            p99_latency_ms=percentile(0.99)
        )


class ModelPerformanceAnalyzer:
    """Analyzer for model performance metrics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_all_models_performance(self) -> List[ModelPerformance]:
        """Get performance metrics for all models"""
        results = self.db.query(
            LLMRequest.model,
            func.count(LLMRequest.id).label("count"),
            func.avg(LLMRequest.latency_ms).label("avg_latency"),
            func.avg(LLMRequest.cost).label("avg_cost"),
            func.avg(LLMRequest.total_tokens).label("avg_tokens"),
            func.sum(LLMRequest.cost).label("total_cost")
        ).group_by(LLMRequest.model).all()
        
        return [
            ModelPerformance(
                model=r.model,
                request_count=r.count,
                avg_latency_ms=float(r.avg_latency or 0),
                avg_cost=float(r.avg_cost or 0),
                avg_tokens=float(r.avg_tokens or 0),
                total_cost=float(r.total_cost or 0)
            )
            for r in results
        ]
