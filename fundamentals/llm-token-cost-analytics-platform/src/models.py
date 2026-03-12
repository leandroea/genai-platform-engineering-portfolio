"""
Database models for LLM Token & Cost Analytics Platform
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class LLMRequest(Base):
    """Model for tracking LLM request metrics"""
    __tablename__ = "llm_requests"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String(255), nullable=False, index=True)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    latency_ms = Column(Float, nullable=False)
    cost = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    
    # Indexes for common query patterns
    __table_args__ = (
        Index('idx_model_timestamp', 'model', 'timestamp'),
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_date', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<LLMRequest(id={self.id}, model={self.model}, tokens={self.total_tokens}, cost=${self.cost:.4f})>"
