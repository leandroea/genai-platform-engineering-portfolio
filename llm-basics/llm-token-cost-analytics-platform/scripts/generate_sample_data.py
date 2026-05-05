"""
Generate sample data for testing the LLM Token & Cost Analytics Platform
"""
import random
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import SessionLocal, init_db
from src.models import LLMRequest
from src.cost_calculator import calculate_cost


# Sample models to use
MODELS = [
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-3-sonnet",
    "claude-3-haiku",
    "llama-3-70b",
    "llama-3-8b",
    "mixtral-8x7b",
    "gemini-pro"
]

# Sample users
USERS = [
    "user_001",
    "user_002",
    "user_003",
    "user_004",
    "user_005",
    "user_admin",
    "user_demo"
]


def generate_sample_data(num_requests: int = 1000, days_back: int = 30):
    """Generate sample LLM request data"""
    print(f"Generating {num_requests} sample requests...")
    
    # Initialize database
    init_db()
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Clear existing data
        db.query(LLMRequest).delete()
        db.commit()
        print("Cleared existing data.")
        
        # Generate requests
        now = datetime.utcnow()
        
        for i in range(num_requests):
            # Random model
            model = random.choice(MODELS)
            
            # Random tokens
            input_tokens = random.randint(100, 5000)
            output_tokens = random.randint(50, 2000)
            total_tokens = input_tokens + output_tokens
            
            # Random latency (varies by model)
            base_latency = {
                "gpt-4": 3000,
                "gpt-3.5-turbo": 800,
                "claude-3-sonnet": 1500,
                "claude-3-haiku": 400,
                "llama-3-70b": 2500,
                "llama-3-8b": 600,
                "mixtral-8x7b": 1800,
                "gemini-pro": 1200
            }.get(model, 1000)
            
            latency_ms = random.gauss(base_latency, base_latency * 0.3)
            latency_ms = max(100, latency_ms)  # Minimum 100ms
            
            # Calculate cost
            cost = calculate_cost(model, input_tokens, output_tokens)
            
            # Random timestamp within the last N days
            days_ago = random.uniform(0, days_back)
            timestamp = now - timedelta(days=days_ago)
            
            # Random user
            user_id = random.choice(USERS) if random.random() > 0.1 else None
            
            # Create request
            request = LLMRequest(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost=cost,
                user_id=user_id,
                timestamp=timestamp
            )
            db.add(request)
            
            # Commit in batches
            if (i + 1) % 100 == 0:
                db.commit()
                print(f"  Generated {i + 1}/{num_requests} requests...")
        
        # Final commit
        db.commit()
        print(f"Successfully generated {num_requests} sample requests!")
        
        # Print summary
        print("\n--- Data Summary ---")
        total_cost = db.query(func.sum(LLMRequest.cost)).scalar()
        total_tokens = db.query(func.sum(LLMRequest.total_tokens)).scalar()
        total_requests = db.query(LLMRequest).count()
        
        print(f"Total requests: {total_requests}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Total tokens: {total_tokens:,}")
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    from sqlalchemy import func
    
    parser = argparse.ArgumentParser(description="Generate sample data")
    parser.add_argument("-n", "--num-requests", type=int, default=1000,
                        help="Number of requests to generate")
    parser.add_argument("-d", "--days", type=int, default=30,
                        help="Number of days to spread data over")
    
    args = parser.parse_args()
    
    generate_sample_data(args.num_requests, args.days)
