"""
Cost calculation module for LLM requests
Uses model pricing to calculate cost per request
"""
from typing import Dict, Optional
import requests


# Default model pricing (per 1M tokens)
# These are common HuggingFace and OpenAI compatible models
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4": {"input": 10.0, "output": 30.0},  # $10/1M input, $30/1M output
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4-32k": {"input": 60.0, "output": 120.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
    
    # Anthropic models
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-2.1": {"input": 8.0, "output": 24.0},
    "claude-2": {"input": 8.0, "output": 24.0},
    "claude-instant": {"input": 1.63, "output": 5.51},
    
    # Google models
    "gemini-pro": {"input": 1.25, "output": 5.0},
    "gemini-pro-vision": {"input": 1.25, "output": 5.0},
    "palm-2": {"input": 1.25, "output": 5.0},
    
    # Meta/Llama models (HuggingFace)
    "llama-2-70b": {"input": 0.9, "output": 0.9},
    "llama-2-13b": {"input": 0.24, "output": 0.24},
    "llama-2-7b": {"input": 0.2, "output": 0.2},
    "llama-3-70b": {"input": 0.9, "output": 0.9},
    "llama-3-8b": {"input": 0.2, "output": 0.2},
    "llama-3.1-405b": {"input": 3.5, "output": 3.5},
    "llama-3.1-70b": {"input": 0.9, "output": 0.9},
    "llama-3.1-8b": {"input": 0.2, "output": 0.2},
    
    # Mistral models
    "mistral-7b": {"input": 0.24, "output": 0.24},
    "mixtral-8x7b": {"input": 0.24, "output": 0.24},
    "mixtral-8x22b": {"input": 1.2, "output": 1.2},
    
    # Cohere models
    "command-r": {"input": 0.5, "output": 1.5},
    "command-r-plus": {"input": 3.0, "output": 15.0},
    
    # AI21 models
    "j2-ultra": {"input": 3.0, "output": 15.0},
    "j2-mid": {"input": 0.6, "output": 0.6},
    
    # Default fallback pricing (per 1M tokens)
    "default": {"input": 1.0, "output": 1.0},
}


def fetch_huggingface_pricing() -> Dict[str, Dict[str, float]]:
    """
    Fetch model pricing from HuggingFace Inference API
    Note: This is a simplified version - in production you'd want to cache this
    """
    # HuggingFace provides pricing info through their Inference API
    # For now, we use the default pricing defined above
    # In production, you could query HF API for real-time pricing
    return MODEL_PRICING


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    """
    Calculate cost for a single LLM request
    
    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        custom_pricing: Optional custom pricing dictionary
        
    Returns:
        Cost in dollars
    """
    pricing = custom_pricing or MODEL_PRICING
    
    # Normalize model name for lookup
    model_key = model.lower()
    
    # Try exact match first
    model_pricing = pricing.get(model_key)
    
    if not model_pricing:
        # Try partial match (e.g., "gpt-4" matches "gpt-4-turbo")
        for key, value in pricing.items():
            if key != "default" and key in model_key:
                model_pricing = value
                break
    
    if not model_pricing:
        # Use default pricing
        model_pricing = pricing.get("default", {"input": 1.0, "output": 1.0})
    
    # Calculate cost: (tokens / 1,000,000) * price_per_million
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    return round(input_cost + output_cost, 6)


def calculate_batch_costs(
    requests_data: list,
    custom_pricing: Optional[Dict[str, Dict[str, float]]] = None
) -> list:
    """
    Calculate costs for a batch of requests
    
    Args:
        requests_data: List of dicts with 'model', 'input_tokens', 'output_tokens'
        custom_pricing: Optional custom pricing dictionary
        
    Returns:
        List of costs
    """
    return [
        calculate_cost(
            req["model"],
            req["input_tokens"],
            req["output_tokens"],
            custom_pricing
        )
        for req in requests_data
    ]


def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a specific model"""
    model_key = model.lower()
    model_pricing = MODEL_PRICING.get(model_key)
    
    if not model_pricing:
        for key, value in MODEL_PRICING.items():
            if key != "default" and key in model_key:
                model_pricing = value
                break
    
    return model_pricing or MODEL_PRICING["default"]


def list_available_models() -> list:
    """List all models with defined pricing"""
    return [k for k in MODEL_PRICING.keys() if k != "default"]
