"""Configuration utilities for loading environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Path to .env file
ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def load_env() -> None:
    """Load environment variables from .env file."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        # Try parent directories for development
        possible_paths = [
            Path.cwd() / ".env",
            Path(__file__).parent.parent / ".env",
        ]
        for path in possible_paths:
            if path.exists():
                load_dotenv(path)
                break


def get_minimax_api_key() -> str:
    """Get MiniMax API key from environment."""
    load_env()
    api_key = os.getenv("MINIMAX_API_KEY", "")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY not found in environment")
    return api_key


def get_minimax_endpoint() -> str:
    """Get MiniMax API endpoint from environment."""
    load_env()
    return os.getenv("MINIMAX_ENDPOINT", "https://api.minimax.io/v1")


def get_model_name() -> str:
    """Get model name from environment."""
    load_env()
    return os.getenv("MODEL_NAME", "minimax-m2.7")


def get_jooble_api_key() -> str:
    """Get Jooble API key from environment."""
    load_env()
    api_key = os.getenv("JOOBLE_API_KEY", "")
    if not api_key:
        raise ValueError("JOOBLE_API_KEY not found in environment")
    return api_key


def get_max_applications() -> int:
    """Get max applications setting."""
    load_env()
    return int(os.getenv("MAX_APPLICATIONS", "20"))


def get_max_jobs_per_search() -> int:
    """Get max jobs per search setting."""
    load_env()
    return int(os.getenv("MAX_JOBS_PER_SEARCH", "20"))