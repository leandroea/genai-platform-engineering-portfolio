"""
Utility functions for Prompt Engineering Playground.
"""
import streamlit as st
from datetime import datetime

from app.models import LLMParameters


def render_stars(rating: int, max_stars: int = 5) -> str:
    """Render a star rating display."""
    filled = "⭐" * rating
    empty = "☆" * (max_stars - rating)
    return filled + empty


def format_datetime(dt_str: str) -> str:
    """Format a datetime string for display."""
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def get_parameter_defaults() -> LLMParameters:
    """Get default LLM parameters."""
    return LLMParameters(
        temperature=0.7,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )


def render_llm_parameters_sidebar() -> LLMParameters:
    """Render LLM parameter controls in sidebar and return the parameters."""
    st.sidebar.header("LLM Parameters")

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. Higher values make output more random."
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=1,
        max_value=4096,
        value=2048,
        step=1,
        help="Maximum number of tokens to generate."
    )

    top_p = st.sidebar.slider(
        "Top P (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Controls diversity via nucleus sampling."
    )

    frequency_penalty = st.sidebar.slider(
        "Frequency Penalty",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Decreases likelihood of repeating same words."
    )

    presence_penalty = st.sidebar.slider(
        "Presence Penalty",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Increases likelihood of talking about new topics."
    )

    return LLMParameters(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def show_error(message: str):
    """Show an error message in Streamlit."""
    st.error(message)


def show_success(message: str):
    """Show a success message in Streamlit."""
    st.success(message)


def show_warning(message: str):
    """Show a warning message in Streamlit."""
    st.warning(message)
