"""
LLM Service module for Prompt Engineering Playground.
Handles integration with NVIDIA API using LangChain.
"""
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import LLMParameters


# Global LLM instance
_llm: Optional[ChatNVIDIA] = None


def get_llm() -> ChatNVIDIA:
    """Get or create the LLM instance."""
    global _llm

    if _llm is None:
        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        model = os.getenv("MODEL", "meta/llama-3.3-70b-instruct")

        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables")

        _llm = ChatNVIDIA(
            model=model,
            api_key=api_key,
            base_url=base_url
        )

    return _llm


def call_llm(
    prompt: str,
    system_message: Optional[str] = None,
    parameters: Optional[LLMParameters] = None
) -> str:
    """
    Call the LLM with a prompt and optional parameters.

    Args:
        prompt: The user prompt
        system_message: Optional system message
        parameters: LLM parameters

    Returns:
        The LLM response as a string
    """
    llm = get_llm()

    # Build messages
    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(HumanMessage(content=prompt))

    # Build invocation params
    invoke_params = {}
    if parameters:
        invoke_params = {
            "temperature": parameters.temperature,
            "max_tokens": parameters.max_tokens,
            "top_p": parameters.top_p,
            "frequency_penalty": parameters.frequency_penalty,
            "presence_penalty": parameters.presence_penalty
        }

    # Call LLM
    response = llm.invoke(messages, **invoke_params)

    return response.content


def call_llm_with_messages(
    messages: list[dict],
    parameters: Optional[LLMParameters] = None
) -> str:
    """
    Call the LLM with a list of messages (for multi-turn conversations).

    Args:
        messages: List of message dicts with 'role' and 'content'
        parameters: LLM parameters

    Returns:
        The LLM response as a string
    """
    llm = get_llm()

    # Convert message dicts to LangChain messages
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    # Build invocation params
    invoke_params = {}
    if parameters:
        invoke_params = {
            "temperature": parameters.temperature,
            "max_tokens": parameters.max_tokens,
            "top_p": parameters.top_p,
            "frequency_penalty": parameters.frequency_penalty,
            "presence_penalty": parameters.presence_penalty
        }

    # Call LLM
    response = llm.invoke(lc_messages, **invoke_params)

    return response.content


def reset_llm():
    """Reset the LLM instance (useful for testing or config changes)."""
    global _llm
    _llm = None
