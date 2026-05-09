"""Conversation module - Natural language interface for Job Search Assistant."""

from .intent_classifier import IntentClassifier, Intent
from .conversation_context import ConversationContext
from .conversational_agent import ConversationalAgent

__all__ = [
    "IntentClassifier",
    "Intent",
    "ConversationContext",
    "ConversationalAgent",
]