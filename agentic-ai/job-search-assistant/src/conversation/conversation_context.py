"""Conversation context - Tracks conversation history and state."""

from typing import Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class TurnType(Enum):
    """Type of conversation turn."""
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    turn_type: TurnType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None
    entities: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Tracks the current conversation context and history.
    
    This class maintains:
    - Conversation history (list of turns)
    - Current state (pending intents, awaiting responses, etc.)
    - Extracted entities from the conversation
    - User preferences discovered during conversation
    """
    
    # Maximum turns to keep in history
    MAX_HISTORY = 20
    
    def __init__(self):
        """Initialize empty conversation context."""
        self.turns: list[ConversationTurn] = []
        self.pending_intent: Optional[str] = None
        self.awaiting_response: bool = False
        self.last_bot_message: Optional[str] = None
        self.extracted_entities: dict = {}
        self.user_preferences: dict = {}
        self.confirmation_needed: bool = False
        self.confirmation_context: Optional[dict] = None
    
    def add_user_turn(self, content: str, intent: Optional[str] = None, entities: dict = None) -> None:
        """Add a user turn to the conversation history.
        
        Args:
            content: The user's message
            intent: Classified intent
            entities: Extracted entities from the message
        """
        turn = ConversationTurn(
            turn_type=TurnType.USER,
            content=content,
            intent=intent,
            entities=entities or {}
        )
        self.turns.append(turn)
        self._prune_history()
    
    def add_bot_turn(self, content: str, metadata: dict = None) -> None:
        """Add a bot turn to the conversation history.
        
        Args:
            content: The bot's response
            metadata: Additional metadata about the response
        """
        turn = ConversationTurn(
            turn_type=TurnType.BOT,
            content=content,
            metadata=metadata or {}
        )
        self.turns.append(turn)
        self.last_bot_message = content
        self._prune_history()
    
    def set_pending_intent(self, intent: str, context: dict = None) -> None:
        """Set an intent that is awaiting user confirmation or additional input.
        
        Args:
            intent: The pending intent
            context: Additional context about what's needed
        """
        self.pending_intent = intent
        self.awaiting_response = True
        if context:
            self.confirmation_context = context
    
    def clear_pending_intent(self) -> None:
        """Clear the pending intent after it's been resolved."""
        self.pending_intent = None
        self.awaiting_response = False
        self.confirmation_needed = False
        self.confirmation_context = None
    
    def request_confirmation(self, message: str, context: dict = None) -> None:
        """Request user confirmation for an action.
        
        Args:
            message: The confirmation request message
            context: Details about what needs confirmation
        """
        self.confirmation_needed = True
        self.awaiting_response = True
        self.confirmation_context = context or {}
        self.confirmation_context["request_message"] = message
    
    def get_recent_turns(self, count: int = 5) -> list[ConversationTurn]:
        """Get the most recent conversation turns.
        
        Args:
            count: Number of recent turns to return
            
        Returns:
            List of recent ConversationTurn objects
        """
        return self.turns[-count:] if self.turns else []
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far.
        
        Returns:
            Formatted summary string
        """
        if not self.turns:
            return "No conversation yet."
        
        summary_lines = []
        for turn in self.turns[-5:]:  # Last 5 turns
            prefix = "You: " if turn.turn_type == TurnType.USER else "Bot: "
            content = turn.content[:80] + "..." if len(turn.content) > 80 else turn.content
            summary_lines.append(f"{prefix}{content}")
        
        return "\n".join(summary_lines)
    
    def update_entity(self, entity_type: str, value: Any) -> None:
        """Update an extracted entity in the context.
        
        Args:
            entity_type: Type of entity (e.g., "location", "job_title")
            value: The entity value
        """
        if entity_type in self.extracted_entities:
            # Append if not already present
            if value not in self.extracted_entities[entity_type]:
                self.extracted_entities[entity_type].append(value)
        else:
            self.extracted_entities[entity_type] = [value]
    
    def set_user_preference(self, key: str, value: Any) -> None:
        """Store a user preference discovered during conversation.
        
        Args:
            key: Preference key
            value: Preference value
        """
        self.user_preferences[key] = value
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a stored user preference.
        
        Args:
            key: Preference key
            default: Default value if not found
            
        Returns:
            The preference value or default
        """
        return self.user_preferences.get(key, default)
    
    def needs_information(self, information_type: str) -> bool:
        """Check if we need specific information before proceeding.
        
        Args:
            information_type: Type of information needed
            
        Returns:
            True if the information is missing and needed
        """
        return self.extracted_entities.get(information_type) is None
    
    def get_missing_information(self) -> list[str]:
        """Get list of missing information types needed for pending intent.
        
        Returns:
            List of missing information types
        """
        missing = []
        
        if self.pending_intent:
            required_map = {
                "upload_resume": [],
                "search_jobs": ["location", "job_title"],
                "score_resume": [],
                "tailor_resume": [],
                "generate_cover_letter": ["job_id"],
                "interview_prep": ["job_id"],
            }
            
            required = required_map.get(self.pending_intent, [])
            for info in required:
                if not self.extracted_entities.get(info):
                    missing.append(info)
        
        return missing
    
    def _prune_history(self) -> None:
        """Remove old turns if history exceeds maximum."""
        if len(self.turns) > self.MAX_HISTORY:
            self.turns = self.turns[-self.MAX_HISTORY:]
    
    def reset(self) -> None:
        """Reset the conversation context."""
        self.turns = []
        self.pending_intent = None
        self.awaiting_response = False
        self.last_bot_message = None
        self.extracted_entities = {}
        self.user_preferences = {}
        self.confirmation_needed = False
        self.confirmation_context = None