"""State management for Content Creation Studio."""

from typing import TypedDict, NotRequired


class ContentState(TypedDict):
    """Shared state passed between all agents in the content creation pipeline.
    
    Attributes:
        topic: User's requested content topic
        keywords: SEO keywords to target in the content
        search_query: Optimized search query for research
        facts: Extracted facts from research [{source, fact, relevance}]
        draft: Initial draft from Writer Agent
        edited_content: After Editor proofreads
        final_content: After SEO optimization
        approval_status: "pending", "approved", or "rejected"
        revision_notes: Human feedback if rejected
        current_agent: Which agent is currently active (for debugging/logging)
    """
    topic: str
    keywords: NotRequired[list[str]]
    search_query: NotRequired[str]
    facts: NotRequired[list[dict]]
    draft: NotRequired[str]
    edited_content: NotRequired[str]
    final_content: NotRequired[str]
    approval_status: NotRequired[str]
    revision_notes: NotRequired[str]
    current_agent: NotRequired[str]
