"""Tests for ContentState TypedDict definition."""

import pytest
from src.state.content_state import ContentState


class TestContentState:
    """Tests for the ContentState TypedDict."""
    
    def test_content_state_has_required_fields(self):
        """Test that ContentState accepts all required fields."""
        state: ContentState = {
            "topic": "Test Topic",
            "keywords": ["keyword1", "keyword2"],
            "search_query": "test query",
            "facts": [
                {"source": "source1", "fact": "fact1", "relevance": "high"},
                {"source": "source2", "fact": "fact2", "relevance": "medium"}
            ],
            "draft": "Draft content",
            "edited_content": "Edited content",
            "final_content": "Final content",
            "approval_status": "pending",
            "revision_notes": "Revision notes",
            "current_agent": "supervisor"
        }
        
        assert state["topic"] == "Test Topic"
        assert len(state["keywords"]) == 2
        assert len(state["facts"]) == 2
        assert state["draft"] == "Draft content"
        assert state["edited_content"] == "Edited content"
        assert state["final_content"] == "Final content"
        assert state["approval_status"] == "pending"
        assert state["revision_notes"] == "Revision notes"
        assert state["current_agent"] == "supervisor"
    
    def test_content_state_empty_values(self):
        """Test that ContentState accepts empty initial values."""
        state: ContentState = {
            "topic": "",
            "keywords": [],
            "search_query": "",
            "facts": [],
            "draft": "",
            "edited_content": "",
            "final_content": "",
            "approval_status": "pending",
            "revision_notes": "",
            "current_agent": "none"
        }
        
        assert state["topic"] == ""
        assert state["keywords"] == []
        assert state["facts"] == []
        assert state["draft"] == ""
        assert state["final_content"] == ""
    
    def test_content_state_facts_structure(self):
        """Test facts field structure."""
        state: ContentState = {
            "topic": "Test",
            "facts": [
                {"source": "https://example.com", "fact": "A factual statement", "relevance": "high"},
                {"source": "https://test.com", "fact": "Another fact", "relevance": "medium"},
                {"source": "Notes", "fact": "Low priority fact", "relevance": "low"}
            ],
            "keywords": [],
            "search_query": "",
            "draft": "",
            "edited_content": "",
            "final_content": "",
            "approval_status": "pending",
            "revision_notes": "",
            "current_agent": "none"
        }
        
        # Verify each fact has the expected structure
        for fact in state["facts"]:
            assert "source" in fact
            assert "fact" in fact
            assert "relevance" in fact
    
    def test_content_state_approval_status_values(self):
        """Test various approval status values."""
        statuses = ["pending", "approved", "rejected"]
        
        for status in statuses:
            state: ContentState = {
                "topic": "Test",
                "keywords": [],
                "search_query": "",
                "facts": [],
                "draft": "",
                "edited_content": "",
                "final_content": "",
                "approval_status": status,
                "revision_notes": "",
                "current_agent": "none"
            }
            assert state["approval_status"] == status
    
    def test_content_state_agent_values(self):
        """Test various agent values."""
        agents = ["none", "supervisor", "research_agent", "writer_agent", 
                  "editor_agent", "research_complete", "writer_complete", 
                  "writer_error", "editor_complete", "approval"]
        
        for agent in agents:
            state: ContentState = {
                "topic": "Test",
                "keywords": [],
                "search_query": "",
                "facts": [],
                "draft": "",
                "edited_content": "",
                "final_content": "",
                "approval_status": "pending",
                "revision_notes": "",
                "current_agent": agent
            }
            assert state["current_agent"] == agent
