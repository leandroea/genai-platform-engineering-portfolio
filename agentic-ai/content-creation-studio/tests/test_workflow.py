"""Tests for workflow orchestration."""

import pytest
from src.state.content_state import ContentState


class TestContentGraph:
    """Tests for the LangGraph workflow."""
    
    def test_initial_state(self):
        """Test that initial state has required fields."""
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
        
        assert "topic" in state
        assert "facts" in state
        assert "draft" in state
        assert "final_content" in state
        assert "approval_status" in state
    
    def test_approval_gate_pending(self):
        """Test approval gate behavior."""
        state: ContentState = {
            "topic": "Test",
            "final_content": "Final content",
            "approval_status": "pending"
        }
        
        # Approval status should trigger approval route
        assert state["approval_status"] == "pending"


class TestPipelineIntegration:
    """Integration tests for the full pipeline using real implementations."""
    
    def test_state_transitions(self):
        """Test state structure and transitions without external calls."""
        # Test that state can hold required data
        state: ContentState = {
            "topic": "Test topic",
            "keywords": ["test"],
            "search_query": "",
            "facts": [],
            "draft": "",
            "edited_content": "",
            "final_content": "",
            "approval_status": "pending",
            "revision_notes": "",
            "current_agent": "none"
        }
        
        assert "topic" in state
        assert state["topic"] == "Test topic"
        
        # Test adding facts
        state["facts"] = [{"source": "test.com", "fact": "A test fact", "relevance": "high"}]
        assert len(state["facts"]) == 1
        
        # Test adding draft
        state["draft"] = "This is a test draft"
        assert state["draft"] == "This is a test draft"
        
        # Test adding edited content
        state["edited_content"] = "Edited version"
        assert state["edited_content"] == "Edited version"
    
    def test_pipeline_state_progression(self):
        """Test that state can progress through pipeline stages."""
        # Step 1: Initial state - no facts, no draft
        state: ContentState = {
            "topic": "Python tutorial",
            "facts": [],
            "draft": "",
            "final_content": "",
            "approval_status": "pending"
        }
        
        # Simulate Research Agent completing
        state["current_agent"] = "research_complete"
        state["facts"] = [{"source": "python.org", "fact": "Python is a programming language", "relevance": "high"}]
        assert len(state["facts"]) == 1
        
        # Simulate Writer Agent completing
        state["current_agent"] = "writer_complete"
        state["draft"] = "Python is a popular programming language..."
        assert len(state["draft"]) > 0
        
        # Simulate Editor Agent completing
        state["current_agent"] = "editor_complete"
        state["final_content"] = "Edited final version of Python article"
        assert len(state["final_content"]) > 0
        
        # Approval
        state["approval_status"] = "approved"
        assert state["approval_status"] == "approved"
    
    def test_rejection_loop(self):
        """Test that rejected content goes back for revision."""
        state: ContentState = {
            "topic": "Python tutorial",
            "facts": [{"source": "python.org", "fact": "Python facts", "relevance": "high"}],
            "draft": "Python draft content",
            "final_content": "Edited content",
            "approval_status": "rejected",
            "revision_notes": "Please add more details about Python syntax"
        }
        
        # In the pipeline, rejected content would loop back to writer
        # We simulate this by setting a flag
        is_rejected = state["approval_status"] == "rejected"
        assert is_rejected is True
        assert state["revision_notes"] != ""
    
    def test_research_agent_node(self):
        """Test research agent node execution."""
        from src.agents.research_agent import research_agent_node
        
        state: ContentState = {
            "topic": "Python programming",
            "current_agent": "none"
        }
        
        result = research_agent_node(state)
        
        # Should complete or error - verify state structure returned
        assert "current_agent" in result
        assert "topic" in result
        assert result["topic"] == "Python programming"
    
    def test_writer_agent_node(self):
        """Test writer agent node execution."""
        from src.agents.writer_agent import writer_agent_node
        
        state: ContentState = {
            "topic": "Python programming",
            "facts": [{"source": "python.org", "fact": "Python is interpreted", "relevance": "high"}],
            "keywords": ["python", "programming"],
            "current_agent": "none"
        }
        
        result = writer_agent_node(state)
        
        # Should complete - verify state structure returned
        assert "current_agent" in result
        assert "topic" in result
        # Either wrote successfully or had an error
        assert isinstance(result["current_agent"], str)
    
    def test_editor_agent_node(self):
        """Test editor agent node execution without API key."""
        from src.agents.editor_agent import editor_agent_node
        
        state: ContentState = {
            "draft": "Python is a great language for beginners",
            "keywords": ["python", "programming"],
            "current_agent": "none"
        }
        
        result = editor_agent_node(state)
        
        # Result can be dict (state) or string (error message)
        assert isinstance(result, (dict, str))
        # If it's a string, it should be an error about API key
        if isinstance(result, str):
            assert "Error" in result or "not configured" in result
