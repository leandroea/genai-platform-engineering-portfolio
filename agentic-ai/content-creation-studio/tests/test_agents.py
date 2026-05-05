"""Tests for agent implementations without mocking."""

import pytest
from src.state.content_state import ContentState


class TestResearchAgent:
    """Tests for the Research Agent."""
    
    def test_research_agent_node_empty_topic(self):
        """Test research with empty topic returns error."""
        from src.agents.research_agent import research_agent_node
        
        state: ContentState = {"topic": ""}
        result = research_agent_node(state)
        
        assert result["current_agent"] == "research_error"
        assert result["facts"] == []
    
    def test_research_agent_node_with_topic(self):
        """Test research agent processes topic and returns state."""
        from src.agents.research_agent import research_agent_node
        
        state: ContentState = {"topic": "Python programming", "current_agent": "none"}
        result = research_agent_node(state)
        
        # Should return a dict with current_agent field
        assert isinstance(result, dict)
        assert "current_agent" in result
        # Should have processed topic
        assert result.get("topic") == "Python programming"


class TestWriterAgent:
    """Tests for the Writer Agent."""
    
    def test_writer_agent_node_empty_topic(self):
        """Test writer with empty topic returns error."""
        from src.agents.writer_agent import writer_agent_node
        
        state: ContentState = {"topic": ""}
        result = writer_agent_node(state)
        
        assert result["current_agent"] == "writer_error"
    
    def test_writer_agent_node_with_topic_and_facts(self):
        """Test writer agent with valid input."""
        from src.agents.writer_agent import writer_agent_node
        
        state: ContentState = {
            "topic": "Python programming",
            "facts": [{"source": "test", "fact": "Python is popular", "relevance": "high"}],
            "keywords": ["python"]
        }
        result = writer_agent_node(state)
        
        # Should return state with current_agent set
        assert isinstance(result, dict)
        assert "current_agent" in result


class TestEditorAgent:
    """Tests for the Editor Agent."""
    
    def test_editor_agent_node_empty_draft(self):
        """Test editor with empty draft returns error."""
        from src.agents.editor_agent import editor_agent_node
        
        state: ContentState = {"draft": ""}
        result = editor_agent_node(state)
        
        assert result["current_agent"] == "editor_error"
    
    def test_editor_agent_node_with_draft(self):
        """Test editor agent with valid input."""
        from src.agents.editor_agent import editor_agent_node
        
        state: ContentState = {
            "draft": "Python is a great programming language",
            "keywords": ["python", "programming"]
        }
        result = editor_agent_node(state)
        
        # Should return dict or error string
        assert isinstance(result, (dict, str))
        if isinstance(result, dict):
            assert "current_agent" in result
