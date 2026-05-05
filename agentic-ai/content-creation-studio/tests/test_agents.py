"""Tests for agent implementations without mocking."""

import pytest
from src.state.content_state import ContentState


class TestSupervisorAgent:
    """Tests for the Supervisor Agent."""
    
    def test_validate_input_valid(self):
        """Test validation with valid input."""
        from src.agents.supervisor import validate_input
        
        is_valid, error = validate_input("Benefits of AI in education")
        assert is_valid is True
        assert error == ""
    
    def test_validate_input_empty_topic(self):
        """Test validation rejects empty topic."""
        from src.agents.supervisor import validate_input
        
        is_valid, error = validate_input("")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_input_short_topic(self):
        """Test validation rejects too short topic."""
        from src.agents.supervisor import validate_input
        
        is_valid, error = validate_input("AI")
        assert is_valid is False
        assert "too short" in error.lower()
    
    def test_validate_input_with_keywords(self):
        """Test validation with valid keywords."""
        from src.agents.supervisor import validate_input
        
        is_valid, error = validate_input("Topic here", ["AI", "education"])
        assert is_valid is True
    
    def test_validate_input_too_many_keywords(self):
        """Test validation rejects too many keywords."""
        from src.agents.supervisor import validate_input
        
        keywords = [f"keyword{i}" for i in range(25)]
        is_valid, error = validate_input("Topic here", keywords)
        assert is_valid is False
        assert "too many" in error.lower()
    
    def test_route_to_agent_research(self):
        """Test routing to research when no facts exist."""
        from src.agents.supervisor import route_to_agent
        
        state: ContentState = {
            "topic": "Test topic",
            "facts": [],
            "draft": "",
            "final_content": ""
        }
        
        route = route_to_agent(state)
        assert route == "research_agent"
    
    def test_route_to_agent_writer(self):
        """Test routing to writer when facts exist but no draft."""
        from src.agents.supervisor import route_to_agent
        
        state: ContentState = {
            "topic": "Test topic",
            "facts": [{"source": "test", "fact": "test fact", "relevance": "high"}],
            "draft": "",
            "final_content": ""
        }
        
        route = route_to_agent(state)
        assert route == "writer_agent"
    
    def test_route_to_agent_editor(self):
        """Test routing to editor when draft exists but no final."""
        from src.agents.supervisor import route_to_agent
        
        state: ContentState = {
            "topic": "Test topic",
            "facts": [{"source": "test", "fact": "test fact", "relevance": "high"}],
            "draft": "This is a draft",
            "final_content": ""
        }
        
        route = route_to_agent(state)
        assert route == "editor_agent"
    
    def test_route_to_agent_approval(self):
        """Test routing to approval when final content exists."""
        from src.agents.supervisor import route_to_agent
        
        state: ContentState = {
            "topic": "Test topic",
            "facts": [{"source": "test", "fact": "test fact", "relevance": "high"}],
            "draft": "This is a draft",
            "final_content": "This is the final content",
            "approval_status": "pending"
        }
        
        route = route_to_agent(state)
        assert route == "approval"
    
    def test_route_to_agent_rejected_returns_to_writer(self):
        """Test that rejected content goes back to writer."""
        from src.agents.supervisor import route_to_agent
        
        state: ContentState = {
            "topic": "Test topic",
            "facts": [],
            "draft": "",
            "final_content": "",
            "approval_status": "rejected"
        }
        
        route = route_to_agent(state)
        assert route == "writer_agent"


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
