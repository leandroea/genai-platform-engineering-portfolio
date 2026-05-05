"""Tests for tool implementations - integration tests without mocks."""

import os
import pytest


class TestSearchTools:
    """Tests for search tools using real implementations."""
    
    def test_web_search_returns_string(self):
        """Test that web_search returns a string result."""
        try:
            from src.tools.search_tools import web_search
        except ImportError as e:
            if "duckduckgo_search" in str(e):
                pytest.skip("duckduckgo-search not installed")
            raise
        
        result = web_search("Python programming")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_web_search_handles_query(self):
        """Test web_search with a specific query."""
        try:
            from src.tools.search_tools import web_search
        except ImportError as e:
            if "duckduckgo_search" in str(e):
                pytest.skip("duckduckgo-search not installed")
            raise
        
        result = web_search("What is Python programming language")
        
        # Should return formatted results or error message
        assert isinstance(result, str)
        # DuckDuckGo may return results or error - both are valid strings
        assert len(result) > 0
    
    def test_extract_facts_requires_api_key(self):
        """Test that extract_facts returns error when API key not configured."""
        try:
            from src.tools.search_tools import extract_facts
        except ImportError as e:
            if "duckduckgo_search" in str(e):
                pytest.skip("duckduckgo-search not installed")
            raise
        
        # Save original API key
        original_key = os.environ.get("MINIMAX_API_KEY")
        
        # Ensure API key is not set
        if "MINIMAX_API_KEY" in os.environ:
            del os.environ["MINIMAX_API_KEY"]
        
        result = extract_facts("Some content", "test topic")
        
        assert "not configured" in result or "Error" in result
        
        # Restore original API key
        if original_key:
            os.environ["MINIMAX_API_KEY"] = original_key


class TestWritingTools:
    """Tests for writing tools using real implementations."""
    
    def test_write_draft_requires_api_key(self):
        """Test that write_draft returns error when API key not configured."""
        from src.tools.writing_tools import write_draft
        
        # Save original API key
        original_key = os.environ.get("MINIMAX_API_KEY")
        
        # Ensure API key is not set
        if "MINIMAX_API_KEY" in os.environ:
            del os.environ["MINIMAX_API_KEY"]
        
        result = write_draft("Test topic", "Some facts", ["keyword"])
        
        assert "not configured" in result or "Error" in result
        
        # Restore original API key
        if original_key:
            os.environ["MINIMAX_API_KEY"] = original_key
    
    def test_structure_outline_requires_api_key(self):
        """Test that structure_outline returns error when API key not configured."""
        from src.tools.writing_tools import structure_outline
        
        # Save original API key
        original_key = os.environ.get("MINIMAX_API_KEY")
        
        # Ensure API key is not set
        if "MINIMAX_API_KEY" in os.environ:
            del os.environ["MINIMAX_API_KEY"]
        
        result = structure_outline("Test topic")
        
        assert "not configured" in result or "Error" in result
        
        # Restore original API key
        if original_key:
            os.environ["MINIMAX_API_KEY"] = original_key


class TestSEOTools:
    """Tests for SEO tools using real implementations."""
    
    def test_grammar_check_requires_api_key(self):
        """Test that grammar_check returns error when API key not configured."""
        from src.tools.seo_tools import grammar_check
        
        # Save original API key
        original_key = os.environ.get("MINIMAX_API_KEY")
        
        # Ensure API key is not set
        if "MINIMAX_API_KEY" in os.environ:
            del os.environ["MINIMAX_API_KEY"]
        
        result = grammar_check("He go to school yesterday")
        
        assert "not configured" in result or "Error" in result
        
        # Restore original API key
        if original_key:
            os.environ["MINIMAX_API_KEY"] = original_key
    
    def test_format_seo_requires_api_key(self):
        """Test that format_seo returns error when API key not configured."""
        from src.tools.seo_tools import format_seo
        
        # Save original API key
        original_key = os.environ.get("MINIMAX_API_KEY")
        
        # Ensure API key is not set
        if "MINIMAX_API_KEY" in os.environ:
            del os.environ["MINIMAX_API_KEY"]
        
        result = format_seo("Original content", ["keyword"])
        
        assert "not configured" in result or "Error" in result
        
        # Restore original API key
        if original_key:
            os.environ["MINIMAX_API_KEY"] = original_key
    
    def test_rewrite_professional_requires_api_key(self):
        """Test that rewrite_professional returns error when API key not configured."""
        from src.tools.seo_tools import rewrite_professional
        
        # Save original API key
        original_key = os.environ.get("MINIMAX_API_KEY")
        
        # Ensure API key is not set
        if "MINIMAX_API_KEY" in os.environ:
            del os.environ["MINIMAX_API_KEY"]
        
        result = rewrite_professional("Casual content", "professional")
        
        assert "not configured" in result or "Error" in result
        
        # Restore original API key
        if original_key:
            os.environ["MINIMAX_API_KEY"] = original_key
