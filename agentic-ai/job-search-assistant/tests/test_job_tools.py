"""Tests for job tools."""

import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import os

from src.tools.job_tools import (
    search_jobs,
    deduplicate_jobs,
    score_job_match,
    _parse_salary,
    save_jobs_to_file,
    load_jobs_from_file,
)


class TestJobTools:
    """Test suite for job_tools module."""
    
    def test_parse_salary_with_k(self):
        """Test salary parsing with 'k' suffix."""
        assert _parse_salary("$120k") == 120000
        assert _parse_salary("120K") == 120000
        assert _parse_salary("$120K - $150K") == 120000
    
    def test_parse_salary_with_commas(self):
        """Test salary parsing with commas."""
        assert _parse_salary("$120,000") == 120000
        assert _parse_salary("120,000") == 120000
    
    def test_parse_salary_empty(self):
        """Test salary parsing with empty string."""
        assert _parse_salary("") is None
        assert _parse_salary(None) is None
    
    def test_parse_salary_no_number(self):
        """Test salary parsing when no number found."""
        assert _parse_salary("Competitive") is None
        assert _parse_salary("Negotiable") is None
    
    def test_deduplicate_jobs(self):
        """Test job deduplication."""
        jobs = [
            {"title": "Python Developer", "company": "TechCorp", "location": "Remote"},
            {"title": "Python Developer", "company": "TechCorp", "location": "Remote"},  # Duplicate
            {"title": "Java Developer", "company": "OtherCorp", "location": "NYC"},
        ]
        
        unique = deduplicate_jobs(jobs)
        
        assert len(unique) == 2
        assert unique[0]["title"] == "Python Developer"
        assert unique[1]["title"] == "Java Developer"
    
    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        assert deduplicate_jobs([]) == []
    
    def test_score_job_match_with_role(self):
        """Test job matching with target roles."""
        job = {"title": "Senior Python Developer", "location": "Remote", "salary_min": 120000}
        user_profile = {"preferred_locations": ["Remote"]}
        target_roles = ["Python Developer", "Software Engineer"]
        
        score = score_job_match(job, user_profile, target_roles)
        
        assert score > 0.5  # Should have decent match
    
    def test_score_job_match_no_match(self):
        """Test job matching with no role match."""
        job = {"title": "Marketing Manager", "location": "NYC"}
        user_profile = {}
        target_roles = ["Python Developer"]
        
        score = score_job_match(job, user_profile, target_roles)
        
        assert score < 0.3  # Should have low match
    
    def test_score_job_match_with_location(self):
        """Test job matching with location bonus."""
        job = {"title": "Developer", "location": "Remote", "salary_min": 100000}
        user_profile = {"preferred_locations": ["Remote"]}
        target_roles = ["Developer"]
        
        score = score_job_match(job, user_profile, target_roles)
        
        # Should get bonus for location match
        assert score > 0.3


class TestJobFileOperations:
    """Test suite for job file operations."""
    
    def test_save_and_load_jobs(self):
        """Test saving and loading jobs to/from file."""
        jobs = [
            {"title": "Python Developer", "company": "TechCorp", "location": "Remote"},
            {"title": "Java Developer", "company": "OtherCorp", "location": "NYC"},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "jobs.json")
            
            save_jobs_to_file(jobs, filepath)
            
            assert os.path.exists(filepath)
            
            loaded = load_jobs_from_file(filepath)
            
            assert len(loaded) == 2
            assert loaded[0]["title"] == "Python Developer"
    
    def test_load_jobs_nonexistent_file(self):
        """Test loading from non-existent file."""
        result = load_jobs_from_file("/nonexistent/path/jobs.json")
        
        assert result == []
    
    def test_load_jobs_corrupted_file(self):
        """Test loading from corrupted JSON file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            result = load_jobs_from_file(temp_path)
            assert result == []  # Should return empty on error
        finally:
            os.unlink(temp_path)


class TestJoobleAPI:
    """Test suite for Jooble API integration (requires real API)."""
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock Jooble API response."""
        return {
            "jobs": [
                {
                    "title": "Python Developer",
                    "company": "TechCorp",
                    "location": "Remote",
                    "snippet": "We are looking for a Python Developer...",
                    "link": "https://example.com/job/1",
                    "salary": "$100k - $120k",
                },
                {
                    "title": "Senior Python Developer",
                    "company": "BigTech",
                    "location": "New York",
                    "snippet": "Join our team as a Senior Python Developer...",
                    "link": "https://example.com/job/2",
                },
            ]
        }
    
    def test_search_jobs_requires_api_key(self):
        """Test that search_jobs requires API key."""
        with patch('src.tools.job_tools.get_jooble_api_key') as mock_get_key:
            mock_get_key.side_effect = ValueError("No API key")
            
            with pytest.raises(ValueError) as exc_info:
                search_jobs("Python Developer", "Remote")
            
            assert "API key" in str(exc_info.value)
    
    def test_search_jobs_integration(self):
        """Integration test for job search (requires real API key).
        
        This test is skipped by default unless a real API key is available.
        Run with: pytest -v -k test_search_jobs_integration
        """
        pytest.skip("Requires real Jooble API key - run manually")
        
        # This would be a real test:
        # result = search_jobs("Python Developer", "Remote")
        # assert result["success"] == True
        # assert len(result["jobs"]) > 0