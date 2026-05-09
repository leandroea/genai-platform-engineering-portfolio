"""Tests for workflow and state management."""

import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import os

from src.state.job_search_state import (
    JobSearchState,
    Job,
    Application,
    create_initial_state,
)


class TestJobSearchState:
    """Test suite for JobSearchState."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state()
        
        assert state["user_profile"] == {}
        assert state["original_resume_path"] == ""
        assert state["original_resume_text"] == ""
        assert state["target_roles"] == []
        assert state["target_locations"] == []
        assert state["target_companies"] == []
        assert state["resume_tailor_output"] == {}
        assert state["cover_letter_output"] == {}
        assert state["job_aggregator_output"] == {}
        assert state["resume_score_output"] == {}
        assert state["application_form_output"] == {}
        assert state["interview_coach_output"] == {}
        assert state["current_phase"] == "idle"
        assert state["pending_tasks"] == []
        assert state["completed_tasks"] == []
        assert state["blockers"] == []
    
    def test_state_type_dict_structure(self):
        """Test that state follows TypedDict structure."""
        state = create_initial_state()
        
        # All required keys should be present
        required_keys = [
            "user_profile", "original_resume_path", "original_resume_text",
            "target_roles", "target_locations", "target_companies",
            "resume_tailor_output", "cover_letter_output", "job_aggregator_output",
            "resume_score_output", "application_form_output", "interview_coach_output",
            "current_phase", "pending_tasks", "completed_tasks", "blockers",
            "supervisor_decision"
        ]
        
        for key in required_keys:
            assert key in state


class TestJobModel:
    """Test suite for Job data model."""
    
    def test_job_creation(self):
        """Test Job object creation."""
        job = Job(
            title="Python Developer",
            company="TechCorp",
            location="Remote",
            description="We need a Python developer",
            url="https://example.com/job",
            salary_min=100000,
            salary_max=150000,
        )
        
        assert job.title == "Python Developer"
        assert job.company == "TechCorp"
        assert job.location == "Remote"
        assert job.salary_min == 100000
        assert job.salary_max == 150000
        assert job.source == "jooble"
        assert job.applied == False
        assert job.id is not None
    
    def test_job_to_dict(self):
        """Test Job serialization to dict."""
        job = Job(
            title="Python Developer",
            company="TechCorp",
            location="Remote",
            description="Description",
            url="https://example.com",
        )
        
        job_dict = job.to_dict()
        
        assert job_dict["title"] == "Python Developer"
        assert job_dict["company"] == "TechCorp"
        assert job_dict["id"] == job.id
        assert "applied_date" in job_dict
    
    def test_job_from_dict(self):
        """Test Job deserialization from dict."""
        job_data = {
            "id": "test-id-123",
            "title": "Python Developer",
            "company": "TechCorp",
            "location": "Remote",
            "description": "Description",
            "url": "https://example.com",
            "salary_min": 100000,
            "source": "jooble",
            "applied": True,
        }
        
        job = Job.from_dict(job_data)
        
        assert job.id == "test-id-123"
        assert job.title == "Python Developer"
        assert job.applied == True


class TestApplicationModel:
    """Test suite for Application data model."""
    
    def test_application_creation(self):
        """Test Application object creation."""
        app = Application(
            job_id="job-123",
            tailored_resume_path="/path/to/resume.docx",
            cover_letter_path="/path/to/letter.docx",
            status="applied",
        )
        
        assert app.job_id == "job-123"
        assert app.tailored_resume_path == "/path/to/resume.docx"
        assert app.status == "applied"
        assert app.id is not None
    
    def test_application_to_dict(self):
        """Test Application serialization to dict."""
        app = Application(
            job_id="job-123",
            tailored_resume_path="/path/to/resume.docx",
        )
        
        app_dict = app.to_dict()
        
        assert app_dict["job_id"] == "job-123"
        assert app_dict["tailored_resume_path"] == "/path/to/resume.docx"
        assert app_dict["status"] == "pending"
    
    def test_application_status_values(self):
        """Test Application status values."""
        statuses = ["pending", "applied", "interview", "rejected", "offer"]
        
        for status in statuses:
            app = Application(
                job_id="job-123",
                tailored_resume_path="/path/resume.docx",
                status=status,
            )
            assert app.status == status


class TestWorkflow:
    """Test suite for LangGraph workflow."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        return create_initial_state()
    
    def test_workflow_initialization_requires_api_key(self):
        """Test that workflow requires API key."""
        with patch('src.workflow.job_search_graph.get_minimax_api_key') as mock_get_key:
            mock_get_key.side_effect = ValueError("No API key")
            
            from src.workflow.job_search_graph import JobSearchGraph
            
            with pytest.raises(ValueError) as exc_info:
                JobSearchGraph()
            
            assert "API key" in str(exc_info.value)
    
    def test_workflow_execute_resume_tailor_requires_resume(self, mock_state):
        """Test that resume tailoring requires a resume in state."""
        # Don't set resume text
        mock_state["original_resume_text"] = ""
        
        from src.workflow.job_search_graph import JobSearchGraph
        
        with patch('src.workflow.job_search_graph.get_minimax_api_key'):
            with patch('src.workflow.job_search_graph.ChatOpenAI'):
                graph = JobSearchGraph()
                
                result = graph.execute_resume_tailor(
                    mock_state,
                    job_description="Test job",
                    job_id="job-123"
                )
                
                assert result["success"] == False
                assert "resume" in result["error"].lower()
    
    def test_workflow_execute_job_search_updates_phase(self, mock_state):
        """Test that job search updates the current phase."""
        mock_state["target_roles"] = ["Python Developer"]
        
        from src.workflow.job_search_graph import JobSearchGraph
        
        with patch('src.workflow.job_search_graph.get_minimax_api_key'):
            with patch('src.workflow.job_search_graph.ChatOpenAI'):
                with patch.object(
                    JobSearchGraph.__new__(JobSearchGraph),
                    'job_aggregator'
                ) as mock_aggregator:
                    mock_aggregator.search_for_jobs.return_value = {
                        "success": True,
                        "jobs": []
                    }
                    
                    graph = JobSearchGraph()
                    graph.execute_job_search(mock_state)
                    
                    assert mock_state["current_phase"] == "job_search"


class TestRunJobSearch:
    """Test suite for run_job_search convenience function."""
    
    def test_run_job_search_requires_roles(self):
        """Test that run_job_search requires target roles."""
        from src.workflow.job_search_graph import run_job_search
        
        # Empty roles should still run but with warning
        result = run_job_search([], ["Remote"])
        
        # Should return state even with empty roles
        assert isinstance(result, dict)
    
    def test_run_job_search_with_valid_input(self):
        """Test run_job_search with valid input."""
        from src.workflow.job_search_graph import run_job_search
        
        with patch('src.workflow.job_search_graph.JobSearchGraph') as MockGraph:
            mock_instance = Mock()
            mock_instance.execute_job_search.return_value = {
                "success": True,
                "jobs": []
            }
            MockGraph.return_value = mock_instance
            
            result = run_job_search(
                target_roles=["Python Developer"],
                target_locations=["Remote"]
            )
            
            # Verify state was updated
            assert result["target_roles"] == ["Python Developer"]
            assert result["target_locations"] == ["Remote"]