"""JobSearchState - Shared state for all agents in the job search system."""

from typing import TypedDict, Optional, Literal
from datetime import datetime


class JobSearchState(TypedDict):
    """Shared state passed between supervisor and all agents.
    
    This is the central data structure that all agents read from and write to.
    The supervisor coordinates agents by querying and updating this state.
    """
    
    # User Profile
    user_profile: dict              # Name, email, phone, location, experience summary
    original_resume_path: str        # Path to uploaded resume (PDF or DOCX)
    original_resume_text: str        # Extracted text from resume
    
    # Job Search Parameters
    target_roles: list[str]          # e.g., ["Product Manager", "Senior Developer"]
    target_locations: list[str]      # e.g., ["Remote", "New York"]
    target_companies: list[str]     # Optional: specific companies to target
    
    # Local Job Descriptions
    local_job_descriptions: list[dict]  # [{id, title, company, description, path}]
    selected_job_description_id: str    # ID of currently selected job description
    
    # Agent Outputs (each agent writes here)
    resume_tailor_output: dict       # {tailored_resume_docx_path, job_id}
    cover_letter_output: dict       # {cover_letter_text, job_id}
    job_aggregator_output: dict     # {jobs: list[job], search_query_used}
    resume_score_output: dict        # {score, breakdown, recommendations}
    application_form_output: dict    # {field_answers, platform_type, job_id}
    interview_coach_output: dict    # {questions: list, answers: list, mock_session}
    
    # Supervisor Decision State
    current_phase: str               # "job_search", "application", "interview_prep", "idle"
    pending_tasks: list[str]         # Tasks the supervisor has assigned
    completed_tasks: list[str]       # Tasks completed by subordinates
    blockers: list[str]              # Issues that need attention
    supervisor_decision: str        # Current decision/reasoning from supervisor


class Job:
    """Job posting data model."""
    
    def __init__(
        self,
        title: str,
        company: str,
        location: str,
        description: str,
        url: str,
        salary_min: Optional[int] = None,
        salary_max: Optional[int] = None,
        posted_date: Optional[str] = None,
        source: str = "jooble",
        match_score: float = 0.0,
        applied: bool = False,
        applied_date: Optional[datetime] = None,
    ):
        import uuid
        self.id = str(uuid.uuid4())
        self.title = title
        self.company = company
        self.location = location
        self.description = description
        self.url = url
        self.salary_min = salary_min
        self.salary_max = salary_max
        self.posted_date = posted_date
        self.source = source
        self.match_score = match_score
        self.applied = applied
        self.applied_date = applied_date
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "url": self.url,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "posted_date": self.posted_date,
            "source": self.source,
            "match_score": self.match_score,
            "applied": self.applied,
            "applied_date": self.applied_date.isoformat() if self.applied_date else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create Job from dictionary."""
        job = cls(
            title=data["title"],
            company=data["company"],
            location=data["location"],
            description=data["description"],
            url=data["url"],
            salary_min=data.get("salary_min"),
            salary_max=data.get("salary_max"),
            posted_date=data.get("posted_date"),
            source=data.get("source", "jooble"),
            match_score=data.get("match_score", 0.0),
            applied=data.get("applied", False),
            applied_date=datetime.fromisoformat(data["applied_date"]) if data.get("applied_date") else None,
        )
        if "id" in data:
            job.id = data["id"]
        return job


class Application:
    """Application tracking data model."""
    
    def __init__(
        self,
        job_id: str,
        tailored_resume_path: str,
        cover_letter_path: Optional[str] = None,
        status: Literal["pending", "applied", "interview", "rejected", "offer"] = "pending",
        applied_date: Optional[datetime] = None,
        notes: str = "",
        interview_date: Optional[datetime] = None,
        feedback: str = "",
    ):
        import uuid
        self.id = str(uuid.uuid4())
        self.job_id = job_id
        self.tailored_resume_path = tailored_resume_path
        self.cover_letter_path = cover_letter_path
        self.status = status
        self.applied_date = applied_date
        self.notes = notes
        self.interview_date = interview_date
        self.feedback = feedback
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "tailored_resume_path": self.tailored_resume_path,
            "cover_letter_path": self.cover_letter_path,
            "status": self.status,
            "applied_date": self.applied_date.isoformat() if self.applied_date else None,
            "notes": self.notes,
            "interview_date": self.interview_date.isoformat() if self.interview_date else None,
            "feedback": self.feedback,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Application":
        """Create Application from dictionary."""
        app = cls(
            job_id=data["job_id"],
            tailored_resume_path=data["tailored_resume_path"],
            cover_letter_path=data.get("cover_letter_path"),
            status=data.get("status", "pending"),
            applied_date=datetime.fromisoformat(data["applied_date"]) if data.get("applied_date") else None,
            notes=data.get("notes", ""),
            interview_date=datetime.fromisoformat(data["interview_date"]) if data.get("interview_date") else None,
            feedback=data.get("feedback", ""),
        )
        if "id" in data:
            app.id = data["id"]
        return app


def create_initial_state() -> JobSearchState:
    """Create an initial JobSearchState with default values."""
    return JobSearchState(
        user_profile={},
        original_resume_path="",
        original_resume_text="",
        target_roles=[],
        target_locations=[],
        target_companies=[],
        local_job_descriptions=[],
        selected_job_description_id="",
        resume_tailor_output={},
        cover_letter_output={},
        job_aggregator_output={},
        resume_score_output={},
        application_form_output={},
        interview_coach_output={},
        current_phase="idle",
        pending_tasks=[],
        completed_tasks=[],
        blockers=[],
        supervisor_decision="",
    )