"""CLI I/O helper functions for user interaction."""

import sys
from typing import Optional, Any
from pathlib import Path
import json

from ..state.job_search_state import Job, Application


def print_header(title: str) -> None:
    """Print a formatted header."""
    border = "=" * 60
    print(f"\n+{border}+")
    print(f"|{title:^60}|")
    print(f"+{border}+\n")


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print(f"\n>> {title}")
    print("-" * 40)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"[OK] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"[ERROR] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"[WARNING] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"[INFO] {message}")


def print_job(job: dict, index: int = 1) -> None:
    """Print a job listing in a formatted way.
    
    Args:
        job: Job dictionary
        index: Job index number
    """
    print(f"\n  [{index}] {job.get('title', 'N/A')}")
    print(f"      Company: {job.get('company', 'N/A')}")
    print(f"      Location: {job.get('location', 'N/A')}")
    
    salary = job.get('salary_min')
    if salary:
        print(f"      Salary: ${salary:,}+")
    
    # Truncate description
    desc = job.get('description', '')[:100]
    if desc:
        print(f"      Description: {desc}...")
    
    print(f"      URL: {job.get('url', 'N/A')[:50]}...")


def print_jobs(jobs: list[dict], limit: Optional[int] = None) -> None:
    """Print a list of jobs.
    
    Args:
        jobs: List of job dictionaries
        limit: Optional limit on number of jobs to print
    """
    if not jobs:
        print_info("No jobs found.")
        return
    
    jobs_to_show = jobs[:limit] if limit else jobs
    
    print_subheader(f"Jobs Found ({len(jobs_to_show)} of {len(jobs)} total)")
    
    for i, job in enumerate(jobs_to_show, 1):
        print_job(job, i)
    
    if limit and len(jobs) > limit:
        print(f"\n  ... and {len(jobs) - limit} more jobs")


def print_score(score_data: dict) -> None:
    """Print resume score results.
    
    Args:
        score_data: Score dictionary with score, breakdown, recommendations
    """
    print_subheader("Resume Score Results")
    
    score = score_data.get("score", 0)
    print(f"\n  Overall Score: {score}/100")
    
    breakdown = score_data.get("breakdown", {})
    if breakdown:
        print("\n  Breakdown:")
        for category, value in breakdown.items():
            print(f"    - {category}: {value}/25")
    
    recommendations = score_data.get("recommendations", [])
    if recommendations:
        print("\n  Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")


def print_menu(options: list[tuple[str, str]]) -> None:
    """Print a numbered menu.
    
    Args:
        options: List of (number, description) tuples
    """
    print("\n" + "=" * 40)
    for num, desc in options:
        print(f"  {num}. {desc}")
    print("=" * 40)


def get_input(prompt: str, required: bool = True) -> str:
    """Get user input with optional validation.
    
    Args:
        prompt: Input prompt text
        required: Whether input is required
        
    Returns:
        User input string
    """
    while True:
        user_input = input(f"\n{prompt}: ").strip()
        
        if not user_input and required:
            print_warning("This field is required. Please enter a value.")
            continue
        
        return user_input


def get_yes_no(prompt: str) -> bool:
    """Get a yes/no answer from user.
    
    Args:
        prompt: Question prompt
        
    Returns:
        True for yes, False for no
    """
    while True:
        answer = input(f"\n{prompt} (yes/no): ").strip().lower()
        
        if answer in ["yes", "y"]:
            return True
        elif answer in ["no", "n"]:
            return False
        else:
            print_warning("Please enter 'yes' or 'no'.")


def confirm_action(action: str) -> bool:
    """Ask user to confirm an action.
    
    Args:
        action: Description of the action
        
    Returns:
        True if confirmed, False otherwise
    """
    return get_yes_no(f"Confirm {action}")


def print_application_status(application: Application) -> None:
    """Print application status information.
    
    Args:
        application: Application object
    """
    status_icons = {
        "pending": "[PENDING]",
        "applied": "[APPLIED]",
        "interview": "[INTERVIEW]",
        "rejected": "[REJECTED]",
        "offer": "[OFFER]",
    }
    
    icon = status_icons.get(application.status, "[UNKNOWN]")
    print(f"\n  {icon} Application Status: {application.status.upper()}")
    print(f"     Job ID: {application.job_id}")
    print(f"     Resume: {application.tailored_resume_path}")
    
    if application.applied_date:
        print(f"     Applied: {application.applied_date.strftime('%Y-%m-%d')}")
    
    if application.notes:
        print(f"     Notes: {application.notes}")


def save_state(state: dict, file_path: str) -> bool:
    """Save application state to a JSON file.
    
    Args:
        state: State dictionary to save
        file_path: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print_error(f"Failed to save state: {e}")
        return False


def load_state(file_path: str) -> Optional[dict]:
    """Load application state from a JSON file.
    
    Args:
        file_path: Path to the state file
        
    Returns:
        State dictionary or None if not found
    """
    if not Path(file_path).exists():
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print_error(f"Failed to load state: {e}")
        return None


def clear_screen() -> None:
    """Clear the terminal screen."""
    print("\n" * 100)


def print_interview_questions(questions_data: dict) -> None:
    """Print interview questions and answers.
    
    Args:
        questions_data: Dictionary with questions and answers
    """
    print_subheader("Interview Preparation")
    
    technical = questions_data.get("technical_questions", [])
    if technical:
        print("\n  Technical Questions:")
        for i, q in enumerate(technical, 1):
            print(f"    {i}. {q}")
    
    behavioral = questions_data.get("behavioral_questions", [])
    if behavioral:
        print("\n  Behavioral Questions (STAR Method):")
        for i, q in enumerate(behavioral, 1):
            print(f"    {i}. {q}")
    
    answers = questions_data.get("sample_answers", [])
    if answers:
        print("\n  Sample Answers:")
        for i, a in enumerate(answers, 1):
            # Truncate long answers
            a_display = a[:200] + "..." if len(a) > 200 else a
            print(f"    {i}. {a_display}")


def print_job_description(job_desc: dict, index: int = 1) -> None:
    """Print a job description in a formatted way.
    
    Args:
        job_desc: Job description dictionary
        index: Job description index number
    """
    print(f"\n  [{index}] {job_desc.get('title', 'N/A')}")
    if job_desc.get('company'):
        print(f"      Company: {job_desc.get('company', 'N/A')}")
    if job_desc.get('location'):
        print(f"      Location: {job_desc.get('location', 'N/A')}")
    
    # Truncate description
    desc = job_desc.get('description', '')[:150]
    if desc:
        print(f"      Description: {desc}...")
    
    print(f"      ID: {job_desc.get('id', 'N/A')}")


def print_job_descriptions(job_descriptions: list[dict], limit: Optional[int] = None) -> None:
    """Print a list of job descriptions.
    
    Args:
        job_descriptions: List of job description dictionaries
        limit: Optional limit on number to print
    """
    if not job_descriptions:
        print_info("No job descriptions found.")
        return
    
    to_show = job_descriptions[:limit] if limit else job_descriptions
    
    print_subheader(f"Job Descriptions ({len(to_show)} of {len(job_descriptions)} total)")
    
    for i, jd in enumerate(to_show, 1):
        print_job_description(jd, i)
    
    if limit and len(job_descriptions) > limit:
        print(f"\n  ... and {len(job_descriptions) - limit} more job descriptions")