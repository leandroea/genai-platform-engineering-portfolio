"""Job search tools using Jooble API."""

import os
import time
import logging
from typing import Optional
from pathlib import Path

import requests
from ..utils.config import get_jooble_api_key, get_max_jobs_per_search
from ..state.job_search_state import Job

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Jooble API configuration
JOOBLE_API_BASE = "https://www.jooble.org/api"
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds


def search_jobs(
    keywords: str,
    location: Optional[str] = None,
    max_jobs: Optional[int] = None,
) -> dict:
    """Search for jobs using Jooble API.
    
    Args:
        keywords: Job search keywords (e.g., "Python Developer")
        location: Job location (e.g., "Remote", "New York")
        max_jobs: Maximum number of jobs to return (default from config)
        
    Returns:
        Dictionary with jobs list and search metadata
    """
    if max_jobs is None:
        max_jobs = get_max_jobs_per_search()
    
    api_key = get_jooble_api_key()
    
    # Build request payload
    payload = {
        "keywords": keywords,
        "location": location or "",
        "page": 1,
        "resultPerPage": min(max_jobs, 20),  # Jooble limit
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Make API request with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            response = _make_request(f"{JOOBLE_API_BASE}/{api_key}", headers, payload)
            
            if response.status_code == 429:
                # Rate limit hit
                logger.warning(f"Jooble API rate limit hit. Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
                continue
            
            if response.status_code != 200:
                logger.error(f"Jooble API error: {response.status_code}")
                return {"jobs": [], "error": f"API error: {response.status_code}", "search_query_used": keywords}
            
            data = response.json()
            jobs = _parse_jooble_response(data)
            
            return {
                "jobs": jobs,
                "search_query_used": keywords,
                "location_used": location or "any",
                "total_found": len(jobs),
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"jobs": [], "error": str(e), "search_query_used": keywords}
    
    return {"jobs": [], "error": "Max retries exceeded", "search_query_used": keywords}


def _make_request(url: str, headers: dict, payload: dict) -> requests.Response:
    """Make HTTP request to Jooble API."""
    return requests.post(url, json=payload, headers=headers, timeout=30)


def _parse_jooble_response(data: dict) -> list[dict]:
    """Parse Jooble API response into standardized job format.
    
    Args:
        data: Raw API response
        
    Returns:
        List of job dictionaries
    """
    jobs = []
    
    # Jooble returns jobs in "jobs" key
    raw_jobs = data.get("jobs", [])
    
    for job_data in raw_jobs:
        try:
            job = {
                "title": job_data.get("title", ""),
                "company": job_data.get("company", ""),
                "location": job_data.get("location", ""),
                "description": job_data.get("snippet", job_data.get("description", "")),
                "url": job_data.get("link", ""),
                "salary_min": _parse_salary(job_data.get("salary", "")),
                "salary_max": None,
                "posted_date": job_data.get("updated", ""),
                "source": "jooble",
            }
            jobs.append(job)
        except Exception as e:
            logger.warning(f"Failed to parse job: {e}")
            continue
    
    return jobs


def _parse_salary(salary_str: str) -> Optional[int]:
    """Parse salary string into minimum numeric value.
    
    Args:
        salary_str: Salary string like "$120k - $150k"
        
    Returns:
        Minimum salary as integer, or None if parsing fails
    """
    if not salary_str:
        return None
    
    import re
    
    # Try to find numbers in the string
    # Handle formats like "$120,000", "120k", "$120K - $150K"
    numbers = re.findall(r"[\d,]+", salary_str.replace("k", "000").replace("K", "000"))
    
    if numbers:
        try:
            # Get the first number and remove commas
            first_num = numbers[0].replace(",", "")
            return int(first_num)
        except ValueError:
            pass
    
    return None


def deduplicate_jobs(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs based on title, company, and location.
    
    Args:
        jobs: List of job dictionaries
        
    Returns:
        Deduplicated list of jobs
    """
    seen = set()
    unique_jobs = []
    
    for job in jobs:
        key = (job.get("title", ""), job.get("company", ""), job.get("location", ""))
        
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)
    
    return unique_jobs


def score_job_match(job: dict, user_profile: dict, target_roles: list[str]) -> float:
    """Score how well a job matches user preferences.
    
    Args:
        job: Job dictionary
        user_profile: User profile information
        target_roles: Target job roles
        
    Returns:
        Match score between 0.0 and 1.0
    """
    score = 0.0
    
    title_lower = job.get("title", "").lower()
    
    # Check if title matches target roles
    for role in target_roles:
        role_lower = role.lower()
        if role_lower in title_lower:
            score += 0.5
            break
        # Partial match
        words = role_lower.split()
        if any(word in title_lower for word in words if len(word) > 3):
            score += 0.2
    
    # Location match bonus
    location = job.get("location", "").lower()
    user_locations = user_profile.get("preferred_locations", [])
    if any(loc.lower() in location for loc in user_locations):
        score += 0.2
    
    # Has salary info bonus
    if job.get("salary_min"):
        score += 0.1
    
    # Normalize to max 1.0
    return min(score, 1.0)


def save_jobs_to_file(jobs: list[dict], file_path: str) -> None:
    """Save jobs list to a JSON file.
    
    Args:
        jobs: List of job dictionaries
        file_path: Path to save the JSON file
    """
    import json
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"jobs": jobs, "count": len(jobs)}, f, indent=2, ensure_ascii=False)


def load_jobs_from_file(file_path: str) -> list[dict]:
    """Load jobs list from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of job dictionaries
    """
    import json
    
    if not Path(file_path).exists():
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("jobs", [])