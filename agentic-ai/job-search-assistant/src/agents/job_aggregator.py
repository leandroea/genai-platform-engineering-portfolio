"""Job Aggregator Agent - Finds and manages job postings from Jooble API."""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name
from ..state.job_search_state import JobSearchState, Job
from ..tools.job_tools import search_jobs, deduplicate_jobs, score_job_match


class JobAggregatorAgent:
    """Agent that finds and manages job postings.
    
    This agent:
    - Receives target roles, locations, and preferences
    - Calls Jooble API to search for relevant job postings
    - Deduplicates results and filters by relevance
    - Returns structured job listings with metadata
    """
    
    def __init__(self):
        """Initialize the job aggregator agent."""
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.5,
        )
    
    def search_for_jobs(
        self,
        state: JobSearchState,
        roles: Optional[list[str]] = None,
        locations: Optional[list[str]] = None,
        max_jobs_per_role: int = 20
    ) -> dict:
        """Search for jobs matching user preferences.
        
        Args:
            state: Current JobSearchState
            roles: List of target job roles (uses state if not provided)
            locations: List of target locations (uses state if not provided)
            max_jobs_per_role: Maximum jobs to return per role
            
        Returns:
            Dictionary with jobs list and search metadata
        """
        # Use provided values or fall back to state
        if roles is None:
            roles = state.get("target_roles", [])
        
        if locations is None:
            locations = state.get("target_locations", [])
        
        if not roles:
            return {
                "success": False,
                "error": "No target roles specified. Please set job preferences first.",
                "jobs": [],
            }
        
        all_jobs = []
        search_queries = []
        
        # Search for each role/location combination
        for role in roles:
            for location in locations:
                if location.lower() == "any" or not location:
                    location_query = ""
                else:
                    location_query = location
                
                # Build search query
                query = f"{role} {location_query}".strip()
                search_queries.append(query)
                
                # Make the search
                result = search_jobs(query, location_query, max_jobs_per_role)
                
                if result.get("jobs"):
                    all_jobs.extend(result["jobs"])
        
        # Also do a general search with just the first role
        if roles:
            general_result = search_jobs(roles[0], "", max_jobs_per_role)
            if general_result.get("jobs"):
                all_jobs.extend(general_result["jobs"])
                search_queries.append(roles[0])
        
        # Deduplicate
        unique_jobs = deduplicate_jobs(all_jobs)
        
        # Score job matches
        user_profile = state.get("user_profile", {})
        for job in unique_jobs:
            job["match_score"] = score_job_match(job, user_profile, roles)
        
        # Sort by match score
        unique_jobs.sort(key=lambda j: j.get("match_score", 0), reverse=True)
        
        # Update state
        state["job_aggregator_output"] = {
            "jobs": unique_jobs,
            "search_queries_used": search_queries,
            "total_found": len(unique_jobs),
        }
        state["current_phase"] = "job_search"
        
        return {
            "success": True,
            "jobs": unique_jobs,
            "search_queries_used": search_queries,
            "total_found": len(unique_jobs),
        }
    
    def get_top_jobs(
        self,
        state: JobSearchState,
        count: int = 5,
        min_score: float = 0.5
    ) -> list[dict]:
        """Get top matching jobs from the aggregated results.
        
        Args:
            state: Current JobSearchState
            count: Maximum number of jobs to return
            min_score: Minimum match score threshold
            
        Returns:
            List of top job dictionaries
        """
        job_output = state.get("job_aggregator_output", {})
        jobs = job_output.get("jobs", [])
        
        # Filter by minimum score
        filtered_jobs = [j for j in jobs if j.get("match_score", 0) >= min_score]
        
        # Sort by score and return top N
        top_jobs = sorted(filtered_jobs, key=lambda j: j.get("match_score", 0), reverse=True)[:count]
        
        return top_jobs
    
    def filter_jobs_by_keyword(
        self,
        jobs: list[dict],
        keywords: list[str],
        match_all: bool = False
    ) -> list[dict]:
        """Filter jobs by keywords in title or description.
        
        Args:
            jobs: List of job dictionaries
            keywords: Keywords to search for
            match_all: If True, job must contain all keywords; if False, any keyword
            
        Returns:
            Filtered list of jobs
        """
        if match_all:
            return [
                job for job in jobs
                if all(kw.lower() in job.get("title", "").lower() or kw.lower() in job.get("description", "").lower()
                       for kw in keywords)
            ]
        else:
            return [
                job for job in jobs
                if any(kw.lower() in job.get("title", "").lower() or kw.lower() in job.get("description", "").lower()
                       for kw in keywords)
            ]
    
    def get_jobs_by_company(
        self,
        jobs: list[dict],
        company: str
    ) -> list[dict]:
        """Get all jobs from a specific company.
        
        Args:
            jobs: List of job dictionaries
            company: Company name to filter by
            
        Returns:
            List of jobs from that company
        """
        company_lower = company.lower()
        return [
            job for job in jobs
            if company_lower in job.get("company", "").lower()
        ]
    
    def get_jobs_by_location(
        self,
        jobs: list[dict],
        location: str
    ) -> list[dict]:
        """Get all jobs in a specific location.
        
        Args:
            jobs: List of job dictionaries
            location: Location to filter by
            
        Returns:
            List of jobs in that location
        """
        location_lower = location.lower()
        return [
            job for job in jobs
            if location_lower in job.get("location", "").lower()
        ]
    
    def rank_jobs_by_preference(
        self,
        state: JobSearchState,
        preference_weights: dict
    ) -> list[dict]:
        """Rank jobs based on weighted preferences.
        
        Args:
            state: Current JobSearchState
            preference_weights: Dict with weights for location, salary, company, etc.
                e.g., {"location": 0.4, "salary": 0.3, "company": 0.3}
                
        Returns:
            List of jobs sorted by weighted score
        """
        job_output = state.get("job_aggregator_output", {})
        jobs = job_output.get("jobs", [])
        
        # Default weights
        weights = {
            "location": 0.3,
            "salary": 0.3,
            "company": 0.2,
            "match_score": 0.2,
        }
        weights.update(preference_weights)
        
        target_locations = state.get("target_locations", [])
        
        ranked_jobs = []
        for job in jobs:
            score = 0.0
            
            # Location match
            job_location = job.get("location", "").lower()
            for loc in target_locations:
                if loc.lower() in job_location:
                    score += weights["location"]
                    break
            
            # Salary
            if job.get("salary_min"):
                score += weights["salary"]  # Simple binary - has salary or not
            
            # Match score
            score += job.get("match_score", 0) * weights["match_score"]
            
            # Company preference
            target_companies = state.get("target_companies", [])
            job_company = job.get("company", "").lower()
            for tc in target_companies:
                if tc.lower() in job_company:
                    score += weights["company"]
                    break
            
            job["weighted_score"] = score
            ranked_jobs.append(job)
        
        # Sort by weighted score
        ranked_jobs.sort(key=lambda j: j.get("weighted_score", 0), reverse=True)
        
        return ranked_jobs
    
    def get_search_summary(self, state: JobSearchState) -> str:
        """Get a summary of the job search results.
        
        Args:
            state: Current JobSearchState
            
        Returns:
            Summary string
        """
        job_output = state.get("job_aggregator_output", {})
        jobs = job_output.get("jobs", [])
        queries = job_output.get("search_queries_used", [])
        
        if not jobs:
            return "No jobs found yet. Run a job search first."
        
        summary_parts = [
            f"Job Search Summary:",
            f"- Total jobs found: {len(jobs)}",
            f"- Search queries used: {len(queries)}",
            f"- Average match score: {sum(j.get('match_score', 0) for j in jobs) / len(jobs):.2f}",
            f"",
        ]
        
        # Top 3 jobs
        top_jobs = self.get_top_jobs(state, count=3)
        if top_jobs:
            summary_parts.append("Top 3 Jobs:")
            for i, job in enumerate(top_jobs, 1):
                summary_parts.append(
                    f"  {i}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}"
                    f" (Score: {job.get('match_score', 0):.2f})"
                )
        
        return "\n".join(summary_parts)