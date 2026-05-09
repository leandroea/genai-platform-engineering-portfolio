"""Job Search Graph - LangGraph workflow for orchestrating agents."""

from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..state.job_search_state import JobSearchState, create_initial_state
from ..agents.supervisor import SupervisorAgent
from ..agents.resume_tailor import ResumeTailorAgent
from ..agents.cover_letter import CoverLetterAgent
from ..agents.job_aggregator import JobAggregatorAgent
from ..agents.resume_scorer import ResumeScorerAgent
from ..agents.application_form import ApplicationFormAgent
from ..agents.interview_coach import InterviewCoachAgent
from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name


class JobSearchGraph:
    """LangGraph workflow for the Job Search Assistant.
    
    This graph orchestrates all agents through a supervisor-coordinated workflow.
    The supervisor decides which agents to activate based on user requests.
    """
    
    def __init__(self):
        """Initialize the job search graph with all agents."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.7,
        )
        
        # Initialize agents
        self.supervisor = SupervisorAgent()
        self.resume_tailor = ResumeTailorAgent()
        self.cover_letter = CoverLetterAgent()
        self.job_aggregator = JobAggregatorAgent()
        self.resume_scorer = ResumeScorerAgent()
        self.application_form = ApplicationFormAgent()
        self.interview_coach = InterviewCoachAgent()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Define nodes
        def supervisor_node(state: JobSearchState) -> JobSearchState:
            """Supervisor node - decides next action."""
            return state
        
        def resume_tailor_node(state: JobSearchState) -> JobSearchState:
            """Resume tailor node - handles resume customization."""
            return state
        
        def cover_letter_node(state: JobSearchState) -> JobSearchState:
            """Cover letter node - generates cover letters."""
            return state
        
        def job_aggregator_node(state: JobSearchState) -> JobSearchState:
            """Job aggregator node - searches for jobs."""
            return state
        
        def resume_scorer_node(state: JobSearchState) -> JobSearchState:
            """Resume scorer node - scores resumes."""
            return state
        
        def application_form_node(state: JobSearchState) -> JobSearchState:
            """Application form node - helps with forms."""
            return state
        
        def interview_coach_node(state: JobSearchState) -> JobSearchState:
            """Interview coach node - prepares for interviews."""
            return state
        
        # Create graph
        workflow = StateGraph(JobSearchState)
        
        # Add nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("resume_tailor", resume_tailor_node)
        workflow.add_node("cover_letter", cover_letter_node)
        workflow.add_node("job_aggregator", job_aggregator_node)
        workflow.add_node("resume_scorer", resume_scorer_node)
        workflow.add_node("application_form", application_form_node)
        workflow.add_node("interview_coach", interview_coach_node)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add edges from supervisor (simplified - supervisor decides)
        workflow.add_edge("supervisor", END)
        
        return workflow.compile()
    
    def run(
        self,
        initial_state: Optional[JobSearchState] = None,
        user_request: str = ""
    ) -> JobSearchState:
        """Run the workflow.
        
        Args:
            initial_state: Optional initial state
            user_request: User's request to process
            
        Returns:
            Final JobSearchState after workflow completes
        """
        if initial_state is None:
            initial_state = create_initial_state()
        
        if user_request:
            # Process request through supervisor
            decision = self.supervisor.process_request(initial_state, user_request)
            initial_state["supervisor_decision"] = decision
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return result
    
    def execute_resume_tailor(
        self,
        state: JobSearchState,
        job_description: str,
        job_id: str,
        operation: str = "full_tailor"
    ) -> dict:
        """Execute resume tailoring workflow.
        
        Args:
            state: Current state
            job_description: Target job description
            job_id: Job identifier
            operation: Tailoring operation type
            
        Returns:
            Tailoring result
        """
        result = self.resume_tailor.tailor_resume(
            state, job_description, job_id, operation
        )
        
        # Update phase
        state["current_phase"] = "application"
        
        return result
    
    def execute_cover_letter(
        self,
        state: JobSearchState,
        job: dict,
        save_docx: bool = False
    ) -> dict:
        """Execute cover letter generation workflow.
        
        Args:
            state: Current state
            job: Job dictionary
            save_docx: Whether to save as DOCX
            
        Returns:
            Cover letter result
        """
        result = self.cover_letter.generate_cover_letter(state, job, save_docx)
        
        return result
    
    def execute_job_search(
        self,
        state: JobSearchState,
        roles: Optional[list[str]] = None,
        locations: Optional[list[str]] = None
    ) -> dict:
        """Execute job search workflow.
        
        Args:
            state: Current state
            roles: Target roles
            locations: Target locations
            
        Returns:
            Job search result
        """
        result = self.job_aggregator.search_for_jobs(
            state, roles, locations
        )
        
        state["current_phase"] = "job_search"
        
        return result
    
    def execute_resume_score(
        self,
        state: JobSearchState,
        job_description: str,
        job_id: str = ""
    ) -> dict:
        """Execute resume scoring workflow.
        
        Args:
            state: Current state
            job_description: Job description to score against
            job_id: Optional job identifier
            
        Returns:
            Score result
        """
        result = self.resume_scorer.score_resume(
            state, job_description, job_id
        )
        
        return result
    
    def execute_application_form(
        self,
        state: JobSearchState,
        fields: list[dict],
        platform: str = "generic"
    ) -> dict:
        """Execute application form filling workflow.
        
        Args:
            state: Current state
            fields: List of field dictionaries
            platform: Platform name
            
        Returns:
            Form filling result
        """
        result = self.application_form.fill_form_fields(
            state, fields, platform
        )
        
        return result
    
    def execute_interview_prep(
        self,
        state: JobSearchState,
        job_description: str,
        company: str = ""
    ) -> dict:
        """Execute interview preparation workflow.
        
        Args:
            state: Current state
            job_description: Target job description
            company: Company name
            
        Returns:
            Interview prep result
        """
        result = self.interview_coach.prepare_interview(
            state, job_description, company
        )
        
        state["current_phase"] = "interview_prep"
        
        return result


def run_job_search(
    target_roles: list[str],
    target_locations: list[str],
    initial_state: Optional[JobSearchState] = None
) -> JobSearchState:
    """Convenience function to run a job search workflow.
    
    Args:
        target_roles: List of target job roles
        target_locations: List of target locations
        initial_state: Optional initial state
        
    Returns:
        Final JobSearchState
    """
    if initial_state is None:
        initial_state = create_initial_state()
    
    # Set search parameters
    initial_state["target_roles"] = target_roles
    initial_state["target_locations"] = target_locations
    
    # Create graph and run
    graph = JobSearchGraph()
    
    # Execute job search
    graph.execute_job_search(initial_state)
    
    return initial_state