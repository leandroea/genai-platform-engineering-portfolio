"""Conversational agent - Natural language interface that routes to agents."""

import logging
import re
from typing import Optional
from pathlib import Path

# Suppress httpx and httpcore logging to keep output clean
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from ..agents.supervisor import SupervisorAgent
from ..agents.resume_tailor import ResumeTailorAgent
from ..agents.cover_letter import CoverLetterAgent
from ..agents.job_aggregator import JobAggregatorAgent
from ..agents.resume_scorer import ResumeScorerAgent
from ..agents.application_form import ApplicationFormAgent
from ..agents.interview_coach import InterviewCoachAgent
from ..state.job_search_state import JobSearchState
from ..tools.resume_tools import extract_resume_text, validate_resume_file
from ..tools.job_description_tools import (
    upload_job_description_file, validate_job_description_file, list_job_descriptions
)
from .intent_classifier import IntentClassifier, Intent
from .conversation_context import ConversationContext


class ConversationalAgent:
    """Natural language interface for the Job Search Assistant.
    
    This agent provides a conversational UI that:
    - Interprets natural language commands
    - Tracks conversation context for multi-turn dialogues
    - Routes requests to appropriate specialized agents
    - Asks follow-up questions when needed
    """
    
    # Prompt for explaining capabilities
    CAPABILITIES = """
I can help you with your job search. Here's what I can do:

RESUME MANAGEMENT:
- Upload your resume (PDF or DOCX)
- Tailor your resume for specific jobs
- Score your resume against job descriptions

JOB SEARCH:
- Search for jobs based on your preferences
- Manage job descriptions you've saved

APPLICATION HELP:
- Generate cover letters for specific jobs
- Help fill out application forms

INTERVIEW PREP:
- Generate interview questions
- Provide STAR method answer templates

EXAMPLES:
- "upload my resume from resume.pdf"
- "find me senior developer jobs in NYC"
- "score my resume against the senior role"
- "help me write a cover letter"
- "prepare me for an interview"
"""

    def __init__(self, state: JobSearchState):
        """Initialize the conversational agent.
        
        Args:
            state: The JobSearchState to operate on
        """
        self.state = state
        self.classifier = IntentClassifier()
        self.context = ConversationContext()
        
        # Initialize all agents
        self._init_agents()
        
        # Intent handlers - maps intents to handler methods
        self._intent_handlers = {
            Intent.GREETING: self._handle_greeting,
            Intent.HELP: self._handle_help,
            Intent.UPLOAD_RESUME: self._handle_upload_resume,
            Intent.SET_PREFERENCES: self._handle_set_preferences,
            Intent.SEARCH_JOBS: self._handle_search_jobs,
            Intent.SCORE_RESUME: self._handle_score_resume,
            Intent.TAILOR_RESUME: self._handle_tailor_resume,
            Intent.GENERATE_COVER_LETTER: self._handle_generate_cover_letter,
            Intent.FILL_FORM: self._handle_fill_form,
            Intent.INTERVIEW_PREP: self._handle_interview_prep,
            Intent.VIEW_JOBS: self._handle_view_jobs,
            Intent.UPLOAD_JOB_DESCRIPTION: self._handle_upload_job_description,
            Intent.ADD_JOB_DESCRIPTION: self._handle_add_job_description,
            Intent.MANAGE_JOB_DESCRIPTIONS: self._handle_manage_job_descriptions,
            Intent.EXIT: self._handle_exit,
        }
    
    def _init_agents(self) -> None:
        """Initialize all specialized agents."""
        try:
            self.supervisor = SupervisorAgent()
            self.resume_tailor = ResumeTailorAgent()
            self.cover_letter = CoverLetterAgent()
            self.job_aggregator = JobAggregatorAgent()
            self.resume_scorer = ResumeScorerAgent()
            self.application_form = ApplicationFormAgent()
            self.interview_coach = InterviewCoachAgent()
        except ValueError as e:
            # API keys not configured - will handle gracefully
            self.supervisor = None
            self.resume_tailor = None
            self.cover_letter = None
            self.job_aggregator = None
            self.resume_scorer = None
            self.application_form = None
            self.interview_coach = None
    
    def process_message(self, user_input: str) -> str:
        """Process a user message and return a response.
        
        Uses the SupervisorAgent's LLM to understand intent naturally,
        rather than keyword-based regex pattern matching.
        
        Args:
            user_input: The user's natural language input
            
        Returns:
            The bot's response
        """
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you rephrase?"
        
        # Use SupervisorAgent's LLM-based chat capability
        if self.supervisor:
            try:
                result = self.supervisor.chat(self.state, user_input)
                
                # Apply any state updates returned
                state_updates = result.get("state_updates", {})
                for key, value in state_updates.items():
                    self.state[key] = value
                
                return result.get("response", "I'm not sure how to help with that.")
            except Exception as e:
                # Fall back to basic response if supervisor fails
                return f"I had trouble understanding that: {e}"
        
        return "I'm not available right now. Please try again later."
    
    def _handle_greeting(self, user_input: str, entities: dict) -> str:
        """Handle a greeting."""
        has_resume = bool(self.state.get("original_resume_text"))
        has_preferences = bool(self.state.get("target_roles"))
        job_descriptions = self.state.get("local_job_descriptions", [])
        has_job_description = len(job_descriptions) > 0
        
        response = "Hello! I'm your job search assistant. "
        
        if has_resume and has_preferences:
            response += "I see you have a resume and job preferences set. "
            response += "Would you like me to search for jobs, or is there something else I can help with?"
        elif has_resume:
            response += "I see you've uploaded a resume. "
            if has_job_description:
                response += f"I also see you have {len(job_descriptions)} job description(s) loaded. "
                response += "Would you like to score your resume against one, or set job preferences?"
            else:
                response += "Would you like to set job preferences so I can find relevant opportunities?"
        elif has_preferences:
            response += "I see you have job preferences set. "
            response += "Would you like to upload your resume to get personalized job matches?"
        elif has_job_description:
            response += f"I see you have {len(job_descriptions)} job description(s) loaded. "
            response += "Would you like to upload your resume so we can work on applications?"
        else:
            response += f"I can help you with your job search. {self.CAPABILITIES}"
        
        return response
    
    def _handle_help(self, user_input: str, entities: dict) -> str:
        """Handle help request."""
        return self.CAPABILITIES
    
    def _handle_upload_resume(self, user_input: str, entities: dict) -> str:
        """Handle resume upload request."""
        # Check if user provided a file path in the message
        file_path = self._extract_file_path(user_input, [".pdf", ".docx"])
        
        if not file_path:
            self.context.set_pending_intent(Intent.UPLOAD_RESUME.value)
            return ("I'll help you upload your resume. "
                    "Please provide the path to your resume file (PDF or DOCX).")
        
        # Validate file
        is_valid, error = validate_resume_file(file_path)
        if not is_valid:
            return f"I couldn't upload the resume: {error}"
        
        try:
            text = extract_resume_text(file_path)
            words = len(text.split())
            
            self.state["original_resume_path"] = file_path
            self.state["original_resume_text"] = text
            
            self.context.clear_pending_intent()
            
            response = (f"Great! I've uploaded and parsed your resume "
                       f"({words} words). ")
            
            # Check if we can do more
            if self.state.get("target_roles"):
                response += ("I can now help you find matching jobs, "
                            "score your resume against positions, "
                            "or tailor your resume for specific roles.")
            else:
                response += ("To get personalized job matches, "
                             "please set your target roles and locations.")
            
            return response
            
        except Exception as e:
            return f"I had trouble uploading your resume: {e}"
    
    def _handle_set_preferences(self, user_input: str, entities: dict) -> str:
        """Handle setting job preferences."""
        # Extract potential preferences from entities
        locations = entities.get("locations", [])
        job_titles = entities.get("job_titles", [])
        
        if locations or job_titles:
            # User provided some preferences inline
            if locations:
                self.state["target_locations"] = locations
                self.state["user_profile"]["target_locations"] = locations
            
            if job_titles:
                self.state["target_roles"] = job_titles
                self.state["user_profile"]["target_roles"] = job_titles
            
            return (f"I've set your preferences: "
                    f"Roles: {', '.join(job_titles) if job_titles else 'not specified'}, "
                    f"Locations: {', '.join(locations) if locations else 'not specified'}. "
                    f"Would you like me to search for jobs matching these preferences?")
        else:
            # Need more info - ask for it
            self.context.set_pending_intent(Intent.SET_PREFERENCES.value)
            return ("I'd be happy to help set your job preferences. "
                    "What type of role are you looking for? "
                    "And what locations are you interested in?")
    
    def _handle_search_jobs(self, user_input: str, entities: dict) -> str:
        """Handle job search request."""
        # Check if we have the necessary info
        roles = self.state.get("target_roles", [])
        locations = self.state.get("target_locations", [])
        
        if not roles:
            self.context.set_pending_intent(Intent.SEARCH_JOBS.value)
            return ("I need to know what type of jobs you're looking for. "
                    "What roles or positions interest you?")
        
        if not self.job_aggregator:
            return "Job search is not available - API keys not configured."
        
        try:
            result = self.job_aggregator.search_for_jobs(self.state)
            
            if result.get("success"):
                jobs = result.get("jobs", [])
                if jobs:
                    response = f"I found {len(jobs)} jobs matching your preferences!\n\n"
                    response += "Here are the top 5:\n"
                    for i, job in enumerate(jobs[:5], 1):
                        title = job.get("title", "Unknown")
                        company = job.get("company", "Unknown")
                        location = job.get("location", "")
                        response += f"{i}. {title} at {company}"
                        if location:
                            response += f" ({location})"
                        response += "\n"
                    
                    response += "\nWould you like me to:\n"
                    response += "- Score your resume against any of these jobs\n"
                    response += "- Generate a tailored resume for a specific role\n"
                    response += "- Create a cover letter for a position"
                    
                    self.context.set_pending_intent(Intent.SCORE_RESUME.value)
                    return response
                else:
                    return "I didn't find any jobs matching your preferences. Try adjusting your target roles or locations."
            else:
                return f"Job search failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"I had trouble searching for jobs: {e}"
    
    def _handle_score_resume(self, user_input: str, entities: dict) -> str:
        """Handle resume scoring request."""
        if not self.state.get("original_resume_text"):
            self.context.set_pending_intent(Intent.SCORE_RESUME.value)
            return ("I need your resume to score it. "
                    "Would you like to upload your resume first?")
        
        # Check for jobs to score against
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        local_jds = list_job_descriptions()
        
        if not jobs and not local_jds:
            return ("I don't have any job descriptions to score against. "
                    "Would you like me to search for jobs first?")
        
        # Get job description
        job_desc = ""
        job_id = ""
        
        if jobs:
            job = jobs[0]  # Use first job
            job_desc = job.get("description", "")
            job_id = job.get("id", "")
        elif local_jds:
            from ..tools.job_description_tools import read_job_description
            jd = read_job_description(local_jds[0]["id"])
            if jd:
                job_desc = jd.get("description", "")
                job_id = local_jds[0]["id"]
        
        if not job_desc:
            return "I couldn't find a job description to score against."
        
        if not self.resume_scorer:
            return "Resume scoring is not available - API keys not configured."
        
        try:
            result = self.resume_scorer.score_resume(self.state, job_desc, job_id)
            
            if result.get("success"):
                score = result.get("score", 0)
                summary = result.get("summary", "")
                
                response = f"Your resume scored {score}/100 for this position.\n\n"
                response += f"Summary: {summary}\n\n"
                
                if score >= 70:
                    response += "This is a good match! I'd recommend tailoring your resume and applying."
                elif score >= 50:
                    response += "There's potential here. Consider tailoring your resume before applying."
                else:
                    response += "The match seems low. You might want to significantly tailor your resume or consider different positions."
                
                return response
            else:
                return f"Scoring failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"I had trouble scoring your resume: {e}"
    
    def _handle_tailor_resume(self, user_input: str, entities: dict) -> str:
        """Handle resume tailoring request."""
        if not self.state.get("original_resume_text"):
            return ("I need your resume to tailor it. "
                    "Please upload your resume first.")
        
        return ("To tailor your resume, I need to know which job you want to target. "
                "Would you like me to search for jobs, or do you have a specific job description?")
    
    def _handle_generate_cover_letter(self, user_input: str, entities: dict) -> str:
        """Handle cover letter generation request."""
        if not self.state.get("original_resume_text"):
            return ("I need your resume to generate a cover letter. "
                    "Please upload your resume first.")
        
        return ("I'll help you create a cover letter. "
                "Which job would you like to write it for?")
    
    def _handle_fill_form(self, user_input: str, entities: dict) -> str:
        """Handle application form assistance request."""
        if not self.state.get("original_resume_text"):
            return ("I need your resume to help fill out forms. "
                    "Please upload your resume first.")
        
        return ("I can help you fill out application forms. "
                "Which platform are you applying on (LinkedIn, Greenhouse, etc.)?")
    
    def _handle_interview_prep(self, user_input: str, entities: dict) -> str:
        """Handle interview preparation request."""
        return ("I'll help you prepare for your interview. "
                "Which job role are you interviewing for? "
                "Or would you like general interview prep?")
    
    def _handle_view_jobs(self, user_input: str, entities: dict) -> str:
        """Handle viewing saved jobs."""
        jobs = self.state.get("job_aggregator_output", {}).get("jobs", [])
        
        if not jobs:
            return ("You don't have any saved jobs yet. "
                    "Would you like me to search for jobs?")
        
        response = f"You have {len(jobs)} saved jobs:\n\n"
        for i, job in enumerate(jobs[:10], 1):
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            response += f"{i}. {title} at {company}\n"
        
        return response
    
    def _handle_upload_job_description(self, user_input: str, entities: dict) -> str:
        """Handle job description upload."""
        file_path = self._extract_file_path(user_input, [".pdf", ".docx", ".txt", ".md"])
        
        if not file_path:
            self.context.set_pending_intent(Intent.UPLOAD_JOB_DESCRIPTION.value)
            return ("I'll help you upload a job description. "
                    "Please provide the path to the file (PDF, DOCX, TXT, or MD).")
        
        is_valid, error = validate_job_description_file(file_path)
        if not is_valid:
            return f"I couldn't upload the job description: {error}"
        
        try:
            result = upload_job_description_file(file_path)
            
            self.context.clear_pending_intent()
            
            # Update state with the new job description
            job_descriptions = list_job_descriptions()
            self.state["local_job_descriptions"] = job_descriptions
            
            # Find the newly added job description
            new_jd = None
            for jd in job_descriptions:
                if result.get("title") in jd.get("title", "") or result.get("path") == jd.get("path"):
                    new_jd = jd
                    break
            
            if new_jd:
                self.state["selected_job_description_id"] = new_jd.get("id", "")
            
            return (f"Job description uploaded successfully!\n"
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"Company: {result.get('company', 'N/A')}\n\n"
                    f"Would you like me to score your resume against this job?")
            
        except Exception as e:
            return f"I had trouble uploading the job description: {e}"
    
    def _handle_add_job_description(self, user_input: str, entities: dict) -> str:
        """Handle adding a job description manually."""
        return ("To add a job description, I'll need you to provide:\n"
                "- Job title\n"
                "- Company name\n"
                "- Job description text\n\n"
                "Would you like to upload a file instead (option C)?")
    
    def _handle_manage_job_descriptions(self, user_input: str, entities: dict) -> str:
        """Handle managing job descriptions."""
        job_descriptions = list_job_descriptions()
        
        if not job_descriptions:
            return ("You don't have any saved job descriptions. "
                    "Would you like to upload one?")
        
        response = f"You have {len(job_descriptions)} saved job descriptions:\n\n"
        for i, jd in enumerate(job_descriptions[:5], 1):
            title = jd.get("title", jd.get("id", "Unknown"))
            response += f"{i}. {title}\n"
        
        return response
    
    def _handle_exit(self, user_input: str, entities: dict) -> str:
        """Handle exit request."""
        self.context.reset()
        return ("Goodbye! Your job search progress has been saved. "
                "Type 'start' to begin a new conversation.")
    
    def _handle_unknown(self, user_input: str, entities: dict) -> str:
        """Handle unrecognized intents by passing to supervisor for natural response."""
        # Try to pass to supervisor for interpretation
        if self.supervisor:
            try:
                response = self.supervisor.process_request(self.state, user_input)
                # Clean up the response - remove thinking tags and coordination messages
                response = self._clean_response(response)
                return response
            except Exception:
                pass
        
        return ("I'm not sure I understood that. "
                f"Here's what I can help with:\n{self.CAPABILITIES}")
    
    def _clean_response(self, response: str) -> str:
        """Remove thinking tags and clean up LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Cleaned response without thinking tags
        """
        import re
        # Remove <think>... tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*?<\/think>', '', cleaned, flags=re.DOTALL)
        # Remove XML-style tags
        cleaned = re.sub(r'<.*?>.*?<\/.*?>', '', cleaned, flags=re.DOTALL)
        # Remove "No agents need to be activated" type messages
        cleaned = re.sub(r'\*\*No agents need to be activated.*?(?=\n|$)', '', cleaned, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = cleaned.strip()
        
        if not cleaned:
            cleaned = "I'm ready to help you with your job search. What would you like to do?"
        
        return cleaned
    
    def _handle_confirmation(self, user_input: str) -> str:
        """Handle user confirmation response."""
        input_lower = user_input.lower()
        
        # Check for positive/negative confirmation
        positive_words = ["yes", "yeah", "sure", "ok", "yep", "yup", "confirm", "proceed"]
        negative_words = ["no", "nope", "nah", "skip", "cancel", "never"]
        
        is_positive = any(word in input_lower for word in positive_words)
        is_negative = any(word in input_lower for word in negative_words)
        
        if is_positive:
            # User confirmed - execute the pending action
            intent = self.context.pending_intent
            self.context.clear_pending_intent()
            
            # Re-execute the pending intent
            return self.process_message(f"yes, {intent}")
        
        elif is_negative:
            self.context.clear_pending_intent()
            return ("No problem. What else can I help you with?")
        
        else:
            return ("I didn't catch that. Please confirm with yes/no.")
    
    def _extract_file_path(self, text: str, extensions: list[str]) -> Optional[str]:
        """Extract a file path from text.
        
        Args:
            text: Input text
            extensions: List of valid extensions to look for
            
        Returns:
            Extracted file path or None
        """
        import re
        
        # Look for file paths with extensions
        for ext in extensions:
            pattern = rf'[^\s]+{ext}'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        # Also check for common path patterns
        path_pattern = r'(?:[A-Za-z]:\\|\/)[^\s]+'
        matches = re.findall(path_pattern, text)
        for match in matches:
            path = Path(match)
            if path.exists() or any(ext in match.lower() for ext in extensions):
                return match
        
        return None
    
    def reset_conversation(self) -> None:
        """Reset the conversation context."""
        self.context.reset()