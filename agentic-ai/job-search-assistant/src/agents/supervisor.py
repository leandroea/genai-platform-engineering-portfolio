"""Supervisor Agent - Coordinates all other agents and makes decisions.""" 

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name
from ..state.job_search_state import JobSearchState
from ..tools.job_description_tools import list_job_descriptions, read_job_description
from ..tools.resume_tools import extract_resume_text, validate_resume_file


class SupervisorAgent:
    """Supervisor agent that coordinates all other agents.
    
    The supervisor is the central coordinator that:
    - Receives user requests and decides which agents to activate
    - Queries each subordinate agent for their current status
    - Synthesizes information from multiple agents
    - Makes decisions about prioritization and strategy
    """
    
    def __init__(self):
        """Initialize the supervisor with LLM capabilities."""
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.7,
        )
        
        self.system_prompt = """You are the Supervisor Agent for a Job Search Assistant system.

Your role is to COORDINATE and DECIDE, not to execute tasks directly.

COORDINATION RESPONSIBILITIES:
1. When a user request comes in, determine which agents need to be activated
2. Query agent outputs to understand current state
3. Synthesize information from multiple agents to make decisions
4. Prioritize tasks when multiple urgent items exist
5. Decide application strategy (which jobs to prioritize)

DECISION MAKING EXAMPLES:
- "Resume Score is 45/100 - too low. Ask Resume Tailor to improve before applying"
- "Found 3 high-matching jobs - prioritize these for immediate application"
- "Cover letter generated but score is low - regenerate with more personalization"

YOU COORDINATE THESE AGENTS:
- Resume Tailor Agent: Customizes resumes for specific jobs
- Cover Letter Agent: Writes personalized cover letters
- Job Aggregator Agent: Finds job postings
- Resume Score Checker Agent: Scores resumes against job descriptions
- Application Form Agent: Assists with form field completion
- Interview Coach Agent: Prepares interview questions and practice

STATE QUERY: You can query the current state to see what each agent has produced.

CONVERSATIONAL MODE:
When the user asks general questions (like "who are you", "what can you do", "help me"), 
respond naturally and directly WITHOUT mentioning agent coordination. 
Keep responses friendly and conversational. Focus on what you can help them accomplish.

Remember: You are the brain, not the hands. Delegate execution to specialized agents."""
    
    def process_request(self, state: JobSearchState, user_request: str) -> str:
        """Process a user request and determine supervisor response.
        
        Args:
            state: Current JobSearchState with all agent outputs
            user_request: The user's request or question
            
        Returns:
            Supervisor's response and any state updates needed
        """
        # Build context from current state
        context = self._build_context(state)
        
        # Create prompt for supervisor decision
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Current State Context:
{context}

User Request: {user_request}

Based on the current state and user request, what should happen? 
Identify which agents need to be activated and what decisions need to be made.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        return response.content
    
    def _build_context(self, state: JobSearchState) -> str:
        """Build context string from current state.
        
        Args:
            state: Current JobSearchState
            
        Returns:
            Formatted context string
        """
        lines = []
        
        # User profile
        if state.get("user_profile"):
            lines.append("User Profile:")
            for key, value in state["user_profile"].items():
                lines.append(f"  - {key}: {value}")
        
        # Resume status
        if state.get("original_resume_text"):
            word_count = len(state["original_resume_text"].split())
            lines.append(f"\nResume: Uploaded ({word_count} words)")
        else:
            lines.append("\nResume: Not uploaded")
        
        # Target roles/locations
        if state.get("target_roles"):
            lines.append(f"\nTarget Roles: {', '.join(state['target_roles'])}")
        if state.get("target_locations"):
            lines.append(f"Target Locations: {', '.join(state['target_locations'])}")
        
        # Jobs found
        job_output = state.get("job_aggregator_output", {})
        if job_output.get("jobs"):
            lines.append(f"\nJobs Found: {len(job_output['jobs'])} jobs via Jooble")
        else:
            lines.append("\nJobs Found: None yet")
        
        # Resume tailor output
        tailor_output = state.get("resume_tailor_output", {})
        if tailor_output:
            lines.append(f"\nTailored Resumes: {len(tailor_output)} generated")
        
        # Cover letter output
        cover_output = state.get("cover_letter_output", {})
        if cover_output:
            lines.append(f"Cover Letters: {len(cover_output)} generated")
        
        # Resume scores
        score_output = state.get("resume_score_output", {})
        if score_output:
            score = score_output.get("score", "N/A")
            lines.append(f"\nResume Score: {score}/100")
        
        # Current phase
        lines.append(f"\nCurrent Phase: {state.get('current_phase', 'idle')}")
        
        # Pending tasks
        pending = state.get("pending_tasks", [])
        if pending:
            lines.append(f"Pending Tasks: {', '.join(pending)}")
        
        # Blockers
        blockers = state.get("blockers", [])
        if blockers:
            lines.append(f"Blockers: {', '.join(blockers)}")
        
        return "\n".join(lines)
    
    def make_decision(
        self,
        state: JobSearchState,
        decision_type: str,
        context: Optional[dict] = None
    ) -> str:
        """Make a specific type of decision.
        
        Args:
            state: Current JobSearchState
            decision_type: Type of decision needed (e.g., "prioritize_jobs", "apply_now")
            context: Additional context for the decision
            
        Returns:
            Supervisor's decision
        """
        decision_prompts = {
            "prioritize_jobs": """Analyze the found jobs and prioritize which ones to apply to first.
Consider: match score, user preferences, application deadline.""",
            
            "apply_now": """Should we apply to this job now? Check:
- Resume score (should be > 60)
- Cover letter quality
- Job match to user targets""",
            
            "tailor_resume": """The resume needs to be tailored for a specific job. 
Decide what changes are needed and activate the Resume Tailor Agent.""",
            
            "interview_prep": """The user is preparing for an interview. 
Decide what preparation is needed and activate the Interview Coach Agent.""",
        }
        
        base_prompt = decision_prompts.get(decision_type, "Make a decision based on the current state.")
        
        context_str = f"\nAdditional Context: {context}" if context else ""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""{base_prompt}

Current State:
{self._build_context(state)}
{context_str}

Provide your decision and reasoning.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Update state with decision
        state["supervisor_decision"] = response.content
        
        return response.content
    
    def query_agent_status(self, state: JobSearchState, agent_name: str) -> str:
        """Query a specific agent's current status and outputs.
        
        Args:
            state: Current JobSearchState
            agent_name: Name of the agent to query
            
        Returns:
            Status summary of the agent
        """
        agent_outputs = {
            "resume_tailor": state.get("resume_tailor_output", {}),
            "cover_letter": state.get("cover_letter_output", {}),
            "job_aggregator": state.get("job_aggregator_output", {}),
            "resume_scorer": state.get("resume_score_output", {}),
            "application_form": state.get("application_form_output", {}),
            "interview_coach": state.get("interview_coach_output", {}),
        }
        
        output = agent_outputs.get(agent_name, {})
        
        if not output:
            return f"{agent_name}: No output yet"
        
        return f"{agent_name} Status:\n{self._format_output(output)}"
    
    def _format_output(self, output: dict) -> str:
        """Format agent output for display."""
        lines = []
        for key, value in output.items():
            if isinstance(value, list):
                lines.append(f"  - {key}: {len(value)} items")
            elif isinstance(value, dict):
                lines.append(f"  - {key}: {len(value)} sub-items")
            else:
                value_str = str(value)[:100]
                lines.append(f"  - {key}: {value_str}")
        return "\n".join(lines)
    
    def add_task(self, state: JobSearchState, task: str) -> None:
        """Add a task to the pending tasks list.
        
        Args:
            state: Current JobSearchState
            task: Task description
        """
        pending = state.get("pending_tasks", [])
        pending.append(task)
        state["pending_tasks"] = pending
    
    def complete_task(self, state: JobSearchState, task: str) -> None:
        """Move a task from pending to completed.
        
        Args:
            state: Current JobSearchState
            task: Task description
        """
        pending = state.get("pending_tasks", [])
        if task in pending:
            pending.remove(task)
            state["pending_tasks"] = pending
        
        completed = state.get("completed_tasks", [])
        completed.append(task)
        state["completed_tasks"] = completed
    
    def chat(self, state: JobSearchState, user_message: str) -> dict:
        """Handle a conversational user message using LLM understanding.
        
        This method uses the LLM to understand user intent naturally,
        then executes the appropriate actions through specialized agents.
        
        Args:
            state: Current JobSearchState
            user_message: The user's natural language message
            
        Returns:
            dict with 'response' (bot message) and 'state_updates' (any state changes)
        """
        # Build comprehensive context
        context = self._build_full_context(state)
        
        # System prompt for chat handling
        chat_system_prompt = """You are a helpful job search assistant chatbot. You understand natural language
and can help users with their job search needs. When a user asks something, you should naturally
understand their intent and help them accomplish their goals.

AVAILABLE ACTIONS:
1. If user wants to upload/see their resume:
   - Check if resume exists in state
   - If uploaded, provide an overview (word count, path, summary of experience sections)
   - If not uploaded, ask them to provide the file path
   
2. If user wants to search for jobs:
   - Use the target_roles and target_locations from state
   - Return a message that you'll search for jobs with those parameters
   
3. If user wants to score/tailor resume:
   - Check if resume and job description are available
   - Guide user to provide what's missing
   
4. If user asks general questions like "who are you", "what can you do":
   - Respond naturally and conversationally
   
5. If user wants to set preferences:
   - Acknowledge and explain what preferences are currently set
   
6. If user asks about their current status:
   - Report what data is available in the state

IMPORTANT: You respond as the chatbot, not as a supervisor coordinating agents.
Keep responses friendly, conversational, and helpful. DO NOT mention agents or coordination.
Just help the user directly."""

        # Create prompt for the chat
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=chat_system_prompt),
            HumanMessage(content=f"""Current State:
{context}

User Message: {user_message}

Respond naturally as a helpful assistant. Provide the information requested or
explain what needs to be done. If something is missing, ask for it in a friendly way.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Clean up the response - remove any thinking tags or reasoning content
        response_text = response.content
        response_text = self._strip_thinking_tags(response_text)
        
        # Determine if any state updates are needed based on the message
        state_updates = {}
        user_lower = user_message.lower()
        
        # Check if user is providing a file path for resume
        if state.get("original_resume_text"):
            pass  # Resume already uploaded
        elif any(ext in user_lower for ext in [".pdf", ".docx", ".txt"]):
            # User might be providing a path
            import re
            paths = re.findall(r'[A-Za-z]:\\[^\s]+|\/[^\s]+', user_message)
            if paths:
                file_path = paths[0]
                if validate_resume_file(file_path)[0]:
                    try:
                        text = extract_resume_text(file_path)
                        state_updates["original_resume_text"] = text
                        state_updates["original_resume_path"] = file_path
                    except:
                        pass
        
        return {
            "response": response_text,
            "state_updates": state_updates
        }
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove thinking/reasoning content from LLM response.
    
        Removes ALL content between <think> and </think> tags (inclusive).
        Uses DOTALL flag so . matches newlines too.
        """
        import re
        # print("Original response:")
        # print(text)
        # print("Stripped response:")
        # Remove all <think>...</think> blocks (non-greedy to handle multiple blocks)
        result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
        # Clean up any orphaned tags (just in case)
        result = re.sub(r'<think>', '', result)
        result = re.sub(r'</think>', '', result)
        return result.strip()
    
    def _build_full_context(self, state: JobSearchState) -> str:
        """Build comprehensive context for chat interactions."""
        lines = []
        
        # Resume status
        if state.get("original_resume_text"):
            word_count = len(state["original_resume_text"].split())
            lines.append(f"Resume: Uploaded ({word_count} words)")
            if state.get("original_resume_path"):
                lines.append(f"Resume Path: {state['original_resume_path']}")
            # Include FULL resume text
            lines.append(f"\n--- FULL RESUME TEXT ---")
            lines.append(state["original_resume_text"])
            lines.append(f"--- END RESUME TEXT ---\n")
        else:
            lines.append("Resume: Not uploaded yet")
        
        # Target preferences
        if state.get("target_roles"):
            lines.append(f"Target Roles: {', '.join(state['target_roles'])}")
        else:
            lines.append("Target Roles: Not set")
            
        if state.get("target_locations"):
            lines.append(f"Target Locations: {', '.join(state['target_locations'])}")
        else:
            lines.append("Target Locations: Not set")
        
        # Job descriptions
        local_jds = list_job_descriptions()
        if local_jds:
            lines.append(f"\nJob Descriptions Loaded: {len(local_jds)}")
            for jd in local_jds[:3]:
                lines.append(f"\n--- JOB DESCRIPTION: {jd.get('title', 'N/A')} at {jd.get('company', 'N/A')} ---")
                # Include FULL job description text
                full_jd = read_job_description(jd.get("id", ""))
                if full_jd and full_jd.get("description"):
                    lines.append(full_jd["description"])
                lines.append(f"--- END JOB DESCRIPTION ---\n")
        else:
            lines.append("\nJob Descriptions: None uploaded")
        
        # Jobs from search
        job_output = state.get("job_aggregator_output", {})
        if job_output.get("jobs"):
            lines.append(f"\nJobs Found: {len(job_output['jobs'])} jobs from search")
        else:
            lines.append("\nJobs Found: No searches performed yet")
        
        # Recent outputs
        if state.get("resume_tailor_output"):
            lines.append(f"\nTailored Resumes: {len(state['resume_tailor_output'])} generated")
        if state.get("cover_letter_output"):
            lines.append(f"Cover Letters: {len(state['cover_letter_output'])} generated")
        if state.get("resume_score_output"):
            score = state["resume_score_output"].get("score", "N/A")
            lines.append(f"Resume Score: {score}/100")
        
        return "\n".join(lines)
    
    def add_blocker(self, state: JobSearchState, blocker: str) -> None:
        """Add a blocker that needs attention.
        
        Args:
            state: Current JobSearchState
            blocker: Blocker description
        """
        blockers = state.get("blockers", [])
        blockers.append(blocker)
        state["blockers"] = blockers
    
    def resolve_blocker(self, state: JobSearchState, blocker: str) -> None:
        """Remove a resolved blocker.
        
        Args:
            state: Current JobSearchState
            blocker: Blocker description
        """
        blockers = state.get("blockers", [])
        if blocker in blockers:
            blockers.remove(blocker)
            state["blockers"] = blockers