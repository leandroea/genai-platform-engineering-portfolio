"""Interview Coach Agent - Prepares interview questions and practice sessions."""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name
from ..state.job_search_state import JobSearchState


class InterviewCoachAgent:
    """Agent that prepares interview questions and practice sessions.
    
    This agent:
    - Receives job description and company research
    - Generates relevant interview questions
    - Prepares STAR-method answers for behavioral questions
    - Runs mock interview sessions on demand
    - Tracks which questions have been practiced
    """
    
    def __init__(self):
        """Initialize the interview coach agent."""
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.7,
        )
        
        self.system_prompt = """You are the Interview Coach Agent, specialized in preparing candidates for job interviews.

YOUR EXPERTISE:
- Generating relevant technical questions based on job description
- Creating behavioral questions using STAR method
- Providing sample answers and talking points
- Running mock interview Q&A sessions
- Tracking which questions have been practiced

INTERVIEW QUESTION TYPES:

1. TECHNICAL QUESTIONS (10-15)
   - Job-specific technical knowledge
   - Problem-solving scenarios
   - System design questions
   - Coding challenges (if applicable)
   - Tool-specific questions

2. BEHAVIORAL QUESTIONS (5-7) using STAR Method
   - Situation: Set the scene
   - Task: Describe your responsibility
   - Action: Explain what you did
   - Result: Share the outcome

Common behavioral themes:
   - Leadership
   - Conflict resolution
   - Problem solving
   - Teamwork
   - Communication
   - Time management
   - Adaptability

3. COMPANY/CULTURE QUESTIONS (3-5)
   - Why this company?
   - Role understanding
   - Career goals
   - Values alignment

STAR METHOD FORMAT:
For each behavioral question, provide:
S: "I was working as a [role] at [company]..."
T: "My team needed to [goal], but [challenge]..."
A: "I decided to [action] by [specific steps]..."
R: "As a result, [quantified outcome]..."

Keep answers concise: 2-3 minutes when speaking."""
    
    def prepare_interview(
        self,
        state: JobSearchState,
        job_description: str,
        company: str = ""
    ) -> dict:
        """Prepare comprehensive interview materials.
        
        Args:
            state: Current JobSearchState
            job_description: Target job description
            company: Optional company name for customization
            
        Returns:
            Dictionary with questions, answers, and talking points
        """
        resume_text = state.get("original_resume_text", "")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: Prepare comprehensive interview preparation materials.

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

{"COMPANY: " + company if company else ""}

Generate:

1. TECHNICAL QUESTIONS (10-15)
   - Questions relevant to the job description
   - Include difficulty level (easy, medium, hard)
   - Mark must-know vs nice-to-know

2. BEHAVIORAL QUESTIONS (5-7) with STAR answers
   - One from each category: Leadership, Conflict, Problem-solving, Teamwork
   - Provide full STAR format answers
   - Include follow-up variations

3. QUESTIONS TO ASK INTERVIEWER (5)
   - Insightful questions that show research/interest
   - Mix of company, role, and culture questions

4. TALKING POINTS
   - Key strengths to highlight
   - Potential weaknesses and how to address
   - elevator pitch for "Tell me about yourself"

Format clearly with headers.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        result = {
            "success": True,
            "materials": response.content,
            "job_description": job_description,
            "company": company,
        }
        
        # Update state
        state["interview_coach_output"] = result
        state["current_phase"] = "interview_prep"
        
        return result
    
    def generate_technical_questions(
        self,
        job_description: str,
        count: int = 10
    ) -> list[dict]:
        """Generate technical interview questions.
        
        Args:
            job_description: Target job description
            count: Number of questions to generate
            
        Returns:
            List of question dictionaries with text and difficulty
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a technical interview expert. Generate relevant interview questions."),
            HumanMessage(content=f"""TASK: Generate {count} technical interview questions.

JOB DESCRIPTION:
{job_description}

Generate {count} technical questions relevant to this job.
For each question include:
- Question text
- Difficulty (Easy/Medium/Hard)
- Topic area (e.g., Python, AWS, System Design)

Format as a numbered list.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Parse questions
        questions = self._parse_technical_questions(response.content)
        
        return questions
    
    def _parse_technical_questions(self, response: str) -> list[dict]:
        """Parse technical questions from LLM output."""
        import re
        
        questions = []
        lines = response.split("\n")
        
        for line in lines:
            # Match patterns like "1. Question [Easy/Medium/Hard]"
            match = re.match(r"^\d+\.\s*(.+?)\s*\[?(\w+)\]?\s*$", line)
            if match:
                questions.append({
                    "question": match.group(1),
                    "difficulty": match.group(2) if match.lastindex >= 1 else "Medium",
                })
        
        return questions
    
    def generate_behavioral_questions(
        self,
        categories: list[str] = None,
        count: int = 5
    ) -> list[dict]:
        """Generate behavioral interview questions with STAR answers.
        
        Args:
            categories: List of categories (Leadership, Conflict, Problem-solving, etc.)
            count: Number of questions to generate
            
        Returns:
            List of question dictionaries with full STAR answers
        """
        if categories is None:
            categories = ["Leadership", "Conflict Resolution", "Problem Solving", "Teamwork"]
        
        categories_str = ", ".join(categories)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: Generate {count} behavioral interview questions with STAR answers.

CATEGORIES: {categories_str}

For each question:
1. Question text
2. Full STAR answer (2-3 minutes when spoken)

Format:
Q: [Question]
STAR:
Situation: ...
Task: ...
Action: ...
Result: ...""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        return {
            "questions": response.content,
            "categories": categories,
        }
    
    def run_mock_interview(
        self,
        state: JobSearchState,
        question_count: int = 5,
        include_technical: bool = True,
        include_behavioral: bool = True
    ) -> dict:
        """Run a mock interview session.
        
        Args:
            state: Current JobSearchState
            question_count: Number of questions per type
            include_technical: Whether to include technical questions
            include_behavioral: Whether to include behavioral questions
            
        Returns:
            Dictionary with mock interview questions
        """
        job_output = state.get("job_aggregator_output", {})
        jobs = job_output.get("jobs", [])
        
        # Get first job for context if available
        job_description = ""
        company = ""
        if jobs:
            job_description = jobs[0].get("description", "")
            company = jobs[0].get("company", "")
        
        result = {
            "success": True,
            "technical_questions": [],
            "behavioral_questions": [],
            "instructions": "Answer each question as if in an interview. Take 2-3 minutes per question.",
        }
        
        if include_technical and job_description:
            result["technical_questions"] = self.generate_technical_questions(
                job_description, question_count
            )
        
        if include_behavioral:
            behavioral = self.generate_behavioral_questions(count=question_count)
            result["behavioral_questions"] = behavioral.get("questions", "")
        
        return result
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        job_description: str = ""
    ) -> dict:
        """Evaluate a candidate's answer to an interview question.
        
        Args:
            question: The interview question
            answer: The candidate's answer
            job_description: Optional job description for context
            
        Returns:
            Evaluation dictionary with score and feedback
        """
        context = f"\nJOB DESCRIPTION:\n{job_description}" if job_description else ""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an interview coach evaluating candidate answers."),
            HumanMessage(content=f"""TASK: Evaluate this interview answer.

QUESTION: {question}

CANDIDATE'S ANSWER:
{answer}
{context}

Evaluate the answer on:
1. Content relevance (does it address the question?)
2. Clarity and structure
3. Specificity (examples, metrics)
4. STAR method usage (for behavioral)
5. Overall impression

Provide:
- Score (1-10)
- Strengths
- Areas for improvement
- Suggested additions""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        return {
            "evaluation": response.content,
            "question": question,
        }
    
    def generate_questions_to_ask(
        self,
        job_description: str,
        company: str = "",
        count: int = 5
    ) -> list[str]:
        """Generate insightful questions to ask the interviewer.
        
        Args:
            job_description: Target job description
            company: Company name
            count: Number of questions to generate
            
        Returns:
            List of questions to ask
        """
        company_context = f"\nCOMPANY: {company}" if company else ""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an interview coach. Generate insightful questions candidates should ask interviewers."),
            HumanMessage(content=f"""TASK: Generate {count} questions to ask the interviewer.

JOB DESCRIPTION:
{job_description}
{company_context}

Generate {count} insightful questions that show:
- Genuine interest in the role
- Research about the company
- Understanding of the job requirements
- Long-term thinking

Avoid basic questions answerable by reading the job description.
Focus on team dynamics, growth opportunities, and challenges.

Format as a numbered list.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Parse questions
        questions = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove leading number or bullet
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    questions.append(clean)
        
        return questions
    
    def get_talking_points(
        self,
        state: JobSearchState,
        job_description: str
    ) -> dict:
        """Generate key talking points for the interview.
        
        Args:
            state: Current JobSearchState
            job_description: Target job description
            
        Returns:
            Dictionary with strengths, weaknesses, and elevator pitch
        """
        resume_text = state.get("original_resume_text", "")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an interview coach helping candidates prepare talking points."),
            HumanMessage(content=f"""TASK: Generate key talking points for an interview.

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Generate:

1. TOP 3 STRENGTHS TO HIGHLIGHT
   - Why each is relevant to this role
   - Specific example for each

2. POTENTIAL WEAKNESSES
   - How to address them honestly
   - Spin them as growth opportunities

3. ELEVATOR PITCH
   - "Tell me about yourself" (2 minutes)
   - Highlight relevant experience
   - End with enthusiasm for role

Format clearly with headers.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        return {
            "talking_points": response.content,
            "job_description": job_description,
        }