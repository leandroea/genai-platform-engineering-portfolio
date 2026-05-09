"""Application Form Agent - Assists with application form field completion."""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name
from ..state.job_search_state import JobSearchState


class ApplicationFormAgent:
    """Agent that assists with filling out application forms.
    
    This agent:
    - Receives requests to fill application form fields
    - Extracts relevant information from user's resume
    - Generates appropriate responses for specific platforms
    - Handles both simple fields and complex ones
    
    Supported Field Types:
    - Personal: Name, email, phone, address, LinkedIn URL
    - Professional: Job title, company, dates, salary expectations
    - Education: Degree, institution, graduation year, GPA
    - Skills: Technical skills, languages, certifications
    - Free Text: "Why do you want to work here?", "Describe your experience"
    """
    
    def __init__(self):
        """Initialize the application form agent."""
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.7,
        )
        
        self.system_prompt = """You are the Application Form Agent, specialized in helping fill out job application forms.

YOUR EXPERTISE:
- Extracting relevant information from resumes
- Mapping resume data to form field labels
- Generating professional responses for free-text fields
- Formatting information for platform-specific requirements
- Providing copy-paste ready answers

SUPPORTED FIELD TYPES:

1. PERSONAL FIELDS
   - Name, email, phone, address
   - LinkedIn URL, personal website
   - GitHub profile, portfolio

2. PROFESSIONAL FIELDS
   - Current job title
   - Company name
   - Employment dates
   - Salary expectations
   - Notice period

3. EDUCATION FIELDS
   - Degree (BS, MS, PhD)
   - Institution name
   - Graduation year
   - GPA (if impressive)
   - Major/specialization

4. SKILLS FIELDS
   - Technical skills
   - Programming languages
   - Tools and technologies
   - Languages spoken
   - Certifications

5. FREE-TEXT FIELDS (most complex)
   - "Tell us about yourself"
   - "Why do you want to work here?"
   - "Describe your relevant experience"
   - "Greatest weakness/strength"
   - "Salary expectations"

For free-text fields, provide:
- 2-3 sentence responses (not too long)
- Specific examples from experience
- Professional but personable tone
- Keywords from job description

Always maintain truthfulness - only use information from the provided resume."""
    
    def get_field_suggestion(
        self,
        state: JobSearchState,
        field_name: str,
        field_type: str = "text"
    ) -> str:
        """Get a suggested answer for a specific form field.
        
        Args:
            state: Current JobSearchState
            field_name: Name/label of the field
            field_type: Type of field (text, email, phone, date, etc.)
            
        Returns:
            Suggested answer for the field
        """
        resume_text = state.get("original_resume_text", "")
        user_profile = state.get("user_profile", {})
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: Provide a suggested answer for a form field.

FIELD NAME: {field_name}
FIELD TYPE: {field_type}

USER PROFILE:
{user_profile}

RESUME (for reference):
{resume_text[:3000]}  # First 3000 chars for context

Provide a suggested answer for this field. If the field requires info not in the resume/profile, say so and suggest what to fill in.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def fill_form_fields(
        self,
        state: JobSearchState,
        fields: list[dict],
        platform: str = "generic"
    ) -> dict:
        """Generate answers for multiple form fields.
        
        Args:
            state: Current JobSearchState
            fields: List of field dictionaries with 'name' and 'type' keys
            platform: Platform name (e.g., "linkedin", "greenhouse", "generic")
            
        Returns:
            Dictionary mapping field names to suggested answers
        """
        resume_text = state.get("original_resume_text", "")
        user_profile = state.get("user_profile", {})
        
        # Build fields description
        fields_str = "\n".join([f"- {f.get('name', 'Unknown')} ({f.get('type', 'text')})" for f in fields])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: Fill out application form fields for {platform}.

PLATFORM: {platform}

FIELDS TO FILL:
{fields_str}

USER PROFILE:
{user_profile}

RESUME:
{resume_text[:4000]}

For each field, provide a suggested answer. Format as:
FIELD_NAME: ANSWER

For free-text fields, provide 2-3 sentences max. Keep responses professional and concise.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Parse responses into dict
        answers = self._parse_field_responses(response.content)
        
        result = {
            "success": True,
            "field_answers": answers,
            "platform": platform,
        }
        
        # Update state
        state["application_form_output"] = result
        
        return result
    
    def _parse_field_responses(self, response: str) -> dict:
        """Parse field responses from LLM output."""
        import re
        
        answers = {}
        
        # Match patterns like "Field Name: Answer"
        lines = response.split("\n")
        for line in lines:
            match = re.match(r"^\s*(.+?):\s*(.+)$", line)
            if match:
                field_name = match.group(1).strip()
                answer = match.group(2).strip()
                answers[field_name] = answer
        
        return answers
    
    def format_work_history(
        self,
        state: JobSearchState,
        format_type: str = "bullets"
    ) -> str:
        """Format work experience for application forms.
        
        Args:
            state: Current JobSearchState
            format_type: Format style ("bullets", "paragraph", "chronological")
            
        Returns:
            Formatted work history text
        """
        resume_text = state.get("original_resume_text", "")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: Format work history for job application.

RESUME:
{resume_text}

FORMAT TYPE: {format_type}

Extract work experience from the resume and format it for a job application.
Use {format_type} format.

Keep descriptions concise and achievement-focused where possible.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def generate_skills_list(
        self,
        state: JobSearchState,
        max_items: int = 10
    ) -> list[str]:
        """Generate a list of skills for application forms.
        
        Args:
            state: Current JobSearchState
            max_items: Maximum number of skills to list
            
        Returns:
            List of skill strings
        """
        resume_text = state.get("original_resume_text", "")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a resume expert. Extract and list skills from resumes."),
            HumanMessage(content=f"""TASK: Extract top skills from resume.

RESUME:
{resume_text}

Extract the top {max_items} most relevant skills from this resume.
Include technical skills, tools, and relevant soft skills.

Return as a comma-separated list.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Parse skills
        skills = [s.strip() for s in response.content.split(",") if s.strip()]
        
        return skills[:max_items]
    
    def generate_free_text_response(
        self,
        state: JobSearchState,
        question: str,
        job_description: str = ""
    ) -> str:
        """Generate a response for a free-text application field.
        
        Args:
            state: Current JobSearchState
            question: The question/prompt for the field
            job_description: Optional job description for context
            
        Returns:
            Suggested response text
        """
        resume_text = state.get("original_resume_text", "")
        user_profile = state.get("user_profile", {})
        
        context = f"\nJOB DESCRIPTION:\n{job_description}" if job_description else ""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: Write a response for a free-text application field.

QUESTION: {question}

USER PROFILE:
{user_profile}

RESUME:{context}
{resume_text[:3000]}

Write a 2-3 sentence response to this question. Be specific, professional, and concise.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def get_cover_phrases(self, category: str) -> list[str]:
        """Get common cover phrases for application forms.
        
        Args:
            category: Category of phrases (e.g., "strengths", "weaknesses", "goals")
            
        Returns:
            List of phrase options
        """
        phrases = {
            "strengths": [
                "Strong problem-solving skills",
                "Excellent communication abilities",
                "Detail-oriented and organized",
                "Team player with leadership experience",
                "Adaptable to fast-paced environments",
            ],
            "weaknesses": [
                "Sometimes too focused on details",
                "Impatient with slow processes",
                "Learning to delegate more effectively",
                "Working on work-life balance",
            ],
            "goals": [
                "Grow into a leadership role",
                "Develop expertise in emerging technologies",
                "Make a meaningful impact at a growing company",
                "Build scalable solutions",
            ],
        }
        
        return phrases.get(category, [])