"""Cover Letter Agent - Writes personalized cover letters for job applications."""

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name
from ..state.job_search_state import JobSearchState
from ..tools.resume_tools import create_resume_docx


class CoverLetterAgent:
    """Agent that writes personalized cover letters for job applications.
    
    This agent:
    - Receives job details and user's background summary
    - Researches company culture and values from job description
    - Writes professional cover letter with personalization
    - Follows standard business letter structure
    - Can save cover letter as DOCX (optional)
    """
    
    def __init__(self):
        """Initialize the cover letter agent."""
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.7,
        )
        
        self.system_prompt = """You are the Cover Letter Agent, specialized in writing personalized cover letters for job applications.

YOUR EXPERTISE:
- Writing engaging opening paragraphs with specific job references
- Highlighting 2-3 most relevant qualifications
- Connecting candidate experience to company needs
- Following standard business letter structure
- Closing with call-to-action and gratitude

COVER LETTER STRUCTURE:

1. OPENING PARAGRAPH
   - State the specific position you're applying for
   - Mention how you discovered the opportunity
   - Create immediate connection with the company

2. BODY PARAGRAPHS (2-3)
   - Highlight most relevant qualifications for this specific job
   - Use specific examples from experience
   - Connect your skills to company's needs/mission/values
   - Show you've researched the company

3. CLOSING PARAGRAPH
   - Reiterate interest in the role
   - Include call-to-action
   - Thank the reader for their time
   - Mention availability for interview

PERSONALIZATION GUIDELINES:
- Use company name and specific job title
- Reference specific requirements from job description
- Mention specific skills or experiences that match
- Show enthusiasm for the company's mission or products
- Keep it concise (300-400 words)

TONE:
- Professional but personable
- Confident without being arrogant
- Specific rather than generic
- Never use templates or generic phrases

Remember: A good cover letter is personal, specific, and shows you've done your research."""
    
    def generate_cover_letter(
        self,
        state: JobSearchState,
        job: dict,
        save_docx: bool = False
    ) -> dict:
        """Generate a personalized cover letter for a job.
        
        Args:
            state: Current JobSearchState
            job: Job dictionary with title, company, description, etc.
            save_docx: Whether to save the cover letter as a DOCX file
            
        Returns:
            Dictionary with cover letter text and optional file path
        """
        resume_text = state.get("original_resume_text", "")
        user_profile = state.get("user_profile", {})
        
        if not resume_text:
            return {
                "success": False,
                "error": "No resume found in state. Please upload a resume first.",
                "job_id": job.get("id", ""),
            }
        
        job_title = job.get("title", "")
        company = job.get("company", "")
        job_description = job.get("description", "")
        
        # Build prompt with context
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: WRITE PERSONALIZED COVER LETTER

CANDIDATE BACKGROUND (from resume):
{resume_text}

JOB DETAILS:
Position: {job_title}
Company: {company}
Job Description: {job_description}

INSTRUCTIONS:
1. Write a complete, personalized cover letter
2. Address it to "Dear Hiring Manager," or specific name if provided
3. Reference the specific position and company
4. Highlight 2-3 qualifications most relevant to the job
5. Connect candidate experience to company needs
6. Include specific examples from the resume
7. End professionally with call-to-action

Format the letter properly with:
- Your Name
- Your Address (or just City, State)
- Date
- Company Name
- Company Address

Keep it to 300-400 words, single-spaced.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        cover_letter_text = response.content
        
        result = {
            "success": True,
            "cover_letter_text": cover_letter_text,
            "job_id": job.get("id", ""),
            "company": company,
            "position": job_title,
        }
        
        # Optionally save as DOCX
        if save_docx:
            output_dir = Path("data/output/cover_letters")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            company_slug = company.lower().replace(" ", "_").replace(":", "")[:20]
            job_slug = job.get("id", "unknown")[:8]
            output_path = str(output_dir / f"cover_letter_{company_slug}_{job_slug}.docx")
            
            try:
                create_resume_docx(cover_letter_text, output_path)
                result["cover_letter_docx_path"] = output_path
            except Exception as e:
                result["docx_error"] = str(e)
        
        # Update state
        state["cover_letter_output"] = result
        
        return result
    
    def generate_multiple_letters(
        self,
        state: JobSearchState,
        jobs: list[dict],
        save_docx: bool = False
    ) -> list[dict]:
        """Generate cover letters for multiple jobs.
        
        Args:
            state: Current JobSearchState
            jobs: List of job dictionaries
            save_docx: Whether to save letters as DOCX
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for job in jobs:
            result = self.generate_cover_letter(state, job, save_docx)
            results.append(result)
        
        return results
    
    def customize_existing_letter(
        self,
        existing_letter: str,
        job: dict
    ) -> str:
        """Customize an existing cover letter for a different job.
        
        Args:
            existing_letter: The existing cover letter text
            job: New job dictionary
            
        Returns:
            Customized cover letter text
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: CUSTOMIZE EXISTING COVER LETTER FOR NEW POSITION

EXISTING COVER LETTER:
{existing_letter}

NEW JOB DETAILS:
Position: {job.get('title', '')}
Company: {job.get('company', '')}
Job Description: {job.get('description', '')}

INSTRUCTIONS:
1. Keep the overall structure and tone
2. Update the specific position and company references
3. Adjust the highlighted qualifications to match new job
4. Update any company-specific references
5. Make it feel fresh, not just find-replace

Provide the customized cover letter.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content