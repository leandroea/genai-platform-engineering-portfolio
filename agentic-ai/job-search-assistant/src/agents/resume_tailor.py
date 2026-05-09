"""Resume Tailor Agent - Customizes resumes for specific job applications."""

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name
from ..state.job_search_state import JobSearchState
from ..tools.resume_tools import create_resume_docx, extract_resume_text


class ResumeTailorAgent:
    """Agent that customizes resumes for specific job applications.
    
    This agent:
    - Receives original resume text and target job description
    - Analyzes job requirements and identifies gaps in resume
    - Rewrites resume content to highlight relevant skills/experience
    - Can modify specific sections upon request
    - Maintains truthfulness — only emphasizes existing experience
    """
    
    def __init__(self):
        """Initialize the resume tailor agent."""
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.7,
        )
        
        self.system_prompt = """You are the Resume Tailor Agent, specialized in customizing resumes for specific job applications.

YOUR EXPERTISE:
- Analyzing job descriptions to identify key requirements
- Matching candidate experience to job requirements
- Optimizing resumes for ATS (Applicant Tracking Systems)
- Highlighting relevant skills and experience
- Maintaining truthfulness — only emphasize existing experience, never fabricate

OPERATIONS YOU CAN PERFORM:

1. FULL TAILOR: Rewrite entire resume for a job posting
   - Analyze the job description thoroughly
   - Identify all key requirements
   - Map candidate experience to requirements
   - Rewrite resume to highlight relevant experience
   - Ensure ATS-friendly keyword density

2. SECTION EDIT: Modify specific section
   - e.g., "update my experience to highlight Python"
   - Only changes requested section, rest unchanged

3. KEYWORD INJECT: Add missing keywords
   - Find keywords in job description not in resume
   - Naturally incorporate them without changing content

4. FORMAT ADJUST: Change formatting, order, or layout
   - Reorganize sections for maximum impact
   - Adjust formatting for better readability

OUTPUT FORMAT:
When tailoring a resume, provide the complete tailored resume content in markdown-like format:
- Use # for section headings (Name, Summary, Experience, Education, Skills)
- Use ## for subsections (Company names, job titles)
- Use - for subheadings (dates, locations)
- Use • for bullet points (achievements, responsibilities)

Remember: NEVER fabricate experience. Only highlight what the candidate actually has."""
    
    def tailor_resume(
        self,
        state: JobSearchState,
        job_description: str,
        job_id: str,
        operation: str = "full_tailor"
    ) -> dict:
        """Tailor a resume for a specific job.
        
        Args:
            state: Current JobSearchState
            job_description: Target job description
            job_id: Job identifier
            operation: Type of tailoring operation ("full_tailor", "section_edit", "keyword_inject", "format_adjust")
            
        Returns:
            Dictionary with tailored resume path and metadata
        """
        original_resume = state.get("original_resume_text", "")
        
        if not original_resume:
            return {
                "success": False,
                "error": "No original resume found in state. Please upload a resume first.",
                "job_id": job_id,
            }
        
        # Build the prompt based on operation type
        if operation == "full_tailor":
            tailored_content = self._full_tailor(original_resume, job_description)
        elif operation == "section_edit":
            tailored_content = self._section_edit(original_resume, job_description)
        elif operation == "keyword_inject":
            tailored_content = self._keyword_inject(original_resume, job_description)
        elif operation == "format_adjust":
            tailored_content = self._format_adjust(original_resume, job_description)
        else:
            tailored_content = self._full_tailor(original_resume, job_description)
        
        # Save the tailored resume as DOCX
        output_dir = Path("data/output/resumes")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a slug from job_id for the filename
        job_slug = job_id.replace("-", "").replace(" ", "")[:20]
        output_path = str(output_dir / f"resume_{job_slug}_{job_id[:8]}.docx")
        
        try:
            create_resume_docx(tailored_content, output_path)
            
            result = {
                "success": True,
                "tailored_resume_docx_path": output_path,
                "job_id": job_id,
                "operation": operation,
            }
            
            # Update state
            state["resume_tailor_output"] = result
            state["current_phase"] = "application"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id,
            }
    
    def _full_tailor(self, resume_text: str, job_description: str) -> str:
        """Perform full resume tailoring."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: FULL RESUME TAILOR

Original Resume:
{resume_text}

Target Job Description:
{job_description}

INSTRUCTIONS:
1. Analyze the job description to identify key requirements
2. Review the original resume
3. Rewrite the resume to:
   - Lead with the most relevant experience for this job
   - Use keywords from the job description
   - Highlight achievements that match job requirements
   - Remove irrelevant information
   - Keep truthfulness — only emphasize what exists
4. Format using markdown with sections (# heading, ## subsection, • bullet)

Provide the complete tailored resume.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def _section_edit(self, resume_text: str, job_description: str) -> str:
        """Perform section-specific edit."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: SECTION EDIT

Original Resume:
{resume_text}

Target Job Description:
{job_description}

INSTRUCTIONS:
1. Focus on the most relevant section for this job
2. Rewrite only that section while keeping everything else the same
3. Optimize the section for ATS with relevant keywords
4. Maintain truthfulness

Format with markdown, showing only the changed section and what stayed the same.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def _keyword_inject(self, resume_text: str, job_description: str) -> str:
        """Inject missing keywords without changing content."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: KEYWORD INJECT

Original Resume:
{resume_text}

Target Job Description:
{job_description}

INSTRUCTIONS:
1. Extract keywords from job description
2. Identify which keywords are missing from resume
3. Naturally incorporate missing keywords into existing content
4. Do NOT change meaning or fabricate experience
5. Focus on skills, technologies, and qualifications mentioned

Return the resume with keywords added naturally.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def _format_adjust(self, resume_text: str, job_description: str) -> str:
        """Adjust formatting and layout."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: FORMAT ADJUST

Original Resume:
{resume_text}

Target Job Description:
{job_description}

INSTRUCTIONS:
1. Reorganize sections to lead with most relevant experience
2. Adjust formatting for better readability and ATS optimization
3. Keep all content (don't remove, just reorganize)
4. Use clear headings and consistent formatting

Return the resume with adjusted format.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def analyze_gaps(self, resume_text: str, job_description: str) -> dict:
        """Analyze gaps between resume and job description.
        
        Args:
            resume_text: Original resume text
            job_description: Target job description
            
        Returns:
            Dictionary with identified gaps and suggestions
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a resume analysis expert. Analyze the gap between a resume and job description."),
            HumanMessage(content=f"""RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Analyze:
1. Key requirements in the job description
2. What's present in the resume
3. What's missing from the resume (gaps)
4. Suggestions to bridge the gaps

Provide a structured analysis.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        return {
            "analysis": response.content,
            "resume_text_length": len(resume_text),
            "job_description_length": len(job_description),
        }