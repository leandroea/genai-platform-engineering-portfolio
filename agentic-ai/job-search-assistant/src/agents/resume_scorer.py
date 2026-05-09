"""Resume Score Checker Agent - Scores resumes against job descriptions."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import get_minimax_api_key, get_minimax_endpoint, get_model_name
from ..state.job_search_state import JobSearchState


class ResumeScorerAgent:
    """Agent that scores resumes against job descriptions.
    
    This agent:
    - Receives resume text and job description
    - Evaluates match quality across multiple criteria
    - Provides numerical score with breakdown
    - Suggests specific improvements
    
    Scoring criteria (0-25 points each):
    - Keyword Match: Job-specific terms present in resume
    - Experience Relevance: Past roles match job requirements
    - Format & Length: Clean formatting, 1-2 pages
    - Achievements: Quantified results, not just duties
    """
    
    def __init__(self):
        """Initialize the resume scorer agent."""
        self.llm = ChatOpenAI(
            api_key=get_minimax_api_key(),
            base_url=get_minimax_endpoint(),
            model=get_model_name(),
            temperature=0.3,
        )
        
        self.system_prompt = """You are the Resume Score Checker Agent, specialized in evaluating resumes against job descriptions.

SCORING CRITERIA (0-25 points each, total 0-100):

1. KEYWORD MATCH (0-25 points)
   - Job-specific terms present in resume
   - Technical skills mentioned
   - Required qualifications addressed
   - 25 = All major keywords present
   - 0 = No relevant keywords

2. EXPERIENCE RELEVANCE (0-25 points)
   - Past roles match job requirements
   - Industry experience aligned
   - Career progression appropriate
   - 25 = Perfect match
   - 0 = Completely unrelated

3. FORMAT & LENGTH (0-25 points)
   - Clean formatting, professional look
   - 1-2 pages (appropriate length)
   - Clear section organization
   - No spelling/grammar errors
   - 25 = Perfect format
   - 0 = Major formatting issues

4. ACHIEVEMENTS (0-25 points)
   - Quantified results (numbers, percentages)
   - Impact statements, not just duties
   - Demonstrates value added
   - Shows career achievements
   - 25 = Strong quantified achievements
   - 0 = Only lists duties, no achievements

OUTPUT FORMAT:
For each category, provide:
- Score (0-25)
- Brief explanation of the score
- Specific examples from the resume

Then provide 3-5 actionable recommendations for improvement.

Format your response as:
SCORE: X/100
- Keyword Match: X/25 (explanation)
- Experience Relevance: X/25 (explanation)
- Format & Length: X/25 (explanation)
- Achievements: X/25 (explanation)

RECOMMENDATIONS:
1. [Specific recommendation]
2. [Specific recommendation]
3. [Specific recommendation]"""
    
    def score_resume(
        self,
        state: JobSearchState,
        job_description: str,
        job_id: str = ""
    ) -> dict:
        """Score a resume against a job description.
        
        Args:
            state: Current JobSearchState
            job_description: Target job description
            job_id: Optional job identifier
            
        Returns:
            Dictionary with score breakdown and recommendations
        """
        resume_text = state.get("original_resume_text", "")
        
        if not resume_text:
            return {
                "success": False,
                "error": "No resume found in state. Please upload a resume first.",
                "score": 0,
            }
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""TASK: SCORE RESUME AGAINST JOB DESCRIPTION

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Evaluate the resume against the job description using the scoring criteria.
Provide detailed breakdown and specific recommendations.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Parse the response to extract scores
        score_data = self._parse_score_response(response.content)
        
        # Add job_id if provided
        if job_id:
            score_data["job_id"] = job_id
        
        # Update state
        state["resume_score_output"] = score_data
        
        return score_data
    
    def _parse_score_response(self, response: str) -> dict:
        """Parse the LLM response to extract score information.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured score data
        """
        import re
        
        result = {
            "success": True,
            "breakdown": {},
            "recommendations": [],
            "raw_response": response,
        }
        
        # Try to extract overall score
        score_match = re.search(r"SCORE:\s*(\d+)/100", response)
        if score_match:
            result["score"] = int(score_match.group(1))
        
        # Try to extract category scores
        categories = ["Keyword Match", "Experience Relevance", "Format & Length", "Achievements"]
        for category in categories:
            # Match pattern like "Keyword Match: 18/25 (explanation)"
            pattern = rf"{category}:\s*(\d+)/25"
            match = re.search(pattern, response)
            if match:
                result["breakdown"][category] = int(match.group(1))
        
        # Try to extract recommendations
        rec_pattern = r"RECOMMENDATIONS:(.+?)(?:$|\n\n)"
        rec_match = re.search(rec_pattern, response, re.DOTALL)
        if rec_match:
            rec_text = rec_match.group(1)
            # Extract numbered items
            rec_items = re.findall(r"\d+\.\s*(.+?)(?:\n\d+\.|$)", rec_text, re.DOTALL)
            result["recommendations"] = [r.strip() for r in rec_items if r.strip()]
        
        # Calculate score from breakdown if not found
        if "score" not in result and result["breakdown"]:
            result["score"] = sum(result["breakdown"].values())
        
        return result
    
    def compare_resumes(
        self,
        state: JobSearchState,
        job_description: str,
        resume_versions: list[dict]
    ) -> list[dict]:
        """Compare multiple resume versions for the same job.
        
        Args:
            state: Current JobSearchState
            job_description: Target job description
            resume_versions: List of dicts with 'name' and 'text' keys
            
        Returns:
            List of score dictionaries for each version
        """
        results = []
        
        for version in resume_versions:
            name = version.get("name", "Unnamed")
            text = version.get("text", "")
            
            if not text:
                results.append({
                    "name": name,
                    "success": False,
                    "error": "No resume text provided",
                })
                continue
            
            # Temporarily set the resume text
            original = state.get("original_resume_text", "")
            state["original_resume_text"] = text
            
            try:
                score_data = self.score_resume(state, job_description)
                score_data["name"] = name
                results.append(score_data)
            finally:
                # Restore original
                state["original_resume_text"] = original
        
        return results
    
    def suggest_improvements(
        self,
        resume_text: str,
        job_description: str,
        current_score: int
    ) -> list[str]:
        """Get specific suggestions to improve resume score.
        
        Args:
            resume_text: Current resume text
            job_description: Target job description
            current_score: Current score to improve upon
            
        Returns:
            List of specific improvement suggestions
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a resume expert. Provide specific, actionable suggestions to improve a resume."),
            HumanMessage(content=f"""Current Score: {current_score}/100

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Based on the current score, provide 5 specific, actionable recommendations to improve this resume's match with the job description.

Each recommendation should:
- Be specific (mention exact changes)
- Be actionable (something you can actually do)
- Target a different area of improvement

Format as a numbered list.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Parse recommendations
        import re
        rec_items = re.findall(r"\d+\.\s*(.+?)(?:\n\d+\.|$)", response.content, re.DOTALL)
        
        return [r.strip() for r in rec_items if r.strip()]
    
    def get_ats_keywords(
        self,
        job_description: str
    ) -> list[str]:
        """Extract ATS keywords from a job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            List of important keywords for ATS
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an ATS optimization expert. Extract important keywords from job descriptions."),
            HumanMessage(content=f"""JOB DESCRIPTION:
{job_description}

Extract the most important keywords and skills from this job description that should appear in a resume for ATS optimization.

Include:
- Technical skills (Python, AWS, SQL, etc.)
- Soft skills (leadership, communication, etc.)
- Industry terms
- Qualifications and certifications
- Action verbs

Return as a comma-separated list of keywords.""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Parse keywords
        keywords = [kw.strip() for kw in response.content.split(",") if kw.strip()]
        
        return keywords