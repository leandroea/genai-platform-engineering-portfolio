"""SEO optimization, grammar checking, and content rewriting tools."""

import os
import re
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def grammar_check(text: str) -> str:
    """Check and fix grammar, spelling, and punctuation errors in text.
    
    Args:
        text: The content to check and fix
        
    Returns:
        Grammar-corrected content
    """
    try:
        api_key = os.getenv("MINIMAX_API_KEY")
        endpoint = os.getenv("MINIMAX_ENDPOINT", "https://api.minimax.io/v1")
        model_name = os.getenv("MODEL_NAME", "minimax-m2.7")
        
        if not api_key or api_key == "your-minimax-api-key-here":
            return "Error: MINIMAX_API_KEY not configured"
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=endpoint,
            model=model_name,
            temperature=0.3
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional editor. Fix grammar, spelling, and punctuation errors 
            in the provided text. Preserve the original meaning and style.
            
            Rules:
            - Fix spelling mistakes
            - Correct punctuation errors
            - Fix subject-verb agreement issues
            - Fix article usage (a, an, the)
            - Preserve paragraph structure
            - Do NOT add any new content or expand the text
            - Return ONLY the corrected text, no explanations or notes"""),
            ("human", "Text to correct:\n{text}")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"text": text})
        
        # Clean thinking tags and extract content only
        clean_content = _clean_llm_output(result.content)
        return clean_content
        
    except Exception as e:
        return f"Grammar check error: {str(e)}"


def format_seo(text: str, keywords: list[str]) -> str:
    """Optimize content for SEO by integrating keywords naturally.
    
    Args:
        text: The content to optimize
        keywords: List of SEO keywords to integrate
        
    Returns:
        SEO-optimized content with keywords naturally integrated
    """
    try:
        api_key = os.getenv("MINIMAX_API_KEY")
        endpoint = os.getenv("MINIMAX_ENDPOINT", "https://api.minimax.io/v1")
        model_name = os.getenv("MODEL_NAME", "minimax-m2.7")
        
        if not api_key or api_key == "your-minimax-api-key-here":
            return "Error: MINIMAX_API_KEY not configured"
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=endpoint,
            model=model_name,
            temperature=0.5
        )
        
        keyword_str = ", ".join(keywords) if keywords else "None provided"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an SEO expert. Optimize the provided content for search engines 
            by naturally integrating the target keywords.
            
            Requirements:
            - Integrate keywords naturally into headings and body text
            - Maintain keyword density of 1-2% (don't overstuff)
            - Use keywords in the first paragraph and at least one heading
            - Ensure keyword placement sounds natural, not forced
            - Keep the content engaging and readable
            - Preserve the overall structure and length of the original content
            - Return ONLY the optimized content, no explanations or notes"""),
            ("human", "Content to optimize:\n{text}\n\nTarget keywords: {keywords}")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"text": text, "keywords": keyword_str})
        
        # Clean thinking tags and extract content only
        clean_content = _clean_llm_output(result.content)
        return clean_content
        
    except Exception as e:
        return f"SEO formatting error: {str(e)}"


def rewrite_professional(text: str, tone: Optional[str] = "professional") -> str:
    """Rewrite content in a professional tone with improved clarity.
    
    Args:
        text: The content to rewrite
        tone: The target tone (default: "professional")
        
    Returns:
        Professionally rewritten content
    """
    try:
        api_key = os.getenv("MINIMAX_API_KEY")
        endpoint = os.getenv("MINIMAX_ENDPOINT", "https://api.minimax.io/v1")
        model_name = os.getenv("MODEL_NAME", "minimax-m2.7")
        
        if not api_key or api_key == "your-minimax-api-key-here":
            return "Error: MINIMAX_API_KEY not configured"
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=endpoint,
            model=model_name,
            temperature=0.6
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional content editor. Rewrite the provided content 
            in a {tone} tone that is clear, engaging, and well-structured.
            
            Requirements:
            - Improve sentence flow and readability
            - Use professional vocabulary and phrasing
            - Maintain the original meaning and key points
            - Add smooth transitions between paragraphs
            - Keep appropriate length (don't expand unnecessarily)
            - Return ONLY the rewritten content, no explanations or notes"""),
            ("human", "Content to rewrite:\n{text}")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"text": text, "tone": tone})
        
        # Clean thinking tags and extract content only
        clean_content = _clean_llm_output(result.content)
        return clean_content
        
    except Exception as e:
        return f"Rewrite error: {str(e)}"


def _clean_llm_output(content: str) -> str:
    """Remove thinking tags and extract actual content from LLM response.
    
    Args:
        content: Raw LLM response content
        
    Returns:
        Cleaned content with thinking removed
    """
    # Remove <|思索|> tags and their contents
    clean = re.sub(r'<\|思索\|>.*?<\|思索\|>', '', content, flags=re.DOTALL)
    
    # Remove <think> tags and their contents  
    clean = re.sub(r'<think>.*?', '', clean, flags=re.DOTALL)
    
    # If there's content before the first markdown heading, remove it
    # This handles cases where LLM includes reasoning before the actual content
    lines = clean.split('\n')
    first_heading_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('```'):
            first_heading_idx = i
            break
    
    if first_heading_idx > 0:
        clean = '\n'.join(lines[first_heading_idx:])
    
    return clean.strip()


# Tool definitions for LangChain
grammar_check_tool = Tool(
    name="grammar_check",
    description="Fix grammar, spelling, and punctuation errors in text. Use this first to clean up any errors before SEO optimization or rewriting.",
    func=grammar_check
)

format_seo_tool = Tool(
    name="format_seo",
    description="Optimize content for SEO by integrating keywords naturally. Takes text and a list of keywords to incorporate into headings and body text for search engine visibility.",
    func=format_seo
)

rewrite_professional_tool = Tool(
    name="rewrite_professional",
    description="Rewrite content in a professional tone. Improves clarity, flow, and engagement while maintaining the original meaning. Use after grammar check and SEO formatting.",
    func=rewrite_professional
)
