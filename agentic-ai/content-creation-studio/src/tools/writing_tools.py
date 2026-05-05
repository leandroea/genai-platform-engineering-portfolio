"""Writing and drafting tools for content creation."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def write_draft(topic: str, facts: str, keywords: Optional[list[str]] = None) -> str:
    """Write a content draft based on topic and research facts.
    
    Args:
        topic: The content topic
        facts: Extracted research facts in JSON or text format
        keywords: Optional list of SEO keywords to incorporate
        
    Returns:
        The generated draft content
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
            temperature=0.7
        )
        
        keyword_str = ", ".join(keywords) if keywords else "Not specified"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional content writer. Create a well-structured article draft 
            based on the provided topic and research facts.
            
            The draft should:
            - Have a compelling introduction that hooks the reader
            - Include clear section headings
            - Have a logical flow with introduction, body paragraphs, and conclusion
            - Be approximately 500-800 words
            - Naturally incorporate the provided keywords
            - Present facts clearly with good transitions
            - Do NOT include any thinking process, reasoning, or internal analysis in the output
            - Return ONLY the final article content, no explanations, notes, or meta-commentary
            
            Do NOT start with "Here's a draft" or similar filler phrases."""),
            ("human", """Topic: {topic}
            
            Keywords to incorporate: {keywords}
            
            Research Facts:
            {facts}""")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"topic": topic, "keywords": keyword_str, "facts": facts})
        return result.content
        
    except Exception as e:
        return f"Draft writing error: {str(e)}"


def structure_outline(topic: str, target_words: int = 500) -> str:
    """Create a content outline for a given topic.
    
    Args:
        topic: The content topic
        target_words: Target word count for the content (default 500)
        
    Returns:
        A structured outline with sections and key points
    """
    try:
        api_key = os.getenv("MINIMAX_API_KEY")
        endpoint = os.getenv("MINIMAX_ENDPOINT", "https://api.minimax.chat/v1")
        model_name = os.getenv("MODEL_NAME", "minimax-m2.7")
        
        if not api_key or api_key == "your-minimax-api-key-here":
            return "Error: MINIMAX_API_KEY not configured"
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=endpoint,
            model=model_name,
            temperature=0.5
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a content structuring expert. Create a detailed outline for an article
            on the given topic.
            
            The outline should include:
            - A compelling title
            - An introduction hook
            - 3-5 main sections with subsections
            - A conclusion with call-to-action
            
            Format as a hierarchical outline using numbers and bullets.
            Target approximately {target_words} words total.
            
            Do NOT include any thinking process, reasoning, or internal analysis.
            Return ONLY the outline content."""),
            ("human", "Topic: {topic}")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"topic": topic, "target_words": target_words})
        return result.content
        
    except Exception as e:
        return f"Outline structuring error: {str(e)}"


# Tool definitions for LangChain
write_draft_tool = Tool(
    name="write_draft",
    description="Write a full content draft based on topic and research facts. Incorporates keywords naturally and creates a well-structured article with introduction, body, and conclusion.",
    func=write_draft
)

structure_outline_tool = Tool(
    name="structure_outline",
    description="Create a structured outline for content on a given topic. Useful for planning articles before writing, or for structuring long-form content.",
    func=structure_outline
)
