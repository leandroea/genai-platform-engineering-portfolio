"""Editor Agent - polishes, proofreads, and optimizes content for SEO."""

import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from src.state.content_state import ContentState
from src.tools.seo_tools import (
    grammar_check_tool, 
    format_seo_tool, 
    rewrite_professional_tool,
    grammar_check,
    format_seo,
    rewrite_professional
)


def create_editor_agent():
    """Create the Editor Agent with grammar, SEO, and rewriting tools.
    
    The Editor Agent is responsible for:
    - Proofreading and fixing grammar errors
    - Optimizing content for SEO keywords
    - Rewriting content in a professional tone
    
    Tools:
    - grammar_check(text): Fix grammar, spelling, and punctuation
    - format_seo(text, keywords): Optimize for SEO with keyword integration
    - rewrite_professional(text): Rewrite in professional tone
    
    Returns:
        A configured editor agent
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    endpoint = os.getenv("MINIMAX_ENDPOINT", "https://api.minimax.io/v1")
    model_name = os.getenv("MODEL_NAME", "minimax-m2.7")
    
    if not api_key or api_key == "your-minimax-api-key-here":
        raise ValueError("MINIMAX_API_KEY not configured in .env file")
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=endpoint,
        model=model_name,
        temperature=0.4
    )
    
    system_prompt = """You are the Editor Agent in the Content Creation Studio multi-agent system.
Your role is to polish, proofread, and optimize content for SEO.

Available tools:
- grammar_check(text): Fix grammar, spelling, and punctuation errors
- format_seo(text, keywords): Optimize content for SEO keywords
- rewrite_professional(text, tone): Rewrite in a professional tone

ReAct pattern: Reason → Act → Observe → Repeat

Your workflow:
1. First, use grammar_check to fix any errors in the draft
2. Then use format_seo to integrate keywords and optimize structure
3. Optionally use rewrite_professional for tone improvements
4. Verify the final content meets quality standards

Quality checklist:
- Grammatically correct
- Keywords naturally integrated (no keyword stuffing)
- Clear structure with proper headings
- Professional tone
- Readable flow

IMPORTANT: Do NOT include any thinking process, reasoning, or internal analysis in your output.
Return ONLY the final edited content with no explanations, notes, or meta-commentary about your process."""
    
    tools = [grammar_check_tool, format_seo_tool, rewrite_professional_tool]
    agent = create_react_agent(llm, tools=tools, state_modifier=system_prompt)
    return agent


def editor_agent_node(state: ContentState) -> ContentState:
    """Editor agent node function for the LangGraph workflow.
    
    Args:
        state: The current ContentState with draft
        
    Returns:
        Updated ContentState with final_content after editing
    """
    draft = state.get("draft", "")
    keywords = state.get("keywords", [])
    revision_notes = state.get("revision_notes", "")
    
    if not draft:
        state["current_agent"] = "editor_error"
        state["final_content"] = ""
        return state
    
    try:
        # Step 1: Grammar check
        checked = grammar_check(draft)
        
        if "error" in checked.lower() and "not configured" in checked.lower():
            state["current_agent"] = "editor_error"
            state["final_content"] = ""
            state["revision_notes"] = checked
            return checked
        
        # Step 2: SEO optimization
        seo_optimized = format_seo(checked, keywords)
        
        if "error" in seo_optimized.lower() and "not configured" in seo_optimized.lower():
            state["current_agent"] = "editor_error"
            state["final_content"] = ""
            state["revision_notes"] = seo_optimized
            return state
        
        # Step 3: Professional rewrite for final polish
        final = rewrite_professional(seo_optimized, "professional")
        
        if "error" in final.lower() and "not configured" in final.lower():
            state["current_agent"] = "editor_error"
            state["final_content"] = ""
            state["revision_notes"] = final
            return state
        
        # Incorporate any revision notes if provided
        if revision_notes:
            # If there are specific revision requests, note them
            pass
        
        state["edited_content"] = checked
        state["final_content"] = final
        state["current_agent"] = "editor_complete"
        
    except Exception as e:
        state["current_agent"] = "editor_error"
        state["final_content"] = ""
        state["revision_notes"] = f"Editing error: {str(e)}"
    
    return state
