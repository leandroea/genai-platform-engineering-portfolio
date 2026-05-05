"""Writer Agent - generates content drafts from research facts."""

import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from src.state.content_state import ContentState
from src.tools.writing_tools import write_draft_tool, structure_outline_tool, write_draft, structure_outline


def create_writer_agent():
    """Create the Writer Agent with drafting and outlining tools.
    
    The Writer Agent is responsible for:
    - Creating structured drafts from research facts
    - Generating content outlines
    - Incorporating keywords naturally into content
    
    Tools:
    - write_draft(topic, facts, keywords): Generate a full content draft
    - structure_outline(topic): Create a content outline for planning
    
    Returns:
        A configured writer agent
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
        temperature=0.7
    )
    
    system_prompt = """You are the Writer Agent in the Content Creation Studio multi-agent system.
Your role is to create well-structured content drafts based on research facts.

Available tools:
- write_draft(topic, facts, keywords): Generate a full content draft (500-800 words)
- structure_outline(topic): Create a content outline for planning

ReAct pattern: Reason → Act → Observe → Repeat

Your workflow:
1. Review the topic and research facts provided
2. Optionally create an outline first for long content
3. Write a comprehensive draft that:
   - Has a compelling introduction
   - Includes clear section headings
   - Presents facts with good transitions
   - Has a strong conclusion
4. If revision notes exist, incorporate the feedback into a new draft

IMPORTANT: Do NOT include any thinking process, reasoning, or internal analysis in your output.
Return ONLY the final content with no explanations, notes, or meta-commentary about your process."""
    
    tools = [write_draft_tool, structure_outline_tool]
    agent = create_react_agent(llm, tools=tools, state_modifier=system_prompt)
    return agent


def writer_agent_node(state: ContentState) -> ContentState:
    """Writer agent node function for the LangGraph workflow.
    
    Args:
        state: The current ContentState with topic and facts
        
    Returns:
        Updated ContentState with draft generated
    """
    topic = state.get("topic", "")
    facts = state.get("facts", [])
    keywords = state.get("keywords", [])
    revision_notes = state.get("revision_notes", "")
    
    if not topic:
        state["current_agent"] = "writer_error"
        state["draft"] = ""
        return state
    
    # Proceed with writing even if facts are empty - LLM has knowledge about the topic
    # Format facts for the LLM
    import json
    facts_str = ""
    if facts and isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, dict):
                source = fact.get("source", "Unknown")
                fact_text = fact.get("fact", str(fact))
                relevance = fact.get("relevance", "medium")
                facts_str += f"- [{relevance.upper()}] {fact_text} (Source: {source})\n"
            else:
                facts_str += f"- {str(fact)}\n"
    elif not facts:
        facts_str = f"Topic: {topic}. Generate content based on general knowledge about this topic."
    else:
        facts_str = str(facts)
    
    # Include revision notes if this is a revision
    if revision_notes:
        facts_str += f"\n\nRevision Notes to incorporate:\n{revision_notes}"
    
    try:
        # Write the draft
        draft = write_draft(topic, facts_str, keywords)
        
        if "error" in draft.lower() and "not configured" in draft.lower():
            state["current_agent"] = "writer_error"
            state["draft"] = ""
            state["revision_notes"] = draft
            return state
        
        state["draft"] = draft
        state["current_agent"] = "writer_complete"
        # Clear revision notes after successful revision
        state["revision_notes"] = ""
        
    except Exception as e:
        state["current_agent"] = "writer_error"
        state["draft"] = ""
        state["revision_notes"] = f"Writing error: {str(e)}"
    
    return state
