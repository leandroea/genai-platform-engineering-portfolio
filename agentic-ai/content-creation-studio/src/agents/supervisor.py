"""Supervisor Agent - coordinates the content creation workflow."""

import os
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from src.state.content_state import ContentState


def create_supervisor_agent():
    """Create the Supervisor agent that routes tasks to subordinate agents.
    
    The Supervisor is responsible for:
    - Validating user input (topic, keywords)
    - Routing tasks to Research, Writer, or Editor agents
    - Managing the overall workflow state
    
    Returns:
        A configured supervisor agent
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
        temperature=0.3
    )
    
    system_prompt = """You are the Supervisor Agent in the Content Creation Studio multi-agent system.
Your role is to coordinate the content creation workflow by routing tasks to the appropriate subordinate agents.

Available routes:
- "research": Route to Research Agent for web search and fact extraction
- "writer": Route to Writer Agent for drafting content
- "editor": Route to Editor Agent for proofreading and SEO optimization
- "approval": Route to human approval gate
- "finish": Complete the workflow

Workflow sequence:
1. When you receive a topic, validate it (not empty, reasonable length)
2. Route to "research" to gather facts about the topic
3. Route to "writer" to create a draft from the facts
4. Route to "editor" to polish and optimize the content
5. Route to "approval" for human review
6. Based on approval, either "finish" or return to "writer" for revisions

IMPORTANT: Do NOT include any thinking process, reasoning, or internal analysis in your output.
Respond ONLY with the routing decision, no explanations or meta-commentary about your reasoning process.

Respond in the following format:
THOUGHT: [your reasoning]
ROUTE: [research/writer/editor/approval/finish]"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Current state:\n{state}\n\nWhat should be the next step in the content creation workflow?")
    ])
    
    agent = create_react_agent(llm, tools=[], state_modifier=system_prompt)
    return agent


def validate_input(topic: str, keywords: list[str] | None = None) -> tuple[bool, str]:
    """Validate user input for content creation.
    
    Args:
        topic: The topic to validate
        keywords: Optional list of keywords
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not topic or not topic.strip():
        return False, "Topic cannot be empty"
    
    if len(topic.strip()) < 3:
        return False, "Topic is too short (minimum 3 characters)"
    
    if len(topic) > 500:
        return False, "Topic is too long (maximum 500 characters)"
    
    if keywords:
        if not isinstance(keywords, list):
            return False, "Keywords must be a list"
        
        if len(keywords) > 20:
            return False, "Too many keywords (maximum 20)"
        
        for kw in keywords:
            if len(kw) > 100:
                return False, f"Keyword '{kw[:20]}...' is too long (maximum 100 characters)"
    
    return True, ""


def supervisor_node(state: ContentState) -> ContentState:
    """Supervisor node function for the LangGraph workflow.
    
    This function is called at each step to determine routing.
    
    Args:
        state: The current ContentState
        
    Returns:
        Updated ContentState with routing decision
    """
    topic = state.get("topic", "")
    keywords = state.get("keywords", [])
    current_agent = state.get("current_agent", "none")
    
    # Validate input if this is the first step
    if current_agent == "none":
        is_valid, error_msg = validate_input(topic, keywords)
        if not is_valid:
            state["current_agent"] = "error"
            state["revision_notes"] = error_msg
            return state
    
    # Determine next route based on current state
    if current_agent == "error":
        # Stay in error state, let user handle
        return state
    
    # Check workflow progress and route accordingly
    facts = state.get("facts", [])
    draft = state.get("draft", "")
    final_content = state.get("final_content", "")
    
    if not facts:
        next_route = "research"
    elif not draft:
        next_route = "writer"
    elif not final_content:
        next_route = "editor"
    else:
        next_route = "approval"
    
    state["current_agent"] = f"supervisor_{next_route}"
    return state


def route_to_agent(state: ContentState) -> Literal["research_agent", "writer_agent", "editor_agent", "approval", "finish"]:
    """Route to the appropriate agent based on current state.
    
    Args:
        state: The current ContentState
        
    Returns:
        The name of the next agent to route to
    """
    facts = state.get("facts", [])
    draft = state.get("draft", "")
    final_content = state.get("final_content", "")
    approval_status = state.get("approval_status", "pending")
    current_agent = state.get("current_agent", "none")
    
    if approval_status == "rejected":
        # If rejected, send back to writer for revision
        return "writer_agent"
    
    # Check if research is complete (facts exist or research was already done)
    research_done = current_agent in ("research_complete", "research_error") or len(facts) > 0
    
    if not research_done:
        return "research_agent"
    elif not draft:
        return "writer_agent"
    elif not final_content:
        return "editor_agent"
    else:
        return "approval"
