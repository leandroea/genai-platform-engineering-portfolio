"""Research Agent - gathers and extracts facts from web search."""

import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from src.state.content_state import ContentState
from src.tools.search_tools import web_search_tool, extract_facts_tool, web_search, extract_facts


def create_research_agent():
    """Create the Research Agent with web search and fact extraction tools.
    
    The Research Agent is responsible for:
    - Searching the web for relevant information on the topic
    - Extracting structured facts from search results
    - Organizing facts with source attribution and relevance ratings
    
    Tools:
    - web_search: DuckDuckGo search to find relevant sources
    - extract_facts: LLM-based extraction of key facts from content
    
    Returns:
        A configured research agent
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
    
    system_prompt = """You are the Research Agent in the Content Creation Studio multi-agent system.
Your role is to gather factual information about the topic provided by the user.

Available tools:
- web_search(query): Search DuckDuckGo for relevant information. Input a search query string.
- extract_facts(content, topic): Extract structured facts from search results. Returns JSON with source, fact, and relevance.

ReAct pattern: Reason → Act → Observe → Repeat

Your workflow:
1. First, construct an optimized search query based on the topic
2. Use web_search to find relevant sources
3. Use extract_facts to extract key facts from the results
4. If results are insufficient, try different search queries
5. Organize facts by relevance and source quality

IMPORTANT: Do NOT include any thinking process, reasoning, or internal analysis in your output.
Return ONLY the structured facts with no explanations, notes, or meta-commentary about your process."""
    
    tools = [web_search_tool, extract_facts_tool]
    agent = create_react_agent(llm, tools=tools, state_modifier=system_prompt)
    return agent


def research_agent_node(state: ContentState) -> ContentState:
    """Research agent node function for the LangGraph workflow.
    
    Args:
        state: The current ContentState with topic
        
    Returns:
        Updated ContentState with facts extracted
    """
    topic = state.get("topic", "")
    if not topic:
        state["current_agent"] = "research_error"
        state["facts"] = []
        return state
    
    try:
        # Build search query from topic
        search_query = f"{topic} facts information"
        
        # Execute search
        search_results = web_search(search_query)
        
        if "error" in search_results.lower() or "no search results" in search_results.lower():
            state["facts"] = []
            state["current_agent"] = "research_complete"
            return state
        
        # Extract facts from search results
        facts_json = extract_facts(search_results, topic)
        
        # Parse facts into structured format if possible
        import json
        facts = []
        
        # Check if extract_facts returned an error (API key not configured)
        if "not configured" in facts_json.lower() or "error" in facts_json.lower():
            state["facts"] = []
            state["search_query"] = search_query
            state["current_agent"] = "research_complete"
            return state
        
        try:
            # Try to parse as JSON
            raw_facts = json.loads(facts_json)
            # Handle nested array format: [[["source", "url"], ["fact", "text"], ["relevance", "level"]]]
            facts = []
            for fact_array in raw_facts:
                if isinstance(fact_array, list) and len(fact_array) >= 3:
                    # Each fact is [["source", "url"], ["fact", "text"], ["relevance", "level"]]
                    source = fact_array[0][1] if isinstance(fact_array[0], list) and len(fact_array[0]) > 1 else "unknown"
                    fact_text = fact_array[1][1] if isinstance(fact_array[1], list) and len(fact_array[1]) > 1 else ""
                    relevance = fact_array[2][1] if isinstance(fact_array[2], list) and len(fact_array[2]) > 1 else "medium"
                    facts.append({"source": source, "fact": fact_text, "relevance": relevance})
                elif isinstance(fact_array, list):
                    # Handle flat array case
                    facts.append({"source": "web_search", "fact": str(fact_array), "relevance": "medium"})
            if not facts:
                facts = [{"source": "web_search", "fact": facts_json, "relevance": "medium"}]
        except (json.JSONDecodeError, TypeError):
            # If not JSON, store as text
            facts = [{"source": "web_search", "fact": facts_json, "relevance": "medium"}]
        
        state["facts"] = facts
        state["search_query"] = search_query
        state["current_agent"] = "research_complete"
        
    except Exception as e:
        state["facts"] = []
        state["current_agent"] = "research_error"
        state["revision_notes"] = f"Research error: {str(e)}"
    
    return state
