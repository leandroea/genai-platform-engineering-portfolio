"""LangGraph workflow for Content Creation Studio.

This module defines the state graph that orchestrates the multi-agent content creation pipeline.
The workflow follows: Supervisor → Research → Writer → Editor → Approval Gate → Output
"""

import os
from typing import TypedDict, Literal, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.content_state import ContentState
from src.agents.supervisor import supervisor_node, route_to_agent
from src.agents.research_agent import research_agent_node
from src.agents.writer_agent import writer_agent_node
from src.agents.editor_agent import editor_agent_node


def create_content_graph() -> StateGraph:
    """Create the LangGraph StateGraph for content creation workflow.
    
    The workflow graph consists of:
    - supervisor_node: Validates input and manages routing
    - research_agent_node: Gathers facts from web search
    - writer_agent_node: Creates draft content
    - editor_agent_node: Polishes and optimizes content
    - approval_node: Human-in-the-loop approval gate
    
    Edges:
    - START → supervisor
    - supervisor → research_agent (if no facts yet)
    - supervisor → writer_agent (if facts exist, no draft)
    - supervisor → editor_agent (if draft exists, no final)
    - supervisor → approval (if final content exists)
    - editor_agent → approval
    - approval → writer_agent (if rejected, for revision)
    - approval → END (if approved)
    
    Returns:
        A configured StateGraph instance
    """
    # Define the workflow graph
    workflow = StateGraph(ContentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("writer_agent", writer_agent_node)
    workflow.add_node("editor_agent", editor_agent_node)
    workflow.add_node("approval", approval_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor based on state
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "research_agent": "research_agent",
            "writer_agent": "writer_agent",
            "editor_agent": "editor_agent",
            "approval": "approval",
            "finish": END
        }
    )
    
    # Research → back to supervisor for routing
    workflow.add_edge("research_agent", "supervisor")
    
    # Writer → back to supervisor for routing
    workflow.add_edge("writer_agent", "supervisor")
    
    # Editor → approval gate
    workflow.add_edge("editor_agent", "approval")
    
    # Approval conditional routing
    workflow.add_conditional_edges(
        "approval",
        lambda state: state.get("approval_status", "pending"),
        {
            "approved": END,
            "rejected": "writer_agent",  # Send back to writer for revision
            "pending": "supervisor"  # Should not happen, but handle it
        }
    )
    
    # Compile the graph with memory for state persistence
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    return graph


def approval_node(state: ContentState) -> ContentState:
    """Human approval gate node.
    
    This node pauses execution and awaits human input before proceeding.
    In CLI mode, it will prompt the user for approval.
    In API mode, it would return the content and wait for a separate approval call.
    
    Args:
        state: The current ContentState with final_content
        
    Returns:
        Updated ContentState with approval_status set
    """
    # This function will be replaced by the CLI's interactive approval
    # For now, set pending status and let the main.py handle the actual prompt
    if "approval_status" not in state or state.get("approval_status") == "pending":
        # In non-interactive mode (when revision_notes is "auto"), auto-approve
        if state.get("revision_notes") == "auto":
            state["approval_status"] = "approved"
        else:
            state["approval_status"] = "pending"
    
    state["current_agent"] = "approval"
    return state


def run_content_pipeline(topic: str, keywords: list[str] | None = None, 
                         interactive: bool = True) -> ContentState:
    """Run the complete content creation pipeline.
    
    Args:
        topic: The content topic
        keywords: Optional list of SEO keywords
        interactive: If True, use interactive CLI for approval
        
    Returns:
        The final ContentState with all content generated
    """
    # Create initial state
    initial_state: ContentState = {
        "topic": topic,
        "keywords": keywords or [],
        "search_query": "",
        "facts": [],
        "draft": "",
        "edited_content": "",
        "final_content": "",
        "approval_status": "pending",
        "revision_notes": "",
        "current_agent": "none"
    }
    
    # Create and run the graph
    graph = create_content_graph()
    
    # For non-interactive mode, auto-approve after generation
    if not interactive:
        config = {"configurable": {"thread_id": "content-creation"}}
        initial_state["revision_notes"] = "auto"
        # Run the graph
        for state in graph.stream(initial_state, config=config):
            pass
        
        # Return the final state
        return state
    
    # Interactive mode - run until approval needed
    config = {"configurable": {"thread_id": "content-creation"}}
    
    # Run through the pipeline
    current_state = initial_state
    step = 0
    
    while True:
        step += 1
        
        # Run one step of the graph
        try:
            result = graph.invoke(current_state, config=config)
            current_state = result
            
            # Check if we need approval
            if current_state.get("final_content"):
                return current_state
            
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            current_state["revision_notes"] = str(e)
            return current_state
    
    return current_state


# Export graph for direct usage
content_graph = create_content_graph()
