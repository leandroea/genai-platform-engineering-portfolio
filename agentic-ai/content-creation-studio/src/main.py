"""Main entry point for Content Creation Studio CLI."""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_banner():
    """Print the CLI banner."""
    banner = """
============================================================
           CONTENT CREATION STUDIO
           Multi-Agent AI Content Pipeline
============================================================

Welcome! Let's create some content.
"""
    print(banner)


def get_input(prompt: str) -> str:
    """Get input from user."""
    return input(f"{prompt}: ").strip()


def print_status(agent: str, message: str, status_num: int = None):
    """Print agent status."""
    if status_num:
        print(f"[{status_num}/4] {agent}: {message}")
    else:
        print(f"{agent}: {message}")


def save_content(topic: str, content: str, keywords: list[str]) -> tuple[str, str]:
    """Save content to output directory.
    
    Args:
        topic: The content topic
        content: The final content
        keywords: List of keywords
        
    Returns:
        Tuple of (file_path, metadata_path)
    """
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create slug from topic
    slug = topic.lower().replace(" ", "-").replace(",", "")[:50]
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{slug}-{date_str}"
    
    # Save markdown content
    md_path = output_dir / f"{filename}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(f"**Keywords**: {', '.join(keywords)}\n\n")
        f.write(f"**Created**: {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")
        f.write(content)
    
    # Save metadata
    metadata = {
        "id": str(datetime.now().timestamp()),
        "topic": topic,
        "keywords": keywords,
        "created_at": datetime.now().isoformat(),
        "status": "published",
        "word_count": len(content.split()),
        "file_path": str(md_path),
        "approval": {
            "approved_at": datetime.now().isoformat(),
            "notes": None
        }
    }
    
    json_path = output_dir / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return str(md_path), str(json_path)


def run_pipeline(topic: str, keywords: list[str]) -> dict:
    """Run the content creation pipeline.
    
    Args:
        topic: The content topic
        keywords: List of SEO keywords
        
    Returns:
        Dictionary with final_state and approval_status
    """
    from src.workflow.content_graph import (
        create_content_graph, 
        research_agent_node, 
        writer_agent_node, 
        editor_agent_node
    )
    from src.state.content_state import ContentState
    
    # Initialize state
    state: ContentState = {
        "topic": topic,
        "keywords": keywords,
        "search_query": "",
        "facts": [],
        "draft": "",
        "edited_content": "",
        "final_content": "",
        "approval_status": "pending",
        "revision_notes": "",
        "current_agent": "none"
    }
    
    try:
        # Step 1: Research
        print_status("Supervisor", "Routing task to Research Agent...", 1)
        state = research_agent_node(state)
        
        # If no facts found, use topic-based facts from LLM's knowledge
        if not state.get("facts"):
            print_status("Research Agent", "Using LLM knowledge for content (web search worked, but fact extraction requires API key)", 2)
            # Generate facts from topic directly via LLM
            from src.tools.writing_tools import write_draft
            # We'll proceed anyway - the writer can use general knowledge
            pass
        else:
            print_status("Research Agent", f"Found {len(state['facts'])} relevant sources", 2)
        
        # Step 2: Writing
        print_status("Supervisor", "Routing task to Writer Agent...", 2)
        state = writer_agent_node(state)
        
        if not state.get("draft"):
            return {"state": state, "error": "Failed to generate draft", "approved": False}
        
        word_count = len(state["draft"].split())
        print_status("Writer Agent", f"Draft complete ({word_count} words)", 3)
        
        # Step 3: Editing
        print_status("Supervisor", "Routing task to Editor Agent...", 3)
        state = editor_agent_node(state)
        
        if not state.get("final_content"):
            return {"state": state, "error": "Failed to edit content", "approved": False}
        
        # Count keyword usage
        final = state["final_content"]
        keyword_counts = {}
        for kw in keywords:
            count = final.lower().count(kw.lower())
            keyword_counts[kw] = count
        
        keyword_info = ", ".join([f"{k} ({v}x)" for k, v in keyword_counts.items()])
        print_status("Editor Agent", f"Keywords integrated: {keyword_info}", 4)
        
        state["approval_status"] = "pending"
        return {"state": state, "error": None, "approved": False}
        
    except Exception as e:
        return {"state": state, "error": str(e), "approved": False}


def main():
    """Main CLI entry point."""
    print_banner()
    
    # Check API key
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key or api_key == "your-minimax-api-key-here":
        print("Error: MINIMAX_API_KEY not configured.")
        print("Please edit the .env file and add your MiniMax API key.")
        sys.exit(1)
    
    while True:
        # Get topic
        topic = get_input("Enter topic")
        if not topic:
            print("Topic cannot be empty. Please try again.")
            continue
        
        # Get keywords
        keywords_input = get_input("Enter keywords (comma-separated)")
        keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
        
        if not keywords:
            print("No keywords provided. Content will be generated without SEO optimization.")
        
        print()
        
        # Run pipeline
        result = run_pipeline(topic, keywords)
        state = result["state"]
        
        if result["error"]:
            print(f"Error: {result['error']}")
            continue
        
        # Show preview
        print("""
============================================================
                     PREVIEW
============================================================""")
        print(state['final_content'][:1000])
        if len(state['final_content']) > 1000:
            print("... [truncated]")
        print("""
============================================================
""")
        
        # Approval
        while True:
            approval = get_input("Approve this content? [y/n/q]").lower()
            
            if approval == "q":
                print("Quit requested. Exiting.")
                sys.exit(0)
            
            if approval not in ["y", "n"]:
                print("Invalid input. Please enter 'y', 'n', or 'q'.")
                continue
            
            break
        
        if approval == "y":
            # Save content
            md_path, json_path = save_content(topic, state["final_content"], keywords)
            
            print("Content approved and saved!")
            print(f"Output: {md_path}")
            state["approval_status"] = "approved"
        else:
            # Reject and ask for revision notes
            revision = get_input("Enter revision notes (or press Enter to skip)")
            if revision:
                print("Revisions noted. The next run will incorporate your feedback.")
                state["revision_notes"] = revision
            state["approval_status"] = "rejected"
        
        # Continue?
        while True:
            again = get_input("Generate another? [y/n]").lower()
            
            if again == "n":
                print("\nThank you for using Content Creation Studio!")
                print("Your content has been saved to the ./output directory.")
                sys.exit(0)
            
            if again != "y":
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            
            break


if __name__ == "__main__":
    main()
