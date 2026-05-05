"""
LLM CLI Assistant - A command-line interface for interacting with LLMs
Features:
- LLM API integration (NVIDIA API with Qwen model)
- Response streaming
- Chat history
- Custom prompts
"""

import os
import sys
from typing import Optional, List
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from rich.theme import Theme
import typer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk
from config import OPENAI_API_KEY, MODEL, NVIDIA_BASE_URL

# Custom theme for rich
custom_theme = Theme({
    "user": "cyan",
    "assistant": "green",
    "system": "yellow",
    "error": "red bold"
})
console = Console(theme=custom_theme)

app = typer.Typer(help="LLM CLI Assistant - Chat with AI models via NVIDIA API")

# Global chat history
chat_history: List = []
system_prompt: Optional[str] = None


def get_llm(streaming: bool = True) -> ChatOpenAI:
    """Initialize and return the LLM client."""
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=OPENAI_API_KEY,
        base_url=NVIDIA_BASE_URL,
        streaming=streaming,
        temperature=0.7,
        max_tokens=16384
    )


def build_messages(user_input: str) -> List:
    """Build the message list including system prompt and chat history."""
    messages = []
    
    # Add system prompt if set
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    # Add chat history
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Add current user input
    messages.append(HumanMessage(content=user_input))
    
    return messages


def stream_response(llm: ChatOpenAI, messages: List) -> str:
    """Stream the LLM response and return the complete response."""
    full_response = ""
    
    console.print("\n[assistant]Assistant:[/assistant] ", end="")
    
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                rprint(chunk.content, end="")
        console.print()  # New line after streaming
    except Exception as e:
        console.print(f"\n[error]Error: {str(e)}[/error]")
        return ""
    
    return full_response


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send to the AI"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable/disable response streaming"),
    clear: bool = typer.Option(False, "--clear", help="Clear chat history"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="Set a system prompt"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the model"),
):
    """
    Start a chat session with the LLM.
    
    If no message is provided, enters interactive mode.
    """
    global chat_history, system_prompt
    
    # Handle clear option
    if clear:
        chat_history = []
        system_prompt = None
        console.print("[yellow]Chat history cleared.[/yellow]")
        return
    
    # Handle system prompt
    if system:
        system_prompt = system
        console.print(f"[yellow]System prompt set to:[/yellow] {system}")
    
    # If no message provided, enter interactive mode
    if message is None:
        interactive_mode(model, stream)
        return
    
    # Single message mode
    process_message(message, stream, model)


def interactive_mode(model_override: Optional[str] = None, stream: bool = True):
    """Run the chat in interactive mode."""
    global chat_history, system_prompt
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold cyan]LLM CLI Assistant[/bold cyan]\n"
        f"Model: [yellow]{model_override or MODEL}[/yellow]\n"
        "Type 'quit' or 'exit' to end the session\n"
        "Type 'clear' to clear chat history\n"
        "Type 'system <prompt>' to set a system prompt\n"
        "Type 'history' to view chat history\n"
        "Type 'help' for more commands",
        title="Welcome"
    ))
    
    while True:
        try:
            user_input = Prompt.ask("\n[user]You[/user]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        
        if not user_input.strip():
            continue
        
        # Handle special commands
        if user_input.lower() in ["quit", "exit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break
        elif user_input.lower() == "clear":
            chat_history = []
            console.print("[yellow]Chat history cleared.[/yellow]")
            continue
        elif user_input.lower() == "history":
            show_history()
            continue
        elif user_input.lower().startswith("system "):
            system_prompt = user_input[7:].strip()
            console.print(f"[yellow]System prompt set to:[/yellow] {system_prompt}")
            continue
        elif user_input.lower() == "help":
            show_help()
            continue
        
        # Process the message
        process_message(user_input, stream, model_override)


def process_message(user_input: str, stream: bool = True, model_override: Optional[str] = None):
    """Process a single user message and get AI response."""
    global chat_history
    
    # Display user message
    console.print(f"\n[user]You:[/user] {user_input}")
    
    # Initialize LLM with optional model override
    llm = get_llm(streaming=stream)
    if model_override:
        llm.model = model_override
    
    # Build messages
    messages = build_messages(user_input)
    
    # Get response
    if stream:
        response = stream_response(llm, messages)
    else:
        try:
            result = llm.invoke(messages)
            response = result.content
            console.print(f"\n[assistant]Assistant:[/assistant] {response}")
        except Exception as e:
            console.print(f"[error]Error: {str(e)}[/error]")
            return
    
    # Add to chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})


def show_history():
    """Display chat history."""
    if not chat_history:
        console.print("[yellow]No chat history.[/yellow]")
        return
    
    console.print("\n[bold]Chat History:[/bold]")
    for i, msg in enumerate(chat_history):
        role = "[user]User[/user]" if msg["role"] == "user" else "[assistant]Assistant[/assistant]"
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        console.print(f"{i+1}. {role}: {content}")
    console.print()


def show_help():
    """Display help information."""
    console.print(Panel.fit(
        "[bold]Available Commands:[/bold]\n"
        "- quit/exit: End the session\n"
        "- clear: Clear chat history\n"
        "- system <prompt>: Set a system prompt\n"
        "- history: View chat history\n"
        "- help: Show this help message\n\n"
        "[bold]Options:[/bold]\n"
        "- --stream/--no-stream: Enable/disable streaming\n"
        "- --system, -s: Set system prompt\n"
        "- --model, -m: Override model\n"
        "- --clear: Clear chat history",
        title="Help"
    ))


@app.command()
def clear_history():
    """Clear chat history."""
    global chat_history, system_prompt
    chat_history = []
    system_prompt = None
    console.print("[yellow]Chat history and system prompt cleared.[/yellow]")


@app.command()
def set_system(prompt: str = typer.Argument(..., help="System prompt to set")):
    """Set a custom system prompt."""
    global system_prompt
    system_prompt = prompt
    console.print(f"[yellow]System prompt set to:[/yellow]\n{prompt}")


@app.command()
def show_system():
    """Show current system prompt."""
    if system_prompt:
        console.print(f"[yellow]Current system prompt:[/yellow]\n{system_prompt}")
    else:
        console.print("[yellow]No system prompt set.[/yellow]")


if __name__ == "__main__":
    app()
