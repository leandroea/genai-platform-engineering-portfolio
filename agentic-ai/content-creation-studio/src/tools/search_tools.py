"""Web search and fact extraction tools using DuckDuckGo (free, no API key)."""

import os
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.tools import Tool

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return formatted results.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing search results with titles, URLs, and snippets
    """
    if DDGS is None:
        return "Error: No search client available. Please install duckduckgo-search or ddgs package."
    
    try:
        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=10))
            
        if not search_results:
            return "No search results found for query: " + query
        
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "No title")
            url = result.get("href", "No URL")
            body = result.get("body", "No description")
            formatted_results.append(f"[{i}] {title}\nURL: {url}\nSnippet: {body}\n")
        
        return "\n---\n".join(formatted_results)
        
    except Exception as e:
        return f"Search error: {str(e)}"


def extract_facts(content: str, topic: str) -> str:
    """Extract relevant facts from search content based on the topic.
    
    This tool uses the LLM to identify and extract key facts from the content.
    
    Args:
        content: The search results content to extract facts from
        topic: The topic to focus fact extraction on
        
    Returns:
        A JSON-formatted string of extracted facts with source, fact, and relevance
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
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
            ("system", """Extract facts from the search content about the topic. Return ONLY a JSON array like: [[[\"source\", \"url\"], [\"fact\", \"text\"], [\"relevance\", \"high\"]]]. 
            
            Rules:
            - Do NOT include any thinking process, reasoning, or internal analysis
            - Do NOT wrap the response in markdown code blocks
            - Return ONLY the raw JSON array, no explanations or notes"""),
            ("human", "Topic: {topic}\n\nSearch Content:\n{content}")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"topic": topic, "content": content})
        
        # Extract and clean the LLM response content
        raw_content = result.content.strip()
        
        # Strip thinking/reasoning content between various tags
        import re
        clean_content = re.sub(r'<\|思索\|>.*?<\|思索\|>', '', raw_content, flags=re.DOTALL)
        clean_content = re.sub(r'<think>.*?</think>', '', clean_content, flags=re.DOTALL)
        clean_content = clean_content.strip()
        
        # Try to extract JSON array from the response - find the first [ and last ]
        json_start = clean_content.find('[')
        json_end = clean_content.rfind(']')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = clean_content[json_start:json_end+1]
            return json_str
        
        # If no JSON found, return the cleaned content
        return clean_content if clean_content else raw_content
        
    except Exception as e:
        return f"Fact extraction error: {str(e)}"


# Tool definitions for LangChain
web_search_tool = Tool(
    name="web_search",
    description="Search the web using DuckDuckGo. Useful for finding current information, facts, and sources on any topic. Input is a search query string.",
    func=web_search
)

extract_facts_tool = Tool(
    name="extract_facts",
    description="Extract relevant facts from search content based on a topic. Returns structured facts with source, fact text, and relevance rating.",
    func=extract_facts
)
