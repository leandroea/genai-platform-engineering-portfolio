"""
RAG Agent for PDF Question Answering
Uses Chroma for vector storage and NVIDIA's Qwen model for generation.
Uses modern LangChain Expression Language (LCEL).
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "AI Engineering.pdf"
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.5-397b-a17b")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1")


def load_pdf(pdf_path: str):
    """Load PDF document using PyPDFLoader."""
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks for embedding."""
    print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, persist_directory: str):
    """Create and persist Chroma vector store from document chunks."""
    print(f"Creating Chroma vector store at: {persist_directory}")
    
    # Use NVIDIA embeddings
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2",
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    )
    
    # Create or load vector store
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    else:
        print("Creating new vector store...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        vectorstore.persist()
    
    return vectorstore


def create_qa_chain(vectorstore):
    """Create a retrieval-augmented question answering chain using LCEL."""
    print("Creating QA chain...")
    
    # Initialize the LLM (OpenAI-compatible, pointing to NVIDIA)
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=os.getenv("NVIDIA_API_KEY"),
        openai_api_base=OPENAI_API_BASE,
        temperature=0.7,
        max_tokens=2048,
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    
    # Custom prompt using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the provided context.
    If you don't know the answer based on the context, say that you don't know.
    Don't make up information that's not directly supported by the context.

    Context:
    {context}

    Question: {input}

    Answer:""")
    
    # Create the combine documents chain (handles formatting docs into context)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the full retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain


def main():
    """Main function to run the RAG agent."""
    print("=" * 50)
    print("RAG Agent - PDF Question Answering")
    print("=" * 50)
    
    # Check for NVIDIA API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("Error: NVIDIA_API_KEY not found in environment variables")
        print("Please set your API key in the .env file")
        return
    
    # Load PDF and create vector store
    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, PERSIST_DIRECTORY)
    
    # Create QA chain
    qa_chain = create_qa_chain(vectorstore)
    
    print("\n" + "=" * 50)
    print("RAG Agent is ready! Ask questions about the PDF.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50 + "\n")
    
    # Interactive question answering loop
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nSearching for answer...")
            result = qa_chain.invoke({"input": question})
            
            print("\nAnswer:")
            print(result["answer"])
            
            # Show source documents if available
            if result.get("context"):
                print("\n--- Source References ---")
                for i, doc in enumerate(result["context"][:3], 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    print(f"{i}. {source} (Page {page})")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
