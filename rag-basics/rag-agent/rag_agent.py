"""
RAG Agent for PDF Question Answering
Uses LlamaIndex with Chroma vector store and NVIDIA API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

# Configuration
PDF_PATH = "AI Engineering.pdf"
PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.5-397b-a17b")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def main():
    print("=" * 50)
    print("RAG Agent - PDF Q&A (LlamaIndex + Ollama + NVIDIA)")
    print("=" * 50)
    
    # Setup embedding model
    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Load PDF
    print(f"Loading PDF: {PDF_PATH}")
    documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
    print(f"Loaded {len(documents)} document nodes")
    
    # Check for existing index
    chroma_collection = "rag_agent"
    chroma_db_file = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    
    # Load or create index
    if os.path.exists(chroma_db_file):
        try:
            # Load existing index - use chromadb client
            client = chromadb.PersistentClient(path=PERSIST_DIR)
            collection = client.get_collection(name=chroma_collection)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
            print("Loaded existing index")
        except Exception as e:
            print(f"Could not load existing index: {e}")
            raise
    else:
        # Create new index
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        collection = client.get_or_create_collection(name=chroma_collection)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
        print("Created new index")
    
    retriever = index.as_retriever(similarity_top_k=5)
    
    print("\n" + "=" * 50)
    print("RAG Agent is ready! Ask questions about the PDF.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50 + "\n")
    
    # Query loop
    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nSearching...")
            nodes = retriever.retrieve(question)
            
            if not nodes:
                print("No relevant documents found.")
                continue
            
            # Query NVIDIA
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("NVIDIA_API_KEY"), base_url=OPENAI_API_BASE)
            context = "\n\n".join([n.text for n in nodes])
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Answer based on the context. If you don't know, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.7,
            )
            
            print("\nAnswer:", response.choices[0].message.content)
            
            print("\n--- Sources ---")
            for i, n in enumerate(nodes[:3], 1):
                print(f"{i}. Page {n.metadata.get('page_label', '?')}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()