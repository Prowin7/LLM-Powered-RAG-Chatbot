import os
import sys
from dotenv import load_dotenv
from typing import Optional

# Load environment variables first
load_dotenv()

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


def main():
    """Main entry point for the RAG application."""
    try:
        # Configuration
        data_dir = os.getenv("DATA_DIR", "data")
        persist_dir = os.getenv("PERSIST_DIR", "faiss_store")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        print("=" * 50)
        print("RAG System - Document Search and Summarization")
        print("=" * 50)
        
        # Initialize RAG system
        print("\n[INFO] Initializing RAG system...")
        rag_search = RAGSearch(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            llm_model=llm_model
        )
        
        # Interactive query loop
        print("\n" + "-" * 50)
        print("Enter your query (or 'exit' to quit):")
        while True:
            try:
                query = input("\n> ").strip()
                if query.lower() in ('exit', 'quit'):
                    print("\n[INFO] Exiting...")
                    break
                    
                if not query:
                    print("Please enter a valid query.")
                    continue
                    
                print("\n[INFO] Processing your query...")
                summary = rag_search.search_and_summarize(query, top_k=3)
                
                print("\n" + "=" * 50)
                print("RESULT:")
                print("=" * 50)
                print(summary)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n[INFO] Operation cancelled by user.")
                break
            except Exception as e:
                print(f"\n[ERROR] An error occurred: {str(e)}")
                print("Please try again or type 'exit' to quit.")
                
    except Exception as e:
        print(f"\n[FATAL] Failed to initialize RAG system: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
