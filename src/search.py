import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        """Initialize the RAG search system.
        
        Args:
            persist_dir: Directory to store/load the FAISS index
            embedding_model: Name of the embedding model to use
            llm_model: Name of the language model to use for generation
            
        Raises:
            ValueError: If required environment variables are not set
        """
        # Initialize vector store
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Load or build vectorstore if not already loaded
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("[INFO] No existing FAISS index found. Building new index...")
            from data_loader import load_all_documents
            try:
                docs = load_all_documents("data")
                if not docs:
                    raise ValueError("No documents found in the data directory")
                self.vectorstore.build_from_documents(docs)
            except Exception as e:
                print(f"[ERROR] Failed to build vector store: {str(e)}")
                raise
        else:
            print("[INFO] Loading existing FAISS index...")
            try:
                self.vectorstore.load()
            except Exception as e:
                print(f"[ERROR] Failed to load vector store: {str(e)}")
                raise
        
        # Initialize LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        try:
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=llm_model,
                temperature=0.1,
                max_tokens=1024
            )
            print(f"[INFO] Successfully initialized Groq LLM: {llm_model}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Groq client: {str(e)}")
            raise

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Search for relevant documents and generate a summary.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            str: Generated summary or error message
        """
        if not query or not isinstance(query, str):
            return "Error: Invalid query"
            
        try:
            # Search for relevant documents
            results = self.vectorstore.query(query, top_k=top_k)
            
            # Extract and validate text content
            texts = []
            for r in results:
                if not isinstance(r, dict) or "metadata" not in r:
                    continue
                text = r["metadata"].get("text", "").strip()
                if text:
                    texts.append(text)
            
            if not texts:
                return "No relevant documents found for the given query."
                
            # Format context with source information
            context = "\n\n".join(
                f"[Document {i+1}]\n{text}" 
                for i, text in enumerate(texts)
            )
            
            # Create a more detailed prompt
            prompt = f"""You are a helpful AI assistant. Please provide a concise and accurate summary 
            based on the following documents that are relevant to the query: "{query}"
            
            Documents:
            {context}
            
            Instructions:
            1. Focus on information that directly addresses the query
            2. Be concise but comprehensive
            3. If the documents contain conflicting information, note the discrepancies
            4. If the query cannot be answered from the documents, state that clearly
            
            Summary:"""
            
            # Generate response with error handling
            try:
                response = self.llm.invoke(prompt)
                return response.content.strip()
            except Exception as e:
                print(f"[ERROR] Failed to generate response: {str(e)}")
                return "Sorry, I encountered an error while generating a response."
                
        except Exception as e:
            print(f"[ERROR] Search failed: {str(e)}")
            return "An error occurred while processing your request. Please try again later."

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)