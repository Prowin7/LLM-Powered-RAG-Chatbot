from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    # --------------------------------------------------
    # EXISTING METHODS (kept)
    # --------------------------------------------------
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        chunks = self.splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

    # --------------------------------------------------
    # REQUIRED METHODS (for VectorStore compatibility)
    # --------------------------------------------------
    def split(self, documents: List[Any]) -> List[Any]:
        """Alias for chunk_documents (expected by vector store)."""
        return self.chunk_documents(documents)

    def embed(self, chunks: List[Any]) -> np.ndarray:
        """Alias for embed_chunks (expected by vector store)."""
        return self.embed_chunks(chunks)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode(query)
