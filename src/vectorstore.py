import os
import faiss
import pickle
import numpy as np
from typing import List, Any
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = None
        self.metadata = []

        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # -------- BUILD --------
    def build_from_documents(self, documents: List[Any]):
        chunks = self.embedding_pipeline.split(documents)
        embeddings = self.embedding_pipeline.embed(chunks).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = [
            {"text": c.page_content, **c.metadata} for c in chunks
        ]

        self.save()
        print("[INFO] FAISS index built and saved")

    # -------- SAVE / LOAD --------
    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. Build it first."
            )

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print("[INFO] FAISS index loaded")

    # -------- QUERY --------
    def query(self, query_text: str, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")

        q_emb = self.embedding_pipeline.embed_query(query_text)
        q_emb = np.array([q_emb]).astype("float32")

        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    "score": float(dist),
                    "metadata": self.metadata[idx]
                })

        return results
